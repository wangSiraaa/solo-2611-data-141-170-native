"""MCMC 诊断核心：秩正态化 split/folded R-hat、bulk/tail ESS、均值 MCSE。

实现口径（遵循 Vehtari, Gelman, Simpson, Carpenter, Bürkner 2021,
"Rank-normalization, folding, and localization: An improved R-hat for
assessing convergence of MCMC"，核心代码由本项目自行实现，仅调用
NumPy/SciPy 数值原语）：

并列秩处理
    秩一律取“平均秩”（scipy.stats.rankdata, method="average"）：相同取值
    分得相同秩，再经 Blom 变换 z = Φ⁻¹((r − 3/8) / (S + 1/4)) 映射到正态
    分位。常量参数在进入秩变换前已被识别并赋予显式状态，不会产生 NaN 伪装。

split / folded R-hat
    秩正态化在全部链合并样本上进行，随后每条链对半 split（奇数长度丢弃最后
    一个样本）得到 2m 条半链，代入经典 Gelman 方差比公式。folded 变体先对
    |x − median(x)|（中位数取全体合并样本）再秩正态化、split，用于发现
    均值相近但尺度/形状不同的链间不一致。报告 rhat_rank、rhat_folded 及
    rhat_max = max(两者)，判定用 rhat_max。

ESS 与相关序列截断
    多链 ESS 采用链间合并的自相关估计：ρ_t = 1 − (W − mean_m acov_{m,t}) / var⁺，
    其中 var⁺ = (n−1)/n·W + B/n 与 R-hat 共用同一方差分解；自协方差用 FFT
    计算的有偏估计（除以 n）。截断采用 Geyer 初始正序列准则：对成对和
    Γ_k = ρ_{2k} + ρ_{2k+1} 在首个负值处截断，再强制单调不增（初始单调序列），
    τ = −1 + 2·ΣΓ_k，ESS = m·n/τ。对反相关链，ESS 上限取 m·n·log10(m·n)
    （与 Stan 相同的上界），τ ≤ 0 时直接取该上限。

bulk / tail ESS 与 MCSE
    bulk ESS：秩正态化 z 值的多链 ESS。
    tail ESS：对 p ∈ {5%, 95%}，用全体合并样本的经验分位数 q_p 构造指示
    序列 I(x ≤ q_p)，取两个指示序列多链 ESS 的较小者。指示序列为 0/1，
    无需秩变换；该口径衡量分布尾部（分位数）估计的有效信息量。
    均值 MCSE：sd(全体合并样本, ddof=1) / sqrt(ESS_raw)，ESS_raw 为原始
    （未秩变换）值的多链 ESS。

所有函数约定输入为 (m, n) 矩阵：m 条链 × 每条 n 个样本，链不得拼接。
样本不足（m < 1 或 n < MIN_CHAIN_LENGTH）或方差退化时返回 NaN，
由上层赋予显式状态，绝不把 NaN 当作“通过”。
"""

from __future__ import annotations

import numpy as np
from scipy import stats

MIN_CHAIN_LENGTH = 4
MIN_CHAINS = 2
TAIL_PROBS = (0.05, 0.95)


# ---------------------------------------------------------------- 秩变换

def rank_normalize(values: np.ndarray) -> np.ndarray:
    """平均秩 + Blom 变换 z = Φ⁻¹((r − 3/8)/(S + 1/4))，输入输出为一维数组。"""
    x = np.asarray(values, dtype=float).ravel()
    ranks = stats.rankdata(x, method="average")
    return stats.norm.ppf((ranks - 3.0 / 8.0) / (x.size + 0.25))


def split_half(x: np.ndarray) -> np.ndarray:
    """(m, n) -> (2m, n//2)，奇数长度丢弃每条链最后一个样本。"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("期望 (m, n) 二维矩阵")
    n2 = x.shape[1] // 2
    if n2 < 1:
        raise ValueError("链太短，无法 split")
    return np.concatenate([x[:, :n2], x[:, n2 : 2 * n2]], axis=0)


# ---------------------------------------------------------------- R-hat

def gelman_rhat(x: np.ndarray) -> float:
    """经典 Gelman 方差比 R-hat，输入 (m, n) 矩阵（调用方负责 split/变换）。

    W == 0：B > 0 时返回 +inf（各链各自常量但取值不同），B == 0 时返回 NaN
    （全体常量，R-hat 无定义）。
    """
    x = np.asarray(x, dtype=float)
    m, n = x.shape
    if m < 2 or n < 2:
        return float("nan")
    chain_means = x.mean(axis=1)
    chain_vars = x.var(axis=1, ddof=1)
    w = chain_vars.mean()
    b = n * chain_means.var(ddof=1)
    if w == 0.0:
        return float("inf") if b > 0.0 else float("nan")
    var_plus = (n - 1.0) / n * w + b / n
    return float(np.sqrt(var_plus / w))


def rank_normalized_split_rhat(x: np.ndarray) -> float:
    """秩正态化 split R-hat。"""
    x = np.asarray(x, dtype=float)
    z = rank_normalize(x.ravel()).reshape(x.shape)
    return gelman_rhat(split_half(z))


def folded_split_rhat(x: np.ndarray) -> float:
    """folded（|x − median|）秩正态化 split R-hat。"""
    x = np.asarray(x, dtype=float)
    folded = np.abs(x - np.median(x))
    z = rank_normalize(folded.ravel()).reshape(x.shape)
    return gelman_rhat(split_half(z))


# ---------------------------------------------------------------- ESS

def _autocovariance(x: np.ndarray) -> np.ndarray:
    """有偏自协方差（除以 n），FFT 计算，返回 lag 0..n-1。"""
    x = np.asarray(x, dtype=float)
    n = x.size
    xc = x - x.mean()
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xc, nfft)
    acov = np.fft.irfft(f * np.conjugate(f), nfft)[:n].real
    return acov / n


def ess_multichain(x: np.ndarray) -> float:
    """多链 ESS（Vehtari et al. 2021），输入 (m, n) 矩阵。

    方差退化（var⁺ ≤ 0，如全体常量）或样本不足时返回 NaN。
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("期望 (m, n) 二维矩阵")
    m, n = x.shape
    if m < 1 or n < MIN_CHAIN_LENGTH:
        return float("nan")
    chain_means = x.mean(axis=1)
    w = x.var(axis=1, ddof=1).mean()
    if m > 1:
        b = n * chain_means.var(ddof=1)
    else:
        b = 0.0
    var_plus = (n - 1.0) / n * w + b / n
    if not np.isfinite(var_plus) or var_plus <= 0.0:
        return float("nan")

    mean_acov = np.mean([_autocovariance(row) for row in x], axis=0)
    rho = 1.0 - (w - mean_acov) / var_plus
    rho[0] = 1.0

    # Geyer 初始正序列：成对和 Γ_k = ρ_{2k} + ρ_{2k+1}，首个负值处截断
    max_k = (n - 1) // 2
    gammas = rho[0 : 2 * max_k + 1 : 2] + rho[1 : 2 * max_k + 2 : 2]
    keep = len(gammas)
    for k in range(1, len(gammas)):  # Γ_0 含 ρ_0=1，恒为正，从 k=1 起判负
        if gammas[k] < 0.0:
            keep = k
            break
    gammas = gammas[:keep]
    # 初始单调序列：强制不增
    gammas = np.minimum.accumulate(gammas)
    tau = -1.0 + 2.0 * float(gammas.sum())

    ess_cap = m * n * np.log10(m * n)
    if tau <= 0.0:  # 强反相关，取上界
        return float(ess_cap)
    return float(min(m * n / tau, ess_cap))


def ess_bulk(x: np.ndarray) -> float:
    """bulk ESS：秩正态化 z 值的多链 ESS。"""
    x = np.asarray(x, dtype=float)
    z = rank_normalize(x.ravel()).reshape(x.shape)
    return ess_multichain(z)


def ess_tail(x: np.ndarray, probs: tuple[float, float] = TAIL_PROBS) -> float:
    """tail ESS：p ∈ {5%, 95%} 分位指示序列多链 ESS 的较小者。"""
    x = np.asarray(x, dtype=float)
    pooled = x.ravel()
    ess_values = []
    for p in probs:
        q = np.quantile(pooled, p)
        indicator = (x <= q).astype(float)
        ess_values.append(ess_multichain(indicator))
    return float(min(ess_values))


def mcse_mean(x: np.ndarray) -> float:
    """均值 MCSE = sd(合并样本, ddof=1) / sqrt(原始值多链 ESS)。"""
    x = np.asarray(x, dtype=float)
    ess = ess_multichain(x)
    if not np.isfinite(ess) or ess <= 0.0:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(ess))


# ---------------------------------------------------------------- 参数级汇总

def summarize_parameter(x: np.ndarray) -> dict:
    """对 (m, n) 矩阵计算某参数的全部指标与退化标志，不做通过/不通过判定。"""
    x = np.asarray(x, dtype=float)
    m, n = x.shape
    pooled = x.ravel()
    out = {
        "n_chains": m,
        "n_draws_per_chain": n,
        "mean": float("nan"),
        "sd": float("nan"),
        "q05": float("nan"),
        "q25": float("nan"),
        "q50": float("nan"),
        "q75": float("nan"),
        "q95": float("nan"),
        "rhat_rank": float("nan"),
        "rhat_folded": float("nan"),
        "rhat_max": float("nan"),
        "ess_bulk": float("nan"),
        "ess_tail": float("nan"),
        "ess_mean": float("nan"),
        "mcse_mean": float("nan"),
        "nonfinite": bool(~np.isfinite(pooled).all()),
        "all_constant": False,
        "chain_constant_differs": False,
    }
    if out["nonfinite"]:
        return out
    out["mean"] = float(pooled.mean())
    out["sd"] = float(pooled.std(ddof=1)) if pooled.size > 1 else 0.0
    q = np.quantile(pooled, [0.05, 0.25, 0.5, 0.75, 0.95])
    out["q05"], out["q25"], out["q50"], out["q75"], out["q95"] = (float(v) for v in q)

    within_vars = x.var(axis=1, ddof=1) if n > 1 else np.zeros(m)
    if float(pooled.var()) == 0.0:
        out["all_constant"] = True
        out["mcse_mean"] = 0.0  # 常量的均值无 Monte Carlo 误差
        return out
    if m > 1 and np.all(within_vars == 0.0):
        out["chain_constant_differs"] = True
        out["rhat_rank"] = out["rhat_folded"] = out["rhat_max"] = float("inf")
        return out
    if m < MIN_CHAINS or n < MIN_CHAIN_LENGTH:
        return out  # 样本不足：诊断指标保持 NaN

    out["rhat_rank"] = rank_normalized_split_rhat(x)
    out["rhat_folded"] = folded_split_rhat(x)
    out["rhat_max"] = max(out["rhat_rank"], out["rhat_folded"])
    out["ess_bulk"] = ess_bulk(x)
    out["ess_tail"] = ess_tail(x)
    out["ess_mean"] = ess_multichain(x)
    out["mcse_mean"] = mcse_mean(x)
    return out
