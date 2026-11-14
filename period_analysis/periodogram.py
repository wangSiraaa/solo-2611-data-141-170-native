"""频率网格、加权 chi2 周期图、局部细化与候选去重。

频率分辨率依据真实时间跨度 T = t_max - t_min：
Rayleigh 分辨率 df_R = 1/T。网格不用任何插值/补零后的规则时间轴，
只在真实观测时刻上构建设计矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import argrelextrema

from .model import FitResult, bic, weighted_fit

OVERSAMPLE = 10          # 初始网格步长 = 1 / (OVERSAMPLE * T)
REFINE_LEVELS = 3        # 细化轮数
REFINE_SHRINK = 5.0      # 每轮把搜索区间缩到上轮步长的 1/5
REFINE_STEPS = 41        # 每轮细化的网格点数
DEDUP_TOL_FACTOR = 1.0   # 去重容差 = 1 个 Rayleigh 分辨率
MAX_CANDIDATES = 12


@dataclass
class Candidate:
    freq: float
    period: float
    fits: dict[int, FitResult]                      # K -> 拟合结果（K=0 为常数，共享）
    preferred_harmonics: int
    chi2: float
    reduced_chi2: float
    bic: float
    delta_chi2_vs_constant: float
    freq_uncertainty: float                         # 局部 chi2 = chi2_min + 1 的半宽
    score_explained: float                          # 1 - chi2/chi2_const，非概率
    freq_grid_step: float
    labels: list[str] = field(default_factory=list)
    family_id: int | None = None
    half_period_pair: str | None = None
    holdout: dict | None = None

    def to_summary_dict(self) -> dict:
        fit = self.fits[self.preferred_harmonics]
        return {
            "freq_per_day": self.freq,
            "period_day": self.period,
            "preferred_harmonics": self.preferred_harmonics,
            "chi2": self.chi2,
            "dof": fit.dof,
            "reduced_chi2": self.reduced_chi2,
            "bic": self.bic,
            "delta_chi2_vs_constant": self.delta_chi2_vs_constant,
            "explained_score": self.score_explained,
            "freq_1sigma_halfwidth": self.freq_uncertainty,
            "period_1sigma": period_uncertainty(self.freq, self.freq_uncertainty),
            "constant": fit.constant,
            "harmonics": [h.to_dict() for h in fit.harmonic_terms()],
            "labels": self.labels,
            "alias_family_id": self.family_id,
            "half_period_pair_candidate_id": self.half_period_pair,
            "holdout": self.holdout,
            "fit_by_harmonics": {
                str(k): {
                    "chi2": f.chi2,
                    "dof": f.dof,
                    "reduced_chi2": f.reduced_chi2,
                    "bic": bic(f.chi2, f.residual.size, f.n_params),
                    "coefs": f.coefs.tolist(),
                }
                for k, f in self.fits.items() if k > 0
            },
        }


def period_uncertainty(freq: float, freq_halfwidth: float) -> float:
    """一阶传播 sigma_P = sigma_f / f^2。"""
    return float(freq_halfwidth / freq**2)


def frequency_grid(f_min: float, f_max: float, time_span: float,
                   oversample: int = OVERSAMPLE) -> np.ndarray:
    """均匀频率网格，步长 1/(oversample*T)；端点按 bounds 裁剪。"""
    step = 1.0 / (oversample * time_span)
    n = int(np.ceil((f_max - f_min) / step)) + 1
    grid = f_min + step * np.arange(n)
    return grid[grid <= f_max + 1e-12]


def chi2_periodogram(t, y, sigma, grid, n_harmonics: int, t0: float) -> np.ndarray:
    """逐频率加权拟合，返回 chi2 数组（不做 FFT、不做时间插补）。"""
    out = np.empty(grid.size)
    for i, f in enumerate(grid):
        out[i] = weighted_fit(t, y, sigma, float(f), n_harmonics, t0).chi2
    return out


def _local_refine(t, y, sigma, f0, half_width, n_harmonics, t0) -> tuple[float, float, float]:
    """在 f0 附近逐级细化，返回 (最佳频率, 最小chi2, 最终网格步长)。"""
    lo, hi = f0 - half_width, f0 + half_width
    best_f, best_chi2, step = f0, np.inf, (hi - lo)
    for _ in range(REFINE_LEVELS):
        step = (hi - lo) / (REFINE_STEPS - 1)
        grid = np.linspace(lo, hi, REFINE_STEPS)
        chis = chi2_periodogram(t, y, sigma, grid, n_harmonics, t0)
        j = int(np.argmin(chis))
        best_f, best_chi2 = float(grid[j]), float(chis[j])
        lo = max(grid[0], best_f - step * REFINE_SHRINK / 2)
        hi = min(grid[-1], best_f + step * REFINE_SHRINK / 2)
    return best_f, best_chi2, step


def freq_uncertainty_chi2(t, y, sigma, f_best, chi2_min, n_harmonics, t0,
                          rayleigh: float) -> float:
    """沿频率轴找 chi2 = chi2_min + 1 的半宽（Brent 求根），上限截到 1 个 Rayleigh。"""
    from scipy.optimize import brentq

    target = chi2_min + 1.0

    def delta(f):
        return weighted_fit(t, y, sigma, float(f), n_harmonics, t0).chi2 - target

    widths = []
    for direction in (-1.0, 1.0):
        try:
            edge = f_best + direction * rayleigh
            if delta(edge) <= 0:          # 一整个 Rayleigh 内仍未升到 +1
                widths.append(rayleigh)
            else:
                root = brentq(delta, f_best, edge, xtol=1e-10, rtol=1e-12)
                widths.append(abs(root - f_best))
        except ValueError:
            widths.append(rayleigh)
    return float(min(max(widths), rayleigh))


def find_candidates(t, y, sigma, period_bounds, t0,
                    max_harmonics: int = 2) -> tuple[list[Candidate], FitResult, dict]:
    """完整候选搜索：粗网格（按最高谐波模型）→ 局部极小 → 细化 → 去重 → 谐波选择。"""
    p_lo, p_hi = float(period_bounds[0]), float(period_bounds[1])
    f_min, f_max = 1.0 / p_hi, 1.0 / p_lo
    T = float(t.max() - t.min())
    rayleigh = 1.0 / T
    grid = frequency_grid(f_min, f_max, T)

    # 用参数最多（K=2）的模型定峰，避免漏谐波；K=1 随后在同频率重拟合
    scan_h = min(max_harmonics, 2)
    chis = chi2_periodogram(t, y, sigma, grid, scan_h, t0)
    minima = argrelextrema(chis, np.less)[0].tolist()
    if not minima:
        minima = [int(np.argmin(chis))]
    # 粗网格上按 chi2 取前若干局部极小进入细化
    minima.sort(key=lambda i: chis[i])
    minima = minima[: MAX_CANDIDATES * 3]

    refined: list[tuple[float, float, float]] = []
    coarse_step = grid[1] - grid[0]
    for i in minima:
        f_best, chi2_best, fine_step = _local_refine(
            t, y, sigma, float(grid[i]), coarse_step, scan_h, t0
        )
        refined.append((f_best, chi2_best, fine_step))

    # 去重：间距小于 1 个 Rayleigh 的峰合并，保留 chi2 更小者
    refined.sort(key=lambda z: z[1])
    kept: list[tuple[float, float, float]] = []
    for f_best, chi2_best, fine_step in refined:
        if any(abs(f_best - fk) < DEDUP_TOL_FACTOR * rayleigh for fk, _, _ in kept):
            continue
        kept.append((f_best, chi2_best, fine_step))
        if len(kept) >= MAX_CANDIDATES:
            break

    const_fit = weighted_fit(t, y, sigma, 0.0, 0, t0)
    candidates: list[Candidate] = []
    for f_best, _, fine_step in kept:
        fits: dict[int, FitResult] = {0: const_fit}
        for k in range(1, max_harmonics + 1):
            # 在已细化频率附近为每个 K 再做一轮小范围细化
            fk, _, _ = _local_refine(t, y, sigma, f_best, fine_step * 3, k, t0)
            fits[k] = weighted_fit(t, y, sigma, fk, k, t0)
        bics = {k: bic(f.chi2, t.size, f.n_params) for k, f in fits.items() if k > 0}
        k_pref = min(bics, key=bics.get)
        pref = fits[k_pref]
        fu = freq_uncertainty_chi2(t, y, sigma, pref.freq, pref.chi2, k_pref, t0, rayleigh)
        candidates.append(
            Candidate(
                freq=pref.freq,
                period=1.0 / pref.freq,
                fits=fits,
                preferred_harmonics=k_pref,
                chi2=pref.chi2,
                reduced_chi2=pref.reduced_chi2,
                bic=bics[k_pref],
                delta_chi2_vs_constant=const_fit.chi2 - pref.chi2,
                freq_uncertainty=fu,
                score_explained=1.0 - pref.chi2 / const_fit.chi2,
                freq_grid_step=fine_step,
            )
        )
    candidates.sort(key=lambda c: c.chi2)
    meta = {
        "time_span_day": T,
        "rayleigh_resolution_per_day": rayleigh,
        "coarse_grid_step_per_day": coarse_step,
        "grid_oversample": OVERSAMPLE,
        "n_grid": int(grid.size),
        "f_min_per_day": f_min,
        "f_max_per_day": f_max,
        "dedup_tolerance_per_day": DEDUP_TOL_FACTOR * rayleigh,
    }
    return candidates, const_fit, meta
