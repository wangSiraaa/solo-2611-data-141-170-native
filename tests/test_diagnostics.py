"""诊断核心测试：方向性核查（独立短链 / 卡在不同均值的链）、
小数组独立复算核对 R-hat 与 ESS、arviz 独立库核对、退化输入状态。"""

import numpy as np
import pytest

from mcmc_diag import diagnostics as dg


# ---------------------------------------------------------------- 方向性核查

def test_iid_chains_look_healthy():
    """独立生成的短独立链：R-hat≈1，ESS≈N，MCSE≈sd/√N。"""
    rng = np.random.default_rng(42)
    x = rng.normal(size=(4, 1000))
    assert dg.rank_normalized_split_rhat(x) <= 1.01
    assert dg.folded_split_rhat(x) <= 1.01
    ess = dg.ess_bulk(x)
    assert 0.6 * x.size < ess < 1.3 * x.size
    assert dg.mcse_mean(x) == pytest.approx(x.std(ddof=1) / np.sqrt(x.size), rel=0.35)


def test_chains_stuck_at_different_means_flagged():
    """刻意卡在不同均值的链：R-hat 显著超限。"""
    rng = np.random.default_rng(0)
    x = np.array([rng.normal(-3.0, 0.2, 500), rng.normal(-3.0, 0.2, 500),
                  rng.normal(3.0, 0.2, 500), rng.normal(3.0, 0.2, 500)])
    assert dg.rank_normalized_split_rhat(x) > 1.5
    s = dg.summarize_parameter(x)
    assert s["rhat_max"] > 1.5


def test_folded_rhat_catches_scale_difference():
    """均值相同、尺度不同的链：秩 R-hat 不敏感，folded R-hat 升高。"""
    rng = np.random.default_rng(7)
    x = np.array([rng.normal(0, 1, 800), rng.normal(0, 1, 800),
                  rng.normal(0, 3, 800), rng.normal(0, 3, 800)])
    rank_rhat = dg.rank_normalized_split_rhat(x)
    folded = dg.folded_split_rhat(x)
    assert rank_rhat < 1.05
    assert folded > 1.1
    assert folded > rank_rhat


def test_strong_autocorrelation_shrinks_ess():
    """强自相关 AR(1)：ESS 远小于样本数。"""
    rng = np.random.default_rng(1)

    def ar1(rho, n):
        e = rng.normal(size=n)
        out = np.empty(n)
        out[0] = e[0]
        for t in range(1, n):
            out[t] = rho * out[t - 1] + np.sqrt(1 - rho**2) * e[t]
        return out

    x = np.array([ar1(0.98, 2000) for _ in range(4)])
    ess = dg.ess_bulk(x)
    assert ess < 0.05 * x.size


def test_antithetic_chain_ess_capped():
    """完全反相关序列：ESS 取上限 m·n·log10(m·n)，而非无穷或 NaN。"""
    x = np.tile([1.0, -1.0], 500)[None, :]
    ess = dg.ess_multichain(x)
    assert ess == pytest.approx(x.size * np.log10(x.size))


# ---------------------------------------------------------------- 小数组独立复算

def test_rhat_matches_hand_computation():
    """与测试内独立手算的经典 Gelman R-hat 对比（不同代码路径）。"""
    x = np.array([[1.0, 2.0, 3.0, 4.0],
                  [2.0, 3.0, 4.0, 5.0],
                  [0.0, 1.0, 2.0, 3.0]])
    m, n = x.shape
    means = [sum(row) / n for row in x]
    variances = [sum((v - means[i]) ** 2 for v in x[i]) / (n - 1) for i in range(m)]
    W = sum(variances) / m
    grand = sum(means) / m
    B = n * sum((mm - grand) ** 2 for mm in means) / (m - 1)
    var_plus = (n - 1) / n * W + B / n
    expected = (var_plus / W) ** 0.5
    assert dg.gelman_rhat(x) == pytest.approx(expected, rel=1e-12)


def test_ess_matches_brute_force_small_array():
    """与测试内显式循环（非 FFT）的 ESS 复算对比，同一统计口径。"""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 30)) + np.array([ar_noise(rng) for _ in range(2)])

    def brute_ess(x):
        m, n = x.shape
        W = float(np.mean([np.var(row, ddof=1) for row in x]))
        B = n * float(np.var(x.mean(axis=1), ddof=1))
        var_plus = (n - 1) / n * W + B / n
        acovs = []
        for row in x:
            xc = row - row.mean()
            # 有偏自协方差（除以 n），与 Stan 相同口径，但用显式循环而非 FFT
            acovs.append([float(np.sum(xc[: n - t] * xc[t:])) / n for t in range(n)])
        mean_acov = np.mean(acovs, axis=0)
        rho = 1.0 - (W - mean_acov) / var_plus
        rho[0] = 1.0  # lag 0 自相关按定义为 1
        gammas = [rho[2 * k] + rho[2 * k + 1] for k in range((n - 1) // 2)]
        keep = len(gammas)
        for k in range(1, len(gammas)):
            if gammas[k] < 0:
                keep = k
                break
        g = np.minimum.accumulate(gammas[:keep])
        tau = -1.0 + 2.0 * float(g.sum())
        ess = m * n / tau
        return min(ess, m * n * np.log10(m * n))

    assert dg.ess_multichain(x) == pytest.approx(brute_ess(x), rel=1e-10)


def ar_noise(rng, n=30, rho=0.5):
    e = rng.normal(size=n)
    out = np.empty(n)
    out[0] = e[0]
    for t in range(1, n):
        out[t] = rho * out[t - 1] + e[t]
    return out


# ---------------------------------------------------------------- arviz 独立库核对

def test_cross_check_with_arviz():
    """用独立库 arviz 核对 R-hat（精确）与 ESS（同量级、阈值同侧）。"""
    az = pytest.importorskip("arviz")
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(11)
    datasets = {
        "iid": rng.normal(size=(4, 600)),
        "ar1": np.array([ar_noise(rng, n=600, rho=0.9) for _ in range(4)]),
        "shifted": rng.normal(size=(4, 600)) + np.array([[0.0], [0.0], [2.0], [2.0]]),
    }
    for name, x in datasets.items():
        da = xr.DataArray(x, dims=("chain", "draw"))
        az_rhat = float(next(iter(az.rhat(da).data_vars.values())).values)
        az_bulk = float(next(iter(az.ess(da, method="bulk").data_vars.values())).values)
        # arviz 的 rhat = max(秩正态化 split, folded split)，与本文 rhat_max 同口径
        assert dg.summarize_parameter(x)["rhat_max"] == pytest.approx(az_rhat, abs=2e-3), name
        # ESS 截断细节不同（本文不对链 split，arviz 先 split 链），病态数据上绝对值
        # 可有数倍差异，但应同量级且对 400 阈值同侧
        ours = dg.ess_bulk(x)
        assert 0.3 <= ours / az_bulk <= 3.0, name
        assert (ours >= 400) == (az_bulk >= 400), name


# ---------------------------------------------------------------- 退化与边界输入

def test_constant_same_value_everywhere():
    x = np.full((3, 100), 2.5)
    s = dg.summarize_parameter(x)
    assert s["all_constant"] is True
    assert np.isnan(s["rhat_max"]) and np.isnan(s["ess_bulk"])
    assert s["mcse_mean"] == 0.0
    assert s["mean"] == 2.5


def test_constant_differs_across_chains():
    x = np.vstack([np.full(100, 1.0), np.full(100, 2.0), np.full(100, 3.0)])
    s = dg.summarize_parameter(x)
    assert s["chain_constant_differs"] is True
    assert s["rhat_max"] == float("inf")


def test_nonfinite_flagged_not_silently_passed():
    x = np.random.default_rng(0).normal(size=(4, 100))
    x[1, 10] = np.nan
    s = dg.summarize_parameter(x)
    assert s["nonfinite"] is True
    assert np.isnan(s["rhat_max"]) and np.isnan(s["ess_bulk"]) and np.isnan(s["mcse_mean"])


def test_insufficient_samples_give_nan_not_pass():
    x = np.random.default_rng(0).normal(size=(4, 3))  # 每链 3 个，不足 split
    s = dg.summarize_parameter(x)
    assert np.isnan(s["rhat_max"]) and np.isnan(s["ess_bulk"])


def test_split_half_drops_odd_last_draw():
    x = np.arange(15, dtype=float).reshape(3, 5)
    sp = dg.split_half(x)
    assert sp.shape == (6, 2)
    np.testing.assert_array_equal(sp[0], [0, 1])
    np.testing.assert_array_equal(sp[3], [2, 3])  # 丢弃了每链最后的 4


def test_rank_normalize_average_ties():
    z = dg.rank_normalize(np.array([1.0, 1.0, 1.0, 2.0, 3.0]))
    assert z[0] == z[1] == z[2]  # 并列取平均秩，映射到同一 z
    assert np.all(np.isfinite(z))


def test_tail_ess_iid_close_to_n():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 1000))
    assert dg.ess_tail(x) > 0.4 * x.size
