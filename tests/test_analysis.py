"""可复算性与采样行为测试。

运行：.venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
import unittest

import numpy as np

from period_analysis.holdout import leave_one_night_out
from period_analysis.io_utils import load_lightcurve
from period_analysis.model import design_matrix, weighted_fit
from period_analysis.periodogram import chi2_periodogram, find_candidates
from period_analysis.window import spectral_window, window_peaks

FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures")
BOUNDS = (0.35, 5.8)


def load(case):
    return load_lightcurve(os.path.join(FIX, f"{case}.csv"))


class TestIndependentDesignMatrix(unittest.TestCase):
    """在少数频率直接构建设计矩阵，独立核对加权残差与参数协方差。"""

    def test_residuals_recomputed_from_design_matrix(self):
        lc = load("spread")
        t, y, s = lc.time, lc.brightness, lc.sigma
        t0 = float(np.mean(t))
        for f in (0.36621, 0.5, 1.35883):
            for k in (1, 2):
                fit = weighted_fit(t, y, s, f, k, t0)
                X = design_matrix(t, f, k, t0)
                w = 1.0 / s**2
                # 独立正规方程求解
                beta = np.linalg.solve((X * w[:, None]).T @ X, (X * w[:, None]).T @ y)
                r = y - X @ beta
                chi2_indep = float(r @ (w * r))
                self.assertTrue(np.allclose(beta, fit.coefs, rtol=1e-10, atol=1e-12))
                self.assertAlmostEqual(chi2_indep, fit.chi2, places=8)
                self.assertTrue(np.allclose(r, fit.residual, rtol=1e-10, atol=1e-12))
                # 协方差即 (X'WX)^-1，常数项方差可独立核对
                cov_indep = np.linalg.inv((X * w[:, None]).T @ X)
                self.assertTrue(np.allclose(cov_indep, fit.cov, rtol=1e-10))

    def test_astropy_lombscargle_agrees_for_single_harmonic(self):
        """K=1 的加权拟合频率应与 astropy LombScargle（nterms=1）峰位一致。"""
        from astropy.timeseries import LombScargle

        lc = load("spread")
        t, y, s = lc.time, lc.brightness, lc.sigma
        t0 = float(np.mean(t))
        grid = np.arange(1 / BOUNDS[1], 1 / BOUNDS[0], 1 / (10 * (t.max() - t.min())))
        mine = chi2_periodogram(t, y, s, grid, 1, t0)
        ls = LombScargle(t, y, dy=s, normalization="standard")
        # nterms=1 + 逐点误差，astropy auto 选择 cython 实现；标准归一化峰位与最小 chi2 同位置
        power = ls.power(grid, assume_regular_frequency=True)
        self.assertAlmostEqual(grid[np.argmin(mine)], grid[np.argmax(power)], places=4)

    def test_frequency_grid_resolution_follows_time_span(self):
        lc = load("spread")
        T = lc.time_span
        cands, _, meta = find_candidates(lc.time, lc.brightness, lc.sigma,
                                         BOUNDS, float(np.mean(lc.time)))
        self.assertAlmostEqual(meta["rayleigh_resolution_per_day"], 1 / T, places=12)
        self.assertAlmostEqual(meta["coarse_grid_step_per_day"], 1 / (10 * T), places=12)
        # 去重后任意两候选间距至少接近 1 个 Rayleigh
        fs = sorted(c.freq for c in cands)
        gaps = np.diff(fs)
        self.assertGreater(gaps.min(), 0.9 * meta["rayleigh_resolution_per_day"])


class TestExactDailySamplingAlias(unittest.TestCase):
    """精确每日采样小例：别名必须显式出现，不能被流程隐藏。"""

    def _make_exact_daily(self, n_nights=14, pts_per_night=4):
        rng = np.random.default_rng(42)
        f0 = 0.42                     # 真实频率 /天
        rows = []
        for n in range(n_nights):
            # 每晚同一时刻 0.30 天，加极小幅抖动模拟同刻密集点
            for j in range(pts_per_night):
                tt = n + 0.30 + j * 0.002
                y = 10.0 + 1.2 * np.cos(2 * np.pi * f0 * tt) + 0.4 * np.sin(2 * np.pi * 2 * f0 * tt)
                sig = 0.03
                rows.append((f"d-{n:02d}-{j}", f"n{n:02d}", tt, y + rng.normal(0, sig), sig))
        return rows

    def test_alias_peaks_visible_and_linked(self):
        rows = self._make_exact_daily()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "exact_daily.csv")
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["observation_id", "night_id", "time_day", "brightness", "sigma"])
                w.writerows(rows)
            lc = load_lightcurve(path)
            self.assertTrue(lc.clean)

            # 谱窗在 ±1/天 处应接近 1（精确每日采样的指纹）
            nu, power = spectral_window(lc.time)
            pk = {round(p.freq_per_day, 2): p.power for p in window_peaks(nu, power)}
            self.assertGreater(pk.get(1.0, 0.0), 0.95)
            self.assertGreater(pk.get(-1.0, 0.0), 0.95)

            cands, _, meta = find_candidates(lc.time, lc.brightness, lc.sigma,
                                             BOUNDS, float(np.mean(lc.time)))
            # 真实频率与 f0+1/天 都应在候选集中（容差 1 Rayleigh）
            fs = np.array([c.freq for c in cands])
            self.assertTrue(np.min(np.abs(fs - 0.42)) < meta["rayleigh_resolution_per_day"])
            self.assertTrue(np.min(np.abs(fs - 1.42)) < meta["rayleigh_resolution_per_day"])
            # 两候选 chi2 高度接近（每日别名近线性相关）
            c_true = cands[int(np.argmin(np.abs(fs - 0.42)))]
            c_alias = cands[int(np.argmin(np.abs(fs - 1.42)))]
            self.assertLess(abs(c_true.chi2 - c_alias.chi2) / max(c_true.chi2, 1e-9), 0.2)


class TestDataQualityAndHoldout(unittest.TestCase):
    def _write_bad_csv(self, td):
        path = os.path.join(td, "bad.csv")
        good = load("spread")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["observation_id", "night_id", "time_day", "brightness", "sigma"])
            for i in range(good.n_points):
                w.writerow([good.observation_id[i], good.night_id[i],
                            good.time[i], good.brightness[i], good.sigma[i]])
            # 重复 id
            w.writerow([good.observation_id[0], "n99", 12.0, 12.0, 0.05])
            # 非有限亮度、非正误差
            w.writerow(["bad-nan", "n99", 12.1, "nan", 0.05])
            w.writerow(["bad-sig", "n99", 12.2, 12.0, -0.01])
        return path

    def test_issues_located_by_row_and_column(self):
        with tempfile.TemporaryDirectory() as td:
            lc = load_lightcurve(self._write_bad_csv(td))
            self.assertFalse(lc.clean)
            kinds = {(i.kind, i.column, i.csv_row) for i in lc.issues}
            self.assertIn(("duplicate_id", "observation_id", 146), kinds)
            self.assertIn(("non_finite", "brightness", 147), kinds)
            self.assertIn(("non_positive_sigma", "sigma", 148), kinds)

    def test_holdout_uses_whole_nights_and_covers_all_points(self):
        lc = load("half_period")
        cands, _, _ = find_candidates(lc.time, lc.brightness, lc.sigma,
                                      BOUNDS, float(np.mean(lc.time)))
        ho = leave_one_night_out(lc, cands[0])
        self.assertEqual(ho["n_folds"], 12)
        # 每折训练集恰好排除整条留出夜；覆盖点数总和 = 全部观测
        covered = sum(f["n_points"] for f in ho["per_night"])
        train_sum = sum(f["n_train_points"] for f in ho["per_night"])
        self.assertEqual(covered, lc.n_points)
        self.assertEqual(train_sum, 12 * (lc.n_points - 12))
        night_sizes = {nid: len(idx) for nid, idx in lc.night_groups().items()}
        for f in ho["per_night"]:
            self.assertEqual(f["n_train_points"], lc.n_points - night_sizes[f["night_id"]])

    def test_half_period_equivalence_on_fixture(self):
        lc = load("half_period")
        cands, _, meta = find_candidates(lc.time, lc.brightness, lc.sigma,
                                         BOUNDS, float(np.mean(lc.time)))
        for c in cands:
            c.holdout = leave_one_night_out(lc, c)
        from period_analysis.holdout import mark_half_period_pairs
        pairs = mark_half_period_pairs(cands, meta["rayleigh_resolution_per_day"])
        near = [p for p in pairs if p["near_equivalent"]]
        self.assertEqual(len(near), 1)
        p = near[0]
        self.assertAlmostEqual(p["period_long_day"] / p["period_short_day"], 2.0, delta=0.05)
        self.assertLess(p["fundamental_to_second_amplitude_ratio_long_model"], 0.25)


if __name__ == "__main__":
    unittest.main()
