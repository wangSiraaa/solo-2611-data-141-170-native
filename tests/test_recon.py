"""独立测试：tiny 手算路径长度、几何约定、诊断与可辨识性。

运行：.venv/bin/python -m unittest discover -s tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import nnls

from recon2d.dataio import load_grid, load_rays_csv, process_geometry
from recon2d.diagnostics import (angle_coverage, direction_probes,
                                 nullspace_energy_map, probe_visibility,
                                 weighted_svd)
from recon2d.geometry import (GeometryError, RayGeometry, build_forward_operator,
                              clip_ray_to_region, grid_from_dict,
                              ray_cell_segments)
from recon2d.regularize import neighbor_difference_matrix, prior_selector
from recon2d.solve import reconstruct
from recon2d.verify import (check_segment_length_sum, cross_check_operator,
                            independent_prediction, sampled_cell_lengths)

ROOT = Path(__file__).resolve().parent.parent
TINY_GRID = ROOT / "fixtures" / "tiny" / "grid.json"
TINY_RAYS = ROOT / "fixtures" / "tiny" / "rays.csv"
MAIN_GRID = ROOT / "fixtures" / "grid.json"
MAIN_RAYS = ROOT / "fixtures" / "rays.csv"
LIMITED_RAYS = ROOT / "fixtures" / "limited-rays.csv"


def seg(src, dst, grid):
    return ray_cell_segments(np.asarray(src, float), np.asarray(dst, float),
                             grid)


class TinyHandCalculation(unittest.TestCase):
    """2x2、dx=dy=1、mu=1 的微型场景，路径长度全部可手算。"""

    def setUp(self):
        self.grid = load_grid(TINY_GRID)
        # 单元扁平索引：0=row0,col0 (左下) 1=row0,col1 (右下)
        #               2=row1,col0 (左上) 3=row1,col1 (右上)
        self.ids, self.ep, self.atten, self.sigma, diag = \
            load_rays_csv(TINY_RAYS)
        self.assertEqual(diag, [])
        self.geoms, gdiag = process_geometry(self.ids, self.ep, self.grid)
        self.assertEqual(gdiag, [])
        self.by_id = {g.ray_id: g for g in self.geoms}

    def test_horizontal_paths(self):
        # T1 y=0.5：仅穿过 row0 两个格，各长 1，总长 2
        s = self.by_id["T1"].segments
        self.assertEqual(s, {0: 1.0, 1: 1.0})
        # T2 y=1.5：仅穿过 row1 两个格，各长 1
        s = self.by_id["T2"].segments
        self.assertEqual(s, {2: 1.0, 3: 1.0})

    def test_vertical_paths(self):
        # T3 x=0.5：仅穿过 col0 两个格，各长 1
        s = self.by_id["T3"].segments
        self.assertEqual(s, {0: 1.0, 2: 1.0})
        # T4 x=1.5：仅穿过 col1 两个格，各长 1
        s = self.by_id["T4"].segments
        self.assertEqual(s, {1: 1.0, 3: 1.0})

    def test_diagonal_path_no_double_count_at_corners(self):
        # T5 y=x，从 (-1,-1) 到 (3,3)：区域内 (0,0)->(2,2)，长 2sqrt2。
        # 穿过内部角点 (1,1) 时不双计；两段各 sqrt2，分别落在对角两格。
        s = self.by_id["T5"].segments
        h = np.sqrt(2.0)
        self.assertEqual(set(s), {0, 3})
        self.assertAlmostEqual(s[0], h, places=12)
        self.assertAlmostEqual(s[3], h, places=12)
        self.assertAlmostEqual(sum(s.values()), 2 * h, places=12)

    def test_manual_density_system_recovers_known_solution(self):
        # 手算方程组（mu=1，衰减=路径长度·密度）：
        # T1: r00+r01=3  T2: r10+r11=7  T3: r00+r10=4  T4: r01+r11=6
        # T5: sqrt2*(r00+r11)=4*sqrt2 -> r00+r11=4
        # 解得 r00=1, r01=2, r10=3, r11=4
        A, zero = build_forward_operator(self.geoms, self.grid)
        self.assertFalse(zero.any())
        B = A.toarray()
        x, _ = nnls(B, self.atten / self.grid.mass_attenuation)
        np.testing.assert_allclose(x, [1, 2, 3, 4], atol=1e-8)
        # 工具自身的 lambda=0 重建同样应恢复
        D = neighbor_difference_matrix(2, 2)
        P = prior_selector(np.ones(4, dtype=bool))
        res = reconstruct(A, zero, self.atten, self.sigma,
                          self.grid.mass_attenuation, D, P, 0.0)
        self.assertTrue(res.success)
        np.testing.assert_allclose(res.density, [1, 2, 3, 4], atol=1e-6)
        np.testing.assert_allclose(res.predicted, self.atten, atol=1e-6)


class GridLineAndCornerConvention(unittest.TestCase):
    def setUp(self):
        self.grid = load_grid(MAIN_GRID)  # 6x6, 1m 格

    def test_internal_vertical_gridline_assigned_to_higher_x_side(self):
        # 沿 x=3 的内部网格线：全部长度应归入 col=3，col=2 为 0
        s = seg((3.0, -2.0), (3.0, 9.0), self.grid)
        cols = {self.grid.rc_of(idx)[1] for idx in s}
        self.assertEqual(cols, {3})
        self.assertAlmostEqual(sum(s.values()), 6.0, places=10)

    def test_internal_horizontal_gridline_assigned_to_higher_y_side(self):
        # 沿 y=4 的内部网格线：全部长度归入 row=4
        s = seg((-2.0, 4.0), (9.0, 4.0), self.grid)
        rows = {self.grid.rc_of(idx)[0] for idx in s}
        self.assertEqual(rows, {4})
        self.assertAlmostEqual(sum(s.values()), 6.0, places=10)

    def test_diagonal_through_corners_counts_each_cell_once(self):
        # y=x 从区域外穿过 (0,0),(1,1)...(6,6)：6 个对角格，每格 sqrt2
        s = seg((-1.0, -1.0), (7.0, 7.0), self.grid)
        h = np.sqrt(2.0)
        self.assertEqual(len(s), 6)
        for k in range(6):
            idx = self.grid.cell_index(k, k)
            self.assertAlmostEqual(s[idx], h, places=10)
        self.assertAlmostEqual(sum(s.values()), 6 * h, places=10)

    def test_boundary_lines_are_closed_outer_edges(self):
        # 沿外边界 x=0（闭）：归入最外侧 col=0
        s = seg((0.0, 0.5), (0.0, 5.5), self.grid)
        self.assertEqual(set(s), {self.grid.cell_index(r, 0)
                                  for r in range(6)})

    def test_generic_ray_segment_sum_equals_clipped_length(self):
        # 任意斜射线：各段长度之和 == 裁切段长度，且单元索引合法
        src, dst = np.array([-2.7, -1.7]), np.array([10.4, 3.1])
        s = seg(src, dst, self.grid)
        entry, exit_ = clip_ray_to_region(src, dst, self.grid)
        clipped = np.hypot(*(exit_ - entry))
        self.assertAlmostEqual(sum(s.values()), clipped, places=10)
        for idx in s:
            r, c = self.grid.rc_of(idx)
            self.assertTrue(0 <= r < 6 and 0 <= c < 6)


class DisjointAndDegenerateRays(unittest.TestCase):
    def setUp(self):
        self.grid = load_grid(MAIN_GRID)

    def test_completely_disjoint_ray_is_empty_not_full_sweep(self):
        # R045: x=8 的竖线，区域 x in [0,6]，完全不相交
        s = seg((8.0, 1.0), (8.0, 6.0), self.grid)
        self.assertEqual(s, {})

    def test_disjoint_ray_becomes_zero_row_in_operator(self):
        geoms = [
            RayGeometry("ok", np.array([-1.0, 0.5]), np.array([7.0, 0.5]),
                        seg((-1, 0.5), (7, 0.5), self.grid), 0.0, 0.0),
            RayGeometry("miss", np.array([8.0, 1.0]), np.array([8.0, 6.0]),
                        {}, 0.0, 0.0)]
        geoms[0].clipped_length = sum(geoms[0].segments.values())
        A, zero = build_forward_operator(geoms, self.grid)
        self.assertTrue(zero[1])
        self.assertFalse(zero[0])
        self.assertEqual(A[1].nnz, 0)          # 全零行，而非穿满区域的观测
        self.assertEqual(A[0].nnz, 6)

    def test_corner_tangent_only_is_disjoint(self):
        # 仅在角点 (6,6) 接触：内部长度 0，按不相交处理
        s = seg((7.0, 7.0), (6.0, 6.0), self.grid)
        self.assertEqual(s, {})

    def test_zero_length_ray_raises(self):
        with self.assertRaises(GeometryError):
            seg((1.0, 1.0), (1.0, 1.0), self.grid)

    def test_nonfinite_endpoints_raise(self):
        with self.assertRaises(GeometryError):
            seg((np.nan, 0.0), (1.0, 1.0), self.grid)
        with self.assertRaises(GeometryError):
            seg((0.0, np.inf), (1.0, 1.0), self.grid)

    def test_bad_grid_and_unknown_geometry_raise(self):
        with self.assertRaises(GeometryError):
            grid_from_dict({"x_min_m": 0, "y_min_m": 0, "dx_m": 1,
                            "dy_m": 1, "nx": 0, "ny": 6,
                            "mass_attenuation_m2_per_kg": 0.02})
        with self.assertRaises(GeometryError):
            grid_from_dict({"x_min_m": 0, "y_min_m": 0, "dx_m": -1,
                            "dy_m": 1, "nx": 6, "ny": 6,
                            "mass_attenuation_m2_per_kg": 0.02})
        with self.assertRaises(GeometryError):
            grid_from_dict({"x_min_m": 0, "y_min_m": 0, "dx_m": 1,
                            "dy_m": 1, "nx": 13, "ny": 6,
                            "mass_attenuation_m2_per_kg": 0.02})

    def test_csv_nonfinite_and_bad_sigma_rows_are_diagnosed(self):
        content = (
            "ray_id,source_x_m,source_y_m,receiver_x_m,"
            "receiver_y_m,attenuation,sigma\n"
            "B1,0,0,1,1,0.1,0.001\n"
            "B2,nan,0,1,1,0.1,0.001\n"
            "B3,0,0,1,1,0.1,-0.002\n"
            "B4,0,0,1,1,inf,0.001\n"
            "B5,0,0,1,1,0.1,0.001\n")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.csv"
            p.write_text(content, encoding="utf-8")
            ids, ep, atten, sigma, diag = load_rays_csv(p)
        self.assertEqual(ids, ["B1", "B5"])
        self.assertEqual(len(diag), 3)
        self.assertTrue(all("已剔除" in d["issue"]) for d in diag)


class IndependentCrossCheck(unittest.TestCase):
    def setUp(self):
        self.grid = load_grid(MAIN_GRID)
        ids, ep, atten, sigma, _ = load_rays_csv(MAIN_RAYS)
        self.atten, self.sigma = atten, sigma
        self.geoms, _ = process_geometry(ids, ep, self.grid)

    def test_segment_length_self_consistency(self):
        reports = check_segment_length_sum(self.geoms)
        worst = max(r["rel_diff"] for r in reports if r["status"] == "ok")
        self.assertLess(worst, 1e-12)

    def test_sampled_integration_matches_event_method(self):
        reports, worst = cross_check_operator(self.grid, self.geoms,
                                              steps_per_cell=2000)
        self.assertLess(worst, 1e-3)
        # 不相交射线在两套实现里都应为空
        miss = [r for r in reports if r["status"] == "disjoint"]
        self.assertEqual(len(miss), 1)

    def test_independent_prediction_path_agrees(self):
        A, zero = build_forward_operator(self.geoms, self.grid)
        D = neighbor_difference_matrix(6, 6)
        counts = np.asarray((A != 0).sum(axis=0)).ravel()
        P = prior_selector(counts > 0)
        res = reconstruct(A, zero, self.atten, self.sigma,
                          self.grid.mass_attenuation, D, P, 0.01)
        pred_alt = independent_prediction(self.grid, self.geoms,
                                          res.density, steps_per_cell=2000)
        scale = np.maximum(np.abs(res.predicted), 1e-12)
        rel = np.abs(pred_alt - res.predicted) / scale
        valid = ~zero
        self.assertLess(rel[valid].max(), 1e-3)
        self.assertEqual(pred_alt[zero].sum(), 0.0)


class IdentifiabilityAndLimitedAngle(unittest.TestCase):
    def test_limited_angle_has_large_nullspace_and_invisible_pattern(self):
        grid = load_grid(MAIN_GRID)
        ids, ep, atten, sigma, _ = load_rays_csv(LIMITED_RAYS)
        geoms, _ = process_geometry(ids, ep, grid)
        A, zero = build_forward_operator(geoms, grid)
        self.assertFalse(zero.any())
        svd = weighted_svd(A, sigma, grid.mass_attenuation)
        # 5 条水平射线 -> 秩恰为 5，零空间维数 36-5=31
        self.assertEqual(svd["rank"], 5)
        self.assertEqual(svd["n_cols"] - svd["rank"], 31)
        probes = direction_probes(6, 6)
        vis = probe_visibility(A, sigma, grid.mass_attenuation, probes)
        by_name = {v["pattern"]: v for v in vis}
        # 行内交替变化对纯水平射线完全不可见
        self.assertTrue(by_name["row_checker"]["invisible"])
        self.assertAlmostEqual(
            by_name["row_checker"]["visibility_ratio"], 0.0, places=12)
        cov = angle_coverage(ep)
        self.assertTrue(cov["finite_angle_suspect"])

    def test_nullspace_energy_map_marks_unsupported_rows(self):
        # 回归测试：完整 SVD 必须保留右零空间，不能被紧 SVD 丢掉
        grid = load_grid(MAIN_GRID)
        ids, ep, atten, sigma, _ = load_rays_csv(LIMITED_RAYS)
        geoms, _ = process_geometry(ids, ep, grid)
        A, zero = build_forward_operator(geoms, grid)
        svd = weighted_svd(A, sigma, grid.mass_attenuation)
        self.assertEqual(len(svd["singular_values"]), 36)  # 零奇异值已补齐
        energy = nullspace_energy_map(svd["vt"],
                                      svd["singular_values"], 6, 6)
        energy2d = energy.reshape(6, 6)
        # 无射线的最上行 row5：变化 100% 落在零空间
        np.testing.assert_allclose(energy2d[5], 1.0, atol=1e-12)
        # 有射线的 row0..4：每个支持格在 31 维零空间中的方向能量为 5/6
        # （零空间中 1 个方向只作用于无射线的 row5，其余 30 个均匀铺开）
        np.testing.assert_allclose(energy2d[:5], 5.0 / 6.0, atol=1e-12)

    def test_full_case_rank_and_unsupported_cells(self):
        grid = load_grid(MAIN_GRID)
        ids, ep, atten, sigma, _ = load_rays_csv(MAIN_RAYS)
        geoms, _ = process_geometry(ids, ep, grid)
        A, zero = build_forward_operator(geoms, grid)
        self.assertEqual(int(zero.sum()), 1)          # R045 被剔除
        counts = np.asarray((A != 0).sum(axis=0)).ravel()
        # R045 之外所有格都被水平/垂直射线扫过 -> 36 格全有支持
        self.assertEqual((counts > 0).sum(), 36)
        svd = weighted_svd(A[~zero], sigma[~zero], grid.mass_attenuation)
        self.assertGreaterEqual(svd["rank"], 30)

    def test_smoothing_does_not_change_data_fit_claim(self):
        # 正则解的 J_data 不低于 lambda=0 解；且强正则平滑度更小
        grid = load_grid(MAIN_GRID)
        ids, ep, atten, sigma, _ = load_rays_csv(MAIN_RAYS)
        geoms, _ = process_geometry(ids, ep, grid)
        A, zero = build_forward_operator(geoms, grid)
        D = neighbor_difference_matrix(6, 6)
        counts = np.asarray((A != 0).sum(axis=0)).ravel()
        P = prior_selector(counts > 0)
        r0 = reconstruct(A, zero, atten, sigma, grid.mass_attenuation,
                         D, P, 0.0)
        rw = reconstruct(A, zero, atten, sigma, grid.mass_attenuation,
                         D, P, 1.0)
        self.assertGreaterEqual(
            rw.objective["J_data_chi2"] + 1e-8,
            r0.objective["J_data_chi2"])
        self.assertLess(rw.objective["J_smooth_unit_diff2"],
                        r0.objective["J_smooth_unit_diff2"])


if __name__ == "__main__":
    unittest.main()
