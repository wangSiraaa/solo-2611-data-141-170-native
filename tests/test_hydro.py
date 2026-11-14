"""合成小栅格的单元测试：不依赖 fixtures 的固定答案，只检验算法性质。"""

from __future__ import annotations

import math
import os
import tempfile
import unittest

import numpy as np

from hydro.accumulation import accumulate
from hydro.basins import Outlet, delineate_basin
from hydro.fill import fill_depressions
from hydro.flow import EXIT_DIR_OFFSET, compute_flow
from hydro.gridio import GridHeader, read_ascii, write_ascii


def make_header(nrows, ncols, nodata=-9999.0):
    return GridHeader(ncols, nrows, 500000.0, 3300000.0, 10.0, nodata)


def run_pipeline(z, valid, header):
    fr = fill_depressions(z, valid, header)
    fl = compute_flow(fr)
    ac = accumulate(fl, fr, header)
    return fr, fl, ac


class FillTests(unittest.TestCase):
    def test_bowl_fills_to_rim_spill(self):
        # 3x3：四周高、中心低；北边边界开口同高，中心只需升到开口高程
        z = np.array([
            [5, 5, 5],
            [5, 1, 5],
            [5, 5, 5],
        ], dtype=np.float64)
        valid = np.ones_like(z, dtype=bool)
        fr = fill_depressions(z, valid, make_header(3, 3))
        # 全台接触边界 => 全部可解决
        self.assertTrue(fr.resolved.all())
        self.assertEqual(fr.filled[1, 1], 5.0)
        self.assertEqual(fr.depth[1, 1], 4.0)
        self.assertTrue(np.all(fr.filled >= z))

    def test_real_zero_elevation_is_valid(self):
        z = np.zeros((3, 3), dtype=np.float64)
        valid = np.ones_like(z, dtype=bool)
        fr, fl, ac = run_pipeline(z, valid, make_header(3, 3))
        self.assertEqual(np.count_nonzero(fr.resolved), 9)
        # 全域零高程平地：所有链必须终止在边界出流，无环
        self.assertGreaterEqual(np.count_nonzero(fl.is_exit), 1)
        self.assertEqual(int(ac.count[fl.is_exit].sum()), 9)

    def test_enclosed_component_unresolved(self):
        nd = -9999.0
        z = np.full((5, 5), nd)
        z[1:4, 1:4] = np.array([[9, 8, 7], [8, 5, 6], [7, 6, 5]], dtype=float)
        valid = z != nd
        fr = fill_depressions(z, valid, make_header(5, 5, nd))
        self.assertEqual(np.count_nonzero(fr.resolved), 0)
        self.assertEqual(np.count_nonzero(fr.unresolved), 9)
        # 未解决 => 高程原样、无填深
        np.testing.assert_array_equal(fr.filled[valid], z[valid])
        np.testing.assert_array_equal(fr.depth[valid], 0)
        fl = compute_flow(fr)
        # 未解决单元不得有接收格/出流
        self.assertTrue(np.all(fl.receiver[valid] == -1))
        self.assertFalse(fl.is_exit.any())

    def test_two_components_one_open_one_closed(self):
        nd = -9999.0
        z = np.array([
            [1, 1, nd, nd, nd],
            [1, 1, nd, 9, nd],
            [1, 1, nd, 5, nd],
            [1, 1, nd, 9, nd],
            [1, 1, nd, nd, nd],
        ], dtype=np.float64)
        valid = z != nd
        fr, fl, ac = run_pipeline(z, valid, make_header(5, 5, nd))
        # 左 2 列 10 格接触边界可解决；右内 3 格被 nodata 全包围
        self.assertEqual(np.count_nonzero(fr.resolved), 10)
        self.assertEqual(np.count_nonzero(fr.unresolved), 3)
        # 封闭组件不参与累计（nodata 语义）
        self.assertTrue(np.all(ac.count[fr.unresolved] == nd))
        # 开放组件 10 个单位全部守恒出流
        exits = np.flatnonzero(fl.is_exit)
        self.assertEqual(int(ac.count.reshape(-1)[exits].sum()), 10)


class FlowTests(unittest.TestCase):
    def test_d8_uses_center_distance_slope(self):
        # 东侧正交落差 2（距离10，坡度0.2）；东南对角落差 2.5（距离14.14，
        # 坡度≈0.177）=> 必须选正交的东向（方向3）
        z = np.array([[5, 3, 2.5], [5, 5, 5], [5, 5, 5]], dtype=np.float64)
        valid = np.ones_like(z, dtype=bool)
        fr, fl, _ = run_pipeline(z, valid, make_header(3, 3))
        self.assertEqual(int(fl.direction[0, 0]), 3)

    def test_slope_tie_picks_smallest_direction_code(self):
        # 中心 (1,1) 仅 N(0,1) 与 E(1,2) 为严格低邻居，同高程、同正交
        # 距离 => 坡度完全并列，取最小方向码 N(1)。其余邻居不低于中心。
        z = np.array([[5, 4, 5], [5, 5, 4], [5, 5, 5]], dtype=np.float64)
        valid = np.ones_like(z, dtype=bool)
        fr, fl, _ = run_pipeline(z, valid, make_header(3, 3))
        self.assertEqual(int(fl.direction[1, 1]), 1)

    def test_no_cycles_and_all_chains_exit(self):
        rng = np.random.default_rng(42)
        z = rng.integers(0, 5, size=(8, 8)).astype(np.float64)
        valid = np.ones_like(z, dtype=bool)
        fr, fl, ac = run_pipeline(z, valid, make_header(8, 8))
        receiver = fl.receiver.reshape(-1)
        for start in range(64):
            seen = set()
            cur = start
            while receiver[cur] >= 0:
                self.assertNotIn(cur, seen, f"环出现在 {cur}")
                seen.add(cur)
                cur = int(receiver[cur])
            self.assertLess(receiver[cur], 0)  # 每条链终于出流
        # 守恒
        exits = np.flatnonzero(fl.is_exit)
        self.assertEqual(int(ac.count.reshape(-1)[exits].sum()), 64)

    def test_flat_routing_evidence_present(self):
        # 带洼地的图：填平后必须出现 platter，且每个 platter 有距离/父证据
        z = np.array([
            [9, 9, 9, 9, 9],
            [9, 5, 5, 5, 8],
            [9, 5, 5, 5, 7],
            [9, 5, 5, 5, 6],
            [9, 9, 9, 4, 3],
        ], dtype=np.float64)
        valid = np.ones_like(z, dtype=bool)
        fr, fl, _ = run_pipeline(z, valid, make_header(5, 5))
        plat = np.flatnonzero(fl.platter.reshape(-1))
        self.assertGreater(plat.size, 0)
        dl = fl.dist_lower.reshape(-1)
        bp = fl.bfs_parent.reshape(-1)
        for p in plat:
            self.assertNotEqual(int(dl[p]), np.iinfo(np.int32).max)
            if int(dl[p]) > 0:
                self.assertGreaterEqual(int(bp[p]), 0)
                # 父单元距离严格小 1 => 无环结构保证
                self.assertEqual(int(dl[bp[p]]), int(dl[p]) - 1)

    def test_exit_codes_decode_to_valid_directions(self):
        z = np.ones((4, 4), dtype=np.float64)
        valid = np.ones_like(z, dtype=bool)
        fr, fl, _ = run_pipeline(z, valid, make_header(4, 4))
        for p in np.flatnonzero(fl.is_exit.reshape(-1)):
            code = -int(fl.receiver.reshape(-1)[p]) - EXIT_DIR_OFFSET
            self.assertIn(code, range(1, 9))


class BasinTests(unittest.TestCase):
    def test_nested_basins_overlap_allowed(self):
        # 单斜面向南：选两个同行链上的出口，流域应嵌套
        z = np.fromfunction(lambda r, c: (10 - r) * 1.0, (6, 6))
        valid = np.ones_like(z, dtype=bool)
        fr, fl, ac = run_pipeline(z, valid, make_header(6, 6))
        inner = Outlet("inner", 3, 2)
        outer = Outlet("outer", 5, 2)
        b_in = delineate_basin(inner, fl.receiver, fr.resolved, fr.header)
        b_out = delineate_basin(outer, fl.receiver, fr.resolved, fr.header)
        self.assertTrue(np.all(b_in.member <= b_out.member))
        self.assertLess(b_in.cell_count, b_out.cell_count)

    def test_outlet_on_void_rejected(self):
        nd = -9999.0
        z = np.ones((4, 4))
        z[2, 2] = nd
        valid = z != nd
        fr = fill_depressions(z, valid, make_header(4, 4, nd))
        with self.assertRaises(ValueError):
            delineate_basin(Outlet("bad", 2, 2), np.full((4, 4), -1),
                            fr.resolved, fr.header)


class GridIOTests(unittest.TestCase):
    def test_roundtrip_preserves_header_and_nodata(self):
        header = make_header(3, 2)
        z = np.array([[0.0, 1.5], [-9999.0, 2.0], [3.0, 0.0]])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.asc")
            write_ascii(path, z, header)
            z2, valid, h2 = read_ascii(path)
            self.assertEqual(h2, header)
            np.testing.assert_allclose(z2, z)
            self.assertFalse(valid[1, 0])
            self.assertTrue(valid[0, 0])  # 零高程有效
            self.assertTrue(valid[2, 1])


if __name__ == "__main__":
    unittest.main()
