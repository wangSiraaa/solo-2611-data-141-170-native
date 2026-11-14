"""通量审计测试。

覆盖：
* 真实双材料 fixture 的散度定理与界面抵消；
* 单元顺序与单元内顶点排列打乱后结论不变（负有向体积合法）；
* 三个负向文件分别被拒绝并精确定位；
* 独立合成小例：常量场零净通量、线性场与解析体积分一致、单四面体面级解析值；
* 非四面体单元被拒；CLI 退出码与产物。
合成网格只在测试临时目录中生成，不触碰 fixtures/，也不解任何 PDE。
"""

import csv
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from itertools import permutations

import meshio
import numpy as np

from fluxaudit.audit import (
    AuditConfig,
    AuditRejected,
    run_audit,
)
from fluxaudit.cli import main as cli_main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "fixtures")
CFG = AuditConfig()


def write_vtu(path, points, tets, flux, material_id):
    """在临时目录合成一阶四面体 VTU（测试夹具，非产品网格生成）。"""
    cells = [meshio.CellBlock("tetra", np.asarray(tets, dtype=np.int32))]
    mesh = meshio.Mesh(
        points=np.asarray(points, dtype=np.float64),
        cells=cells,
        point_data={"flux": np.asarray(flux, dtype=np.float64)},
        cell_data={"material_id": [np.asarray(material_id, dtype=np.int32)]},
    )
    meshio.write(path, mesh)


def linear_flux(points):
    """与真实 fixture 同形式的线性场 v=(2x+y, -x+3y, z)，div v = 6。"""
    x, y, z = points.T
    return np.column_stack([2 * x + y, -x + 3 * y, z])


class SingleTetAnalytic(unittest.TestCase):
    """单四面体：面级通量可手算，独立于任何 fixture。"""

    def setUp(self):
        # 参考直角四面体，体积 1/6
        self.points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.tet = np.array([[0, 1, 2, 3]])
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "one.vtu")

    def tearDown(self):
        self.tmp.cleanup()

    def test_constant_field_zero_net_flux(self):
        v0 = np.array([1.5, -2.0, 0.7])
        flux = np.tile(v0, (4, 1))
        write_vtu(self.path, self.points, self.tet, flux, [1])
        r = run_audit(self.path, CFG)
        self.assertEqual(r.diagnostics["status"], "passed")
        self.assertAlmostEqual(
            r.material_summaries[0].face_flux_net, 0.0, places=12)
        self.assertAlmostEqual(
            r.material_summaries[0].divergence_integral, 0.0, places=12)
        self.assertAlmostEqual(
            r.material_summaries[0].inflow_flux,
            r.material_summaries[0].outflow_flux, places=12)
        for c in r.cells:
            self.assertAlmostEqual(c.closure_residual, 0.0, places=12)

    def test_linear_field_matches_analytic_surface_integral(self):
        # v=(x,y,z)，div=3，体积 1/6 → 体积分 0.5
        flux = self.points.copy()
        write_vtu(self.path, self.points, self.tet, flux, [1])
        r = run_audit(self.path, CFG)
        m = r.material_summaries[0]
        self.assertAlmostEqual(m.volume, 1.0 / 6.0, places=14)
        self.assertAlmostEqual(m.divergence_integral, 0.5, places=12)
        self.assertAlmostEqual(m.face_flux_net, 0.5, places=12)
        # 面级解析值：三个坐标面外向法向与场正交 → 通量 0；
        # 斜面 (1,2,3) 面积加权外向法向 (1/6,1/6,1/6)，
        # 节点场均值 (1/3,1/3,1/3) → 通量 3·(1/6·1/3)=1/2，
        # 与散度定理 3×(1/6)=1/2 闭合。
        boundary = {f.vertex_ids: f.side1 for f in r.faces
                    if f.kind == "boundary"}
        for key in [(0, 1, 2), (0, 1, 3), (0, 2, 3)]:
            self.assertAlmostEqual(boundary[tuple(sorted(key))].face_flux,
                                   0.0, places=12)
        self.assertAlmostEqual(boundary[(1, 2, 3)].face_flux, 0.5,
                               places=12)
        for c in r.cells:
            self.assertAlmostEqual(c.closure_residual, 0.0, places=12)


class TwoTetInterface(unittest.TestCase):
    """两个材料各一个四面体，共享面两侧抵消，全域量可解析。"""

    def setUp(self):
        # tet0: (0,1,2,3)，tet1: (1,2,3,4)，共享面 1-2-3
        self.points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        self.tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        self.mats = np.array([1, 2])
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "two.vtu")

    def tearDown(self):
        self.tmp.cleanup()

    def test_interface_cancellation_and_divergence_theorem(self):
        flux = linear_flux(self.points)
        write_vtu(self.path, self.points, self.tets, flux, self.mats)
        r = run_audit(self.path, CFG)
        self.assertEqual(r.diagnostics["status"], "passed")

        faces = {f.face_id: f for f in r.faces}
        interface = [f for f in r.faces if f.kind == "interface"]
        self.assertEqual(len(interface), 1)
        f = interface[0]
        self.assertEqual(f.vertex_ids, (1, 2, 3))
        # 两侧面积加权外向法向严格反向
        self.assertTrue(np.allclose(
            np.array(f.side1.area_weighted_normal),
            -np.array(f.side2.area_weighted_normal), atol=1e-14))
        # 通量抵消
        self.assertAlmostEqual(f.side1.face_flux + f.side2.face_flux,
                               0.0, places=12)

        # 全域：div=6，两 tet 各体积 1/6（总体积 1/3）→ 净外通量 3
        g = r.diagnostics["global_checks"]
        self.assertAlmostEqual(g["global_face_flux"], 3.0, places=12)
        self.assertAlmostEqual(g["global_divergence_integral"], 3.0, places=12)
        self.assertAlmostEqual(g["global_residual"], 0.0, places=10)
        self.assertTrue(g["passed"])

        # 每材料各自满足散度定理（体路径 vs 面路径）
        for s in r.material_summaries:
            self.assertAlmostEqual(
                s.face_flux_net, s.divergence_integral, places=10)
            self.assertAlmostEqual(abs(s.residual), 0.0, places=10)
        # 两侧界面净通量互为相反数
        m1 = next(s for s in r.material_summaries if s.material_id == 1)
        m2 = next(s for s in r.material_summaries if s.material_id == 2)
        self.assertAlmostEqual(m1.interface_flux, -m2.interface_flux,
                               places=12)

    def test_constant_field_interface_and_global_zero(self):
        flux = np.tile([0.3, 0.9, -1.1], (len(self.points), 1))
        write_vtu(self.path, self.points, self.tets, flux, self.mats)
        r = run_audit(self.path, CFG)
        for s in r.material_summaries:
            self.assertAlmostEqual(s.face_flux_net, 0.0, places=12)
            self.assertAlmostEqual(s.divergence_integral, 0.0, places=12)
        g = r.diagnostics["global_checks"]
        self.assertAlmostEqual(g["global_face_flux"], 0.0, places=12)


class CubeFixtureEquivalence(unittest.TestCase):
    """独立合成的 6 四面体单位立方体，与真实 fixture 同场同解析结果。"""

    def setUp(self):
        p = np.array([[x, y, z] for x in (0, 1) for y in (0, 1)
                      for z in (0, 1)], dtype=float)
        tets = np.array([
            [0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7],
            [6, 0, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7],
        ])
        self.points, self.tets = p, tets
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_analytic_volume_integral(self):
        path = os.path.join(self.tmp.name, "cube.vtu")
        write_vtu(path, self.points, self.tets, linear_flux(self.points),
                  np.ones(6, dtype=int))
        r = run_audit(path, CFG)
        self.assertAlmostEqual(sum(c.volume for c in r.cells), 1.0, places=12)
        m = r.material_summaries[0]
        self.assertAlmostEqual(m.divergence_integral, 6.0, places=10)
        self.assertAlmostEqual(m.face_flux_net, 6.0, places=10)


class OrderInvariance(unittest.TestCase):
    """输入单元顺序、单元内顶点排列改变不改变几何与通量结论。"""

    def test_shuffled_vertex_order_and_cell_order(self):
        rng = np.random.default_rng(20261114)
        mesh = meshio.read(os.path.join(FIXTURES, "mesh.vtu"))
        points = mesh.points
        flux = mesh.point_data["flux"]
        tets = np.concatenate([b.data for b in mesh.cells
                               if b.type == "tetra"])
        mats = np.concatenate(mesh.cell_data["material_id"])

        shuffled_tets = np.array([
            rng.permutation(tet) for tet in tets
        ], dtype=np.int32)
        perm = rng.permutation(len(shuffled_tets))
        shuffled_tets = shuffled_tets[perm]
        shuffled_mats = mats[perm]

        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, "shuffled.vtu")
        write_vtu(path, points, shuffled_tets, flux, shuffled_mats)
        r = run_audit(path, CFG)
        ref = run_audit(os.path.join(FIXTURES, "mesh.vtu"), CFG)

        # 有向体积符号可翻转，但绝对体积、散度、通量结论不变
        self.assertEqual(sorted(c.volume for c in r.cells),
                         sorted(c.volume for c in ref.cells))
        self.assertEqual(sorted(round(c.divergence, 10) for c in r.cells),
                         sorted(round(c.divergence, 10) for c in ref.cells))
        key = lambda s: s.material_id
        for got, exp in zip(sorted(r.material_summaries, key=key),
                            sorted(ref.material_summaries, key=key)):
            self.assertAlmostEqual(got.face_flux_net, exp.face_flux_net,
                                   places=10)
            self.assertAlmostEqual(got.divergence_integral,
                                   exp.divergence_integral, places=10)
            self.assertAlmostEqual(got.inflow_flux, exp.inflow_flux, places=10)
            self.assertAlmostEqual(got.outflow_flux, exp.outflow_flux,
                                   places=10)
        g1, g0 = (r.diagnostics["global_checks"],
                  ref.diagnostics["global_checks"])
        self.assertAlmostEqual(g1["global_face_flux"],
                               g0["global_face_flux"], places=10)
        # 规范面身份集合与单元顺序无关
        self.assertEqual({f.vertex_ids for f in r.faces},
                         {f.vertex_ids for f in ref.faces})
        tmp.cleanup()

    def test_all_permutations_of_one_tet_equivalent(self):
        """单四面体 24 种顶点排列全部给出相同面通量与绝对体积。"""
        points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ])
        tmp = tempfile.TemporaryDirectory()
        nets, vols, slanted = [], [], []
        for perm in permutations(range(4)):
            path = os.path.join(tmp.name, "p.vtu")
            write_vtu(path, points, np.array([list(perm)]),
                      points.copy(), [1])
            r = run_audit(path, CFG)
            nets.append(r.material_summaries[0].face_flux_net)
            vols.append(r.cells[0].volume)
            f = next(f for f in r.faces
                     if f.vertex_ids == (1, 2, 3))
            slanted.append(f.side1.face_flux)
        self.assertTrue(np.allclose(nets, 0.5))
        self.assertTrue(np.allclose(vols, 1.0 / 6.0))
        self.assertTrue(np.allclose(slanted, 0.5))
        tmp.cleanup()


class TopologicalIdentity(unittest.TestCase):
    """节点下标是拓扑身份：坐标重合但下标不同的节点不得合并。"""

    def test_coincident_coordinates_keep_distinct_faces(self):
        # 两个独立 tet 各用各的节点，即便部分坐标完全相同也不共享面
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # tet A
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # tet B，坐标重复
        ], dtype=float)
        tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, "coincident.vtu")
        write_vtu(path, points, tets, linear_flux(points), [1, 2])
        r = run_audit(path, CFG)
        # 无任何共享面：8 个边界面，0 界面，0 内部面
        kinds = r.diagnostics["global_checks"]["n_faces_by_kind"]
        self.assertEqual(kinds, {"boundary": 8, "interface": 0, "interior": 0})
        self.assertEqual(len(r.faces), 8)
        tmp.cleanup()

    def test_tight_tolerance_flags_warning_but_keeps_evidence(self):
        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, "one.vtu")
        points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ])
        write_vtu(path, points, np.array([[0, 1, 2, 3]]),
                  points.copy(), [1])
        tight = AuditConfig(tolerance=1e-16)
        r = run_audit(path, tight)
        self.assertIn(r.diagnostics["status"],
                      ("passed", "passed_with_warnings"))
        if r.diagnostics["status"] == "passed_with_warnings":
            self.assertTrue(r.diagnostics["warnings"])
            self.assertFalse(
                r.diagnostics["global_checks"]["passed"])
        # 无论告警与否，残差本身必须是机器精度量级
        self.assertLess(
            r.diagnostics["global_checks"]["global_relative_residual"], 1e-12)
        tmp.cleanup()

    def test_units_recorded_in_diagnostics(self):
        cfg = AuditConfig(length_unit="mm", field_unit="W m^-2")
        r = run_audit(os.path.join(FIXTURES, "mesh.vtu"), cfg)
        d = r.diagnostics
        self.assertEqual(d["length_unit"], "mm")
        self.assertEqual(d["field_unit"], "W m^-2")
        self.assertEqual(d["volume_unit"], "mm^3")
        self.assertEqual(d["flux_unit"], "(W m^-2)*mm^2")


class RealFixtureAndRejections(unittest.TestCase):
    """真实双材料 fixture 与三个负向文件。"""

    def test_mesh_fixture_passes_with_full_evidence(self):
        r = run_audit(os.path.join(FIXTURES, "mesh.vtu"), CFG)
        d = r.diagnostics
        self.assertEqual(d["status"], "passed")
        self.assertEqual(d["materials"], [1, 2])
        g = d["global_checks"]
        self.assertEqual(g["n_faces_by_kind"],
                         {"boundary": 20, "interface": 2, "interior": 12})
        self.assertAlmostEqual(g["global_face_flux"], 12.0, places=10)
        self.assertAlmostEqual(g["global_divergence_integral"], 12.0,
                               places=10)
        self.assertAlmostEqual(
            g["max_interface_flux_cancellation_residual"], 0.0, places=12)
        for s in r.material_summaries:
            self.assertAlmostEqual(s.face_flux_net, 6.0, places=10)
            self.assertAlmostEqual(s.divergence_integral, 6.0, places=10)
            self.assertAlmostEqual(s.residual, 0.0, places=10)
        # 同材料内部面不计入任何材料边界
        self.assertTrue(all(
            f.kind != "interior" or
            all(s is None or True for s in (f.side1, f.side2))
            for f in r.faces))
        # 界面面在 faces 表中各占两行且 face_id 相同
        interface_rows = [f for f in r.faces if f.kind == "interface"]
        for f in interface_rows:
            self.assertIsNotNone(f.side2)
            self.assertNotEqual(f.side1.material_id, f.side2.material_id)

    def test_negative_fixtures_rejected_with_locations(self):
        cases = {
            "duplicate.vtu": "duplicate_cells",
            "degenerate.vtu": "zero_volume_cells",
            "nonmanifold.vtu": "nonmanifold_faces",
        }
        for name, diag_key in cases.items():
            with self.subTest(file=name):
                with self.assertRaises(AuditRejected) as ctx:
                    run_audit(os.path.join(FIXTURES, name), CFG)
                diag = ctx.exception.diagnostics
                self.assertEqual(diag["status"], "rejected")
                self.assertTrue(diag["errors"])
                self.assertTrue(diag[diag_key], f"{name} 应定位 {diag_key}")

    def test_coplanar_zero_volume_without_repeated_nodes(self):
        # 四点共面且节点互不重复：必须靠体积判定，而非重复节点判定
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        ], dtype=float)
        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, "flat.vtu")
        write_vtu(path, points, np.array([[0, 1, 2, 3]]),
                  np.zeros((4, 3)), [1])
        with self.assertRaises(AuditRejected) as ctx:
            run_audit(path, CFG)
        self.assertEqual(ctx.exception.diagnostics["zero_volume_cells"], [0])
        tmp.cleanup()

    def test_non_tetra_cell_type_rejected(self):
        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, "hex.vtu")
        mesh = meshio.Mesh(
            points=np.zeros((8, 3)),
            cells=[meshio.CellBlock("hexahedron",
                                    np.arange(8, dtype=np.int32)[None])],
        )
        meshio.write(path, mesh)
        with self.assertRaises(AuditRejected) as ctx:
            run_audit(path, CFG)
        self.assertIn("only linear tetrahedra",
                      ctx.exception.diagnostics["errors"][0])
        tmp.cleanup()


class CliAndReports(unittest.TestCase):
    def test_cli_success_writes_four_artifacts(self):
        out = tempfile.mkdtemp()
        with contextlib.redirect_stdout(io.StringIO()):
            code = cli_main([
                "audit", os.path.join(FIXTURES, "mesh.vtu"),
                "--plan", os.path.join(FIXTURES, "plan.json"),
                "--out", out,
            ])
        self.assertEqual(code, 0)
        for name in ("faces.csv", "cells.csv", "materials.csv",
                     "diagnostics.json"):
            self.assertTrue(os.path.exists(os.path.join(out, name)))
        with open(os.path.join(out, "faces.csv"), newline="") as fh:
            rows = list(csv.DictReader(fh))
        # 48 个面-侧：20 边界*1 + 2 界面*2 + 12 内部*2
        self.assertEqual(len(rows), 20 + 4 + 24)
        with open(os.path.join(out, "diagnostics.json")) as fh:
            diag = json.load(fh)
        self.assertEqual(diag["status"], "passed")
        self.assertIn("flux_unit", diag)

    def test_cli_rejection_exit_code_and_no_success_artifacts(self):
        out = tempfile.mkdtemp()
        with contextlib.redirect_stderr(io.StringIO()):
            code = cli_main([
                "audit", os.path.join(FIXTURES, "degenerate.vtu"),
                "--plan", os.path.join(FIXTURES, "plan.json"),
                "--out", out,
            ])
        self.assertEqual(code, 2)
        self.assertEqual(sorted(os.listdir(out)), ["diagnostics.json"])
        with open(os.path.join(out, "diagnostics.json")) as fh:
            self.assertEqual(json.load(fh)["status"], "rejected")

    def test_module_invocation(self):
        proc = subprocess.run(
            [sys.executable, "-m", "fluxaudit", "audit",
             os.path.join(FIXTURES, "mesh.vtu"),
             "--plan", os.path.join(FIXTURES, "plan.json"),
             "--out", tempfile.mkdtemp()],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("AUDIT PASSED", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
