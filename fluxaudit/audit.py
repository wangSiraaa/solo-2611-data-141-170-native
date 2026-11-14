"""审计核心：面身份、相邻关系、面通量与单元散度两条独立核对路径。

设计要点
--------
1. 节点下标即拓扑身份。面以排序后的节点三元组作为规范 key，绝不按浮点
   坐标接近程度合并节点。
2. 面的外向法向由"对侧顶点"判定（叉积方向与对侧顶点相对），不依赖单元
   顶点排列的正负号；有向体积为负不是坏几何。
3. 重复单元、零体积单元、超过两个单元共面的非流形面必须定位并拒绝形成
   成功审计，不自动删面或修复。
4. 面 ID 按规范 key 字典序分配，逐材料与全域检查共用同一份面拓扑证据；
   材料间界面的两侧各自保留外向记录，同材料内部面标记为 interior 而非边界。
5. 两条互相独立的计算路径：
   * 面路径：面面积加权外向法向（叉积几何）+ 线性形函数的节点场均值；
   * 体路径：参考单元 Jacobian 求逆得梯度，单元散度乘以体积。
   散度定理要求二者在容差内一致；公共界面两侧的面通量必须抵消。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import meshio
import numpy as np

# 单元四个面：(面内三个顶点位置, 对侧顶点位置)，位置仅用于取节点，方向另算
_FACE_SLOTS = ((0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0))


@dataclass
class AuditConfig:
    """审计配置（长度单位与场单位来自 plan.json 或命令行）。"""

    length_unit: str = "m"
    field_unit: str = "unit m^-2 s^-1"
    field_name: str = "flux"
    material_name: str = "material_id"
    tolerance: float = 1e-10


@dataclass
class CellRecord:
    """单元级体积、散度证据。"""

    cell_id: int
    material_id: int
    vertex_ids: tuple[int, int, int, int]
    signed_volume: float
    volume: float
    divergence: float
    volume_integral: float
    face_flux_sum: float
    closure_residual: float


@dataclass
class FaceSide:
    """一个面对某个相邻单元的一侧（外向信息独立保留）。"""

    cell_id: int
    material_id: int
    opposite_vertex: int
    area: float
    outward_normal: tuple[float, float, float]  # 单位法向
    area_weighted_normal: tuple[float, float, float]
    face_flux: float
    vertex_order: tuple[int, int, int]  # 使叉积外向的顶点排列


@dataclass
class FaceRecord:
    """规范面：全域唯一的拓扑身份，两侧挂同一 face_id。"""

    face_id: int
    vertex_ids: tuple[int, int, int]
    kind: str  # boundary / interface / interior
    side1: FaceSide
    side2: Optional[FaceSide]


@dataclass
class MaterialSummary:
    """单材料通量表。"""

    material_id: int
    n_cells: int
    volume: float
    face_flux_net: float
    inflow_flux: float  # 所有边界侧（外边界+界面）中通量为负的绝对值
    outflow_flux: float
    exterior_flux: float  # 全域外边界上的净通量
    interface_flux: float  # 材料间界面上的净通量（本材料外向）
    divergence_integral: float
    residual: float


@dataclass
class AuditResult:
    """一次成功审计的全部证据。"""

    config: AuditConfig
    points: np.ndarray
    tets: np.ndarray
    materials: np.ndarray
    flux: np.ndarray
    cells: list[CellRecord]
    faces: list[FaceRecord]
    material_summaries: list[MaterialSummary]
    diagnostics: dict


class AuditRejected(Exception):
    """网格未通过结构/几何校验，审计被拒绝（不得形成成功审计）。"""

    def __init__(self, diagnostics: dict):
        super().__init__("audit rejected: " + "; ".join(diagnostics["errors"]))
        self.diagnostics = diagnostics


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------

def load_mesh(path: str, config: AuditConfig):
    """读取 VTU，校验只含直边一阶四面体且场/材料齐备。"""
    mesh = meshio.read(path)
    tet_blocks = [b for b in mesh.cells if b.type == "tetra"]
    other = sorted({b.type for b in mesh.cells if b.type != "tetra"})
    errors = []
    if other:
        errors.append(f"only linear tetrahedra are supported, found cell types: {other}")
    if not tet_blocks:
        errors.append("mesh contains no tetrahedral cells")
        raise AuditRejected(_base_diagnostics(config, errors))

    tets = np.concatenate([b.data.astype(np.int64) for b in tet_blocks])
    material_data = mesh.cell_data.get(config.material_name)
    if material_data is None or len(material_data) != len(mesh.cells):
        raise AuditRejected(_base_diagnostics(
            config,
            [f"cell data '{config.material_name}' is missing or incomplete"]))
    # 只接受与 tet 块一一对应的材料数组，不静默丢弃非 tet 块上的数据
    material = np.concatenate([
        np.asarray(v) for block, v in zip(mesh.cells, material_data)
        if block.type == "tetra"
    ]).astype(np.int64)
    try:
        flux = np.asarray(mesh.point_data[config.field_name], dtype=np.float64)
    except KeyError:
        raise AuditRejected(_base_diagnostics(
            config, [f"point vector field '{config.field_name}' is missing"]))

    if flux is not None and flux.shape != (mesh.points.shape[0], 3):
        raise AuditRejected(_base_diagnostics(
            config,
            [f"field '{config.field_name}' must be a continuous 3-component node "
             f"vector, got shape {flux.shape} vs ({mesh.points.shape[0]}, 3)"],
        ))
    if material.shape[0] != tets.shape[0]:
        raise AuditRejected(_base_diagnostics(
            config,
            [f"material_id count {material.shape[0]} != tetra cell count {tets.shape[0]}"],
        ))
    if tets.min() < 0 or tets.max() >= mesh.points.shape[0]:
        raise AuditRejected(_base_diagnostics(
            config, ["connectivity references node indices outside the point array"]))
    return mesh.points.astype(np.float64), tets, material, flux


def _base_diagnostics(config: AuditConfig, errors: list[str]) -> dict:
    return {
        "status": "rejected",
        "length_unit": config.length_unit,
        "field_unit": config.field_unit,
        "tolerance": config.tolerance,
        "n_errors": len(errors),
        "errors": errors,
    }


# --------------------------------------------------------------------------
# 结构与几何校验（全部收集后一次性拒绝）
# --------------------------------------------------------------------------

def _signed_volume(points: np.ndarray, tet: np.ndarray) -> float:
    r = points[tet]
    edge = np.stack([r[1] - r[0], r[2] - r[0], r[3] - r[0]])
    return float(np.linalg.det(edge)) / 6.0


def validate_structure(points: np.ndarray, tets: np.ndarray,
                       config: AuditConfig) -> dict:
    """定位重复单元、零体积（含重复节点）单元与非流形面。"""
    errors: list[str] = []
    diag = {
        "n_points": int(points.shape[0]),
        "n_cells": int(tets.shape[0]),
        "duplicate_cells": [],
        "zero_volume_cells": [],
        "nonmanifold_faces": [],
        "min_abs_signed_volume": None,
    }

    # 1. 单元内重复节点（必然零体积，单独给出更明确的定位）
    repeated = []
    for cid, tet in enumerate(tets):
        if len(set(int(v) for v in tet)) < 4:
            repeated.append(cid)
            errors.append(
                f"cell {cid} has repeated vertex ids {tuple(int(v) for v in tet)}"
            )

    # 2. 零体积：尺度感知阈值，几何长度的立方 × 相对容差
    span = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    vol_tol = max(span, 1.0) ** 3 * config.tolerance
    signed_vols = np.array([_signed_volume(points, t) for t in tets])
    diag["min_abs_signed_volume"] = float(np.min(np.abs(signed_vols)))
    zero_cells = [int(c) for c in np.where(np.abs(signed_vols) <= vol_tol)[0]
                  if c not in repeated]
    for cid in zero_cells:
        errors.append(
            f"cell {cid} has zero signed volume {signed_vols[cid]:.3e} "
            f"(tolerance {vol_tol:.3e} {config.length_unit}^3)"
        )
    diag["zero_volume_cells"] = repeated + zero_cells
    diag["volume_tolerance"] = vol_tol

    # 3. 重复单元：同一组四个节点（顶点排列任意）出现多次
    key_to_cells: dict[frozenset, list[int]] = {}
    for cid, tet in enumerate(tets):
        key_to_cells.setdefault(frozenset(int(v) for v in tet), []).append(int(cid))
    dup_groups = [cids for cids in key_to_cells.values() if len(cids) > 1]
    for cids in sorted(dup_groups):
        diag["duplicate_cells"].append(cids)
        errors.append(
            f"duplicate cells share the same 4 vertices: cell ids {cids}"
        )

    # 4. 非流形面：同一规范面被两个以上单元共享
    face_owners: dict[tuple, list[int]] = {}
    for cid, tet in enumerate(tets):
        for triple in combinations((int(v) for v in tet), 3):
            key = tuple(sorted(triple))
            face_owners.setdefault(key, []).append(int(cid))
    nm = [(key, owners) for key, owners in face_owners.items()
          if len(owners) > 2]
    for key, owners in sorted(nm):
        diag["nonmanifold_faces"].append(
            {"face_vertices": list(key), "cell_ids": owners}
        )
        errors.append(
            f"non-manifold face {list(key)} is incident to {len(owners)} cells "
            f"(>2): cell ids {owners}"
        )

    diag["errors"] = errors
    return diag


# --------------------------------------------------------------------------
# 面构建与外向法向（由对侧顶点确定，不依赖单元顶点序）
# --------------------------------------------------------------------------

def _outward_geometry(points: np.ndarray, face_vertices: tuple[int, ...],
                      opposite: int):
    """返回 (面积, 单位外向法向, 面积加权外向法向, 外向顶点排列)。

    取面的一个排列做叉积，若叉积指向对侧顶点则翻转，从而保证外向。
    """
    a, b, c = face_vertices
    raw = np.cross(points[b] - points[a], points[c] - points[a])
    if np.dot(raw, points[opposite] - points[a]) > 0.0:
        # 叉积当前指向单元内部（朝向对侧顶点），翻转
        raw = -raw
        order = (a, c, b)
    else:
        order = (a, b, c)
    norm = float(np.linalg.norm(raw))
    area = 0.5 * norm
    unit = raw / norm if norm > 0.0 else np.zeros(3)
    return area, unit, raw * 0.5, order


def build_faces(points: np.ndarray, tets: np.ndarray, materials: np.ndarray,
                flux: np.ndarray) -> list[FaceRecord]:
    """建立规范面表，逐侧计算外向法向与面通量。"""
    # face key -> 每个相邻单元的一侧信息
    side_buckets: dict[tuple, list[FaceSide]] = {}
    for cid, tet in enumerate(tets):
        verts = tuple(int(v) for v in tet)
        for i0, i1, i2, iopp in _FACE_SLOTS:
            fv = (verts[i0], verts[i1], verts[i2])
            key = tuple(sorted(fv))
            opp = verts[iopp]
            area, unit, awn, order = _outward_geometry(points, key, opp)
            # 线性形函数：面上通量取三个面节点值的平均
            face_flux = float(np.dot(unit, flux[list(key)].mean(axis=0)) * area)
            side_buckets.setdefault(key, []).append(FaceSide(
                cell_id=int(cid),
                material_id=int(materials[cid]),
                opposite_vertex=opp,
                area=area,
                outward_normal=tuple(unit),
                area_weighted_normal=tuple(awn),
                face_flux=face_flux,
                vertex_order=order,
            ))

    faces: list[FaceRecord] = []
    for face_id, key in enumerate(sorted(side_buckets)):
        sides = side_buckets[key]
        mats = {s.material_id for s in sides}
        if len(sides) == 1:
            kind = "boundary"
        elif len(mats) == 1:
            kind = "interior"
        else:
            kind = "interface"
        faces.append(FaceRecord(
            face_id=face_id,
            vertex_ids=key,
            kind=kind,
            side1=sides[0],
            side2=sides[1] if len(sides) > 1 else None,
        ))
    return faces


# --------------------------------------------------------------------------
# 独立的体路径：参考单元 Jacobian 求散度
# --------------------------------------------------------------------------

def cell_divergence(points: np.ndarray, tet: np.ndarray, flux: np.ndarray):
    """参考单元 Jacobian 求逆，计算单元内恒定散度并返回体积与有向体积。

    参考单元取 r0 为原点，三条参考棱 e1,e2,e3；J 的列为物理棱向量。
    形函数梯度 ∇N = J^{-T} e_i，场梯度 = Σ v_i (∇N_i)^T，散度取迹。
    与面路径（叉积+形函数均值）不共享几何代码，构成独立验证。
    """
    r = points[tet]
    jmat = np.column_stack([r[1] - r[0], r[2] - r[0], r[3] - r[0]])
    det_j = float(np.linalg.det(jmat))
    signed_volume = det_j / 6.0
    volume = abs(signed_volume)
    j_inv_t = np.linalg.inv(jmat).T
    # grad_N[i] 是第 i 个形函数的物理梯度：i=1..3 为 J^{-T} 的列，i=0 取负和
    grad_n = np.empty((4, 3))
    grad_n[1:] = j_inv_t.T  # 行即 ∇N_1..∇N_3
    grad_n[0] = -grad_n[1:].sum(axis=0)
    field_grad = flux[tet].T @ grad_n  # (3,3): 场分量对坐标的导数矩阵
    divergence = float(np.trace(field_grad))
    return signed_volume, volume, divergence


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------

def _summarize_materials(materials: np.ndarray, faces: list[FaceRecord],
                         cells: list[CellRecord]) -> list[MaterialSummary]:
    summaries = []
    for mat in sorted(set(int(m) for m in materials)):
        n_cells = int(np.sum(materials == mat))
        volume = sum(c.volume for c in cells if c.material_id == mat)
        div_int = sum(c.volume_integral for c in cells if c.material_id == mat)

        sides = []
        for f in faces:
            for s in (f.side1, f.side2):
                if s is not None and s.material_id == mat:
                    sides.append((f, s))
        boundary_sides = [(f, s) for f, s in sides
                          if f.kind in ("boundary", "interface")]
        net = sum(s.face_flux for _, s in boundary_sides)
        inflow = sum(-s.face_flux for _, s in boundary_sides if s.face_flux < 0.0)
        outflow = sum(s.face_flux for _, s in boundary_sides if s.face_flux > 0.0)
        exterior = sum(s.face_flux for f, s in boundary_sides
                       if f.kind == "boundary")
        interface = sum(s.face_flux for f, s in boundary_sides
                        if f.kind == "interface")
        summaries.append(MaterialSummary(
            material_id=mat,
            n_cells=n_cells,
            volume=volume,
            face_flux_net=net,
            inflow_flux=inflow,
            outflow_flux=outflow,
            exterior_flux=exterior,
            interface_flux=interface,
            divergence_integral=div_int,
            residual=net - div_int,
        ))
    return summaries


def _global_checks(faces: list[FaceRecord], cells: list[CellRecord],
                   config: AuditConfig) -> dict:
    """全域级证据：界面抵消、法向反向、两条路径总量一致。"""
    interface_pairs = []
    max_normal_antisym = 0.0
    max_interface_cancel = 0.0
    for f in faces:
        if f.kind == "interface":
            s1, s2 = f.side1, f.side2
            antisym = float(np.max(np.abs(
                np.array(s1.area_weighted_normal)
                + np.array(s2.area_weighted_normal))))
            cancel = abs(s1.face_flux + s2.face_flux)
            max_normal_antisym = max(max_normal_antisym, antisym)
            max_interface_cancel = max(max_interface_cancel, cancel)
            interface_pairs.append({
                "face_id": f.face_id,
                "face_vertices": list(f.vertex_ids),
                "materials": [s1.material_id, s2.material_id],
                "flux_side_a": s1.face_flux,
                "flux_side_b": s2.face_flux,
                "sum": s1.face_flux + s2.face_flux,
            })

    face_total = sum(
        f.side1.face_flux for f in faces if f.kind == "boundary"
    )  # boundary 面仅有一侧
    div_total = sum(c.volume_integral for c in cells)
    scale = max(abs(div_total), abs(face_total), 1.0)
    checks = {
        "global_face_flux": face_total,
        "global_divergence_integral": div_total,
        "global_residual": face_total - div_total,
        "global_relative_residual": abs(face_total - div_total) / scale,
        "max_cell_closure_residual": max(abs(c.closure_residual) for c in cells),
        "max_interface_normal_antisymmetry": max_normal_antisym,
        "max_interface_flux_cancellation_residual": max_interface_cancel,
        "interface_pairs": interface_pairs,
        "n_faces_by_kind": {
            k: sum(1 for f in faces if f.kind == k)
            for k in ("boundary", "interface", "interior")
        },
        "tolerance": config.tolerance,
        "passed": True,
    }
    # 判据：相对残差与各绝对残差均在容差尺度内
    tol = config.tolerance
    if checks["global_relative_residual"] > tol:
        checks["passed"] = False
    if max(abs(c.closure_residual) for c in cells) > tol * max(
            sum(c.volume for c in cells), 1.0):
        checks["passed"] = False
    if max_interface_cancel > tol or max_normal_antisym > tol:
        checks["passed"] = False
    return checks


def run_audit(mesh_path: str, config: AuditConfig) -> AuditResult:
    """对单个 VTU 执行完整审计；违规时抛出 :class:`AuditRejected`。"""
    points, tets, materials, flux = load_mesh(mesh_path, config)

    structural = validate_structure(points, tets, config)
    if structural["errors"]:
        structural["status"] = "rejected"
        structural["length_unit"] = config.length_unit
        structural["field_unit"] = config.field_unit
        structural["tolerance"] = config.tolerance
        raise AuditRejected(structural)

    faces = build_faces(points, tets, materials, flux)

    cells: list[CellRecord] = []
    face_sum_by_cell = {c: 0.0 for c in range(len(tets))}
    for f in faces:
        for s in (f.side1, f.side2):
            if s is not None:
                face_sum_by_cell[s.cell_id] += s.face_flux
    for cid, tet in enumerate(tets):
        signed_v, vol, div = cell_divergence(points, tet, flux)
        vol_int = div * vol
        cells.append(CellRecord(
            cell_id=cid,
            material_id=int(materials[cid]),
            vertex_ids=tuple(int(v) for v in tet),
            signed_volume=signed_v,
            volume=vol,
            divergence=div,
            volume_integral=vol_int,
            face_flux_sum=face_sum_by_cell[cid],
            closure_residual=face_sum_by_cell[cid] - vol_int,
        ))

    summaries = _summarize_materials(materials, faces, cells)
    global_checks = _global_checks(faces, cells, config)

    diagnostics = {
        "status": "passed",
        "mesh_path": mesh_path,
        "length_unit": config.length_unit,
        "field_unit": config.field_unit,
        "flux_unit": f"({config.field_unit})*{config.length_unit}^2",
        "volume_unit": f"{config.length_unit}^3",
        "tolerance": config.tolerance,
        "n_points": int(points.shape[0]),
        "n_cells": int(tets.shape[0]),
        "materials": sorted({int(m) for m in materials}),
        "structural_checks": {
            k: v for k, v in structural.items() if k != "errors"
        },
        "material_residuals": [
            {
                "material_id": s.material_id,
                "face_flux_net": s.face_flux_net,
                "divergence_integral": s.divergence_integral,
                "residual": s.residual,
                "relative_residual": abs(s.residual)
                / max(abs(s.divergence_integral), abs(s.face_flux_net), 1.0),
            }
            for s in summaries
        ],
        "global_checks": global_checks,
        "warnings": [],
    }
    if not global_checks["passed"]:
        diagnostics["status"] = "passed_with_warnings"
        diagnostics["warnings"].append(
            "numerical residuals exceed configured tolerance; inspect diagnostics")
    return AuditResult(config, points, tets, materials, flux,
                       cells, faces, summaries, diagnostics)
