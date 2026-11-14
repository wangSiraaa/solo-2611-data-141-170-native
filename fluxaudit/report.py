"""审计证据落盘：faces.csv、单元体积/散度表、材料通量表、诊断 JSON。

CSV 一律长表；faces.csv 每个面-侧一行，材料间界面因此出现两行，各自携带
本侧外向法向与通量，face_id 相同，可回查顶点与相邻单元。
"""

from __future__ import annotations

import csv
import json
import os

from .audit import AuditResult, AuditRejected

FACE_COLUMNS = [
    "face_id", "kind", "side", "cell_id", "material_id",
    "vertex_0", "vertex_1", "vertex_2", "opposite_vertex",
    "outward_vertex_order", "neighbor_cell_id",
    "area", "normal_x", "normal_y", "normal_z",
    "awn_x", "awn_y", "awn_z", "face_flux",
]

CELL_COLUMNS = [
    "cell_id", "material_id", "vertex_0", "vertex_1", "vertex_2", "vertex_3",
    "signed_volume", "volume", "divergence", "volume_integral",
    "face_flux_sum", "closure_residual",
]

MATERIAL_COLUMNS = [
    "material_id", "n_cells", "volume", "inflow_flux", "outflow_flux",
    "face_flux_net", "exterior_flux", "interface_flux",
    "divergence_integral", "residual",
]


def _fmt(value: float) -> str:
    """固定 12 位有效输出，避免 -0.0 噪声，同时保留机器可解析精度。"""
    if value == 0.0:
        return "0.0"
    return f"{value:.12g}"


def write_faces_csv(result: AuditResult, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FACE_COLUMNS)
        writer.writeheader()
        for face in result.faces:
            neighbor = face.side2.cell_id if face.side2 is not None else ""
            sides = [(1, face.side1, neighbor)]
            if face.side2 is not None:
                sides.append((2, face.side2, face.side1.cell_id))
            for side_no, s, neigh in sides:
                writer.writerow({
                    "face_id": face.face_id,
                    "kind": face.kind,
                    "side": side_no,
                    "cell_id": s.cell_id,
                    "material_id": s.material_id,
                    "vertex_0": face.vertex_ids[0],
                    "vertex_1": face.vertex_ids[1],
                    "vertex_2": face.vertex_ids[2],
                    "opposite_vertex": s.opposite_vertex,
                    "outward_vertex_order": " ".join(map(str, s.vertex_order)),
                    "neighbor_cell_id": neigh,
                    "area": _fmt(s.area),
                    "normal_x": _fmt(s.outward_normal[0]),
                    "normal_y": _fmt(s.outward_normal[1]),
                    "normal_z": _fmt(s.outward_normal[2]),
                    "awn_x": _fmt(s.area_weighted_normal[0]),
                    "awn_y": _fmt(s.area_weighted_normal[1]),
                    "awn_z": _fmt(s.area_weighted_normal[2]),
                    "face_flux": _fmt(s.face_flux),
                })


def write_cells_csv(result: AuditResult, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CELL_COLUMNS)
        writer.writeheader()
        for c in result.cells:
            writer.writerow({
                "cell_id": c.cell_id,
                "material_id": c.material_id,
                "vertex_0": c.vertex_ids[0],
                "vertex_1": c.vertex_ids[1],
                "vertex_2": c.vertex_ids[2],
                "vertex_3": c.vertex_ids[3],
                "signed_volume": _fmt(c.signed_volume),
                "volume": _fmt(c.volume),
                "divergence": _fmt(c.divergence),
                "volume_integral": _fmt(c.volume_integral),
                "face_flux_sum": _fmt(c.face_flux_sum),
                "closure_residual": _fmt(c.closure_residual),
            })


def write_materials_csv(result: AuditResult, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MATERIAL_COLUMNS)
        writer.writeheader()
        for m in result.material_summaries:
            writer.writerow({
                "material_id": m.material_id,
                "n_cells": m.n_cells,
                "volume": _fmt(m.volume),
                "inflow_flux": _fmt(m.inflow_flux),
                "outflow_flux": _fmt(m.outflow_flux),
                "face_flux_net": _fmt(m.face_flux_net),
                "exterior_flux": _fmt(m.exterior_flux),
                "interface_flux": _fmt(m.interface_flux),
                "divergence_integral": _fmt(m.divergence_integral),
                "residual": _fmt(m.residual),
            })


def write_diagnostics(diagnostics: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2, ensure_ascii=False, sort_keys=False)
        fh.write("\n")


def write_all(result: AuditResult, out_dir: str) -> dict[str, str]:
    """成功审计：写四份产物，返回文件路径表。"""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "faces": os.path.join(out_dir, "faces.csv"),
        "cells": os.path.join(out_dir, "cells.csv"),
        "materials": os.path.join(out_dir, "materials.csv"),
        "diagnostics": os.path.join(out_dir, "diagnostics.json"),
    }
    write_faces_csv(result, paths["faces"])
    write_cells_csv(result, paths["cells"])
    write_materials_csv(result, paths["materials"])
    write_diagnostics(result.diagnostics, paths["diagnostics"])
    return paths


def write_rejection(exc: AuditRejected, out_dir: str) -> str:
    """拒绝场景：只落诊断，绝不写任何看似成功的通量产物。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "diagnostics.json")
    write_diagnostics(exc.diagnostics, path)
    return path
