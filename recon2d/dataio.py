"""输入读取：rays.csv 与几何批处理，集中产出输入诊断。

rays.csv 列：ray_id, source_x_m, source_y_m, receiver_x_m,
receiver_y_m, attenuation（无量纲对数衰减）, sigma（衰减误差，1 sigma）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import MAX_RAYS
from .geometry import GeometryError, RayGeometry, grid_from_dict, ray_cell_segments


def load_grid(path: str | Path):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return grid_from_dict(cfg)


def load_rays_csv(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray,
                                             np.ndarray, list[dict]]:
    """读取射线 CSV。

    返回 (ray_ids, endpoints[n,4], attenuation[n], sigma[n], 行级诊断)。
    非有限值、非正 sigma 不在这里直接抛掉：记录诊断后交由上层决定，
    但几何层对非有限端点仍会抛 GeometryError——因此本函数先做拦截，
    把问题行从有效集合中剔除并写明原因，保证“未知几何有诊断”。
    """
    ids: list[str] = []
    rows_raw: list[list[float]] = []
    diagnostics: list[dict] = []
    seen_ids: set[str] = set()

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"ray_id", "source_x_m", "source_y_m", "receiver_x_m",
                    "receiver_y_m", "attenuation", "sigma"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise GeometryError(
                f"{path}: 列名不全，需要 {sorted(required)}，"
                f"实际 {reader.fieldnames}")
        for line_no, row in enumerate(reader, start=2):
            rid = row["ray_id"].strip()
            try:
                vals = [float(row[c]) for c in
                        ("source_x_m", "source_y_m", "receiver_x_m",
                         "receiver_y_m", "attenuation", "sigma")]
            except (TypeError, ValueError):
                diagnostics.append({"ray_id": rid or f"line{line_no}",
                                    "line": line_no,
                                    "issue": "字段无法解析为数值，该行已剔除"})
                continue
            if rid in seen_ids:
                diagnostics.append({"ray_id": rid, "line": line_no,
                                    "issue": "ray_id 重复，该行已剔除"})
                continue
            finite = np.isfinite(vals).all()
            sigma_ok = vals[5] > 0.0
            if not finite:
                diagnostics.append({"ray_id": rid, "line": line_no,
                                    "issue": "含 NaN/Inf 非有限值，该行已剔除"})
                continue
            if not sigma_ok:
                diagnostics.append({"ray_id": rid, "line": line_no,
                                    "issue": "sigma 必须为正，该行已剔除"})
                continue
            seen_ids.add(rid)
            ids.append(rid)
            rows_raw.append(vals)

    if len(ids) > MAX_RAYS:
        raise GeometryError(
            f"射线数 {len(ids)} 超过题目约束 {MAX_RAYS} 条")
    if not rows_raw:
        raise GeometryError(f"{path}: 没有任何有效射线行")

    arr = np.asarray(rows_raw, dtype=float)
    endpoints = arr[:, :4]
    attenuation = arr[:, 4]
    sigma = arr[:, 5]
    return ids, endpoints, attenuation, sigma, diagnostics


def process_geometry(ids, endpoints, grid) -> tuple[list[RayGeometry], list[dict]]:
    """对每条射线做裁切与逐格求长，汇总几何诊断。"""
    geoms: list[RayGeometry] = []
    extra: list[dict] = []
    for rid, ep in zip(ids, endpoints):
        src = ep[:2].copy()
        dst = ep[2:].copy()
        total = float(np.hypot(*(dst - src)))
        try:
            segs = ray_cell_segments(src, dst, grid)
        except GeometryError as exc:
            # 兜底：load_rays_csv 已拦截非有限，这里仍显式记录而非静默
            extra.append({"ray_id": rid, "issue": f"几何错误: {exc}"})
            geoms.append(RayGeometry(rid, src, dst, {}, 0.0, total,
                                     error=str(exc)))
            continue
        clipped = sum(segs.values())
        if not segs:
            extra.append({"ray_id": rid,
                          "issue": "射线与重建区域完全不相交（零内部长度），"
                                   "不纳入拟合；其观测不会被当作穿区有效观测"})
        geoms.append(RayGeometry(rid, src, dst, segs, clipped, total))
    return geoms, extra
