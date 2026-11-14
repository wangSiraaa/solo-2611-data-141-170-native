"""独立交叉校验：用与 geometry.py 完全不同的积分路径重算。

geometry.py 用“网格线事件 + 段中点归属”精确求长；本模块改用固定步长
中点采样法：把裁切段等分成大量小步，每一小步按其中点所在单元
（半开 floor，同一归属约定）累计步长。两套实现共享的代码只有
裁切函数与单元索引约定，任何系统性重复计长/漏计都会在对比中暴露。
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import Grid, RayGeometry, clip_ray_to_region


def check_segment_length_sum(geoms: list[RayGeometry]) -> list[dict]:
    """核对各段长度之和与裁切线段长度一致（精确法的自洽性）。"""
    reports = []
    for g in geoms:
        if not g.segments:
            reports.append({"ray_id": g.ray_id, "status": "disjoint",
                            "segment_sum": 0.0, "clipped_length": 0.0,
                            "abs_diff": 0.0})
            continue
        seg_sum = sum(g.segments.values())
        diff = abs(seg_sum - g.clipped_length)
        reports.append({"ray_id": g.ray_id, "status": "ok",
                        "segment_sum": seg_sum,
                        "clipped_length": g.clipped_length,
                        "abs_diff": diff,
                        "rel_diff": diff / max(g.clipped_length, 1e-300)})
    return reports


def sampled_cell_lengths(grid: Grid, src: np.ndarray, dst: np.ndarray,
                         steps_per_cell: int = 200) -> dict[int, float]:
    """固定步长中点采样的逐格长度（独立实现）。"""
    clipped = clip_ray_to_region(np.asarray(src, float),
                                 np.asarray(dst, float), grid)
    if clipped is None:
        return {}
    entry, exit_ = clipped
    seg_vec = exit_ - entry
    seg_len = math.hypot(*seg_vec)
    n_steps = max(1, int(math.ceil(seg_len / (min(grid.dx, grid.dy)
                                              / steps_per_cell))))
    h = seg_len / n_steps
    unit = seg_vec / seg_len
    accum: dict[int, float] = {}
    for k in range(n_steps):
        mid = entry + (k + 0.5) * h * unit
        col = int(math.floor((mid[0] - grid.x_min) / grid.dx))
        row = int(math.floor((mid[1] - grid.y_min) / grid.dy))
        col = min(max(col, 0), grid.nx - 1)
        row = min(max(row, 0), grid.ny - 1)
        idx = grid.cell_index(row, col)
        accum[idx] = accum.get(idx, 0.0) + h
    return accum


def cross_check_operator(grid: Grid, geoms: list[RayGeometry],
                         steps_per_cell: int = 200
                         ) -> tuple[list[dict], float]:
    """逐射线对比事件法与采样法的逐格长度，返回明细与最大相对偏差。"""
    reports = []
    worst = 0.0
    for g in geoms:
        if not g.segments:
            sampled = sampled_cell_lengths(grid, g.src, g.dst,
                                           steps_per_cell)
            reports.append({"ray_id": g.ray_id, "status": "disjoint",
                            "max_abs_cell_diff": 0.0,
                            "length_rel_diff": 0.0})
            continue
        sampled = sampled_cell_lengths(grid, g.src, g.dst, steps_per_cell)
        keys = set(g.segments) | set(sampled)
        max_cell_diff = max(abs(g.segments.get(k, 0.0) - sampled.get(k, 0.0))
                            for k in keys)
        sampled_total = sum(sampled.values())
        rel = abs(sampled_total - g.clipped_length) / g.clipped_length
        worst = max(worst, rel)
        reports.append({"ray_id": g.ray_id, "status": "ok",
                        "max_abs_cell_diff_m": max_cell_diff,
                        "event_total_m": g.clipped_length,
                        "sampled_total_m": sampled_total,
                        "length_rel_diff": rel})
    return reports, worst


def independent_prediction(grid: Grid, geoms: list[RayGeometry],
                           density: np.ndarray,
                           steps_per_cell: int = 200) -> np.ndarray:
    """用采样法独立重算每条射线的预测衰减 mu*sum(length*density)。"""
    mu = grid.mass_attenuation
    pred = np.zeros(len(geoms))
    for i, g in enumerate(geoms):
        if not g.segments:
            continue
        sampled = sampled_cell_lengths(grid, g.src, g.dst, steps_per_cell)
        line_integral = sum(length * density[idx]
                            for idx, length in sampled.items())
        pred[i] = mu * line_integral
    return pred
