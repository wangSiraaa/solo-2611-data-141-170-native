"""指定出口的上游流域。

出口按零基 (row, col) 定位，必须是有效可解决单元（nodata、未解决
封闭组件中的单元一律拒绝）。流域 = 接收格链会经过该出口格的全部
单元，即在“接收格反向图”上从出口可达的所有单元（含出口自身）。
两个出口可以形成嵌套/重叠流域：各自独立反向追踪，不做互斥切分。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Outlet:
    outlet_id: str
    row: int
    col: int

    @property
    def linear(self) -> int:
        return -1  # 由 build 阶段结合 ncols 填充


@dataclass
class BasinResult:
    outlet_id: str
    row: int
    col: int
    member: np.ndarray        # bool 栅格，True 表示属于该流域
    member_cells: np.ndarray  # 线性编号数组，按编号排序
    cell_count: int
    area_m2: float


def read_outlets(path: str) -> list[Outlet]:
    outlets: list[Outlet] = []
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        expected = ["outlet_id", "row", "col"]
        if [h.strip() for h in header] != expected:
            raise ValueError(f"{path}: 表头应为 {expected}，实际 {header}")
        for line_no, line in enumerate(fh, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_no} 字段数不为 3")
            outlets.append(Outlet(parts[0].strip(), int(parts[1]), int(parts[2])))
    if not outlets:
        raise ValueError(f"{path}: 没有读到任何出口")
    return outlets


def build_reverse_adjacency(receiver: np.ndarray, resolved: np.ndarray) -> list[list[int]]:
    """由接收格栅格构造反向邻接表：parents[q] = 直接上游单元列表。"""
    n = resolved.size
    parents: list[list[int]] = [[] for _ in range(n)]
    recv_flat = receiver.reshape(-1)
    for p in np.flatnonzero(resolved.reshape(-1)):
        q = recv_flat[p]
        if q >= 0:  # 出流（负码）不指向任何栅格内单元
            parents[int(q)].append(int(p))
    return parents


def delineate_basin(
    outlet: Outlet,
    receiver: np.ndarray,
    resolved: np.ndarray,
    header,
) -> BasinResult:
    nrows, ncols = resolved.shape
    if not (0 <= outlet.row < nrows and 0 <= outlet.col < ncols):
        raise ValueError(f"出口 {outlet.outlet_id} 越界：({outlet.row},{outlet.col})")
    target = outlet.row * ncols + outlet.col
    if not resolved.reshape(-1)[target]:
        raise ValueError(
            f"出口 {outlet.outlet_id} ({outlet.row},{outlet.col}) "
            "不是有效可解决单元（nodata 或封闭未解决组件）"
        )

    parents = build_reverse_adjacency(receiver, resolved)
    member_flat = np.zeros(nrows * ncols, dtype=bool)
    stack = [target]
    member_flat[target] = True
    while stack:
        q = stack.pop()
        for p in parents[q]:
            if not member_flat[p]:
                member_flat[p] = True
                stack.append(p)

    member_cells = np.flatnonzero(member_flat)
    return BasinResult(
        outlet_id=outlet.outlet_id,
        row=outlet.row,
        col=outlet.col,
        member=member_flat.reshape(nrows, ncols),
        member_cells=member_cells,
        cell_count=int(member_cells.size),
        area_m2=float(member_cells.size) * float(header.cellsize) ** 2,
    )
