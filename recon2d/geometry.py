"""几何核心：直线射线在规则网格中的逐格路径长度。

约定（与 grid.json 一致）：
- 数组行列方向：行号沿 +y 增长，列号沿 +x 增长。
  单元 (row, col) 对应 x 区间 [x_min + col*dx, x_min + (col+1)*dx)、
  y 区间 [y_min + row*dy, y_min + (row+1)*dy)，最外侧最大边界取闭。
- 沿内部网格线的射线归高坐标一侧：x=x_k 上的点归入 col=k，
  y=y_k 上的点归入 row=k。floor() 的半开约定天然满足这一点。
- 穿过角点时，在该参数点发生的两个跨越只产生一个事件，不重复计长。

方法：先把射线用 Liang-Barsky 式 slab 求交裁到区域，再收集裁切段内
所有网格线跨越事件，相邻事件之间的整段落在唯一单元内，按段中点
（取半开 floor）归属单元并累加欧氏长度。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from . import MAX_GRID

# 浮点归并容差（相对事件参数 t 的尺度）与网格线吸附容差（相对格宽）
T_MERGE_REL = 1e-12
GRID_SNAP_REL = 1e-10


class GeometryError(ValueError):
    """输入几何无法解释（非有限值、未知方向、网格非法等）。"""


@dataclass
class Grid:
    """规则矩形网格。单元索引 row=0..ny-1（最低 y 带），col=0..nx-1。"""

    x_min: float
    y_min: float
    dx: float
    dy: float
    nx: int
    ny: int
    mass_attenuation: float
    raw: dict = field(default_factory=dict)

    @property
    def x_max(self) -> float:
        return self.x_min + self.nx * self.dx

    @property
    def y_max(self) -> float:
        return self.y_min + self.ny * self.dy

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def cell_index(self, row: int, col: int) -> int:
        """扁平索引：行优先（同一行的列连续），idx = row*nx + col。"""
        return row * self.nx + col

    def rc_of(self, idx: int) -> tuple[int, int]:
        return int(idx // self.nx), int(idx % self.nx)

    def cell_xy(self, idx: int) -> tuple[float, float]:
        """单元中心坐标，供独立积分路径等使用。"""
        row, col = self.rc_of(idx)
        x = self.x_min + (col + 0.5) * self.dx
        y = self.y_min + (row + 0.5) * self.dy
        return x, y


def grid_from_dict(cfg: dict) -> Grid:
    """从 grid.json 内容构造 Grid，并做完整输入诊断。"""
    required = ("x_min_m", "y_min_m", "dx_m", "dy_m", "nx", "ny",
                "mass_attenuation_m2_per_kg")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise GeometryError(f"grid.json 缺少必要字段: {missing}")
    vals = [cfg[k] for k in required]
    if not all(math.isfinite(float(v)) for v in vals[:4] + [vals[6]]):
        raise GeometryError("grid.json 含非有限数值")
    nx, ny = int(cfg["nx"]), int(cfg["ny"])
    if nx <= 0 or ny <= 0:
        raise GeometryError(f"网格维数必须为正，得到 nx={nx}, ny={ny}")
    if nx > MAX_GRID or ny > MAX_GRID:
        raise GeometryError(
            f"超过题目约束 {MAX_GRID}x{MAX_GRID}: nx={nx}, ny={ny}")
    dx, dy = float(cfg["dx_m"]), float(cfg["dy_m"])
    if dx <= 0 or dy <= 0:
        raise GeometryError(f"格宽必须为正，得到 dx={dx}, dy={dy}")
    mu = float(cfg["mass_attenuation_m2_per_kg"])
    if mu <= 0:
        raise GeometryError(f"质量衰减常数必须为正，得到 {mu}")
    return Grid(float(cfg["x_min_m"]), float(cfg["y_min_m"]),
                dx, dy, nx, ny, mu, raw=dict(cfg))


def clip_ray_to_region(src: np.ndarray, dst: np.ndarray,
                       grid: Grid) -> tuple[np.ndarray, np.ndarray] | None:
    """把射线 P(t)=src+t*(dst-src), t in [0,1] 裁到闭区域 [x_min,x_max]x[y_min,y_max]。

    返回裁切后的 (entry, exit) 端点；完全不相交（含仅在角点相切、
    内部长度为零）返回 None。
    """
    d = dst - src
    t_lo, t_hi = 0.0, 1.0
    # 两个轴分别做 slab 求交，逐轴收紧参数区间
    for axis, lo, hi in ((0, grid.x_min, grid.x_max),
                         (1, grid.y_min, grid.y_max)):
        p0, v = src[axis], d[axis]
        if abs(v) < 1e-300:
            # 与 slab 边界平行：起点必须在闭区间内，否则整线在区域外
            if p0 < lo or p0 > hi:
                return None
            continue
        t1 = (lo - p0) / v
        t2 = (hi - p0) / v
        t_enter, t_exit = (t1, t2) if t1 < t2 else (t2, t1)
        t_lo = max(t_lo, t_enter)
        t_hi = min(t_hi, t_exit)
        if t_lo >= t_hi:
            # 退化（t_lo == t_hi）只可能是单点接触（角点/边线），
            # 内部路径长度为零，按完全不相交处理
            return None
    return src + t_lo * d, src + t_hi * d


def ray_cell_segments(src: np.ndarray, dst: np.ndarray,
                      grid: Grid) -> dict[int, float]:
    """计算一条射线在各格中的实际路径长度。

    返回 {扁平单元索引: 长度}，仅含有正长度的单元。
    射线与区域不相交时返回空 dict。
    输入含非有限值或零长度/未知方向时抛 GeometryError。
    """
    pts = np.asarray([src, dst], dtype=float)
    if not np.isfinite(pts).all():
        raise GeometryError("射线端点含非有限值（NaN/Inf）")
    src, dst = pts[0], pts[1]
    d = dst - src
    length_total = math.hypot(d[0], d[1])
    if length_total == 0.0:
        raise GeometryError("射线源与接收点重合：零长度、方向未知")

    clipped = clip_ray_to_region(src, dst, grid)
    if clipped is None:
        return {}
    entry, exit_ = clipped
    inside_len = math.hypot(*(exit_ - entry))
    if inside_len <= T_MERGE_REL * max(length_total, 1.0):
        return {}

    # 裁切段在原射线参数 t 下的区间（重算保证与端点同参数化）
    t0 = float(np.dot(entry - src, d) / np.dot(d, d))
    t1 = float(np.dot(exit_ - src, d) / np.dot(d, d))

    # 收集裁切段内部跨越的全部内部网格线事件（x、y 两个方向）
    events: list[float] = [t0, t1]
    tol_t = T_MERGE_REL * max(1.0, abs(t1 - t0))
    for axis, p_min, step, n in ((0, grid.x_min, grid.dx, grid.nx),
                                 (1, grid.y_min, grid.dy, grid.ny)):
        v = d[axis]
        if abs(v) < 1e-300:
            continue  # 平行于该组网格线，无跨越
        # 严格位于裁切段内部的内部网格线 k=1..n-1
        k_lo = math.ceil((min(entry[axis], exit_[axis]) - p_min) / step
                         - GRID_SNAP_REL)
        k_hi = math.floor((max(entry[axis], exit_[axis]) - p_min) / step
                          + GRID_SNAP_REL)
        for k in range(max(1, k_lo), min(n, k_hi) + 1):
            line = p_min + k * step
            t = (line - src[axis]) / v
            if t0 + tol_t < t < t1 - tol_t:
                events.append(t)

    # 排序并归并重合事件：穿过角点时 x/y 跨越同 t，只保留一个事件，
    # 因此角点两侧各算一段、各归一个单元，绝不双计
    events.sort()
    merged = [events[0]]
    for t in events[1:]:
        if t - merged[-1] > tol_t:
            merged.append(t)
        # 重合（角点）事件直接丢弃，不产生零长段

    out: dict[int, float] = {}
    for ta, tb in zip(merged[:-1], merged[1:]):
        tm = 0.5 * (ta + tb)
        mid = src + tm * d
        # 吸附到恰好位于网格线上的段中点（理论上仅边界/数值情形）
        gx = (mid[0] - grid.x_min) / grid.dx
        gy = (mid[1] - grid.y_min) / grid.dy
        if abs(gx - round(gx)) <= GRID_SNAP_REL:
            gx = round(gx)
        if abs(gy - round(gy)) <= GRID_SNAP_REL:
            gy = round(gy)
        col = int(math.floor(gx))
        row = int(math.floor(gy))
        # 最外侧最大边界为闭区间；数值误差下夹回合法范围
        col = min(max(col, 0), grid.nx - 1)
        row = min(max(row, 0), grid.ny - 1)
        seg_len = (tb - ta) * length_total
        if seg_len > 0.0:
            idx = grid.cell_index(row, col)
            out[idx] = out.get(idx, 0.0) + seg_len
    return out


@dataclass
class RayGeometry:
    """一条射线的几何处理结果。"""

    ray_id: str
    src: np.ndarray
    dst: np.ndarray
    segments: dict[int, float]          # 空 dict 表示与区域不相交
    clipped_length: float               # 区域内裁切段总长；不相交为 0
    total_length: float                 # 源-接收点全长
    error: str | None = None            # 几何诊断信息（非有限等）


def build_forward_operator(geoms: list[RayGeometry], grid: Grid
                           ) -> tuple[sparse.csr_matrix, np.ndarray]:
    """根据每条射线的逐格长度构建稀疏前向算子 A（路径长度，米）。

    A 的行按 geoms 顺序、列按扁平单元索引。完全不相交的射线得到全零行
    （而不是被误填成穿满区域的观测）；其行同时由返回的布尔数组标记。
    """
    rows, cols, data = [], [], []
    zero_row = np.zeros(len(geoms), dtype=bool)
    for i, g in enumerate(geoms):
        if not g.segments:
            zero_row[i] = True
            continue
        for idx, length in g.segments.items():
            rows.append(i)
            cols.append(idx)
            data.append(length)
    A = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(geoms), grid.n_cells), dtype=float).tocsr()
    return A, zero_row
