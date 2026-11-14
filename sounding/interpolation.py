"""时间序列插值。

规则（题目协议）：

* 只在目标时刻被两条合法记录**包围**时才插值，精确命中采样时刻直接取该记录；
* 包围间隔超过 ``max_interpolation_gap_ms`` 时拒绝插值，报告缺口；
* 范围外一律报告越界，**不**回退到最近邻；
* 位置、潮位线性插值；姿态使用单位四元数**短弧** slerp——``q`` 与 ``-q``
  表示同一姿态，插值前先把右端四元数翻转到与左端同一半球，避免线性混成零。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 包围状态码
EXACT = "exact_sample"          # 精确命中采样时刻，直接使用
INTERPOLATED = "interpolated"   # 合法包围且间隔在门槛内
OUT_OF_RANGE = "out_of_range"   # 目标时刻早于首条或晚于末条记录
GAP_TOO_LARGE = "gap_too_large"  # 包围间隔超过门槛
NO_DATA = "no_data"             # 序列为空

# 四元数范数容差：单位四元数允许的相对偏差
QUAT_NORM_TOL = 1e-6


@dataclass(frozen=True)
class Bracket:
    """目标时刻在时间序列中的包围信息。

    Attributes:
        status: EXACT / INTERPOLATED / OUT_OF_RANGE / GAP_TOO_LARGE / NO_DATA。
        left_index, right_index: 包围记录下标；精确命中时二者相等。
        alpha: 区间内线性比例 ``(t - t_left) / (t_right - t_left)``，
            精确命中时为 0。
        left_time_ms, right_time_ms: 包围记录时间戳；无包围时为 NaN。
        gap_ms: 包围间隔（毫秒）；无包围时为 NaN。
    """

    status: str
    left_index: int | None
    right_index: int | None
    alpha: float
    left_time_ms: float
    right_time_ms: float
    gap_ms: float

    @property
    def usable(self) -> bool:
        """该包围能否支撑一次取值（精确命中或合法插值）。"""
        return self.status in (EXACT, INTERPOLATED)


def find_bracket(times: np.ndarray, t_ms: float, max_gap_ms: float) -> Bracket:
    """在单调不减的时间序列中定位 ``t_ms`` 的包围。

    绝不返回最近邻：范围外给 :data:`OUT_OF_RANGE`，缺口超门槛给
    :data:`GAP_TOO_LARGE`，只有精确命中或合法包围才标记可用。
    """
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return Bracket(NO_DATA, None, None, np.nan, np.nan, np.nan, np.nan)

    j = int(np.searchsorted(times, t_ms, side="left"))

    # 精确命中（含重复时间戳，取第一个命中位置）
    if j < times.size and times[j] == t_ms:
        return Bracket(EXACT, j, j, 0.0, float(times[j]), float(times[j]), 0.0)

    # 范围外：不做最近邻
    if j == 0 or j == times.size:
        return Bracket(
            OUT_OF_RANGE, None, None, np.nan, np.nan, np.nan, np.nan
        )

    i = j - 1
    t_left, t_right = float(times[i]), float(times[j])
    gap = t_right - t_left
    if gap > max_gap_ms + 1e-9:
        return Bracket(GAP_TOO_LARGE, i, j, np.nan, t_left, t_right, gap)

    alpha = (t_ms - t_left) / gap
    return Bracket(INTERPOLATED, i, j, alpha, t_left, t_right, gap)


def linear_interpolate(
    values: np.ndarray, bracket: Bracket
) -> tuple[np.ndarray | None, str | None]:
    """按包围结果对位置/潮位做线性插值。

    Args:
        values: 一维或二维（行对应记录）数组。
        bracket: :func:`find_bracket` 的结果。

    Returns:
        (插值结果, None) 或 (None, 原因代码)。精确命中时直接返回该记录。
    """
    values = np.asarray(values, dtype=float)
    if bracket.status == EXACT:
        return values[bracket.left_index].copy(), None
    if bracket.status != INTERPOLATED:
        return None, bracket.status
    v0 = values[bracket.left_index]
    v1 = values[bracket.right_index]
    return v0 + bracket.alpha * (v1 - v0), None


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """单位四元数最短弧球面线性插值，输入/输出均为 (w, x, y, z)。

    先把 ``q1`` 翻转到与 ``q0`` 同一半球（``q`` 与 ``-q`` 同一姿态），
    再走标准 slerp；两姿态几乎重合时退化为 ``q0``，避免数值抖动。
    本函数不做线性混合，因此不会出现零四元数。
    """
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # 同一姿态选短弧半球
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)

    if 1.0 - dot < 1e-10:  # 旋转角可忽略
        return q0.copy()

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w0 = np.sin((1.0 - alpha) * theta) / sin_theta
    w1 = np.sin(alpha * theta) / sin_theta
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q)


def interpolate_quaternion(
    quats_wxyz: np.ndarray, bracket: Bracket
) -> tuple[np.ndarray | None, str | None]:
    """按包围结果取姿态四元数（精确命中或短弧 slerp）。

    同时校验端点四元数的范数；非法姿态单独报告，不被插值悄悄掩盖。
    """
    quats = np.asarray(quats_wxyz, dtype=float)
    if not bracket.usable:
        return None, bracket.status

    indices = (
        (bracket.left_index,)
        if bracket.status == EXACT
        else (bracket.left_index, bracket.right_index)
    )
    for k in indices:
        norm = float(np.linalg.norm(quats[k]))
        if abs(norm - 1.0) > QUAT_NORM_TOL or not np.isfinite(norm):
            return None, f"attitude_not_unit_quaternion_at_index_{k}"

    if bracket.status == EXACT:
        q = quats[bracket.left_index]
        return q / np.linalg.norm(q), None
    return (
        slerp_wxyz(
            quats[bracket.left_index],
            quats[bracket.right_index],
            bracket.alpha,
        ),
        None,
    )
