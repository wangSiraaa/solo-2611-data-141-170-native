"""姿态几何：body-FRD -> world-NED。

旋转矩阵由单位四元数显式构造（不依赖外部旋转约定），
杆臂与波束初始方向在**射线追踪之前**就旋到世界系，
姿态影响进入射线的初始方向，而不是事后旋转射线结果。
"""

from __future__ import annotations

import numpy as np


def quat_wxyz_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """单位四元数 (w,x,y,z) 转 3x3 旋转矩阵（body -> world）。

    采用 FRD/NED 通用的 Hamilton 约定：world 向量 = R @ body 向量。
    """
    w, x, y, z = (float(v) for v in np.asarray(q_wxyz, dtype=float))
    n = w * w + x * x + y * y + z * z
    if n <= 0.0 or not np.isfinite(n):
        raise ValueError("非法四元数：范数非正或含非有限分量")
    # 归一化，消除输入的微小范数偏差
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=float,
    )


def body_beam_direction(angle_rad: float) -> np.ndarray:
    """船体系波束初始方向。

    配置协议：``[0, sin(angle), cos(angle)]``，angle 为弧度，
    正角指向右舷（+y），竖直向下波束 angle=0。该向量本身为单位向量。
    """
    return np.array([0.0, np.sin(angle_rad), np.cos(angle_rad)], dtype=float)


def rotate_body_to_world(
    rot: np.ndarray, vec_body: np.ndarray
) -> np.ndarray:
    """把 body-FRD 向量旋到 world-NED。"""
    return np.asarray(rot, dtype=float) @ np.asarray(vec_body, dtype=float)


def transducer_position_ned(
    body_position_ned: np.ndarray, rot: np.ndarray, lever_arm_body: np.ndarray
) -> np.ndarray:
    """换能器世界系位置 = 船体参考点位置 + R_body_to_world @ 杆臂。"""
    return (
        np.asarray(body_position_ned, dtype=float)
        + rot @ np.asarray(lever_arm_body, dtype=float)
    )


def is_downward(direction_world: np.ndarray, tol: float = 1e-12) -> bool:
    """世界系波束方向是否向下（NED 的 down 分量为正）。

    水平（down≈0）或向上的波束无法用本归位模型处理，判为非向下。
    """
    return float(np.asarray(direction_world, dtype=float)[2]) > tol
