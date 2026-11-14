"""正演模型、稳健代价与发震时刻估计。

模型(固定,不反演速度结构):
    arrival_i = t0 + |source - station_i| / velocity[phase_i]

稳健目标(对少数误拾取稳健):
    r_i = (t_i^obs - t0 - tt_i) / sigma_i          标准化残差
    cost = sum_i rho(r_i)
    rho(r) = r^2/2            |r| <= c
           = c|r| - c^2/2     |r| >  c            (Huber)
尺度 c 以标准差为单位,默认 1.5。同一代价同时用于局部优化与网格剖面,
使 plan 的 profile_score_delta 对两者口径一致。
"""
from __future__ import annotations

import numpy as np


def huber_rho(r: np.ndarray, c: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= c, 0.5 * r * r, c * a - 0.5 * c * c)


def huber_weight(r: np.ndarray, c: float) -> np.ndarray:
    """IRLS 权重 w = psi(r)/r,|r|<=c 时为 1,否则 c/|r|。"""
    a = np.abs(r)
    return np.where(a <= c, 1.0, c / np.maximum(a, 1e-300))


def travel_time_matrix(points: np.ndarray,
                       pick_station_xyz: np.ndarray,
                       inv_vel: np.ndarray) -> np.ndarray:
    """points (...,3) 到各 pick 台站的走时 (...,n_pick)。"""
    pts = np.asarray(points, dtype=float)
    d = np.linalg.norm(pts[..., None, :] - pick_station_xyz, axis=-1)
    return d * inv_vel


def standardized_residuals(params: np.ndarray,
                           pick_station_xyz: np.ndarray,
                           inv_vel: np.ndarray,
                           obs: np.ndarray,
                           sigma: np.ndarray) -> np.ndarray:
    """params = [x, y, depth, t0]。"""
    tt = travel_time_matrix(np.asarray(params[:3]), pick_station_xyz, inv_vel)
    return (obs - params[3] - tt) / sigma


def cost(params: np.ndarray,
         pick_station_xyz: np.ndarray,
         inv_vel: np.ndarray,
         obs: np.ndarray,
         sigma: np.ndarray,
         c: float) -> float:
    r = standardized_residuals(params, pick_station_xyz, inv_vel, obs, sigma)
    return float(np.sum(huber_rho(r, c)))


def huber_location(a: np.ndarray,
                   sigma: np.ndarray,
                   c: float,
                   lo: float,
                   hi: float,
                   tol: float = 1e-10,
                   max_iter: int = 200) -> tuple[np.ndarray, int]:
    """给定空间位置,求使 sum rho((a_i - t0)/sigma_i) 最小的 t0。

    a: (n_points, n_pick) = obs - travel_time。IRLS 不动点迭代,
    每步裁剪到 [lo, hi](一维凸问题,裁剪即约束最优)。
    返回 (t0 数组 (n_points,), 实际迭代次数)。
    """
    a = np.atleast_2d(a)
    inv_s2 = 1.0 / (sigma * sigma)
    t = (a * inv_s2).sum(axis=1) / inv_s2.sum()
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        r = (a - t[:, None]) / sigma
        w = huber_weight(r, c) * inv_s2
        t_new = np.clip((w * a).sum(axis=1) / w.sum(axis=1), lo, hi)
        if np.max(np.abs(t_new - t)) < tol:
            t = t_new
            break
        t = t_new
    return t, n_iter
