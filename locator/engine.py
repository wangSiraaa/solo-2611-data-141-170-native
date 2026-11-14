"""定位引擎:空间网格剖面(逐点重估发震时刻)+ 多初值局部优化 + 支持区域分析。

搜索策略:
1. 在固定坐标盒内以 plan 给定步长建立三维网格,每个格点用 IRLS 重估最优
   发震时刻并计算 Huber 代价,得到完整代价剖面(不做单次局部优化就下结论);
2. 从支持区域各连通块最小点、代价最优的一批分散格点及盒中心/台阵中心
   等多初值出发做稳健局部优化,聚类去重得到全部不同极小点;
3. 报告代价 <= 最小值 + delta 的近优支持区域、竞争解、深度剖面与简并分析。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, optimize

from .io import Arrival, Plan, Station
from .model import (cost as huber_cost, huber_location, huber_rho,
                    huber_weight, standardized_residuals)

HUBER_C = 1.5               # 稳健尺度,单位为标准差
OPT_TOL = 1e-12             # 局部优化 ftol/xtol/gtol
T0_IRLS_TOL = 1e-10         # 网格上 t0 IRLS 收敛容差 (s)
CLUSTER_POS_TOL_KM = 0.5    # 解聚类半径(位置)
CLUSTER_T0_TOL_S = 0.5      # 解聚类半径(发震时刻)
MAX_REPORTED_MINIMA = 25


@dataclass
class DataSet:
    stations: dict[str, Station]
    arrivals: list[Arrival]
    pick_station_xyz: np.ndarray   # (n_pick, 3)
    inv_vel: np.ndarray            # (n_pick,)
    obs: np.ndarray                # (n_pick,)
    sigma: np.ndarray              # (n_pick,)


def build_dataset(stations: dict[str, Station],
                  arrivals: list[Arrival],
                  plan: Plan) -> DataSet:
    xyz, inv_vel, obs, sigma = [], [], [], []
    for a in arrivals:
        st = stations[a.station_id]
        xyz.append([st.x_km, st.y_km, st.depth_km])
        inv_vel.append(1.0 / plan.velocities[a.phase])
        obs.append(a.time_s)
        sigma.append(a.sigma_s)
    return DataSet(
        stations=stations,
        arrivals=arrivals,
        pick_station_xyz=np.asarray(xyz),
        inv_vel=np.asarray(inv_vel),
        obs=np.asarray(obs),
        sigma=np.asarray(sigma),
    )


def _axis(lo: float, hi: float, step: float) -> np.ndarray:
    n = int(round((hi - lo) / step))
    ax = lo + step * np.arange(n + 1)
    ax[-1] = min(ax[-1], hi)
    return ax


def _residual_fn(data: DataSet):
    def fun(p: np.ndarray) -> np.ndarray:
        return standardized_residuals(p, data.pick_station_xyz, data.inv_vel,
                                      data.obs, data.sigma)
    return fun


def _jacobian_fn(data: DataSet):
    def jac(p: np.ndarray) -> np.ndarray:
        diff = p[:3] - data.pick_station_xyz          # (n,3)
        dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
        ddist = diff / dist[:, None]                  # d dist/d [x,y,z]
        g = np.empty((len(data.obs), 4))
        g[:, :3] = -ddist * data.inv_vel[:, None] / data.sigma[:, None]
        g[:, 3] = -1.0 / data.sigma
        return g
    return jac


def _local_refine(data: DataSet, plan: Plan, c: float, p0: np.ndarray):
    res = optimize.least_squares(
        _residual_fn(data), p0, jac=_jacobian_fn(data),
        bounds=([plan.x_bounds[0], plan.y_bounds[0], plan.depth_bounds[0],
                 plan.origin_time_bounds[0]],
                [plan.x_bounds[1], plan.y_bounds[1], plan.depth_bounds[1],
                 plan.origin_time_bounds[1]]),
        method="trf", loss="huber", f_scale=c,
        xtol=OPT_TOL, ftol=OPT_TOL, gtol=OPT_TOL, max_nfev=2000,
        x_scale="jac",
    )
    p = np.asarray(res.x)
    return p, huber_cost(p, data.pick_station_xyz, data.inv_vel,
                         data.obs, data.sigma, c)


def _t0_at(data: DataSet, plan: Plan, c: float, point) -> float:
    tt = np.linalg.norm(data.pick_station_xyz - np.asarray(point), axis=1) \
        * data.inv_vel
    a = (data.obs - tt)[None, :]
    t0, _ = huber_location(a, data.sigma, c,
                           plan.origin_time_bounds[0],
                           plan.origin_time_bounds[1], tol=T0_IRLS_TOL)
    return float(t0[0])


def _cluster(solutions: list[dict]) -> list[dict]:
    """按 (位置 0.5 km, t0 0.5 s) 聚类,每簇保留代价最小者。"""
    clusters: list[list[dict]] = []
    for s in sorted(solutions, key=lambda d: d["cost"]):
        for cl in clusters:
            r = cl[0]
            if (max(abs(s["x_km"] - r["x_km"]), abs(s["y_km"] - r["y_km"]),
                    abs(s["depth_km"] - r["depth_km"])) <= CLUSTER_POS_TOL_KM
                    and abs(s["origin_time_s"] - r["origin_time_s"]) <= CLUSTER_T0_TOL_S):
                cl.append(s)
                break
        else:
            clusters.append([s])
    reps = [min(cl, key=lambda d: d["cost"]) for cl in clusters]
    return sorted(reps, key=lambda d: d["cost"])


def _sol_dict(p: np.ndarray, cst: float, data: DataSet) -> dict:
    r = standardized_residuals(p, data.pick_station_xyz, data.inv_vel,
                               data.obs, data.sigma)
    return {
        "x_km": float(p[0]), "y_km": float(p[1]), "depth_km": float(p[2]),
        "origin_time_s": float(p[3]), "cost": float(cst),
        "rms_standardized_residual": float(np.sqrt(np.mean(r * r))),
        "max_abs_standardized_residual": float(np.max(np.abs(r))),
    }


def run_location(stations: dict[str, Station],
                 arrivals: list[Arrival],
                 plan: Plan,
                 c: float = HUBER_C) -> dict:
    data = build_dataset(stations, arrivals, plan)

    # ---------- 1. 空间网格剖面(逐格点重估 t0) ----------
    xs = _axis(*plan.x_bounds, plan.profile_grid_step_km)
    ys = _axis(*plan.y_bounds, plan.profile_grid_step_km)
    zs = _axis(*plan.depth_bounds, plan.profile_grid_step_km)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    tt = np.linalg.norm(pts[:, None, :] - data.pick_station_xyz[None, :, :],
                        axis=2) * data.inv_vel[None, :]
    a = data.obs[None, :] - tt
    t0_flat, t0_iters = huber_location(a, data.sigma, c,
                                       *plan.origin_time_bounds,
                                       tol=T0_IRLS_TOL)
    r = (a - t0_flat[:, None]) / data.sigma[None, :]
    cost_flat = huber_rho(r, c).sum(axis=1)
    shape = (len(xs), len(ys), len(zs))
    cost3d = cost_flat.reshape(shape)
    t0_3d = t0_flat.reshape(shape)

    min_cost = float(cost_flat.min())
    delta = plan.profile_score_delta
    support = cost3d <= min_cost + delta

    # ---------- 2. 支持区域:连通块、范围、边界接触 ----------
    labels, n_comp = ndimage.label(support, structure=np.ones((3, 3, 3)))
    components = []
    for cid in range(1, n_comp + 1):
        idx = np.argwhere(labels == cid)
        costs_c = cost3d[labels == cid]
        k = int(np.argmin(costs_c))
        i0, i1, i2 = idx[k]
        components.append({
            "id": cid,
            "n_voxels": int(len(idx)),
            "min_cost": float(costs_c[k]),
            "min_location": {
                "x_km": float(xs[i0]), "y_km": float(ys[i1]),
                "depth_km": float(zs[i2]),
                "origin_time_s": float(t0_3d[i0, i1, i2]),
            },
            "extent_km": {
                "x": [float(xs[idx[:, 0].min()]), float(xs[idx[:, 0].max()])],
                "y": [float(ys[idx[:, 1].min()]), float(ys[idx[:, 1].max()])],
                "depth": [float(zs[idx[:, 2].min()]), float(zs[idx[:, 2].max()])],
            },
        })
    components.sort(key=lambda d: d["min_cost"])

    touches = []
    if support[0, :, :].any():
        touches.append(f"x={plan.x_bounds[0]}")
    if support[-1, :, :].any():
        touches.append(f"x={plan.x_bounds[1]}")
    if support[:, 0, :].any():
        touches.append(f"y={plan.y_bounds[0]}")
    if support[:, -1, :].any():
        touches.append(f"y={plan.y_bounds[1]}")
    if support[:, :, 0].any():
        touches.append(f"depth={plan.depth_bounds[0]}")
    if support[:, :, -1].any():
        touches.append(f"depth={plan.depth_bounds[1]}")

    # ---------- 3. 多初值局部优化 ----------
    starts: list[np.ndarray] = []
    for comp in components:                       # 每个连通块的最小格点
        loc = comp["min_location"]
        starts.append(np.array([loc["x_km"], loc["y_km"], loc["depth_km"],
                                loc["origin_time_s"]]))
    order = np.argsort(cost_flat)                 # 分散的低成本格点
    taken: list[np.ndarray] = []
    sep = 2.0 * plan.profile_grid_step_km
    for gi in order:
        p3 = pts[gi]
        if all(np.max(np.abs(p3 - q)) >= sep for q in taken):
            taken.append(p3)
            starts.append(np.array([p3[0], p3[1], p3[2], t0_flat[gi]]))
        if len(taken) >= 24:
            break
    centroid = np.array([
        0.5 * sum(plan.x_bounds), 0.5 * sum(plan.y_bounds),
        0.5 * sum(plan.depth_bounds)])
    starts.append(np.append(centroid, _t0_at(data, plan, c, centroid)))
    sta_mean = data.pick_station_xyz.mean(axis=0)
    p_mid = np.array([sta_mean[0], sta_mean[1], 0.5 * sum(plan.depth_bounds)])
    starts.append(np.append(p_mid, _t0_at(data, plan, c, p_mid)))

    refined, n_converged = [], 0
    for p0 in starts:
        p0 = np.clip(p0, [plan.x_bounds[0], plan.y_bounds[0],
                          plan.depth_bounds[0], plan.origin_time_bounds[0]],
                     [plan.x_bounds[1], plan.y_bounds[1],
                      plan.depth_bounds[1], plan.origin_time_bounds[1]])
        p, cst = _local_refine(data, plan, c, p0)
        refined.append(_sol_dict(p, cst, data))
        n_converged += 1
    minima = _cluster(refined)
    best = minima[0]
    for rank, m in enumerate(minima):
        m["rank"] = rank + 1
        m["within_support_delta"] = bool(m["cost"] <= min_cost + delta + 1e-9)

    # ---------- 4. 最优解处局部曲率(仅线性化诊断) ----------
    p_best = np.array([best["x_km"], best["y_km"], best["depth_km"],
                       best["origin_time_s"]])
    J = _jacobian_fn(data)(p_best)
    w = huber_weight(_residual_fn(data)(p_best), c)
    H = (J * w[:, None]).T @ J
    eigval, eigvec = np.linalg.eigh(H)
    eigval = np.maximum(eigval, 0.0)
    cov = np.linalg.pinv(H, rcond=1e-12)
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    lam_max = float(eigval[-1]) if eigval.size else 0.0
    lam_min = float(eigval[0]) if eigval.size else 0.0
    curvature = {
        "param_std": {"x_km": float(std[0]), "y_km": float(std[1]),
                      "depth_km": float(std[2]), "origin_time_s": float(std[3])},
        "hessian_eigenvalues": [float(v) for v in eigval],
        "weakest_direction": {
            "x": float(eigvec[0, 0]), "y": float(eigvec[1, 0]),
            "depth": float(eigvec[2, 0]), "origin_time": float(eigvec[3, 0])},
        "condition_number": (lam_max / lam_min) if lam_min > 0 else float("inf"),
        "note": "仅最优解处的局部线性化(稳健加权 J^T W J),"
                "不能替代网格多解检查;弱几何下本矩阵病态即局部不可分辩。",
    }

    # ---------- 5. 残差表(最优解,全部 pick,含降权) ----------
    r_best = _residual_fn(data)(p_best)
    w_best = huber_weight(r_best, c)
    tt_best = np.linalg.norm(data.pick_station_xyz - p_best[:3], axis=1) \
        * data.inv_vel
    residual_rows = []
    for i, arr in enumerate(data.arrivals):
        residual_rows.append({
            "pick_id": arr.pick_id,
            "station_id": arr.station_id,
            "phase": arr.phase,
            "observed_s": arr.time_s,
            "predicted_s": float(p_best[3] + tt_best[i]),
            "travel_time_s": float(tt_best[i]),
            "origin_time_s": float(p_best[3]),
            "residual_s": float(arr.time_s - p_best[3] - tt_best[i]),
            "sigma_s": arr.sigma_s,
            "standardized_residual": float(r_best[i]),
            "robust_weight": float(w_best[i]),
            "downweighted": bool(w_best[i] < 0.999),
        })

    # ---------- 6. 深度剖面与简并分析 ----------
    depth_profile = []
    for k, z in enumerate(zs):
        sl = cost3d[:, :, k]
        i0, i1 = np.unravel_index(int(np.argmin(sl)), sl.shape)
        depth_profile.append({
            "depth_km": float(z),
            "min_cost": float(sl[i0, i1]),
            "x_km": float(xs[i0]),
            "y_km": float(ys[i1]),
            "origin_time_s": float(t0_3d[i0, i1, k]),
        })

    sup_idx = np.argwhere(support)
    degeneracy: dict = {}
    if len(sup_idx) >= 3:
        yy = ys[sup_idx[:, 1]].astype(float)
        zz = zs[sup_idx[:, 2]].astype(float)
        tt0 = np.array([t0_3d[tuple(i)] for i in sup_idx])
        r2 = yy ** 2 + zz ** 2
        degeneracy = {
            "support_origin_time_range_s": [float(tt0.min()), float(tt0.max())],
            "support_depth_range_km": [float(zz.min()), float(zz.max())],
            "support_abs_y_range_km": [float(np.abs(yy).min()),
                                       float(np.abs(yy).max())],
            "y2_plus_z2_mean_km2": float(r2.mean()),
            "y2_plus_z2_std_km2": float(r2.std()),
            "corr_depth_abs_y": (float(np.corrcoef(zz, np.abs(yy))[0, 1])
                                 if zz.std() > 0 and np.abs(yy).std() > 0 else None),
        }

    picks_by_phase: dict[str, int] = {}
    for arr in arrivals:
        picks_by_phase[arr.phase] = picks_by_phase.get(arr.phase, 0) + 1

    return {
        "plan": plan, "data": data, "huber_c": c,
        "grid": {"x_km": xs, "y_km": ys, "depth_km": zs,
                 "cost": cost3d, "origin_time_s": t0_3d},
        "min_cost": min_cost, "support_mask": support,
        "support": {
            "delta": delta,
            "n_voxels": int(support.sum()),
            "volume_km3": float(support.sum() * plan.profile_grid_step_km ** 3),
            "extent_km": {
                "x": [float(xs[sup_idx[:, 0].min()]), float(xs[sup_idx[:, 0].max()])],
                "y": [float(ys[sup_idx[:, 1].min()]), float(ys[sup_idx[:, 1].max()])],
                "depth": [float(zs[sup_idx[:, 2].min()]), float(zs[sup_idx[:, 2].max()])],
            } if len(sup_idx) else None,
            "touches_boundary": touches,
            "components": components,
        },
        "best": best,
        "minima": minima[:MAX_REPORTED_MINIMA],
        "n_distinct_minima": len(minima),
        "local_curvature": curvature,
        "residuals": residual_rows,
        "depth_profile": depth_profile,
        "degeneracy": degeneracy,
        "picks_by_phase": picks_by_phase,
        "diagnostics": {
            "grid_step_km": plan.profile_grid_step_km,
            "grid_shape": list(shape),
            "n_grid_points": int(np.prod(shape)),
            "optimizer": "scipy.optimize.least_squares (trf, loss=huber, "
                         f"f_scale={c}, x_scale=jac)",
            "optimizer_tol": OPT_TOL,
            "t0_irls_tol_s": T0_IRLS_TOL,
            "t0_irls_max_iters_used": int(t0_iters),
            "cluster_pos_tol_km": CLUSTER_POS_TOL_KM,
            "cluster_t0_tol_s": CLUSTER_T0_TOL_S,
            "n_starts": len(starts),
            "n_refined": n_converged,
        },
    }
