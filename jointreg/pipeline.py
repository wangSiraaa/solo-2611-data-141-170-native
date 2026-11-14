"""端到端联合配准流程与结果组装。

输出中的所有残差均回到原始观测坐标系、以米为单位；
holdout 观测只用于跨站一致性检查，从不进入姿态/标靶拟合。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adjust import HUBER_K_SIGMA, JointModel
from .graph import analyze_graph
from .init import initialize
from .io import Dataset, load_dataset


def _world(model: JointModel, x: np.ndarray, station: str, local: np.ndarray):
    R, t = model.station_Rt(station, x)
    return R @ local + t


def _extract(model: JointModel, x: np.ndarray):
    R, t = {}, {}
    for s in model.layout.stations:
        R[s], t[s] = model.station_Rt(s, x)
    X = {tg: model.target_X(tg, x) for tg in model.layout.targets}
    return R, t, X


def _obs_residual_rows(ds: Dataset, model: JointModel, x: np.ndarray,
                       comp_weights: np.ndarray | None, tag: str):
    """逐观测残差表（3 分量 + 范数 + 归一化 + 权重与处置依据）。"""
    rows = []
    e_all = model.residuals(x)
    u_all = e_all / np.repeat(model.sigma, 3)
    for k, o in enumerate(model.obs):
        e = e_all[3 * k:3 * k + 3]
        u = u_all[3 * k:3 * k + 3]
        if comp_weights is not None:
            w = comp_weights[3 * k:3 * k + 3].mean()
            umax = float(np.max(np.abs(u)))
            if umax > HUBER_K_SIGMA:
                action = "downweighted_huber"
                reason = (
                    f"归一化残差最大 {umax:.2f}σ > {HUBER_K_SIGMA:.0f}σ，"
                    f"Huber 平均权重 {w:.3f}；保留观测但降低影响，未删除"
                )
            else:
                action = "kept"
                reason = f"归一化残差 {umax:.2f}σ 未超过 Huber 拐点"
        else:
            w = 1.0
            action = "kept"
            reason = "普通最小二乘对照解，不做稳健降权"
        rows.append({
            "solution": tag,
            "station_id": o.station,
            "target_id": o.target,
            "source_row": o.row,
            "holdout": False,
            "resx_m": e[0], "resy_m": e[1], "resz_m": e[2],
            "residual_norm_m": float(np.linalg.norm(e)),
            "sigma_m": o.sigma,
            "max_abs_normalized_sigma": float(np.max(np.abs(u))),
            "robust_weight": float(w),
            "action": action,
            "basis": reason,
        })
    return rows


def _closure_metrics(ds: Dataset, graph, R, t, X, tag: str, downweighted: set | None = None):
    """站间闭环：共有标靶在两站各自变换后的世界坐标之差。

    只用拟合观测；每条站间边给出 RMS 与逐标靶明细。
    downweighted 给出被稳健解降权的 (station,target) 集合，
    汇总同时报告全部观测 RMS 与剔除降权观测后的内点 RMS——
    保留离群点（不删除）与展示真实闭环质量两件事分开。
    """
    downweighted = downweighted or set()
    obs_lookup = {(o.station, o.target): o for o in ds.fit_observations()}
    edge_rows, summary = [], []
    for edge in graph.edges:
        # 断开分量单独建模：只处理两端都在当前 R/t 字典内的边
        if edge.station_i not in R or edge.station_j not in t:
            continue
        difs, inlier_difs, flagged = [], [], set()
        for tg in edge.shared_fit_targets:
            oa = obs_lookup[(edge.station_i, tg)]
            ob = obs_lookup[(edge.station_j, tg)]
            wa = R[edge.station_i] @ oa.local + t[edge.station_i]
            wb = R[edge.station_j] @ ob.local + t[edge.station_j]
            d = wa - wb
            is_flag = (edge.station_i, tg) in downweighted or (
                edge.station_j, tg
            ) in downweighted
            if is_flag:
                flagged.add(tg)
            else:
                inlier_difs.append(d)
            difs.append(d)
            edge_rows.append({
                "solution": tag,
                "station_i": edge.station_i,
                "station_j": edge.station_j,
                "target_id": tg,
                "dx_m": d[0], "dy_m": d[1], "dz_m": d[2],
                "discrepancy_m": float(np.linalg.norm(d)),
                "downweighted_observation": is_flag,
            })
        if difs:
            rms = float(np.sqrt(np.mean(np.sum(np.array(difs) ** 2, axis=1))))
            row = {
                "solution": tag,
                "station_i": edge.station_i,
                "station_j": edge.station_j,
                "n_shared": edge.n_shared,
                "closure_rms_m": rms,
                "closure_max_m": float(max(np.linalg.norm(d) for d in difs)),
                "n_downweighted_targets": len(flagged),
            }
            if inlier_difs:
                row["inlier_closure_rms_m"] = float(
                    np.sqrt(np.mean(np.sum(np.array(inlier_difs) ** 2, axis=1)))
                )
                row["inlier_closure_max_m"] = float(
                    max(np.linalg.norm(d) for d in inlier_difs)
                )
            else:
                row["inlier_closure_rms_m"] = None
                row["inlier_closure_max_m"] = None
            summary.append(row)
    return edge_rows, summary


def _holdout_metrics(ds: Dataset, R, t, X_fit: dict, tag: str,
                     fit_reference: dict | None = None):
    """holdout 跨站一致性检查（不参与任何拟合）。

    * 若该标靶已由拟合观测估计出世界坐标（X_fit）：holdout 变换结果直接对参考检查；
    * 若标靶只在 holdout 中出现（如 T10）：用 fit_reference（由拟合观测经各自
      最终变换得到的加权世界坐标）作参考，并退化为看到该标靶各站的两两之差——
      仍可检查跨站一致性，只是该参考本身不参与任何平差。
    """
    rows = []
    ho = ds.holdout_observations()
    by_target: dict[str, list] = {}
    for o in ho:
        by_target.setdefault(o.target, []).append(o)
    for tg in sorted(by_target):
        world_pts = [
            (o.station, R[o.station] @ o.local + t[o.station], o.sigma)
            for o in by_target[tg]
        ]
        ref = X_fit.get(tg)
        if ref is None and fit_reference is not None:
            ref = fit_reference.get(tg)
        if ref is not None:
            for st, wp, sig in world_pts:
                d = wp - ref
                rows.append({
                    "solution": tag, "target_id": tg, "station_id": st,
                    "check_type": "against_fitted_target",
                    "dx_m": d[0], "dy_m": d[1], "dz_m": d[2],
                    "discrepancy_m": float(np.linalg.norm(d)),
                    "sigma_m": sig,
                })
        for a in range(len(world_pts)):
            for b in range(a + 1, len(world_pts)):
                d = world_pts[a][1] - world_pts[b][1]
                rows.append({
                    "solution": tag,
                    "target_id": tg,
                    "station_id": f"{world_pts[a][0]}|{world_pts[b][0]}",
                    "check_type": "cross_station_pair",
                    "dx_m": d[0], "dy_m": d[1], "dz_m": d[2],
                    "discrepancy_m": float(np.linalg.norm(d)),
                    "sigma_m": max(world_pts[a][2], world_pts[b][2]),
                })
    if rows:
        rms = float(np.sqrt(np.mean([r["discrepancy_m"] ** 2 for r in rows])))
    else:
        rms = None
    return rows, rms


def _station_status(graph, ident: dict | None, station: str) -> dict:
    """把局部支撑诊断与联合雅可比零空间合并为每站可识别状态。"""
    sup = graph.supports[station]
    status = {
        "connected_to_anchor": sup.connected_to_anchor,
        "hops_to_anchor": sup.distance_hops,
        "support_verdict": sup.report["verdict"],
        "rotational_dof_unconstrained": sup.report["rotational_dof_unconstrained"],
        "weak_axes": sup.report["weak_axes"],
        "singular_values_m": sup.report["singular_values_m"],
        "line_axis_local": sup.report["line_axis"],
        "plane_normal_local": sup.report["plane_normal"],
        "jacobian_null_rotation_axes": [],
        "jacobian_weak_rotation_axes": [],
        "identifiable": True,
        "status": "",
    }
    if not sup.connected_to_anchor:
        status["identifiable"] = False
        status["status"] = (
            "disconnected_from_anchor: 与锚站无共有标靶路径。分量内部姿态在其临时锚"
            "（分量内排序最小站，R=I,t=0）下可识别，分量 SVD 无内部零空间；"
            "但该临时锚相对 A 世界系的 3 旋转+3 平移共 6 个分量间基准自由度"
            "没有任何观测约束，属于图层面的不可识别，不能宣称得到了 A 系下的姿态"
        )
    elif sup.report["verdict"] == "collinear":
        status["identifiable"] = False
        ax = sup.report["line_axis"]
        status["jacobian_null_rotation_axes"].append({
            "axis_local": ax,
            "note": "共线支撑：绕直线轴的旋转在联合雅可比零空间中，优化器无法确定该角",
        })
        status["status"] = "collinear: 平移与另外两个旋转角可识别，绕直线轴旋转不可识别"
    elif sup.report["verdict"] == "single_point":
        status["identifiable"] = False
        status["status"] = "single_point: 仅平移可识别，3 个旋转自由度全部不可识别"
    elif sup.report["weak_axes"]:
        status["identifiable"] = True
        status["status"] = (
            f"full_but_weak: 结构可识别，但主轴 {sup.report['weak_axes']} "
            "方向几何跨度小，对应姿态角估计精度低（看协方差/弱奇异方向）"
        )
    else:
        status["status"] = "identifiable: 完整几何支撑且连通锚站，6 自由度均可识别"
    # 无论哪种状态都附加联合雅可比 SVD 的零空间/弱方向证据（断开站分量单独 SVD）
    if ident is not None:
        for nd in ident["null_directions"]:
            for mix in nd["parameter_mix"]:
                if mix["kind"] == "station_rotation" and mix["station"] == station:
                    status["jacobian_null_rotation_axes"].append({
                        "axis_param_space": mix["vector"],
                        "singular_value_ratio": nd["ratio"],
                    })
        for wd in ident["weak_directions"]:
            for mix in wd["parameter_mix"]:
                if mix["kind"] == "station_rotation" and mix["station"] == station:
                    status["jacobian_weak_rotation_axes"].append({
                        "axis_param_space": mix["vector"],
                        "singular_value_ratio": wd["ratio"],
                    })
    return status


@dataclass
class PipelineResult:
    ds: Dataset
    graph: object
    init: object
    robust: dict
    plain: dict
    ident: dict | None
    ident_components: list
    R: dict
    t: dict
    X: dict
    R_init: dict
    t_init: dict
    X_init: dict
    residual_rows: list
    closure_rows: list
    closure_summary: list
    holdout_rows: list
    holdout_rms: dict
    target_rows: list
    station_rows: list
    summary: dict


def run_pipeline(csv_path: str, anchor_json_path: str) -> PipelineResult:
    ds = load_dataset(csv_path, anchor_json_path)
    graph = analyze_graph(ds)
    init = initialize(ds, graph)

    fit = ds.fit_observations()
    # 长度尺度：全部拟合局部点到各自站心的中位距离，用于无量纲化与协方差解读
    spans = []
    for s in ds.stations:
        pts = np.array([o.local for o in ds.by_station(s)])
        if len(pts) >= 2:
            c = pts.mean(axis=0)
            spans.append(np.median(np.linalg.norm(pts - c, axis=1)))
    length_scale = float(np.median(spans)) if spans else 1.0

    # 断开分量：每个不含锚站的连通分量单独建模（临时锚=分量内排序最小站）
    comp_of: dict[str, str] = {}
    anchor_comp = set(graph.anchor_component)
    for s in graph.anchor_component:
        comp_of[s] = ds.anchor_station
    seen: set[str] = set()
    for s in graph.disconnected_stations:
        if s in seen:
            continue
        comp = {s}
        queue = [s]
        while queue:
            cur = queue.pop()
            for nxt in graph.adjacency[cur]:
                if nxt not in comp and nxt not in anchor_comp:
                    comp.add(nxt)
                    queue.append(nxt)
        local_anchor = sorted(comp)[0]
        for c in comp:
            comp_of[c] = local_anchor
        seen |= comp

    components = sorted(set(comp_of.values()))
    R_final, t_final, X_final = {}, {}, {}
    residual_rows, closure_rows, closure_summary = [], [], []
    ident_combined = None
    ident_components = []
    robust_meta, plain_meta = [], []
    downweighted_keys: set[tuple[str, str]] = set()

    def _init_residual_rows(comp_obs: list):
        """初始化（生成树挂载、闭环尚未参与）的逐观测残差，单位米。"""
        rows = []
        for o in comp_obs:
            e = init.R[o.station] @ o.local + init.t[o.station] - init.targets[o.target]
            rows.append({
                "solution": "initialization",
                "station_id": o.station,
                "target_id": o.target,
                "source_row": o.row,
                "holdout": False,
                "resx_m": e[0], "resy_m": e[1], "resz_m": e[2],
                "residual_norm_m": float(np.linalg.norm(e)),
                "sigma_m": o.sigma,
                "max_abs_normalized_sigma": float(np.max(np.abs(e / o.sigma))),
                "robust_weight": 1.0,
                "action": "kept",
                "basis": "生成树初始化残差（闭环边尚未参与姿态求解）",
            })
        return rows

    for comp_anchor in components:
        members = {s for s, ca in comp_of.items() if ca == comp_anchor}
        comp_obs = [o for o in fit if o.station in members]
        comp_targets = sorted({o.target for o in comp_obs})
        X0 = {tg: init.targets[tg] for tg in comp_targets}
        model = JointModel(comp_obs, comp_anchor, init.R, init.t, X0)

        plain = model.solve_plain()
        robust = model.solve_robust()

        Rp, tp, Xp = _extract(model, plain["x"])
        Rr, tr, Xr = _extract(model, robust["x"])
        R_final.update(Rr)
        t_final.update(tr)
        X_final.update(Xr)

        residual_rows += _init_residual_rows(comp_obs)
        residual_rows += _obs_residual_rows(
            ds, model, robust["x"], robust["component_weights"], "joint_huber"
        )
        residual_rows += _obs_residual_rows(
            ds, model, plain["x"], None, "plain_lsq"
        )
        for k, o in enumerate(comp_obs):
            w_mean = float(robust["component_weights"][3 * k:3 * k + 3].mean())
            if w_mean < 1.0 - 1e-12:
                downweighted_keys.add((o.station, o.target))

        cr, cs = _closure_metrics(
            ds, graph,
            {s: init.R[s] for s in members},
            {s: init.t[s] for s in members},
            init.targets, "initialization",
        )
        closure_rows += cr
        closure_summary += cs
        cr, cs = _closure_metrics(
            ds, graph, Rr, tr, Xr, "joint_huber", downweighted_keys
        )
        closure_rows += cr
        closure_summary += cs
        cr, cs = _closure_metrics(ds, graph, Rp, tp, Xp, "plain_lsq")
        closure_rows += cr
        closure_summary += cs

        ident = model.identifiability(robust["x"], length_scale)
        ident["component_anchor"] = comp_anchor
        ident["component_stations"] = sorted(members)
        ident_components.append(ident)
        if comp_anchor == ds.anchor_station:
            ident_combined = ident
        robust_meta.append({"component_anchor": comp_anchor, **{
            k: robust[k] for k in ("label", "iterations", "history")
        }})
        plain_meta.append({"component_anchor": comp_anchor, **{
            k: plain[k] for k in ("label", "iterations", "history")
        }})

    # ---- holdout：初始化 vs 联合解，两套变换独立代入 ----
    # fit_reference：仅用拟合观测与当前变换构造的世界坐标（加权平均），
    # 供"只在 holdout 出现"的标靶作参考；holdout 数据本身绝不进入其中。
    def _fit_reference(Rr, tr):
        acc, wsum = {}, {}
        for o in fit:
            wpt = Rr[o.station] @ o.local + tr[o.station]
            w = 1.0 / o.sigma**2
            acc[o.target] = acc.get(o.target, np.zeros(3)) + wpt * w
            wsum[o.target] = wsum.get(o.target, 0.0) + w
        return {tg: acc[tg] / wsum[tg] for tg in acc}

    # 普通 LSQ 全量解（分量循环里只保存了最后一个，这里重建），用于对照
    R_plain, t_plain, X_plain_all = {}, {}, {}
    for comp_anchor in components:
        members = {s for s, ca in comp_of.items() if ca == comp_anchor}
        comp_obs = [o for o in fit if o.station in members]
        comp_targets = sorted({o.target for o in comp_obs})
        model = JointModel(
            comp_obs, comp_anchor, init.R, init.t,
            {tg: init.targets[tg] for tg in comp_targets},
        )
        plain = model.solve_plain()
        Rp, tp, Xp = _extract(model, plain["x"])
        R_plain.update(Rp)
        t_plain.update(tp)
        X_plain_all.update(Xp)

    fit_ref_final = _fit_reference(R_final, t_final)
    fit_ref_init = _fit_reference(init.R, init.t)
    fit_ref_plain = _fit_reference(R_plain, t_plain)
    ho_init, rms_init = _holdout_metrics(
        ds, init.R, init.t, init.targets, "initialization", fit_ref_init
    )
    ho_rob, rms_rob = _holdout_metrics(
        ds, R_final, t_final, X_final, "joint_huber", fit_ref_final
    )
    ho_plain, rms_plain = _holdout_metrics(
        ds, R_plain, t_plain, X_plain_all, "plain_lsq", fit_ref_plain
    )
    holdout_rows = ho_init + ho_rob + ho_plain

    # ---- 标靶表 ----
    target_rows = []
    fit_count: dict[str, int] = {}
    for o in fit:
        fit_count[o.target] = fit_count.get(o.target, 0) + 1
    for tg in ds.targets:
        row = {
            "target_id": tg,
            "n_fit_observations": fit_count.get(tg, 0),
            "holdout_only": tg not in X_final,
        }
        if tg in X_final:
            row.update({
                "x_m": X_final[tg][0], "y_m": X_final[tg][1], "z_m": X_final[tg][2],
                "x_init_m": init.targets[tg][0],
                "y_init_m": init.targets[tg][1],
                "z_init_m": init.targets[tg][2],
            })
        else:
            row.update({"x_m": None, "y_m": None, "z_m": None,
                        "x_init_m": None, "y_init_m": None, "z_init_m": None})
        target_rows.append(row)

    # ---- 每站状态 ----
    station_rows = []
    ident_by_anchor = {ic["component_anchor"]: ic for ic in ident_components}
    for s in ds.stations:
        ca = comp_of[s]
        ident = ident_by_anchor.get(ca) if ca != ds.anchor_station else ident_combined
        st = _station_status(graph, ident, s)
        st["station_id"] = s
        st["component_anchor"] = ca
        # 零空间旋转轴（旋转向量参数位于初始局部系，其增量经 R 作用于世界向量）
        # 与初始化局部系中的几何直线轴直接比较。
        for entry in st["jacobian_null_rotation_axes"]:
            if "axis_param_space" in entry:
                ax = np.array(entry["axis_param_space"])
                nrm = np.linalg.norm(ax)
                entry["axis_local_normalized"] = (
                    ax / nrm
                ).tolist() if nrm > 0 else None
                geo = st.get("line_axis_local")
                if geo is not None and entry["axis_local_normalized"] is not None:
                    g = np.array(geo)
                    a = np.array(entry["axis_local_normalized"])
                    entry["aligns_with_line_axis"] = bool(
                        abs(float(np.dot(g, a))) > 0.95
                    )
        station_rows.append(st)

    def _rms(rows, solution):
        vals = [r["residual_norm_m"] ** 2 for r in rows if r["solution"] == solution]
        return float(np.sqrt(np.mean(vals))) if vals else None

    # 内点（未被稳健降权）上的 RMS 对照：说明普通 LSQ 如何被离群点带偏
    robust_rows = [r for r in residual_rows if r["solution"] == "joint_huber"]
    inlier_keys = {
        (r["station_id"], r["target_id"])
        for r in robust_rows if r["action"] != "downweighted_huber"
    }

    def _inlier_rms(solution: str):
        vals = [
            r["residual_norm_m"] ** 2
            for r in residual_rows
            if r["solution"] == solution
            and (r["station_id"], r["target_id"]) in inlier_keys
        ]
        return float(np.sqrt(np.mean(vals))) if vals else None

    summary = {
        "n_stations": len(ds.stations),
        "n_targets": len(ds.targets),
        "n_fit_observations": len(fit),
        "n_holdout_observations": len(ds.holdout_observations()),
        "anchor_station": ds.anchor_station,
        "units": ds.units,
        "length_scale_m": length_scale,
        "fit_residual_rms_m": {
            "initialization": _rms(residual_rows, "initialization"),
            "joint_huber": _rms(residual_rows, "joint_huber"),
            "plain_lsq": _rms(residual_rows, "plain_lsq"),
        },
        # 同一内点集合（稳健解判定的 kept 观测）上的三方案公平对照；
        # 总 RMS 含被保留的离群点，其上升只说明稳健解没有向粗差弯曲
        "inlier_fit_residual_rms_m": {
            "initialization": _inlier_rms("initialization"),
            "joint_huber": _inlier_rms("joint_huber"),
            "plain_lsq": _inlier_rms("plain_lsq"),
        },
        "holdout_rms_m": {
            "initialization": rms_init,
            "joint_huber": rms_rob,
            "plain_lsq": rms_plain,
        },
        "robust_solver": robust_meta,
        "plain_solver": plain_meta,
        "n_downweighted_observations": sum(
            1 for r in residual_rows if r["solution"] == "joint_huber" and r["action"] == "downweighted_huber"
        ),
    }

    return PipelineResult(
        ds=ds, graph=graph, init=init,
        robust=robust, plain=plain, ident=ident_combined,
        ident_components=ident_components,
        R=R_final, t=t_final, X=X_final,
        R_init=init.R, t_init=init.t, X_init=init.targets,
        residual_rows=residual_rows,
        closure_rows=closure_rows,
        closure_summary=closure_summary,
        holdout_rows=holdout_rows,
        holdout_rms={"initialization": rms_init, "joint_huber": rms_rob, "plain_lsq": rms_plain},
        target_rows=target_rows,
        station_rows=station_rows,
        summary=summary,
    )
