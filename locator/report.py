"""交付物写出:solution.json、residuals.csv、profile.csv、depth_profile.csv、report.md。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

CAVEATS = [
    "支持区域(代价 <= 最小值 + delta)只是数值支持区域,不是已校准的置信水平。",
    "local_curvature 来自最优解处的局部线性化 Hessian,不能替代多解检查;"
    "弱几何下它可能给出误导性的小椭球。",
    "搜索盒以外区域未覆盖;若支持区域触及边界,盒外延续未知。",
    "支持区域范围按 1 个体素量化(±半步长);网格之间的细节由局部优化补充。",
    "代价采用 Huber 稳健损失;换用其他损失会改变支持区域口径,比较时需同一代价定义。",
]


def _round(x, n=6):
    if isinstance(x, float):
        return round(x, n)
    return x


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float):
        return round(obj, 12)   # 保留足够精度,使 solution.json 可独立重算残差
    return obj


def write_solution_json(path: Path, case: str, result: dict) -> None:
    plan = result["plan"]
    doc = {
        "case": case,
        "model": {
            "description": "arrival = origin_time + Euclidean_distance(source, station)"
                           " / velocity[phase]",
            "velocities_km_s": plan.velocities,
            "position_bounds_km": {"x": list(plan.x_bounds),
                                   "y": list(plan.y_bounds),
                                   "depth": list(plan.depth_bounds)},
            "origin_time_bounds_s": list(plan.origin_time_bounds),
            "loss": {
                "type": "huber",
                "c": result["huber_c"],
                "definition": "cost = sum rho(r); rho(r)=r^2/2 (|r|<=c) else "
                              "c|r|-c^2/2; r = (obs - t0 - dist/v)/sigma",
            },
            "support_score_delta": plan.profile_score_delta,
        },
        "data_summary": {
            "n_stations": len(result["data"].stations),
            "n_picks": len(result["data"].arrivals),
            "picks_by_phase": result["picks_by_phase"],
        },
        "best_solution": result["best"],
        "n_distinct_minima": result["n_distinct_minima"],
        "competing_solutions": result["minima"],
        "support_region": {
            **result["support"],
            "grid_min_cost": result["min_cost"],
            "definition": "网格代价 <= 网格最小代价 + support_score_delta;"
                          "优化解代价可低于网格最小代价(解不必落在格点上)",
        },
        "depth_profile": result["depth_profile"],
        "degeneracy_analysis": result["degeneracy"],
        "local_curvature": result["local_curvature"],
        "downweighted_picks": [
            {"pick_id": r["pick_id"], "station_id": r["station_id"],
             "phase": r["phase"],
             "standardized_residual": r["standardized_residual"],
             "robust_weight": r["robust_weight"]}
            for r in result["residuals"] if r["downweighted"]
        ],
        "diagnostics": result["diagnostics"],
        "caveats": CAVEATS,
    }
    path.write_text(json.dumps(_jsonable(doc), ensure_ascii=False, indent=2),
                    encoding="utf-8")


def write_residuals_csv(path: Path, result: dict) -> None:
    cols = ["pick_id", "station_id", "phase", "observed_s", "predicted_s",
            "travel_time_s", "origin_time_s", "residual_s", "sigma_s",
            "standardized_residual", "robust_weight", "downweighted"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in result["residuals"]:
            w.writerow({k: (f"{row[k]:.12f}" if isinstance(row[k], float) else row[k])
                        for k in cols})


def write_profile_csv(path: Path, result: dict) -> None:
    g = result["grid"]
    xs, ys, zs = g["x_km"], g["y_km"], g["depth_km"]
    cost, t0 = g["cost"], g["origin_time_s"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x_km", "y_km", "depth_km", "origin_time_s", "cost"])
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    w.writerow([f"{x:.3f}", f"{y:.3f}", f"{z:.3f}",
                                f"{t0[i, j, k]:.6f}", f"{cost[i, j, k]:.6f}"])


def write_depth_profile_csv(path: Path, result: dict) -> None:
    cols = ["depth_km", "min_cost", "x_km", "y_km", "origin_time_s"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in result["depth_profile"]:
            w.writerow({k: f"{row[k]:.6f}" for k in cols})


def _fmt(v, n=4):
    return f"{v:.{n}f}"


def write_case_report_md(path: Path, case: str, result: dict) -> None:
    plan = result["plan"]
    best = result["best"]
    sup = result["support"]
    curv = result["local_curvature"]
    deg = result["degeneracy"]
    lines = []
    A = lines.append
    A(f"# 案例 `{case}` 定位报告")
    A("")
    A("## 数据与模型")
    pb = result["picks_by_phase"]
    A(f"- 台站 {len(result['data'].stations)} 个,pick {len(result['data'].arrivals)} 条"
      f"({', '.join(f'{k}: {v}' for k, v in sorted(pb.items()))})。")
    vel = ", ".join(f"{k}={v} km/s" for k, v in sorted(plan.velocities.items()))
    A(f"- 模型固定:`arrival = t0 + 欧氏距离 / 相位波速`,波速 {vel},不反演速度结构。")
    A(f"- 稳健目标:Huber 代价 `C = Σρ(r)`,c = {result['huber_c']}(以标准化残差为单位);"
      "同一代价用于优化与剖面。")
    A(f"- 搜索盒:x∈{list(plan.x_bounds)}, y∈{list(plan.y_bounds)}, "
      f"depth∈{list(plan.depth_bounds)} km,t0∈{list(plan.origin_time_bounds)} s。")
    A("")
    A("## 解(联合估计 x, y, depth, t0)")
    A(f"- 最优解:x = {_fmt(best['x_km'])} km,y = {_fmt(best['y_km'])} km,"
      f"depth = {_fmt(best['depth_km'])} km,t0 = {_fmt(best['origin_time_s'])} s,"
      f"代价 = {_fmt(best['cost'])}。")
    A(f"- 标准化残差 RMS = {_fmt(best['rms_standardized_residual'])},"
      f"最大 |r| = {_fmt(best['max_abs_standardized_residual'])}。")
    n_min = result["n_distinct_minima"]
    if n_min > 1:
        A(f"- ⚠ **解不唯一**:多初值搜索发现 {n_min} 个不同极小点(下表)。"
          "它们是优化在近简并谷上的采样,真实近优集合是连续弧/区域(见支持区域),"
          "任一单点都不能当作唯一定位结果;此处最优解只是代价最小的代表。")
    A(f"- 多初值({result['diagnostics']['n_starts']} 个)局部优化后聚类得到 "
      f"**{n_min} 个不同极小点**;代价 ≤ 最小值 + "
      f"{plan.profile_score_delta} 的竞争解:")
    A("")
    A("| rank | x (km) | y (km) | depth (km) | t0 (s) | 代价 | Δ代价 | 支持区域内 |")
    A("|---|---|---|---|---|---|---|---|")
    ref_cost = best["cost"]
    for m in result["minima"]:
        A(f"| {m['rank']} | {_fmt(m['x_km'])} | {_fmt(m['y_km'])} | "
          f"{_fmt(m['depth_km'])} | {_fmt(m['origin_time_s'])} | {_fmt(m['cost'])} | "
          f"{_fmt(m['cost'] - ref_cost)} | "
          f"{'是' if m['within_support_delta'] else '否'} |")
    A("")
    A("## 近优支持区域(代价 ≤ 最小值 + "
      f"{plan.profile_score_delta},**数值支持区域,非置信水平**)")
    A(f"- 定义口径:网格最小代价 {result['min_cost']:.4f};优化后的解代价 "
      f"{best['cost']:.4f} 可低于它(解不必落在格点上),支持区域按"
      "**网格**代价 ≤ 网格最小值 + delta 划定。")
    if sup["n_voxels"]:
        ext = sup["extent_km"]
        A(f"- 体素 {sup['n_voxels']} 个(约 {sup['volume_km3']:.0f} km³),"
          f"连通块 {len(sup['components'])} 个。")
        A(f"- 范围:x ∈ [{ext['x'][0]:.1f}, {ext['x'][1]:.1f}] km,"
          f"y ∈ [{ext['y'][0]:.1f}, {ext['y'][1]:.1f}] km,"
          f"depth ∈ [{ext['depth'][0]:.1f}, {ext['depth'][1]:.1f}] km。")
        if sup["touches_boundary"]:
            A(f"- ⚠ 支持区域触及搜索盒边界:{', '.join(sup['touches_boundary'])};"
              "盒外延续未覆盖。")
        else:
            A("- 支持区域未触及搜索盒边界。")
        A("")
        A("| 连通块 | 体素数 | 块内最小代价 | 块内最优位置 (x, y, depth, t0) |")
        A("|---|---|---|---|")
        for comp in sup["components"]:
            loc = comp["min_location"]
            A(f"| {comp['id']} | {comp['n_voxels']} | {_fmt(comp['min_cost'])} | "
              f"({_fmt(loc['x_km'], 1)}, {_fmt(loc['y_km'], 1)}, "
              f"{_fmt(loc['depth_km'], 1)}, {_fmt(loc['origin_time_s'], 2)}) |")
    else:
        A("- 支持区域为空(不应发生:最小代价点总在区域内)。")
    A("")
    A("## 残差与降权(全部 pick,未删除任何台站)")
    A("")
    A("| pick | 台站 | 相位 | 观测 (s) | 预测 (s) | 标准化残差 | 稳健权重 |")
    A("|---|---|---|---|---|---|---|")
    for r in result["residuals"]:
        flag = f"**{r['robust_weight']:.3f}**" if r["downweighted"] else \
               f"{r['robust_weight']:.3f}"
        A(f"| {r['pick_id']} | {r['station_id']} | {r['phase']} | "
          f"{r['observed_s']:.4f} | {r['predicted_s']:.4f} | "
          f"{r['standardized_residual']:+.2f} | {flag} |")
    dw = [r for r in result["residuals"] if r["downweighted"]]
    A("")
    if dw:
        A("被降权记录:" + ", ".join(
            f"{r['pick_id']}(r={r['standardized_residual']:+.1f}, w={r['robust_weight']:.2f})"
            for r in dw) + "。仅降权,未剔除。")
    else:
        A("无被降权记录(全部权重为 1)。")
    A("")
    A("## 深度可解性与参数补偿")
    if deg:
        A(f"- 支持区域内 depth 范围 [{deg['support_depth_range_km'][0]:.1f}, "
          f"{deg['support_depth_range_km'][1]:.1f}] km,"
          f"|y| 范围 [{deg['support_abs_y_range_km'][0]:.1f}, "
          f"{deg['support_abs_y_range_km'][1]:.1f}] km,"
          f"t0 范围 [{deg['support_origin_time_range_s'][0]:.3f}, "
          f"{deg['support_origin_time_range_s'][1]:.3f}] s。")
        A(f"- y²+depth² 均值 {deg['y2_plus_z2_mean_km2']:.2f} km²,"
          f"标准差 {deg['y2_plus_z2_std_km2']:.2f} km²;"
          f"depth 与 |y| 相关系数 "
          f"{deg['corr_depth_abs_y'] if deg['corr_depth_abs_y'] is not None else float('nan'):.3f}"
          "(近共线台阵下两者沿 y²+depth²≈常数 的弧互相补偿)。")
    A(f"- 局部曲率(仅线性化):参数标准差 x={curv['param_std']['x_km']:.3f} km,"
      f"y={curv['param_std']['y_km']:.3f} km,depth={curv['param_std']['depth_km']:.3f} km,"
      f"t0={curv['param_std']['origin_time_s']:.3f} s;"
      f"Hessian 条件数 {curv['condition_number']:.3g}。")
    wd = curv["weakest_direction"]
    A(f"- 最弱约束方向 (x, y, depth, t0) ≈ ({wd['x']:+.2f}, {wd['y']:+.2f}, "
      f"{wd['depth']:+.2f}, {wd['origin_time']:+.2f});"
      "该小椭球只是局部近似,多解检查以上述网格支持区域为准。")
    A("")
    A("## 数值说明")
    d = result["diagnostics"]
    A(f"- 网格步长 {d['grid_step_km']} km,网格 {d['grid_shape'][0]}×"
      f"{d['grid_shape'][1]}×{d['grid_shape'][2]} = {d['n_grid_points']} 点,"
      "每个格点用 IRLS 重估 t0(容差 "
      f"{d['t0_irls_tol_s']:.0e} s,实际最多 {d['t0_irls_max_iters_used']} 次迭代)。")
    A(f"- 局部优化:{d['optimizer']},收敛容差 {d['optimizer_tol']:.0e};"
      f"解聚类半径 {d['cluster_pos_tol_km']} km / {d['cluster_t0_tol_s']} s。")
    A("- 未覆盖区域:搜索盒以外不评估;支持区域范围为体素量化(±0.5 km)。")
    A("")
    A("## 注意事项")
    for c in CAVEATS:
        A(f"- {c}")
    A("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_all(out_dir: Path, case: str, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_solution_json(out_dir / "solution.json", case, result)
    write_residuals_csv(out_dir / "residuals.csv", result)
    write_profile_csv(out_dir / "profile.csv", result)
    write_depth_profile_csv(out_dir / "depth_profile.csv", result)
    write_case_report_md(out_dir / "report.md", case, result)


def write_combined_report_md(path: Path, results: list[tuple[str, dict]]) -> None:
    """三个案例的对比总报告(简短静态)。"""
    lines = []
    A = lines.append
    A("# 单事件到时定位总报告")
    A("")
    A("## 方法(三组共用)")
    A("")
    first = results[0][1]
    plan = first["plan"]
    vel = ", ".join(f"{k}={v} km/s" for k, v in sorted(plan.velocities.items()))
    A(f"- 模型固定:`arrival = t0 + 欧氏距离(源, 台站) / 相位波速`,波速 {vel};"
      "不反演速度结构,各组独立分析。")
    A(f"- 目标函数:Huber 稳健代价 `C = Σρ(r)`,c = {first['huber_c']}"
      "(以标准化残差为单位),对少数误拾取稳健;优化与剖面使用同一代价定义。")
    A(f"- 搜索策略:先在 {plan.profile_grid_step_km} km 三维网格上逐点重估 t0 "
      "并计算代价,再从各连通块最小点、分散低成本格点、盒中心/台阵中心等多初值"
      "做局部优化并聚类,不以单次局部收敛断言唯一解。")
    A(f"- 支持区域:代价 ≤ 最小值 + {plan.profile_score_delta} 的格点集合,"
      "**只是数值支持区域,不是已校准置信水平**。")
    A("")
    A("## 三组结果对比")
    A("")
    A("| 案例 | 最优解 (x, y, depth) km | t0 (s) | 代价 | 不同极小点 | "
      "支持区域体素 | 连通块 | 触边界 | 被降权 pick |")
    A("|---|---|---|---|---|---|---|---|---|")
    for case, r in results:
        b = r["best"]
        dw = [x for x in r["residuals"] if x["downweighted"]]
        A(f"| {case} | ({b['x_km']:.3f}, {b['y_km']:.3f}, {b['depth_km']:.3f}) | "
          f"{b['origin_time_s']:.3f} | {b['cost']:.3f} | "
          f"{r['n_distinct_minima']} | {r['support']['n_voxels']} | "
          f"{len(r['support']['components'])} | "
          f"{'是' if r['support']['touches_boundary'] else '否'} | {len(dw)} |")
    A("")
    A("## 几何与误拾取的影响")
    A("")
    by_case = dict(results)
    good_best = by_case["good"]["best"] if "good" in by_case else None
    for case, r in results:
        b = r["best"]
        sup = r["support"]
        ext = sup["extent_km"] or {"x": [0, 0], "y": [0, 0], "depth": [0, 0]}
        A(f"### {case}")
        A(f"- 代表解:({b['x_km']:.3f}, {b['y_km']:.3f}, {b['depth_km']:.3f}) km, "
          f"t0 = {b['origin_time_s']:.3f} s;支持区域:x ∈ [{ext['x'][0]:.0f}, "
          f"{ext['x'][1]:.0f}] km,y ∈ [{ext['y'][0]:.0f}, {ext['y'][1]:.0f}] km,"
          f"depth ∈ [{ext['depth'][0]:.0f}, {ext['depth'][1]:.0f}] km。")
        dw = [x for x in r["residuals"] if x["downweighted"]]
        if r["n_distinct_minima"] > 1:
            A(f"- **多解**:共 {r['n_distinct_minima']} 个等代价/近等代价极小点;"
              "台阵近共线且缺少 S 相,y 符号完全不可分辩,depth 与 |y| 沿 "
              "y²+depth²≈常数 的弧互相补偿——深度在此几何下不可唯一确定,"
              "只能给出支持区域。")
        elif not dw:
            A("- 台阵几何充分:多初值搜索仅得 1 个极小点,支持区域收缩到体素级,"
              "深度可解(见案例报告的局部曲率,仅为线性化参考)。")
        if r["degeneracy"]:
            deg = r["degeneracy"]
            A(f"- 支持区域内 y²+depth² = {deg['y2_plus_z2_mean_km2']:.2f} ± "
              f"{deg['y2_plus_z2_std_km2']:.2f} km²,depth–|y| 相关 "
              f"{deg['corr_depth_abs_y'] if deg['corr_depth_abs_y'] is not None else float('nan'):.2f};"
              f"t0 在 [{deg['support_origin_time_range_s'][0]:.3f}, "
              f"{deg['support_origin_time_range_s'][1]:.3f}] s 内仅小幅变化"
              "(x 与 t0 仍被走时水平与到时绝对值较好约束)。")
        if dw:
            A("- 稳健降权:" + ", ".join(
                f"{x['pick_id']}(标准化残差 {x['standardized_residual']:+.1f}, "
                f"权重 {x['robust_weight']:.2f})" for x in dw)
              + ";仅降权未剔除,全部台站仍参与反演。")
        if case != "good" and good_best is not None and r["n_distinct_minima"] == 1:
            dx = ((b["x_km"] - good_best["x_km"]) ** 2
                  + (b["y_km"] - good_best["y_km"]) ** 2
                  + (b["depth_km"] - good_best["depth_km"]) ** 2) ** 0.5
            A(f"- 与 good 案例解相距 {dx:.3f} km(同台阵几何,误拾取被降权后"
              "解保持稳定;Huber 影响有界但非零,残差仍轻微拉动解)。")
        A("")
    A("## 交付物清单")
    A("")
    A("每个案例目录含:`solution.json`(解/竞争解/支持区域/诊断)、"
      "`residuals.csv`(观测、预测、标准化残差、权重)、`profile.csv`(三维网格代价剖面)、"
      "`depth_profile.csv`(逐深度最优代价)、`report.md`(案例报告)。")
    A("")
    A("## 注意事项")
    for c in CAVEATS:
        A(f"- {c}")
    A("")
    path.write_text("\n".join(lines), encoding="utf-8")
