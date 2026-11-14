"""命令行入口：读取输入 -> 几何 -> 算子 -> 诊断 -> 两组正则求解 -> 落盘。

用法：
  python -m recon2d --rays fixtures/rays.csv --grid fixtures/grid.json --out out/full
  python -m recon2d --rays fixtures/limited-rays.csv --grid fixtures/grid.json --out out/limited
  python -m recon2d --rays fixtures/tiny/rays.csv --grid fixtures/tiny/grid.json --out out/tiny
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from . import __version__
from .dataio import load_grid, load_rays_csv, process_geometry
from .diagnostics import (angle_coverage, direction_probes,
                          nullspace_energy_map, probe_visibility,
                          stability_sweep, weighted_svd, cell_support)
from .geometry import build_forward_operator
from .regularize import neighbor_difference_matrix, prior_selector
from .solve import reconstruct
from .verify import (check_segment_length_sum, cross_check_operator,
                     independent_prediction)


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"无法序列化 {type(obj)}")


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2,
                  default=_json_default)


def write_operator_manifest(path: Path, geoms, grid) -> None:
    """稀疏算子清单：每个非零 (射线, 单元) 一项路径长度。"""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ray_id", "cell_row", "cell_col", "cell_flat_index",
                    "path_length_m"])
        for g in geoms:
            for idx, length in sorted(g.segments.items(),
                                      key=lambda kv: kv[0]):
                row, col = grid.rc_of(idx)
                w.writerow([g.ray_id, row, col, idx,
                            f"{length:.12g}"])
            if not g.segments:
                w.writerow([g.ray_id, "", "", "",
                            "DISJOINT_ZERO_LENGTH_INSIDE_REGION"])


def run_case(rays_path: str, grid_path: str, out_dir: str,
             lambdas: list[float] | None = None,
             prior_weight: float = 0.0, prior_value: float = 0.0,
             sweep_points: int = 5, name: str | None = None) -> dict:
    rays_path, grid_path, out_dir = Path(rays_path), Path(grid_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if name:
        case = name
    else:
        # rays.csv -> 父目录名（tiny/rays.csv -> tiny）；其余取文件名词干
        stem = rays_path.stem
        if stem == "rays":
            case = rays_path.parent.name or "case"
        else:
            case = stem.replace("-rays", "").replace("rays", "")
            case = case or "case"

    grid = load_grid(grid_path)
    if lambdas is None:
        lambdas = list(grid.raw.get("regularization_strengths", [0.01, 1.0]))
    ids, endpoints, atten, sigma, input_diag = load_rays_csv(rays_path)
    geoms, geom_diag = process_geometry(ids, endpoints, grid)
    A, zero_row = build_forward_operator(geoms, grid)
    valid = ~zero_row

    # ---- 算子与支持 -----------------------------------------------------
    support = cell_support(A)
    support_mask = support["ray_count_per_cell"] > 0
    D = neighbor_difference_matrix(grid.nx, grid.ny)
    P = prior_selector(support_mask)

    # ---- 秩 / 近零空间 --------------------------------------------------
    svd = weighted_svd(A[valid], sigma[valid], grid.mass_attenuation)
    null_energy = nullspace_energy_map(svd["vt"], svd["singular_values"],
                                       grid.nx, grid.ny)
    angles = angle_coverage(endpoints)
    probes = direction_probes(grid.nx, grid.ny)
    visibility = probe_visibility(A[valid], sigma[valid],
                                  grid.mass_attenuation, probes)

    # ---- 两组给定强度求解 ----------------------------------------------
    solutions = {}
    for lam in lambdas:
        res = reconstruct(A, zero_row, atten, sigma, grid.mass_attenuation,
                          D, P, lam, prior_weight, prior_value)
        solutions[lam] = res

    # 纯数据解（lambda=0）：正则只能牺牲数据拟合，用于佐证目标量分解
    data_only = reconstruct(A, zero_row, atten, sigma, grid.mass_attenuation,
                            D, P, 0.0, prior_weight, prior_value)

    # ---- 正则扫描：哪些区域不稳定 --------------------------------------
    lmin, lmax = min(lambdas), max(lambdas)
    sweep_lams = np.unique(np.logspace(np.log10(lmin), np.log10(lmax),
                                       sweep_points))
    sweep = stability_sweep(
        lambda lam: reconstruct(A, zero_row, atten, sigma,
                                grid.mass_attenuation, D, P, float(lam),
                                prior_weight, prior_value),
        sweep_lams)

    # ---- 独立校验 -------------------------------------------------------
    length_check = check_segment_length_sum(geoms)
    cross_reports, cross_worst = cross_check_operator(
        grid, geoms, steps_per_cell=2000)
    primary = solutions[lambdas[0]]
    pred_indep = independent_prediction(grid, geoms, primary.density,
                                        steps_per_cell=2000)
    indep_diff = np.abs(pred_indep - primary.predicted)
    indep_scale = np.maximum(np.abs(primary.predicted), 1e-12)
    indep_reports = [
        {"ray_id": g.ray_id,
         "status": "disjoint" if not g.segments else "ok",
         "predicted_event_m2_per_kg": float(primary.predicted[i]),
         "predicted_sampled_m2_per_kg": float(pred_indep[i]),
         "abs_diff": float(indep_diff[i]),
         "rel_diff": float(indep_diff[i] / indep_scale[i])
                     if g.segments else 0.0}
        for i, g in enumerate(geoms)]

    # ---- 单元可信度标签（数据支持 / 近零空间 / 正则不稳定 / 无支持） ---
    labels = np.full(grid.n_cells, "", dtype=object)
    labels[~support_mask] = "no_ray_support_prior_only"
    weak = support_mask & (null_energy > 0.5)
    labels[weak] = "ray_supported_but_near_nullspace"
    unstable = (sweep["relative_spread"] > 0.25) & support_mask & ~weak
    labels[unstable] = "ray_supported_unstable_under_regularization"
    labels[(labels == "")] = "data_determined"

    # ---- 落盘：稀疏算子 -------------------------------------------------
    sparse.save_npz(out_dir / f"operator_{case}.npz", A)
    write_operator_manifest(out_dir / f"operator_manifest_{case}.csv",
                            geoms, grid)

    # ---- 落盘：结果数组 -------------------------------------------------
    np.savez_compressed(
        out_dir / f"results_{case}.npz",
        grid_x_min=grid.x_min, grid_y_min=grid.y_min,
        dx=grid.dx, dy=grid.dy, nx=grid.nx, ny=grid.ny,
        ray_count_per_cell=support["ray_count_per_cell"].reshape(
            grid.ny, grid.nx),
        path_length_per_cell=support["path_length_per_cell"].reshape(
            grid.ny, grid.nx),
        support_mask=support_mask.reshape(grid.ny, grid.nx),
        nullspace_energy=null_energy.reshape(grid.ny, grid.nx),
        sweep_lambdas=sweep["lambdas"],
        sweep_relative_spread=sweep["relative_spread"].reshape(
            grid.ny, grid.nx),
        sweep_std_density=sweep["std_density"].reshape(grid.ny, grid.nx),
        **{f"density_lambda_{lam}": solutions[lam].density.reshape(
            grid.ny, grid.nx) for lam in lambdas},
        density_data_only_lambda0=data_only.density.reshape(grid.ny, grid.nx),
        cell_labels=labels.reshape(grid.ny, grid.nx))

    # ---- 落盘：逐射线预测与残差 ----------------------------------------
    # weighted_residual 只按有效射线排列，这里映射回全部射线行
    wres_by_ray = np.full(len(geoms), np.nan)
    wres_by_ray[valid] = primary.weighted_residual
    with open(out_dir / f"predictions_{case}.csv", "w",
              encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ray_id", "attenuation_observed", "sigma",
                    "attenuation_predicted", "residual",
                    "weighted_residual", "clipped_path_length_m",
                    "full_ray_length_m", "status"])
        for i, g in enumerate(geoms):
            status = "disjoint_excluded_from_fit" if zero_row[i] else "valid"
            if g.error:
                status = f"geometry_error:{g.error}"
            wres = "" if zero_row[i] else f"{wres_by_ray[i]:.6g}"
            w.writerow([g.ray_id, f"{atten[i]:.12g}", f"{sigma[i]:.12g}",
                        f"{primary.predicted[i]:.12g}",
                        f"{primary.residual[i]:.12g}", wres,
                        f"{g.clipped_length:.12g}", f"{g.total_length:.12g}",
                        status])

    # ---- 落盘：诊断 JSON -----------------------------------------------
    payload = {
        "tool_version": __version__,
        "case": case,
        "inputs": {"rays_csv": str(rays_path), "grid_json": str(grid_path)},
        "grid": {"nx": grid.nx, "ny": grid.ny, "dx_m": grid.dx,
                 "dy_m": grid.dy, "x_min_m": grid.x_min,
                 "y_min_m": grid.y_min,
                 "mass_attenuation_m2_per_kg": grid.mass_attenuation,
                 "n_cells": grid.n_cells,
                 "array_axes": "rows increase y, columns increase x",
                 "cell_convention": "half-open [lo,hi); internal grid-line "
                                    "rays assigned to higher-coordinate side"},
        "rays": {
            "n_total": len(geoms),
            "n_valid_in_fit": int(valid.sum()),
            "n_disjoint_excluded": int(zero_row.sum()),
            "input_diagnostics": input_diag,
            "geometry_diagnostics": geom_diag,
        },
        "operator": {
            "shape": list(A.shape), "nnz": int(A.nnz),
            "density_fraction": float(A.nnz) / (A.shape[0] * A.shape[1]),
            "nnz_per_valid_row_min": int(A[valid].getnnz(axis=1).min())
                                      if valid.any() else 0,
            "nnz_per_valid_row_max": int(A[valid].getnnz(axis=1).max())
                                      if valid.any() else 0,
            "npz_file": f"operator_{case}.npz",
            "manifest_csv": f"operator_manifest_{case}.csv",
        },
        "support": {
            "ray_count_per_cell_2d": support["ray_count_per_cell"].reshape(
                grid.ny, grid.nx).tolist(),
            "path_length_per_cell_2d": support["path_length_per_cell"]
                .reshape(grid.ny, grid.nx).tolist(),
            "supported_cells": support["supported_cell_count"],
            "unsupported_cells": support["unsupported_cell_count"],
        },
        "rank_and_nullspace": {
            "weighted_operator_shape": [svd["n_rows"], svd["n_cols"]],
            "rank": svd["rank"],
            "numeric_rank_tol": svd["numeric_tol"],
            "near_null_threshold_rel_1e-8": svd["near_null_tol"],
            "near_null_singular_value_count": svd["near_null_count"],
            "nullity_n_cols_minus_rank": svd["n_cols"] - svd["rank"],
            "singular_values": svd["singular_values"].tolist(),
            "relative_nullspace_energy_2d": null_energy.reshape(
                grid.ny, grid.nx).tolist(),
            "note": "success=True 仅表示最小二乘收敛；秩亏时解不唯一，"
                    "正则只是在零空间中挑了一个平滑解，不是数据证明的真值",
        },
        "angle_coverage": {
            "angles_deg_mod180": angles["angles_deg_mod180"].tolist(),
            "histogram_10deg": angles["histogram_10deg"].tolist(),
            "occupied_sectors": angles["occupied_sectors"],
            "max_empty_sector_run_deg": angles["max_empty_sector_run"],
            "finite_angle_suspect": angles["finite_angle_suspect"],
        },
        "finite_angle_probes": visibility,
        "regularization": {
            "form": "min J_data + lambda*||D x||^2 + ||w_p P(x-x0)||^2, "
                    "x>=0; D = nearest-neighbor unit differences",
            "strengths_run": lambdas,
            "prior_weight": prior_weight,
            "prior_value": prior_value,
            "prior_selector_rows": int(P.shape[0]),
            "sweep_lambdas": sweep["lambdas"].tolist(),
            "sweep_relative_spread_2d": sweep["relative_spread"].reshape(
                grid.ny, grid.nx).tolist(),
        },
        "solutions": {
            str(lam): {
                "success": res.success, "status": res.status,
                "message": res.message, "iterations": res.n_iter,
                "objective": res.objective,
                "density_min": float(res.density.min()),
                "density_max": float(res.density.max()),
                "density_sum": float(res.density.sum()),
                "mean_density_unsupported_cells": float(
                    res.density[~support_mask].mean())
                    if (~support_mask).any() else None,
                "mass_fraction_in_unsupported_cells": float(
                    res.density[~support_mask].sum() / res.density.sum())
                    if res.density.sum() > 0 else None,
            } for lam, res in solutions.items()},
        "data_only_lambda0_check": {
            "J_data_chi2": data_only.objective["J_data_chi2"],
            "reduced_chi2": data_only.objective["reduced_chi2"],
            "note": "正则解的 J_data 必须不小于该值（正则以拟合换平滑）",
        },
        "cell_confidence_labels_2d": labels.reshape(
            grid.ny, grid.nx).tolist(),
        "verification": {
            "segment_sum_vs_clipped": {
                "max_abs_diff_m": float(max(
                    (r["abs_diff"] for r in length_check), default=0.0)),
                "max_rel_diff": float(max(
                    (r["rel_diff"] for r in length_check
                     if r["status"] == "ok"), default=0.0))},
            "independent_sampled_integration": {
                "max_rel_diff_ray_total": float(max(
                    (r["length_rel_diff"] for r in cross_reports
                     if r["status"] == "ok"), default=0.0)),
                "max_prediction_rel_diff": float(max(
                    (r["rel_diff"] for r in indep_reports
                     if r["status"] == "ok"), default=0.0)),
                "per_ray": indep_reports,
            },
        },
        "outputs": {
            "results_npz": f"results_{case}.npz",
            "predictions_csv": f"predictions_{case}.csv",
            "diagnostics_json": f"diagnostics_{case}.json",
        },
    }
    write_json(out_dir / f"diagnostics_{case}.json", payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="二维直线射线层析重建（非负 NNLS + 邻格平滑）")
    ap.add_argument("--rays", required=True, help="rays.csv 路径")
    ap.add_argument("--grid", required=True, help="grid.json 路径")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--lambdas", nargs="*", type=float, default=None,
                    help="两组（或多组）正则强度；缺省读 grid.json")
    ap.add_argument("--prior-weight", type=float, default=0.0,
                    help="无射线格先验锚定权重，默认 0（不凭空锚定）")
    ap.add_argument("--prior-value", type=float, default=0.0,
                    help="无射线格先验密度值")
    ap.add_argument("--sweep-points", type=int, default=5,
                    help="正则稳定性扫描的对数采样点数")
    ap.add_argument("--name", default=None,
                    help="案例名（输出文件前缀），缺省由射线文件名推导")
    args = ap.parse_args(argv)

    payload = run_case(args.rays, args.grid, args.out, args.lambdas,
                       args.prior_weight, args.prior_value,
                       args.sweep_points, args.name)
    _print_summary(payload)


def _print_summary(p: dict) -> None:
    print(f"\n=== 案例 {p['case']} ===")
    g = p["grid"]
    print(f"网格 {g['nx']}x{g['ny']}（行=y+，列=x+），"
          f"mu={g['mass_attenuation_m2_per_kg']} m^2/kg")
    r = p["rays"]
    print(f"射线 {r['n_total']} 条：有效 {r['n_valid_in_fit']}，"
          f"完全不相交剔除 {r['n_disjoint_excluded']}")
    rn = p["rank_and_nullspace"]
    print(f"加权算子秩 {rn['rank']}/{rn['weighted_operator_shape'][1]}，"
          f"零空间维数 {rn['nullity_n_cols_minus_rank']}，"
          f"近零奇异值 {rn['near_null_singular_value_count']} 个")
    s = p["support"]
    print(f"单元支持：有射线 {s['supported_cells']}，"
          f"无射线（仅先验/平滑补出）{s['unsupported_cells']}")
    for lam, sol in p["solutions"].items():
        o = sol["objective"]
        print(f"[lambda={lam}] success={sol['success']} "
              f"chi2={o['J_data_chi2']:.4g} "
              f"reduced_chi2={o['reduced_chi2']:.4g} "
              f"smooth={o['J_smooth_unit_diff2']:.4g} "
              f"reg/data={o['regularization_to_data_ratio']:.4g} "
              f"lambda_eff={o['lambda_eff_per_data_edge']:.4g}")
    for probe in p["finite_angle_probes"]:
        flag = "不可区分" if probe["invisible"] else "可部分区分"
        print(f"角度探测 {probe['pattern']}: 可见比 "
              f"{probe['visibility_ratio']:.3e} -> {flag}")
    v = p["verification"]
    print(f"校验：段长和-裁切段最大相对差 "
          f"{v['segment_sum_vs_clipped']['max_rel_diff']:.2e}；"
          f"独立积分预测最大相对差 "
          f"{v['independent_sampled_integration']['max_prediction_rel_diff']:.2e}")
    print("提醒：success 不等于唯一可辨识；无射线/近零空间格的数值是"
          "平滑先验的选择，不是测量证明的密度。\n")


if __name__ == "__main__":
    main()
