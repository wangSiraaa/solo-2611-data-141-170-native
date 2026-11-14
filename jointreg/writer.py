"""结果写出：每案例一个目录，CSV（表格）+ JSON（结构化全量信息）。"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

from .pipeline import PipelineResult


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"不可序列化的类型 {type(obj)}")


def write_outputs(result: PipelineResult, out_dir: str) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, str] = {}

    # 1) 每站 R/t（初始化与联合解并列），并附正交性/行列式检查列
    station_rows = []
    for s in result.ds.stations:
        R, t = result.R[s], result.t[s]
        Ri, ti = result.R_init[s], result.t_init[s]
        station_rows.append({
            "station_id": s,
            "R00": R[0, 0], "R01": R[0, 1], "R02": R[0, 2],
            "R10": R[1, 0], "R11": R[1, 1], "R12": R[1, 2],
            "R20": R[2, 0], "R21": R[2, 1], "R22": R[2, 2],
            "tx_m": t[0], "ty_m": t[1], "tz_m": t[2],
            "R00_init": Ri[0, 0], "R01_init": Ri[0, 1], "R02_init": Ri[0, 2],
            "R10_init": Ri[1, 0], "R11_init": Ri[1, 1], "R12_init": Ri[1, 2],
            "R20_init": Ri[2, 0], "R21_init": Ri[2, 1], "R22_init": Ri[2, 2],
            "tx_init_m": ti[0], "ty_init_m": ti[1], "tz_init_m": ti[2],
            "RTR_dev": float(np.max(np.abs(R.T @ R - np.eye(3)))),
            "det_R": float(np.linalg.det(R)),
        })
    p = os.path.join(out_dir, "station_poses.csv")
    _write_csv(p, station_rows, list(station_rows[0].keys()))
    paths["station_poses_csv"] = p

    # 2) 拟合标靶
    p = os.path.join(out_dir, "targets.csv")
    _write_csv(p, result.target_rows, list(result.target_rows[0].keys()))
    paths["targets_csv"] = p

    # 3) 逐观测残差（含 holdout 之外的全部拟合观测，两种解并列行）
    p = os.path.join(out_dir, "observation_residuals.csv")
    _write_csv(p, result.residual_rows, list(result.residual_rows[0].keys()))
    paths["residuals_csv"] = p

    # 4) 闭环明细与汇总
    p = os.path.join(out_dir, "closure_details.csv")
    _write_csv(p, result.closure_rows, list(result.closure_rows[0].keys()))
    paths["closure_csv"] = p
    p = os.path.join(out_dir, "closure_summary.csv")
    _write_csv(p, result.closure_summary, list(result.closure_summary[0].keys()))
    paths["closure_summary_csv"] = p

    # 5) holdout 跨站检查
    p = os.path.join(out_dir, "holdout_checks.csv")
    if result.holdout_rows:
        _write_csv(p, result.holdout_rows, list(result.holdout_rows[0].keys()))
    else:
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("no_holdout_observations\n")
    paths["holdout_csv"] = p

    # 6) JSON 全量结果
    payload = {
        "summary": result.summary,
        "graph": result.graph.as_dict(),
        "initialization": {
            "spanning_tree": [list(e) for e in result.init.spanning_tree],
            "records": [r.__dict__ for r in result.init.records],
        },
        "stations": result.station_rows,
        "station_poses": {
            s: {
                "R": result.R[s].tolist(),
                "t_m": result.t[s].tolist(),
                "R_init": result.R_init[s].tolist(),
                "t_init_m": result.t_init[s].tolist(),
            }
            for s in result.ds.stations
        },
        "targets": result.target_rows,
        "observation_residuals": result.residual_rows,
        "closure_summary": result.closure_summary,
        "closure_details": result.closure_rows,
        "holdout_checks": result.holdout_rows,
        "identifiability_svd": result.ident,
        "identifiability_svd_by_component": result.ident_components,
        "convention": result.ds.convention,
    }
    p = os.path.join(out_dir, "report.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_json_default)
    paths["report_json"] = p
    return paths
