"""本地 `analyze` 命令入口：python -m period_analysis.cli analyze ...

三份输入各自独立分析，输出到独立子目录，互不拼接。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .holdout import (
    constant_holdout,
    leave_one_night_out,
    mark_half_period_pairs,
)
from .io_utils import load_lightcurve
from .periodogram import (
    MAX_CANDIDATES,
    chi2_periodogram,
    find_candidates,
    frequency_grid,
)
from .report import (
    build_report,
    write_json,
    write_periodogram_csv,
    write_phase_csv,
    write_window_csv,
)
from .window import (
    alias_table,
    assign_alias_families,
    sampling_summary,
    spectral_window,
    window_peaks,
)


def analyze_one(csv_path: str, period_bounds, max_harmonics: int, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    case_name = os.path.splitext(os.path.basename(csv_path))[0]
    lc = load_lightcurve(csv_path)

    payload: dict = {
        "case": case_name,
        "source_csv": os.path.basename(csv_path),
        "period_bounds_day": list(period_bounds),
        "max_harmonics": max_harmonics,
        "data_issues": [i.to_dict() for i in lc.issues],
        "data_clean": lc.clean,
    }
    if not lc.clean:
        # 问题已定位并写出；不带着坏点继续拟合
        write_json(os.path.join(out_dir, "candidates.json"), payload)
        return {"case": case_name, "status": "rejected", "n_issues": len(lc.issues)}

    t, y, s = lc.time, lc.brightness, lc.sigma
    t0 = float(np.mean(t))

    candidates, const_fit, meta = find_candidates(t, y, s, period_bounds, t0, max_harmonics)
    rayleigh = meta["rayleigh_resolution_per_day"]

    # 整夜留出（先算，半周期配对需要引用留出结果）
    const_ho = constant_holdout(lc)
    for c in candidates:
        c.holdout = leave_one_night_out(lc, c)

    # 采样窗与别名族
    nu, power = spectral_window(t)
    peaks = window_peaks(nu, power)
    families = assign_alias_families(candidates, rayleigh)
    alias_ev = alias_table([c.freq for c in candidates], nu, power, rayleigh)
    half_pairs = mark_half_period_pairs(candidates, rayleigh)

    # ---- 交付物 ----
    payload["grid_meta"] = meta
    payload["constant_model"] = {
        "chi2": const_fit.chi2,
        "dof": const_fit.dof,
        "reduced_chi2": const_fit.reduced_chi2,
        "constant": const_fit.constant,
        "holdout": const_ho,
    }
    payload["candidates"] = [c.to_summary_dict() for c in candidates]
    payload["candidate_id_note"] = "candidate_N 编号按全数据 chi2 升序，与 CSV 文件名一致"
    payload["scoring_note"] = (
        "explained_score = 1 - chi2/chi2_constant 仅为描述性拟合分数；"
        "它不是后验概率，候选间分数差不得解释为真实概率或后验胜算。")
    payload["sampling_window"] = {
        "sampling_summary": sampling_summary(t, lc.night_id),
        "window_peaks": [p.to_dict() for p in peaks],
        "alias_evidence": alias_ev,
        "alias_families": families,
    }
    payload["half_period_pairs"] = half_pairs

    write_json(os.path.join(out_dir, "candidates.json"), payload)
    write_json(os.path.join(out_dir, "sampling_window.json"), payload["sampling_window"])
    write_window_csv(os.path.join(out_dir, "window_function.csv"), nu, power)

    # 粗网格周期图（K=1、K=2），供独立复算峰位置
    grid = frequency_grid(meta["f_min_per_day"], meta["f_max_per_day"],
                          meta["time_span_day"])
    chi_by_k = {k: chi2_periodogram(t, y, s, grid, k, t0)
                for k in range(1, max_harmonics + 1)}
    write_periodogram_csv(os.path.join(out_dir, "periodogram.csv"), grid, chi_by_k)

    for i, c in enumerate(candidates):
        write_phase_csv(os.path.join(out_dir, f"candidate_{i}_phase.csv"), lc, c)

    report = build_report(case_name, lc, payload)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(report)

    return {"case": case_name, "status": "ok",
            "n_candidates": len(candidates),
            "best_period_day": candidates[0].period}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="period_analysis",
        description="不规则光变曲线的加权谐波周期分析（本地、无联网、无 FFT 插补）",
    )
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="分析一份或多份光变 CSV")
    a.add_argument("inputs", nargs="+", help="输入 CSV 路径（每份独立分析）")
    a.add_argument("--plan", help="plan.json 路径（提供 period_bounds_day/max_harmonics）")
    a.add_argument("--period-bounds", nargs=2, type=float, metavar=("P_MIN", "P_MAX"))
    a.add_argument("--max-harmonics", type=int, choices=(1, 2), default=2)
    a.add_argument("--out-dir", default="output", help="输出根目录（默认 ./output）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "analyze":
        return 2
    period_bounds, max_h = args.period_bounds, args.max_harmonics
    if args.plan:
        with open(args.plan, encoding="utf-8") as fh:
            plan = json.load(fh)
        period_bounds = period_bounds or plan["period_bounds_day"]
        max_h = plan.get("max_harmonics", max_h)
    if not period_bounds:
        print("错误：需要 --period-bounds 或 --plan 提供周期范围", file=sys.stderr)
        return 2

    summaries = []
    for path in args.inputs:
        case = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(args.out_dir, case)
        summaries.append(analyze_one(path, period_bounds, max_h, out_dir))
    write_json(os.path.join(args.out_dir, "run_summary.json"),
               {"command": "analyze", "results": summaries})
    for s in summaries:
        print(json.dumps(s, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
