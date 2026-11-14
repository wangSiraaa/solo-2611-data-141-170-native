"""命令行入口。

退出码约定（见 :mod:`expdesign.errors`）：

* 0 成功；2 配置错误（未知项/矛盾输入，先拒绝）；
* 3 有证据的不可行（结构混淆/预算不足/锁定冲突，证据写入 JSON）；
* 4 有限搜索未找到方案（不是不可行证明）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any

from .design import (
    DesignBundle, design_matrix_records, diagnostics_payload,
    load_bundle, randomized_order, run_design, run_table_rows,
)
from .errors import (
    EXIT_CONFIG_ERROR, EXIT_INFEASIBLE, EXIT_OK, EXIT_SEARCH_FAILURE,
    ConfigError, InfeasibleError, SearchFailure,
)
from .invariance import check_coding_invariance
from .io_utils import ensure_dir, write_csv, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="expdesign",
        description="有限运行次数下的实验运行设计工具（条件 D 准则坐标交换）",
    )
    parser.add_argument("--factors", default="fixtures/factors.json")
    parser.add_argument("--model", default="fixtures/model.json")
    parser.add_argument("--constraints", default="fixtures/constraints.json")
    parser.add_argument("--plan", default="fixtures/plan.json")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--random-starts", type=int, default=8)
    parser.add_argument("--max-evaluations", type=int, default=20000)
    parser.add_argument("--time-budget", type=float, default=120.0)
    parser.add_argument(
        "--json-only", action="store_true",
        help="只把结果打印为 JSON（仍会写 outdir 工件），便于脚本集成",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bundle = load_bundle(
            args.factors, args.model, args.constraints, args.plan
        )
        bundle = run_design(
            bundle,
            n_random_starts=args.random_starts,
            max_evaluations=args.max_evaluations,
            time_budget_seconds=args.time_budget,
        )
    except InfeasibleError as exc:
        payload = {
            "status": "infeasible_proven",
            "reason": exc.reason,
            "message": exc.message,
            "evidence": exc.evidence,
        }
        _emit_failure(args.outdir, "infeasible.json", payload)
        print(_format_infeasible(exc), file=sys.stderr)
        return EXIT_INFEASIBLE
    except ConfigError as exc:
        payload = {"status": "config_error", "message": str(exc)}
        _emit_failure(args.outdir, "config_error.json", payload)
        print(f"配置错误，已拒绝: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except SearchFailure as exc:
        payload = {"status": "search_failure", "message": str(exc)}
        _emit_failure(args.outdir, "search_failure.json", payload)
        print(f"搜索失败（非不可行证明）: {exc}", file=sys.stderr)
        return EXIT_SEARCH_FAILURE

    _write_artifacts(bundle, args.outdir)
    if not args.json_only:
        print(_format_success(bundle, args.outdir))
    else:
        print(json.dumps(diagnostics_payload(bundle), ensure_ascii=False,
                         indent=2, default=str))
    return EXIT_OK


# ---------------------------------------------------------------------- #
def _write_artifacts(bundle: DesignBundle, outdir: str) -> None:
    ensure_dir(outdir)
    rows = run_table_rows(bundle)
    write_csv(os.path.join(outdir, "run_table.csv"), rows)
    write_csv(os.path.join(outdir, "randomized_runs.csv"),
              randomized_order(bundle))
    names, matrix_records = design_matrix_records(bundle)
    write_csv(os.path.join(outdir, "design_matrix.csv"), matrix_records)

    payload = diagnostics_payload(bundle)
    # 等价编码不变性核对（对最终设计的完整组合）。
    combos = [
        bundle.candidates.combos[cid]
        for bid in bundle.plan.block_ids
        for cid in bundle.result.design[bid].candidate_ids
    ]
    blocks = [
        bid for bid in bundle.plan.block_ids
        for _ in bundle.result.design[bid].candidate_ids
    ]
    inv = check_coding_invariance(
        bundle.factor_set, bundle.model, bundle.candidates,
        combos, blocks, bundle.plan.block_ids,
    )
    payload["coding_invariance_check"] = asdict(inv)
    write_json(os.path.join(outdir, "diagnostics.json"), payload)


def _emit_failure(outdir: str, filename: str, payload: dict[str, Any]) -> None:
    ensure_dir(outdir)
    write_json(os.path.join(outdir, filename), payload)


def _format_infeasible(exc: InfeasibleError) -> str:
    lines = [
        "=" * 72,
        f"不可行（有结构性证据，非搜索失败） 原因: {exc.reason}",
        exc.message,
        "证据:",
        json.dumps(exc.evidence, ensure_ascii=False, indent=2, default=str),
        "=" * 72,
    ]
    return "\n".join(lines)


def _format_success(bundle: DesignBundle, outdir: str) -> str:
    d = diagnostics_payload(bundle)
    sel = d["selected_design"]
    base = d["baseline_round_robin"]
    report = d["search_report"]
    verify = d["independent_verification"]
    model_n_treat = bundle.model.n_treatment
    lines = [
        "=" * 72,
        "实验设计完成",
        f"准则: {d['criterion']}",
        "-" * 72,
        f"选中设计 log det(M) = {sel['log_det_conditional_information']:.6f}"
        f"  (每参数 det 根 = {sel['det_root_per_parameter']:.6f},"
        f" D 效率 = {sel['d_efficiency_vs_n']:.6f})",
        "朴素基线 log det(M) = "
        + (f"{base['log_det_conditional_information']:.6f}"
           if base["feasible_full_rank"] else "不满秩"),
        f"改进: {d['improvement_over_baseline']}",
        "-" * 72,
        f"起点: {report['start_points']}",
        f"停止条件: {report['stop_reason']}；sweep 轮数: {report['n_sweeps']}；"
        f"交换次数: {report['n_exchange_moves']}",
        f"目标评估次数: {report['n_objective_evaluations']}；"
        f"实际检查的不同方案: {report['n_distinct_designs_checked']}；"
        f"耗时: {report['elapsed_seconds']:.3f}s",
        "-" * 72,
        f"处理矩阵秩 {sel['treatment_rank']}/{model_n_treat}，"
        f"块调整后目标秩 {sel['target_rank_adjusted_for_blocks']}/"
        f"{sel['n_target_columns']}，全矩阵秩 {sel['full_rank']}/"
        f"{sel['full_rank_expected_adjusted']}（期望调整秩），"
        f"最大绝对列相关 {sel['max_abs_pairwise_correlation']:.4f}",
        f"独立重建核对全部通过: {verify['all_passed']}",
        f"块内重复名额占用: {verify['repeat_slots']}",
        "-" * 72,
        f"枚举: {json.dumps(d['enumeration'], ensure_ascii=False)}",
        f"工件目录: {outdir}/ (run_table.csv, randomized_runs.csv, "
        f"design_matrix.csv, diagnostics.json)",
        "=" * 72,
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
