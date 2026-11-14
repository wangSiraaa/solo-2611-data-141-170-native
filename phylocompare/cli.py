"""命令行入口：``python -m phylocompare compare ...``。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compare import NEAR_TIE_LL, run_comparison, write_report
from .errors import CompareError


def _load_plan(plan_path: Path) -> dict:
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    required = ["cases", "candidate_file", "branch_bounds", "bootstrap_replicates", "bootstrap_seed"]
    missing = [key for key in required if key not in data]
    if missing:
        raise CompareError(f"plan.json 缺少字段: {missing}")
    return data


def _print_summary(report: dict) -> None:
    print(f"输入: {report['input']['fasta']}")
    sp = report["site_patterns"]
    print(
        f"位点 {sp['n_sites']} 个，分类单元 {sp['n_taxa']} 个，"
        f"不同模式 {sp['distinct_patterns']} 个，含 N 位点 {sp['sites_with_unknown_N']} 个"
    )
    print(f"全数据排名（似然由高到低）: {' > '.join(report['full_data_ranking'])}")
    for cand in report["candidates"]:
        cid = cand["candidate_id"]
        if not cand["optimization_ok"]:
            print(f"  {cid}: 拟合失败 — {cand['failure_reason']}")
            continue
        delta = cand["delta_loglik_vs_best"]
        tie = "（近乎并列）" if cand["near_tied_best_full_data"] else ""
        print(
            f"  {cid}: lnL={cand['log_likelihood']:.4f}, "
            f"ΔlnL={delta:+.4f}{tie}, 触界边 {len(cand['boundary_parameters'])} 条"
        )
    print(
        f"bootstrap 重采样 {report['run_parameters']['bootstrap_replicates']} 次"
        f"（指纹 {report['bootstrap_resample_fingerprint']}，"
        f"近乎并列阈值 |ΔlnL|≤{NEAR_TIE_LL:g}）"
    )
    for cid, item in report["bootstrap_support_frequencies"].items():
        print(
            f"  {cid}: 严格最优频率 {item['strict_win_frequency']:.3f}, "
            f"最优或近乎并列频率 {item['best_or_near_tied_frequency']:.3f}, "
            f"拟合失败 {item['fit_failures']} 次"
        )
    print("注：支持频率是 bootstrap 经验频率，不是贝叶斯后验概率。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phylocompare",
        description="在固定的少量无根二叉候选树之间，用 JC69 树似然与共享 bootstrap 做比较。",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("compare", help="比较候选拓扑并输出 JSON 报告")
    p.add_argument("--plan", required=True, type=Path, help="plan.json 路径")
    p.add_argument(
        "--case",
        help="plan.json cases 中的单个 FASTA 文件名；省略时依次运行全部 cases",
    )
    p.add_argument(
        "--fasta",
        type=Path,
        help="显式指定单个 FASTA（与 --case 二选一，仍需 --plan 提供参数）",
    )
    p.add_argument("--out-dir", type=Path, default=Path("reports"), help="报告输出目录")
    p.add_argument(
        "--replicates",
        type=int,
        help="覆盖 plan.json 的 bootstrap 次数（计划上限 24）",
    )
    p.add_argument("--seed", type=int, help="覆盖 plan.json 的 bootstrap 种子")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan_path: Path = args.plan
        plan = _load_plan(plan_path)
        base_dir = plan_path.parent
        newick_path = base_dir / plan["candidate_file"]
        bounds = tuple(plan["branch_bounds"])
        if len(bounds) != 2 or not (0 < bounds[0] < bounds[1]):
            raise CompareError(f"branch_bounds 必须是 [正数下界, 上界]，得到 {bounds}")
        n_boot = args.replicates if args.replicates is not None else plan["bootstrap_replicates"]
        if not 1 <= n_boot <= 24:
            raise CompareError("bootstrap 次数限制在 1..24（计划上限 24）")
        seed = args.seed if args.seed is not None else plan["bootstrap_seed"]

        if args.fasta is not None:
            cases = [args.fasta]
        elif args.case is not None:
            if args.case not in plan["cases"]:
                raise CompareError(f"--case {args.case!r} 不在 plan.json cases 中")
            cases = [base_dir / args.case]
        else:
            cases = [base_dir / name for name in plan["cases"]]

        for fasta_path in cases:
            report = run_comparison(
                fasta_path=fasta_path,
                newick_path=newick_path,
                n_bootstrap=n_boot,
                seed=seed,
                bounds=bounds,  # type: ignore[arg-type]
            )
            out_path = args.out_dir / f"{Path(fasta_path).stem}.report.json"
            write_report(report, out_path)
            _print_summary(report)
            print(f"报告已写入: {out_path}\n")
    except CompareError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
