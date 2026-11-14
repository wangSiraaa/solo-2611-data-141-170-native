#!/usr/bin/env python3
"""端到端演示：对 resolved / weak 两份数据运行 compare，并核对共享重采样。

运行：

    .venv/bin/python scripts/run_demo.py

内容：
1. 两个 fixture 各跑一次完整比较（24 次共享 bootstrap）；
2. 显式核对每次 bootstrap 三个候选用到的重采样列完全相同；
3. 打印全数据排名、配对似然差与支持频率，保留近乎并列的证据；
4. 报告写入 reports/。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phylocompare.bootstrap import make_bootstrap_plan, resampled_patterns  # noqa: E402
from phylocompare.compare import NEAR_TIE_LL, load_inputs, run_comparison, write_report  # noqa: E402
from phylocompare.optim import fit_branch_lengths  # noqa: E402

CASES = ("resolved.fasta", "weak.fasta")


def verify_shared_columns(fasta: Path, nwk: Path, n_boot: int, seed: int) -> None:
    """不依赖报告内部结构，独立重放一遍 bootstrap，核对共享列。"""
    alignment, trees = load_inputs(fasta, nwk)
    plan = make_bootstrap_plan(alignment, n_boot, seed)
    for rep in range(n_boot):
        columns = plan.columns[rep]
        per_candidate = []
        for tree in trees:
            patterns = resampled_patterns(alignment, columns)
            # 每次都重新拟合枝长，而不是复用全数据的长度。
            fit = fit_branch_lengths(tree, patterns)
            if not fit.ok:
                raise AssertionError(f"rep {rep} 出现拟合失败，未按预期完成演示")
            per_candidate.append(
                hashlib.sha256(patterns.patterns.tobytes()).hexdigest()
            )
        if len(set(per_candidate)) != 1:
            raise AssertionError(f"rep {rep} 各候选的重采样模式不一致")
    print(f"  共享列核对：{n_boot} 次重采样、{len(trees)} 个候选，模式指纹全部一致")


def main() -> int:
    plan = json.loads((ROOT / "fixtures" / "plan.json").read_text(encoding="utf-8"))
    nwk = ROOT / "fixtures" / plan["candidate_file"]
    n_boot = plan["bootstrap_replicates"]
    seed = plan["bootstrap_seed"]

    for case in CASES:
        fasta = ROOT / "fixtures" / case
        print(f"=== {case} ===")
        verify_shared_columns(fasta, nwk, n_boot, seed)
        report = run_comparison(fasta, nwk, n_bootstrap=n_boot, seed=seed)
        out = write_report(report, ROOT / "reports" / f"{Path(case).stem}.report.json")

        print(f"  位点数 {report['site_patterns']['n_sites']}，"
              f"模式数 {report['site_patterns']['distinct_patterns']}，"
              f"含 N 位点 {report['site_patterns']['sites_with_unknown_N']}")
        print(f"  全数据排名: {' > '.join(report['full_data_ranking'])}")
        for cand in report["candidates"]:
            if not cand["optimization_ok"]:
                print(f"  {cand['candidate_id']}: 拟合失败（{cand['failure_reason']}）")
                continue
            tie = "，近乎并列" if cand["near_tied_best_full_data"] else ""
            print(
                f"  {cand['candidate_id']}: lnL={cand['log_likelihood']:.3f}, "
                f"ΔlnL={cand['delta_loglik_vs_best']:+.3f}{tie}, "
                f"触界边 {len(cand['boundary_parameters'])} 条"
            )
        for pair in report["pairwise_bootstrap"]:
            wins = pair["wins"]
            print(
                f"  配对 {pair['pair'][0]}-{pair['pair'][1]}: "
                f"平均ΔlnL={pair['delta_mean']:+.3f}±{pair['delta_sd']:.3f}, "
                f"胜负 {wins[pair['pair'][0]]}:{wins[pair['pair'][1]]} / {pair['n_evaluated']}"
            )
        for cid, item in report["bootstrap_support_frequencies"].items():
            print(
                f"  {cid} 支持频率: 严格最优 {item['strict_win_frequency']:.3f}, "
                f"最优或近乎并列(|ΔlnL|≤{NEAR_TIE_LL:g}) "
                f"{item['best_or_near_tied_frequency']:.3f}"
            )
        print(f"  报告: {out}\n")

    print("提示：支持频率是 bootstrap 经验频率，不是贝叶斯后验概率。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
