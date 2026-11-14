"""命令行入口：python -m rnacount build [--bam ... --exons ... --plan ... --out ...]"""
from __future__ import annotations

import argparse
import sys

from .build import STATUS_LABELS, run_build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rnacount",
        description="已比对双端片段 → 转录本兼容集合 / 等价类计数",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser(
        "build",
        help="从坐标排序 BAM 构建片段兼容集合、唯一支持与等价类计数",
    )
    b.add_argument("--bam", default="fixtures/reads.bam", help="坐标排序的 BAM")
    b.add_argument("--exons", default="fixtures/exons.tsv", help="零基半开外显子注释 TSV")
    b.add_argument("--plan", default="fixtures/plan.json", help="实验计划/过滤规则 JSON")
    b.add_argument("--out", default="out", help="输出目录")
    args = parser.parse_args(argv)

    if args.command == "build":
        diag = run_build(args.bam, args.exons, args.plan, args.out)
        fr = diag["fragments"]
        print(f"输出目录: {args.out} (fragments.csv, counts_unique.csv, counts_equivalence.csv, diagnostics.json, summary.txt)")
        print(f"片段总数 {fr['total_query_names']}: " + ", ".join(f"{STATUS_LABELS[k]} {fr[k]}" for k in STATUS_LABELS))
        ok = diag["closure"]["fragment_ok"] and diag["closure"]["alignment_ok"]
        print(f"闭合检查: {'OK' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
