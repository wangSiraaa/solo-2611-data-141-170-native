"""命令行入口：python -m rulelearn fit|apply|verify ..."""

from __future__ import annotations

import argparse
import json
import sys

from .api import apply, fit, verify


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="rulelearn",
        description="有代价约束的可读分流规则精确学习工具（不调用外部模型）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="从标记案例学习规则并落盘")
    p_fit.add_argument("--input-dir", required=True)
    p_fit.add_argument("--output-dir", required=True)

    p_apply = sub.add_parser("apply", help="从保存的规则 JSON 重新分流 CSV")
    p_apply.add_argument("--rules", required=True)
    p_apply.add_argument("--input-csv", required=True)
    p_apply.add_argument("--output", help="可选：把分流结果写入 JSON 文件")

    p_verify = sub.add_parser("verify", help="独立重放核验 + 穷举基准")
    p_verify.add_argument("--input-dir", required=True)
    p_verify.add_argument("--output-dir", required=True)

    args = parser.parse_args(argv)
    if args.command == "fit":
        result = fit(args.input_dir, args.output_dir)
        summary = result["report"]
        print(json.dumps(
            {
                "objective_value": summary["objective_value"],
                "n_optimal_terminals": summary["n_optimal_terminals"],
                "canonical_plan": summary["canonical_plan"],
            },
            ensure_ascii=False, indent=2,
        ))
        print(f"已写出: {args.output_dir}/rules.json, exceptions.json, report.json")
        return 0

    if args.command == "apply":
        result = apply(args.rules, args.input_csv)
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"已写出: {args.output}（{len(result)} 行）")
        else:
            print(text)
        return 0

    payload = verify(args.input_dir, args.output_dir)
    print(json.dumps(
        {"all_passed": payload["all_passed"],
         "checks": [c["check"] + (" OK" if c["passed"] else " FAIL")
                    for c in payload["checks"]]},
        ensure_ascii=False, indent=2,
    ))
    print(f"已写出: {args.output_dir}/verification.json")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
