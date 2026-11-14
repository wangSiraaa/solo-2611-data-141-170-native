"""命令行入口：python -m merge3 {plan,resolve,export} ..."""

from __future__ import annotations

import argparse
import json
import sys

from .errors import Merge3Error, UnresolvedConflictsError
from .export import export as run_export
from .plan import build_plan, write_plan
from .resolve import resolve as run_resolve


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="merge3", description="固定两表的一次离线三方合并")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="识别三方变更并产出冲突计划")
    p_plan.add_argument("--input", required=True, help="input.json 路径")
    p_plan.add_argument("--schema", required=True, help="schema.sql 路径")
    p_plan.add_argument("--out", required=True, help="计划 JSON 输出路径")

    p_res = sub.add_parser("resolve", help="套用裁决并全量复查约束")
    p_res.add_argument("--plan", required=True)
    p_res.add_argument("--decisions", required=True)
    p_res.add_argument("--out", required=True, help="resolved JSON 输出路径")

    p_exp = sub.add_parser("export", help="导出 SQLite 与来源并重新打开校验")
    p_exp.add_argument("--resolved", required=True)
    p_exp.add_argument("--out", required=True, help="输出 SQLite 路径")

    args = parser.parse_args(argv)

    try:
        if args.cmd == "plan":
            cfg = json.loads(open(args.input, encoding="utf-8").read())
            plan = build_plan(cfg["base"], cfg["left"], cfg["right"], args.schema)
            write_plan(plan, args.out)
            _print(
                {
                    "status": "ok",
                    "plan": args.out,
                    "plan_digest": plan["plan_digest"],
                    "stats": plan["stats"],
                    "conflicts": [
                        {"id": g["id"], "type": g["type"], "title": g["title"]}
                        for g in plan["conflicts"]
                    ],
                }
            )
        elif args.cmd == "resolve":
            result = run_resolve(args.plan, args.decisions, args.out)
            _print(
                {
                    "status": "ok",
                    "resolved": args.out,
                    "resolved_digest": result["resolved_digest"],
                    "decisions_applied": result["decisions_applied"],
                    "merged_rows": {t: len(v) for t, v in result["merged"].items()},
                }
            )
        else:
            info = run_export(args.resolved, args.out)
            _print({"status": "ok", **info})
    except UnresolvedConflictsError as exc:
        _print({"status": "unresolved", "error": str(exc), "residual": exc.residual})
        return 2
    except Merge3Error as exc:
        _print({"status": "error", "error": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
