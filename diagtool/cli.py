"""命令行入口。

用法：
  python3 -m diagtool explain --db work-input/report.sqlite \
      --query work-input/query.sql --plan work-input/plan.json \
      --allowed work-input/allowed_changes.json [--format json|text] [--max-cost N]

只读：数据库以 mode=ro 打开，绝不写回。规划/输入错误返回码 2，未知语法返回码 3。
"""

import argparse
import json
import sys

from .config import load_allowed_changes, load_plan
from .errors import DiagError, PlanError, UnsupportedSyntaxError
from .report import run_diagnosis


def _fmt_value(v):
    if v is None:
        return "NULL"
    if isinstance(v, str):
        return repr(v)
    return str(v)


def render_text(report) -> str:
    lines = []
    q = report["query_structure"]
    lines.append("== 原查询解释 ==")
    lines.append(q["sql"].strip())
    lines.append("")
    lines.append(f"基础表: {q['from']['table']} (别名 {q['from']['alias']})")
    for j in q["joins"]:
        lines.append(f"  {j['join_type'].upper()} JOIN {j['table']} "
                     f"(别名 {j['alias']}) ON {j['on']}")
    lines.append("SELECT 列:")
    for item in q["select_items"]:
        lines.append(f"  {item['expression']} -> {item['output_name']}")
    lines.append("WHERE 原子谓词（按源码顺序）:")
    for a in q["where"]["atoms"]:
        lines.append(f"  [{a['atom_index']}] {a['sql']}  "
                     f"(span {a['source_span'][0]}..{a['source_span'][1]})")
    if q.get("parameters_used"):
        lines.append(f"绑定参数: {', '.join(':' + p for p in q['parameters_used'])}")
    lines.append("")

    b = report["baseline"]
    lines.append("== 原报表基线（真实 SQL 执行）==")
    lines.append(f"结果 {b['row_count']} 行；列: {', '.join(b['columns'])}")
    for r in b["rows"]:
        lines.append("  " + " | ".join(_fmt_value(r.get(c)) for c in b["columns"]))
    lines.append("目标可见次数: " +
                 ", ".join(f"{k}={v}" for k, v in
                           b["target_visibility_counts"].items()))
    lines.append("")

    lines.append("== 目标排除诊断 ==")
    for d in report["target_diagnoses"]:
        lines.append(f"[{d['target_pk']}] 类别={d['category']} "
                     f"基础行存在={d['base_row_exists']} "
                     f"原报表可见={d['visible_in_original']}")
        lines.append("  说明: " + d["explanation"].replace("\n", "\n  "))
        if d.get("join_trace"):
            for st in d["join_trace"]:
                if st["operation"] == "base_table":
                    lines.append(f"  步骤{st['step']} base_table {st['table']}: 命中")
                else:
                    lines.append(
                        f"  步骤{st['step']} {st['operation']} {st['table']} "
                        f"ON {st['on']}: 匹配 {st['matched_row_count']} 行"
                        + ("（LEFT 补 NULL）" if st.get("null_extended") else ""))
                    for mr in st.get("matched_rows", []):
                        pk = mr.get("primary_key")
                        lines.append(
                            f"      - 行 {mr['table']} {pk}: "
                            + ", ".join(f"{k}={_fmt_value(v)}"
                                        for k, v in mr["values"].items()))
        for c in d.get("candidate_rows", []):
            synth = c.get("null_extended_aliases") or []
            lines.append(f"  候选 {c.get('candidate_index', '-')}"
                         + (f"（NULL 扩展: {', '.join(synth)}）" if synth else ""))
            lines.append("      原子状态: " + ", ".join(
                f"[{k}]={v}" for k, v in c["atom_states"].items()))
            for ad in c.get("atom_details", []):
                lines.append(
                    f"      [{ad['atom_index']}] {ad['sql']}: "
                    f"左={_fmt_value(ad['left'].get('value'))} "
                    f"({ad['left'].get('alias', ad['left'].get('kind'))}) "
                    f"op {ad['op']} "
                    f"右={_fmt_value(ad['right'].get('value'))} "
                    f"({ad['right'].get('alias', ad['right'].get('kind'))}) "
                    f"=> {ad['result']}")
            lines.append(f"      WHERE 结果: {c['where_result']}")
        lines.append("")

    lines.append("== 最少条件调整方案（均已用实际 SQL 重跑验证）==")
    ss = report["solutions"]["search_space"]
    lines.append("可删谓词: " + "; ".join(
        f"{p['id']}={p['declared_sql']} (原子[{p['resolved_atom_index']}])"
        for p in ss["removable_predicates"]))
    lines.append("参数备选: " + json.dumps(ss["parameter_alternatives"],
                                       ensure_ascii=False))
    for t in report["solutions"]["per_target"]:
        lines.append("")
        if not t["feasible"]:
            lines.append(f"[{t['target']}] 无可行方案：{t['rejected_reason']}")
            continue
        lines.append(f"[{t['target']}] 最小成本={t['minimum_cost']}，"
                     f"并列方案 {t['parallel_minimum_count']} 个")
        for i, sol in enumerate(t["verified_solutions"]):
            lines.append(f"  方案 {i}: 成本 {sol['cost']}")
            for op in sol["operations"]:
                if op["kind"] == "change_parameter":
                    lines.append(f"    - 改参数 :{op['parameter']} "
                                 f"{_fmt_value(op['from'])} -> {_fmt_value(op['to'])}")
                else:
                    lines.append(f"    - 删除谓词 {op['predicate_id']} "
                                 f"({op['removed_sql']}) => TRUE")
            lines.append("    建议 SQL（未应用）:")
            for ln in sol["suggested_sql"].splitlines():
                lines.append("      " + ln)
            imp = sol["revalidation"]["impact"]
            lines.append(f"    验证: 目标可见={sol['revalidation']['target_visible']}，"
                         f"结果 {imp['solution_row_count']} 行"
                         f"（原 {imp['baseline_row_count']} 行）")
            lines.append(f"    新增 {imp['added_row_count']} 行:")
            for a in imp["added"]:
                lines.append(f"      + x{a['count']} " + " | ".join(
                    _fmt_value(a["row"].get(c)) for c in imp["columns"]))
            lines.append(f"    减少 {imp['removed_row_count']} 行:")
            for rm in imp["removed"]:
                lines.append(f"      - x{rm['count']} " + " | ".join(
                    _fmt_value(rm["row"].get(c)) for c in imp["columns"]))
    lines.append("")
    lines.append("注意: 以上方案仅为诊断建议，均未写入源数据。")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="diagtool",
        description="只读诊断受限 SELECT 报表为何排除指定订单，并枚举最少条件调整。")
    parser.add_argument("--version", action="version",
                        version="%(prog)s 1.0.0")
    sub = parser.add_subparsers(dest="command", required=True)

    explain = sub.add_parser(
        "explain", help="解释原查询、逐目标排除路径与验证后的最少调整方案")
    explain.add_argument("--db", required=True, help="只读 SQLite 数据库路径")
    explain.add_argument("--query", required=True, help="query.sql 路径")
    explain.add_argument("--plan", required=True, help="plan.json 路径")
    explain.add_argument("--allowed", required=True, help="allowed_changes.json 路径")
    explain.add_argument("--format", choices=("json", "text"), default="text")
    explain.add_argument("--max-cost", type=int, default=None,
                         help="最多枚举的操作成本上限（默认自动）")
    explain.add_argument("--output", default=None, help="把报告写入文件而非标准输出")

    args = parser.parse_args(argv)
    try:
        from pathlib import Path
        query_sql = Path(args.query).read_text(encoding="utf-8")
        plan = load_plan(Path(args.plan))
        allowed = load_allowed_changes(Path(args.allowed))
        report = run_diagnosis(args.db, query_sql, plan, allowed,
                               max_cost=args.max_cost)
    except UnsupportedSyntaxError as exc:
        print(f"错误：不支持的 SQL 语法：{exc}", file=sys.stderr)
        return 3
    except PlanError as exc:
        print(f"错误：输入/计划不一致：{exc}", file=sys.stderr)
        return 2
    except DiagError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    else:
        text = render_text(report)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(text + ("\n" if not text.endswith("\n") else ""),
                                     encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
