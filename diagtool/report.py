"""诊断报告编排：先解释原查询，再逐个目标给出排除证据，再列验证后的方案及影响。

严格只读：不修改数据库，不自动把建议写成正式 SQL。方案以“建议 / 未应用”呈现。
"""

import datetime
import hashlib

from .config import AllowedChanges, Plan
from .db import Database
from .diagnose import diagnose_target
from .errors import PlanError
from .parser import parse_sql
from .analysis import AnalysisContext
from .runner import execute_rows, identity_rows
from .solutions import search_target

TOOL_VERSION = "1.0.0"


def build_context(db, query_sql, plan: Plan) -> AnalysisContext:
    select = parse_sql(query_sql)
    ctx = AnalysisContext(db, select, plan.parameters,
                          target_pk=plan.target_pk,
                          identity_column=plan.result_identity_column)
    _validate_plan_against_query(ctx, plan)
    return ctx


def _validate_plan_against_query(ctx, plan: Plan) -> None:
    if ctx.target_table.name.lower() != plan.target_table.lower():
        raise PlanError(
            f"plan.target_table={plan.target_table!r} 与查询基础表 "
            f"{ctx.target_table.name!r} 不一致（工具以 FROM 的首个表为基础表）"
        )
    if not ctx.target_table.has_column(plan.target_pk):
        raise PlanError(
            f"plan.target_pk={plan.target_pk!r} 不是基础表 "
            f"{ctx.target_table.name} 的列"
        )
    ctx.target_pk = ctx.target_table.canonical_column(plan.target_pk)
    names = {item.alias or item.expression.name for item in ctx.select.items}
    names |= {item.expression.name for item in ctx.select.items}
    if plan.result_identity_column not in names:
        raise PlanError(
            f"plan.result_identity_column={plan.result_identity_column!r} "
            f"不在 SELECT 列表 {sorted(names)} 中，无法判定目标可见性"
        )


def run_diagnosis(db_path, query_sql, plan: Plan, allowed: AllowedChanges,
                  max_cost=None):
    with Database(db_path) as db:
        ctx = build_context(db, query_sql, plan)
        baseline_rows = execute_rows(db, ctx.select.original_sql,
                                     ctx.bindings(plan.parameters))
        columns = [item.alias or item.expression.name for item in ctx.select.items]
        baseline_counts = identity_rows(baseline_rows,
                                        plan.result_identity_column)

        target_reports = []
        searches = []
        for t in plan.targets:
            diag = diagnose_target(ctx, t)
            search = None
            if diag["category"] == "visible":
                diag["visible_row_count"] = baseline_counts.get(t, 0)
            else:
                search = search_target(ctx, allowed, t, baseline_rows,
                                       max_cost=max_cost, diagnosis=diag)
                searches.append(search)
                diag["adjustment_search"] = {
                    "feasible": search.feasible,
                    "minimum_cost": search.min_cost if search.feasible else None,
                    "parallel_minimum_count": len(search.solutions),
                    "reason": search.reason or None,
                }
            target_reports.append(diag)

        report = {
            "tool": "order-exclusion-diagnoser",
            "version": TOOL_VERSION,
            "generated_at": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "inputs": {
                "database": {"path": str(db_path), "read_only": True},
                "query_sha256": hashlib.sha256(
                    ctx.select.original_sql.encode("utf-8")).hexdigest(),
                "parameters": plan.parameters,
                "target_table": plan.target_table,
                "target_pk": ctx.target_pk,
                "targets": plan.targets,
                "result_identity_column": plan.result_identity_column,
                "allowed_changes": allowed.raw,
            },
            "query_structure": describe_query(ctx),
            "baseline": {
                "sql": ctx.select.original_sql,
                "columns": columns,
                "row_count": len(baseline_rows),
                "rows": [{c: r.get(c) for c in columns} for r in baseline_rows],
                "target_visibility_counts":
                    {t: baseline_counts.get(t, 0) for t in plan.targets},
            },
            "target_diagnoses": target_reports,
            "solutions": {
                "search_space": describe_search_space(ctx, allowed),
                "per_target": [describe_search(ctx, s) for s in searches],
                "any_feasible": any(s.feasible for s in searches) if searches
                                else True,
                "note": ("每个方案均已用实际 SQL 在只读库上重跑并核实目标可见；"
                         "impact 为相对原报表的完整结果行新增/减少及重复次数。"
                         "suggested_sql 仅作建议，未应用到任何数据。"),
            },
        }
        return report


def describe_query(ctx) -> dict:
    sel = ctx.select
    return {
        "sql": sel.original_sql,
        "from": {"table": sel.base_table.name, "alias": sel.base_table.alias},
        "joins": [
            {"join_type": j.join_type.lower(), "table": j.right.name,
             "alias": j.right.alias, "on": j.on_text}
            for j in sel.joins
        ],
        "select_items": [
            {"expression": item.raw_text,
             "output_name": item.alias or item.expression.name}
            for item in sel.items
        ],
        "where": {
            "sql": _where_sql(sel),
            "atoms": [
                {"atom_index": a.atom_index, "sql": a.sql_text,
                 "node_id": a.node_id, "source_span": [a.start, a.end],
                 "uses_parameters": sorted(
                     p.name for p in _atom_params(a))}
                for a in ctx.atoms
            ],
        },
        "parameters_used": sorted(
            p.name for p in _collect_params(sel.where)),
    }


def _atom_params(atom):
    from .ast_nodes import collect_params
    return collect_params(atom)


def _collect_params(node):
    from .ast_nodes import collect_params
    return collect_params(node)


def _where_sql(sel):
    if sel.where is None or sel.where_start < 0:
        return ""
    return sel.original_sql[sel.where_start:sel.where_end].strip()


def describe_search_space(ctx, allowed) -> dict:
    preds = []
    for decl in allowed.removable_predicates:
        preds.append({
            "id": decl.id, "declared_sql": decl.sql,
            "resolved_atom_index": decl.atom_index,
            "resolved_atom_sql": decl.atom_sql,
            "cost": 1,
        })
    return {
        "parameter_alternatives": allowed.parameter_alternatives,
        "removable_predicates": preds,
        "cost_rule": "每个参数换值或删除一个可删谓词各计 1",
        "remove_semantics": "将被删除原子谓词的 AST 位置替换为 TRUE，保留其余括号结构",
    }


def describe_search(ctx, search) -> dict:
    out = {
        "target": search.target,
        "feasible": search.feasible,
        "minimum_cost": search.min_cost if search.feasible else None,
        "parallel_minimum_count": len(search.solutions),
        "rejected_reason": search.reason or None,
        "verified_solutions": [describe_solution(ctx, sol) for sol in search.solutions],
    }
    return out


def describe_solution(ctx, sol) -> dict:
    return {
        "cost": sol.cost,
        "operations": [op.describe() for op in sol.operations],
        "parameters": sol.params,
        "removed_atom_indices": list(sol.removed_ids),
        "suggested_sql": sol.sql,
        "applied": False,
        "note": "建议方案，未修改源数据；如需采用请人工审阅后自行执行",
        "revalidation": sol.revalidation,
    }
