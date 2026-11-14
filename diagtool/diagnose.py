"""目标订单的排除路径诊断。

分类：
  * base_row_missing      —— 基础表（orders）中不存在该主键
  * inner_join_no_match   —— INNER 连接某步无匹配行（含 NULL 键）
  * where_false           —— 至少一个候选行的 WHERE 为 FALSE
  * where_unknown         —— 候选行 WHERE 为 UNKNOWN（仅 UNKNOWN 或混合）
  * visible               —— 该订单本应出现在原报表中（不应进入缺失诊断）
"""

from . import ast_nodes as ast
from .semantics import tri_label


def _atom_state_summary(cand) -> dict:
    states = {}
    for idx in sorted(cand.atom_values):
        states[idx] = tri_label(cand.atom_values[idx])
    return states


def _where_path(cand, select) -> dict:
    """给出完整布尔结构的求值路径，并标注 OR 分支与 LEFT NULL 扩展。"""
    return {
        "where_sql": _where_sql(select),
        "tri_value": tri_label(cand.where_value),
        "tree": cand.where_trace,
        "atoms": [
            {
                "atom_index": idx,
                **cand.atom_details[idx],
            }
            for idx in sorted(cand.atom_details)
        ],
    }


def _where_sql(select) -> str:
    if select.where is None or select.where_start < 0:
        return ""
    return select.original_sql[select.where_start:select.where_end].strip()


def _candidate_evidence(cand) -> dict:
    return {
        "join_rows": {
            alias: (snap.evidence() if snap is not None else {
                "alias": alias,
                "null_extended": True,
                "reason": "LEFT JOIN 无匹配时按 SQL 语义补出的 NULL 行",
            })
            for alias, snap in cand.sources.items()
        },
        "null_extended_aliases": list(cand.synthetic_aliases),
        "atom_states": _atom_state_summary(cand),
        "where_result": tri_label(cand.where_value),
        "synthetic_result_identity": {
            "columns": cand.identity["columns"],
            "values": list(cand.identity["values"]),
        },
    }


def diagnose_target(ctx, target_pk_value: str) -> dict:
    """对单个目标主键做完整排除诊断。"""
    sel = ctx.select
    base_table = ctx.target_table
    # 通过主键列在基础表中定位该行。
    all_base = ctx.db.rows(base_table.name)
    matched_base = [r for r in all_base if str(r.get(ctx.target_pk)) == str(target_pk_value)]
    result = {
        "target_pk": target_pk_value,
        "target_table": base_table.name,
        "target_pk_column": ctx.target_pk,
        "base_row_exists": bool(matched_base),
        "category": None,
        "explanation": "",
        "join_trace": [],
        "candidate_rows": [],
        "visible_in_original": False,
    }
    if not matched_base:
        result["category"] = "base_row_missing"
        result["explanation"] = (
            f"目标 {ctx.target_pk}={target_pk_value!r} 在基础表 "
            f"{base_table.name} 中不存在，故不可能出现在任何结果中。"
        )
        return result

    base_row = matched_base[0]
    candidates, traces = ctx.enumerate_candidates(target_values=base_row)
    trace = traces[0]
    result["join_trace"] = _render_trace(ctx, trace)

    if not trace["alive"]:
        failing = [s for s in trace["steps"] if s.join_type == "INNER"
                   and not s.matched_rows]
        last = failing[-1] if failing else None
        result["category"] = "inner_join_no_match"
        if last is not None:
            result["explanation"] = (
                f"基础行存在，但 INNER JOIN {last.alias}（{last.table.name}）"
                f" 在 ON {last.on_text} 上没有匹配行；INNER 连接会直接丢弃该行，"
                "WHERE 尚未参与判断。"
            )
        else:
            result["explanation"] = "基础行存在，但连接链上无任何匹配元组。"
        return result

    visible_cands = [c for c in candidates if c.visible]
    result["visible_in_original"] = bool(visible_cands)
    if visible_cands:
        result["category"] = "visible"
        result["explanation"] = (
            "目标在原始查询中可见；无需诊断排除路径。"
        )
        result["candidate_rows"] = [
            _candidate_evidence(c) for c in candidates
        ]
        return result

    # 所有候选行都被 WHERE 排除。区分 FALSE 与 UNKNOWN。
    has_false = any(c.where_value is False for c in candidates)
    has_unknown = any(c.where_value is None for c in candidates)
    if has_false and not has_unknown:
        result["category"] = "where_false"
    elif has_unknown and not has_false:
        result["category"] = "where_unknown"
    else:
        result["category"] = "where_false_or_unknown"
    result["explanation"] = _explain_where(ctx, candidates)
    result["candidate_rows"] = [
        {
            "candidate_index": i,
            "join_rows": {
                alias: (snap.evidence() if snap is not None else {
                    "alias": alias,
                    "null_extended": True,
                    "reason": "LEFT JOIN 无匹配时按 SQL 语义补出的 NULL 行",
                })
                for alias, snap in c.sources.items()
            },
            "null_extended_aliases": list(c.synthetic_aliases),
            "atom_states": _atom_state_summary(c),
            "atom_details": [
                c.atom_details[idx] for idx in sorted(c.atom_details)
            ],
            "where_result": tri_label(c.where_value),
            "where_path": _where_path(c, sel),
            "synthetic_result_identity": {
                "columns": c.identity["columns"],
                "values": list(c.identity["values"]),
            },
        }
        for i, c in enumerate(candidates)
    ]
    return result


def _render_trace(ctx, trace) -> list[dict]:
    out = []
    for step in trace["steps"]:
        if step.join_type == "BASE":
            out.append({
                "step": step.step_index,
                "operation": "base_table",
                "alias": step.alias,
                "table": step.table.name,
                "matched": True,
                "row": step.matched_rows[0].evidence() if step.matched_rows else None,
            })
            continue
        out.append({
            "step": step.step_index,
            "operation": step.join_type.lower() + "_join",
            "alias": step.alias,
            "table": step.table.name,
            "on": step.on_text,
            "matched_row_count": len(step.matched_rows),
            "matched_rows": [r.evidence() for r in step.matched_rows],
            "null_extended": step.synthetic and not step.matched_rows,
        })
    return out


def _explain_where(ctx, candidates) -> str:
    lines = []
    n = len(candidates)
    lines.append(f"共重建 {n} 条候选关联行，全部被 WHERE 排除：")
    for i, c in enumerate(candidates):
        states = ", ".join(
            f"[{idx}] {c.atom_details[idx]['sql']} = {tri_label(c.atom_values[idx])}"
            for idx in sorted(c.atom_values)
        )
        synth = ""
        if c.synthetic_aliases:
            synth = f"（LEFT 为 {', '.join(c.synthetic_aliases)} 补出 NULL 行）"
        lines.append(
            f"  候选 {i}{synth}: {states}；WHERE = {tri_label(c.where_value)}"
        )
    return "\n".join(lines)
