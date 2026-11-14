"""resolve 入口：套用用户裁决文件并全量复查。

原则：
  1. 裁决文件必须显式点名 plan_digest；重新计算三份输入与 schema 的摘要，
     源变化后拒绝套用旧裁决；
  2. 每个计划冲突组都必须被显式裁决，选项形状逐组校验，不接受未知组/未知动作；
  3. 裁决后从零重建合并快照，重新检查外键、唯一、NOT NULL、非负金额；
  4. 一个裁决引出新冲突时，返回仍需裁决的原因（含合法选项），不写任何产物；
  5. 绝不自动删订单、恢复删除行或编造业务值——这些动作只能来自用户裁决。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .canonical import MISSING, digest_of, file_sha256, read_json, write_json
from .errors import DecisionError, InputChangedError, Merge3Error, UnresolvedConflictsError
from .schema import load_schema

FORMAT_VERSION = 1
MAX_FIXPOINT = 8

# 字段状态 -> 落库值来源
STATUS_SOURCE = {
    "unchanged": ["base"],
    "left_only": ["left"],
    "right_only": ["right"],
    "both_same": ["left", "right"],
    "left_set": ["left"],
    "right_set": ["right"],
    "both_set_same": ["left", "right"],
}


def verify_inputs(plan: dict[str, Any]) -> None:
    for role in ("base", "left", "right", "schema_sql"):
        spec = plan["inputs"][role]
        actual = file_sha256(spec["path"])
        if actual != spec["sha256"]:
            raise InputChangedError(
                f"{role} 输入 {spec['path']} 的摘要已变化："
                f"计划绑定 {spec['sha256'][:12]}…，当前 {actual[:12]}…；旧裁决不得套用"
            )


def _plan_group_index(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {g["id"]: g for g in plan["conflicts"]}


def _plan_row_index(plan: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        tname: {row["key"]: row for row in payload["rows"]}
        for tname, payload in plan["tables"].items()
    }


def resolve(plan_path: str, decisions_path: str, resolved_out_path: str) -> dict[str, Any]:
    plan = read_json(plan_path)
    decisions_doc = read_json(decisions_path)

    if decisions_doc.get("plan_digest") != plan["plan_digest"]:
        raise DecisionError(
            "裁决文件绑定的 plan_digest 与当前计划不一致，拒绝套用（请基于当前计划重新裁决）"
        )
    verify_inputs(plan)
    schema = load_schema(plan["inputs"]["base"]["path"])

    raw_decisions = decisions_doc.get("decisions")
    if not isinstance(raw_decisions, list):
        raise DecisionError("裁决文件必须包含 decisions 列表（无冲突时可为空列表）")
    decisions: dict[str, dict[str, Any]] = {}
    for d in raw_decisions:
        gid = d.get("group_id")
        if not gid:
            raise DecisionError(f"裁决缺少 group_id：{d}")
        if gid in decisions:
            raise DecisionError(f"冲突组 {gid} 被重复裁决")
        decisions[gid] = d.get("decision")
        if not isinstance(decisions[gid], dict) or "action" not in decisions[gid]:
            raise DecisionError(f"冲突组 {gid} 的裁决缺少合法 action")

    rows_index = _plan_row_index(plan)
    applied: list[dict[str, Any]] = []
    consumed: set[str] = set()
    active_groups = _plan_group_index(plan)

    merged, meta = _initial_snapshot(plan)

    for _ in range(MAX_FIXPOINT):
        # 应用所有匹配当前活动组的裁决（按 id 排序，保证与行序无关的确定性）。
        matched_this_pass = False
        for gid in sorted(set(decisions) - consumed):
            if gid not in active_groups:
                continue
            _apply_group(gid, active_groups[gid], decisions[gid], merged, meta, rows_index)
            consumed.add(gid)
            applied.append({"group_id": gid, "decision": decisions[gid]})
            matched_this_pass = True

        # 计划内冲突组必须全部裁决。
        plan_ids = set(_plan_group_index(plan))
        missing = sorted(plan_ids - consumed)
        if missing:
            raise DecisionError(f"以下冲突组尚未裁决：{', '.join(missing)}")

        residual = _global_recheck(schema, plan, merged, meta, rows_index, applied)
        if not residual:
            unknown = sorted(set(decisions) - consumed)
            if unknown:
                raise DecisionError(f"裁决引用了不存在的冲突组：{', '.join(unknown)}")
            result = _build_result(plan, merged, meta, rows_index, applied)
            write_json(resolved_out_path, result)
            return result

        residual_ids = {g["id"] for g in residual}
        if not any(gid in residual_ids for gid in set(decisions) - consumed):
            raise UnresolvedConflictsError(
                "裁决后仍存在约束冲突，且没有可继续套用的裁决", residual
            )
        if not matched_this_pass:
            raise UnresolvedConflictsError("裁决无法消解剩余冲突", residual)
        active_groups = {g["id"]: g for g in residual}

    raise Merge3Error("冲突级联超过最大迭代次数，拒绝发布半成品")


def _initial_snapshot(plan: dict[str, Any]):
    """从计划中的自动合并行建立初始快照与来源元数据。"""
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    meta: dict[str, dict[str, dict[str, Any]]] = {}
    for tname, payload in plan["tables"].items():
        merged[tname] = {}
        meta[tname] = {}
        for row in payload["rows"]:
            if row["auto_row"] is None:
                continue
            key = row["key"]
            merged[tname][key] = deepcopy(row["auto_row"])
            field_sources = {
                f["name"]: STATUS_SOURCE.get(f["status"], ["base"])
                for f in row["fields"]
            }
            kind = "insert" if row["kind"] == "insert_auto" else (
                "unchanged" if row["kind"] == "unchanged" else "auto_merged"
            )
            meta[tname][key] = {
                "row_kind": kind,
                "field_sources": field_sources,
                "inserted_by": row["inserted_by"],
            }
    return merged, meta


def _pk_of(table: str) -> str:
    return "customer_id" if table == "customers" else "order_id"


def _with_pk(table: str, key: str, values: dict[str, Any]) -> dict[str, Any]:
    """计划行的 values 只含值列，落库快照必须补回主键列。"""
    return {_pk_of(table): key, **deepcopy(values)}


def _assemble_conflict_row(table: str, row: dict[str, Any], overrides: dict[str, Any]):
    """FIELD_CONFLICT 行在字段被逐一裁决后整行装配。"""
    pk_col = "customer_id" if table == "customers" else "order_id"
    assembled: dict[str, Any] = {pk_col: row["key"]}
    sources: dict[str, list[str]] = {pk_col: ["base"]}
    for f in row["fields"]:
        col = f["name"]
        if col in overrides:
            assembled[col] = overrides[col]
            sources[col] = [overrides[col + "__side"]]
        elif f["status"] == "unchanged":
            assembled[col] = f["base"]
            sources[col] = ["base"]
        else:
            assembled[col] = f["auto_value"]
            sources[col] = STATUS_SOURCE.get(f["status"], ["base"])
    return assembled, sources


def _apply_group(gid, group, decision, merged, meta, rows_index) -> None:
    gtype = group["type"]
    action = decision.get("action")

    if gtype == "field_conflict":
        if action != "pick":
            raise DecisionError(f"{gid} 只支持 action=pick")
        side = decision.get("side")
        if side not in ("left", "right"):
            raise DecisionError(f"{gid} 的 pick 侧必须是 left/right")
        table, key, col = _affected_single(group)
        val = group["values"][side]
        if val is MISSING:
            raise DecisionError(f"{gid} 所选侧不存在该行")
        row = rows_index[table][key]
        if key not in merged[table]:
            assembled, sources = _assemble_conflict_row(
                table, row, {col: val, col + "__side": side}
            )
            merged[table][key] = assembled
            meta[table][key] = {"row_kind": "conflict_resolved", "field_sources": sources}
        else:
            merged[table][key][col] = val
            meta[table][key]["field_sources"][col] = [side]
            meta[table][key]["row_kind"] = "conflict_resolved"
        return

    if gtype == "delete_vs_modify":
        table, key = _affected_single(group)[:2]
        row = rows_index[table][key]
        if action == "keep_row":
            side = decision.get("side")
            if side != group["modifying_side"]:
                raise DecisionError(f"{gid} 只能保留修改侧 {group['modifying_side']}")
            merged[table][key] = _with_pk(table, key, row["values"][side])
            meta[table][key] = {
                "row_kind": "conflict_resolved",
                "field_sources": {c: [side] for c in row["values"][side]},
            }
        elif action == "delete_row":
            merged[table].pop(key, None)
            meta[table].pop(key, None)
        else:
            raise DecisionError(f"{gid} 不支持动作 {action}")
        return

    if gtype == "insert_collision":
        table, key = _affected_single(group)[:2]
        row = rows_index[table][key]
        if action == "keep_row":
            side = decision.get("side")
            if side not in ("left", "right"):
                raise DecisionError(f"{gid} 的 keep_row 侧必须是 left/right")
            merged[table][key] = _with_pk(table, key, row["values"][side])
            meta[table][key] = {
                "row_kind": "conflict_resolved",
                "field_sources": {c: [side] for c in row["values"][side]},
                "inserted_by": row["inserted_by"],
            }
        elif action == "field_mix":
            fields = decision.get("fields")
            if not isinstance(fields, dict) or not fields:
                raise DecisionError(f"{gid} 的 field_mix 需要非空 fields 映射")
            assembled, sources = {}, {}
            pk = "customer_id" if table == "customers" else "order_id"
            assembled[pk] = key
            diff_cols = {c for c, v in group["per_column"].items()
                         if v["left"] != v["right"]}
            for col, vals in group["per_column"].items():
                if col in fields:
                    side = fields[col]
                    if side not in ("left", "right"):
                        raise DecisionError(f"{gid}.{col} 只能指定 left/right")
                    assembled[col] = vals[side]
                    sources[col] = [side]
                else:
                    if col in diff_cols:
                        raise DecisionError(f"{gid}.{col} 两侧值不同，必须在 field_mix 中指定")
                    assembled[col] = vals["left"]
                    sources[col] = ["left", "right"]
            extra = set(fields) - set(group["per_column"])
            if extra:
                raise DecisionError(f"{gid} 的 field_mix 含未知列 {sorted(extra)}")
            merged[table][key] = assembled
            meta[table][key] = {
                "row_kind": "conflict_resolved", "field_sources": sources,
                "inserted_by": row["inserted_by"],
            }
        else:
            raise DecisionError(f"{gid} 不支持动作 {action}")
        return

    if gtype == "fk_delete_reference":
        key = group["customer"]["key"]
        crow = rows_index["customers"][key]
        if action == "keep_customer":
            side = decision.get("side")
            if side not in ("left", "right") or crow["values"][side] is None:
                raise DecisionError(f"{gid} 的 keep_customer 侧不存在")
            merged["customers"][key] = _with_pk("customers", key, crow["values"][side])
            meta["customers"][key] = {
                "row_kind": "conflict_resolved",
                "field_sources": {c: [side] for c in crow["values"][side]},
            }
        elif action == "restore_customer":
            if decision.get("source") != "base" or crow["values"]["base"] is None:
                raise DecisionError(f"{gid} 只能从 base 显式恢复")
            merged["customers"][key] = _with_pk("customers", key, crow["values"]["base"])
            meta["customers"][key] = {
                "row_kind": "conflict_resolved",
                "field_sources": {c: ["base"] for c in crow["values"]["base"]},
            }
        elif action == "delete_customer_discard_orders":
            order_keys = decision.get("order_keys")
            expected = [a["key"] for a in group["affected"] if a["table"] == "orders"]
            if sorted(order_keys or []) != sorted(expected):
                raise DecisionError(
                    f"{gid} 丢弃订单必须显式且完整列出 {expected}，不得擅自增减"
                )
            merged["customers"].pop(key, None)
            meta["customers"].pop(key, None)
            for ok in order_keys:
                merged["orders"].pop(ok, None)
                meta["orders"].pop(ok, None)
        else:
            raise DecisionError(f"{gid} 不支持动作 {action}")
        return

    if gtype == "unique_conflict":
        if action != "unique_pick":
            raise DecisionError(f"{gid} 只支持 action=unique_pick")
        members = decision.get("members")
        expected = {(m["table"], m["key"], m["column"]) for m in group["members"]}
        got = {(m.get("table"), m.get("key"), m.get("column")) for m in (members or [])}
        if got != expected:
            raise DecisionError(f"{gid} 的成员覆盖必须恰好为 {sorted(expected)}")
        col = group["column"]
        for m in members:
            table, key, side = m["table"], m["key"], m.get("side")
            if side not in ("left", "right", "base"):
                raise DecisionError(f"{gid} 成员 {key} 的侧必须是 left/right/base")
            plan_row = rows_index[table][key]
            vals = next(f for f in plan_row["fields"] if f["name"] == col)
            val = vals[side]
            if val is MISSING:
                raise DecisionError(f"{gid} 成员 {key} 的 {side} 侧无值")
            if key not in merged[table]:
                raise DecisionError(f"{gid} 成员 {key} 当前不在存活行中，无法改列")
            merged[table][key][col] = val
            meta[table][key]["field_sources"][col] = [side]
            # 仅被覆盖的列标记为冲突裁决；行内其余列保留自动合并的原始意图。
            meta[table][key].setdefault("resolved_columns", []).append(col)
        return

    raise DecisionError(f"未知冲突组类型 {gtype}")


def _affected_single(group: dict[str, Any]):
    a = group["affected"][0]
    return a["table"], a["key"], a.get("column")


def _global_recheck(schema, plan, merged, meta, rows_index, applied) -> list[dict[str, Any]]:
    residual: list[dict[str, Any]] = []
    residual.extend(_recheck_unique(schema, merged, rows_index))
    residual.extend(_recheck_fk(plan, merged, rows_index, applied))
    residual.extend(_recheck_integrity(schema, merged))
    return residual


def _recheck_unique(schema, merged, rows_index) -> list[dict[str, Any]]:
    out = []
    for tname, ts in schema.tables.items():
        for ucols in ts.unique:
            if len(ucols) != 1:
                continue
            col = ucols[0]
            buckets: dict[Any, list[str]] = {}
            for key, row in merged[tname].items():
                val = row[col]
                if val is None:
                    continue
                buckets.setdefault(val, []).append(key)
            for val, keys in buckets.items():
                if len(keys) < 2:
                    continue
                members = []
                for key in sorted(keys):
                    plan_row = rows_index[tname][key]
                    fvals = next(f for f in plan_row["fields"] if f["name"] == col)
                    values = {"base": fvals["base"], "left": fvals["left"], "right": fvals["right"]}
                    members.append(
                        {
                            "table": tname, "key": key, "column": col, "values": values,
                            "options": _value_options(values),
                        }
                    )
                out.append(
                    {
                        "id": f"unique:{tname}:{col}:{val}",
                        "type": "unique_conflict",
                        "title": f"裁决后 {tname}.{col} 仍有重复值 {val!r}",
                        "constraint": f"UNIQUE({col})：最多一行可取 {val!r}",
                        "affected": [{"table": tname, "key": k, "column": col} for k in sorted(keys)],
                        "column": col,
                        "colliding_value": val,
                        "members": members,
                        "options": [
                            {"decision": {"action": "unique_pick", "members": "…逐成员选侧…"},
                             "description": "为每个成员行分别选择 left/right/base"}
                        ],
                        "dependencies": [],
                    }
                )
    return out


def _value_options(values: dict[str, Any]) -> list[dict[str, Any]]:
    opts, seen = [], []
    for side in ("left", "right", "base"):
        val = values[side]
        if val is MISSING or any(val == s for s in seen):
            continue
        seen.append(val)
        opts.append({"decision": {"action": "pick", "side": side}, "value": val})
    return opts


def _recheck_fk(plan, merged, rows_index, applied) -> list[dict[str, Any]]:
    out = []
    live_customers = set(merged["customers"])
    dangling: dict[str, list[str]] = {}
    for okey, orow in merged["orders"].items():
        ref = orow["customer_id"]
        if ref not in live_customers:
            dangling.setdefault(ref, []).append(okey)
    for ckey, okeys in sorted(dangling.items()):
        plan_customer = rows_index["customers"].get(ckey)
        options = []
        if plan_customer is not None:
            for side in ("left", "right", "base"):
                if plan_customer["values"][side] is not None:
                    if side == "base":
                        decision_shape = {"action": "restore_customer", "source": "base"}
                        desc = f"恢复 base 客户 {ckey}"
                    else:
                        decision_shape = {"action": "keep_customer", "side": side}
                        desc = f"保留 {side} 侧客户 {ckey}"
                    options.append({"decision": decision_shape, "description": desc})
        options.append(
            {
                "decision": {"action": "delete_customer_discard_orders", "order_keys": sorted(okeys)},
                "description": "显式丢弃这些孤立订单（不会自动删除）",
            }
        )
        out.append(
            {
                "id": f"fk:customers:{ckey}",
                "type": "fk_delete_reference",
                "title": f"裁决后订单 {sorted(okeys)} 引用的客户 {ckey} 不存在",
                "constraint": "外键 orders.customer_id -> customers.customer_id 被违反",
                "affected": [{"table": "customers", "key": ckey}]
                + [{"table": "orders", "key": k} for k in sorted(okeys)],
                "customer": {"key": ckey, "values": plan_customer["values"] if plan_customer else None},
                "orders": [{"table": "orders", "key": k} for k in sorted(okeys)],
                "options": options,
                "dependencies": [],
                "triggered_by": [a["group_id"] for a in applied],
            }
        )
    return out


def _recheck_integrity(schema, merged) -> list[dict[str, Any]]:
    out = []
    for tname, ts in schema.tables.items():
        for key, row in merged[tname].items():
            for col in ts.not_null:
                if row[col] is None:
                    out.append(_integrity_group(tname, key, col, "NOT NULL"))
            for chk in ts.checks:
                if chk["kind"] == "nonnegative":
                    col = chk["columns"][0]
                    if not isinstance(row[col], int) or row[col] < 0:
                        out.append(_integrity_group(tname, key, col, "CHECK(>=0)"))
    return out


def _integrity_group(table, key, col, constraint) -> dict[str, Any]:
    return {
        "id": f"integrity:{table}:{key}:{col}",
        "type": "integrity_violation",
        "title": f"{table}.{key} 的 {col} 违反 {constraint}",
        "constraint": constraint,
        "affected": [{"table": table, "key": key, "column": col}],
        "options": [],
        "dependencies": [],
    }


def _build_result(plan, merged, meta, rows_index, applied) -> dict[str, Any]:
    provenance = []
    for tname in ("customers", "orders"):
        pk_col = "customer_id" if tname == "customers" else "order_id"
        for key in sorted(merged[tname]):
            row = merged[tname][key]
            m = meta[tname][key]
            resolved_cols = set(m.get("resolved_columns", []))
            for col, val in row.items():
                if col == pk_col:
                    continue
                provenance.append(
                    {
                        "table": tname, "key": key, "column": col,
                        "value": val, "sources": m["field_sources"].get(col, []),
                        "change_kind": "conflict_resolved" if col in resolved_cols else m["row_kind"],
                    }
                )
            # 主键身份行：新增行的主键来自插入侧，既有行身份继承 base，主键不可修改。
            identity_sources = m.get("inserted_by") or ["base"]
            provenance.append(
                {
                    "table": tname, "key": key, "column": pk_col,
                    "value": row[pk_col], "sources": identity_sources,
                    "change_kind": "identity",
                }
            )
        # base 中存在但结果中消失的行：显式记录删除来源。
        for key, prow in sorted(rows_index[tname].items()):
            if prow["present"]["base"] and key not in merged[tname]:
                provenance.append(
                    {
                        "table": tname, "key": key, "column": "*",
                        "value": None, "sources": [],
                        "change_kind": "deleted",
                        "note": "base 行未进入合并结果，删除依据见 decisions_applied",
                    }
                )

    result = {
        "format_version": FORMAT_VERSION,
        "tool": "merge3",
        "plan_digest": plan["plan_digest"],
        "inputs": plan["inputs"],
        "merged": {t: [merged[t][k] for k in sorted(merged[t])] for t in merged},
        "provenance": provenance,
        "decisions_applied": applied,
    }
    result["resolved_digest"] = digest_of(
        {k: v for k, v in result.items() if k != "resolved_digest"}
    )
    return result
