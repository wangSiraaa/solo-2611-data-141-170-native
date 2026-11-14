"""plan 入口：把三方差异组织成可解释的冲突组。

冲突组类型：
  field_conflict        同字段改成不同值（单字段裁决）
  unique_conflict       跨行唯一约束碰撞（多个成员必须一起裁决）
  delete_vs_modify      一删一改，禁止静默决定
  insert_collision      同主键新增内容不同，可取整侧或按字段混合
  fk_delete_reference   删客户与引用该客户的订单并存（外键组，内含删除裁决）

组内保留 base/left/right 三侧值、受影响记录、约束说明与全部合法裁决形状。
"""

from __future__ import annotations

from typing import Any

from .canonical import MISSING, digest_of, file_sha256, write_json
from .diff import FieldStatus, RowDiff, RowKind, diff_database
from .schema import Schema, load_schema

FORMAT_VERSION = 1


def _field_values(rd: RowDiff, col: str) -> dict[str, Any]:
    fd = rd.fields[col]
    return {"base": fd.base, "left": fd.left, "right": fd.right}


def _side_options(values: dict[str, Any], allow_base: bool = True) -> list[dict[str, Any]]:
    """生成字段级合法选项：只列出取值互不相同的侧。"""
    opts: list[dict[str, Any]] = []
    seen: list[Any] = []
    for side in ("left", "right", "base"):
        if side == "base" and not allow_base:
            continue
        val = values[side]
        if val is MISSING:
            continue
        if any(val == seen_val for seen_val in seen):
            continue
        seen.append(val)
        opts.append(
            {
                "decision": {"action": "pick", "side": side},
                "description": f"取 {side} 的值",
                "value": val,
            }
        )
    return opts


def _field_conflict_group(rd: RowDiff, col: str) -> dict[str, Any]:
    values = _field_values(rd, col)
    return {
        "id": f"field:{rd.table}:{rd.key}:{col}",
        "type": "field_conflict",
        "title": f"{rd.table} 主键 {rd.key} 的字段 {col} 被两侧改成不同值",
        "constraint": "同一字段相对 base 的两个修改互不一致，必须逐字段选择一侧",
        "affected": [{"table": rd.table, "key": rd.key, "column": col}],
        "values": values,
        "options": _side_options(values),
        "dependencies": [],
    }


def _delete_vs_modify_group(rd: RowDiff) -> dict[str, Any]:
    keep_side = rd.modifying_side
    options = [
        {
            "decision": {"action": "keep_row", "side": keep_side},
            "description": f"保留 {keep_side} 侧版本（拒绝另一侧的删除）",
        },
        {
            "decision": {"action": "delete_row"},
            "description": f"执行 {rd.deleting_side} 侧的删除（放弃 {keep_side} 侧版本）",
        },
    ]
    return {
        "id": f"dvm:{rd.table}:{rd.key}",
        "type": "delete_vs_modify",
        "title": f"{rd.table} 主键 {rd.key}：{rd.deleting_side} 侧删除，{keep_side} 侧保留/修改",
        "constraint": "删除与修改互斥，不允许按时间戳静默取胜",
        "affected": [{"table": rd.table, "key": rd.key}],
        "deleting_side": rd.deleting_side,
        "modifying_side": rd.modifying_side,
        "options": options,
        "dependencies": [],
    }


def _insert_collision_group(rd: RowDiff) -> dict[str, Any]:
    columns = list(rd.fields)
    per_column = {col: _field_values(rd, col) for col in columns}
    mixed_desc = {col: "left | right" for col in columns}
    options = [
        {
            "decision": {"action": "keep_row", "side": "left"},
            "description": "整行采用 left 新增版本",
        },
        {
            "decision": {"action": "keep_row", "side": "right"},
            "description": "整行采用 right 新增版本",
        },
        {
            "decision": {"action": "field_mix", "fields": mixed_desc},
            "description": "按字段分别指定 left/right（仅值不同的字段需要指定）",
        },
    ]
    return {
        "id": f"insert:{rd.table}:{rd.key}",
        "type": "insert_collision",
        "title": f"{rd.table} 主键 {rd.key}：两侧各自新增且内容不同（新增碰撞）",
        "constraint": f"主键 {rd.key} 身份相同，不能同时插入两行；主键本身不可修改",
        "affected": [{"table": rd.table, "key": rd.key}],
        "inserted_by": rd.inserted_by,
        "per_column": per_column,
        "options": options,
        "dependencies": [],
    }


def _row_payload(rd: RowDiff) -> dict[str, Any]:
    return {
        "key": rd.key,
        "kind": rd.kind.value,
        "present": rd.present,
        "values": {
            "base": _full_side(rd, "base"),
            "left": _full_side(rd, "left"),
            "right": _full_side(rd, "right"),
        },
        "fields": [
            {
                "name": fd.name,
                "base": fd.base,
                "left": fd.left,
                "right": fd.right,
                "status": fd.status.value,
                "auto_value": fd.auto_value,
                "sources": fd.sources,
            }
            for fd in rd.fields.values()
        ],
        "auto_row": rd.auto_row,
        "deleting_side": rd.deleting_side,
        "modifying_side": rd.modifying_side,
        "inserted_by": rd.inserted_by,
    }


def _full_side(rd: RowDiff, side: str) -> dict[str, Any] | None:
    if not rd.present[side]:
        return None
    pk_col = "customer_id" if rd.table == "customers" else "order_id"
    payload: dict[str, Any] = {pk_col: rd.key}
    for col, fd in rd.fields.items():
        payload[col] = getattr(fd, side)
    return payload


def build_plan(
    base_path: str, left_path: str, right_path: str, schema_path: str
) -> dict[str, Any]:
    schema = load_schema(base_path)
    # 三个库的 schema 必须一致。
    load_schema(left_path)
    load_schema(right_path)
    diffs = diff_database(base_path, left_path, right_path, schema)

    conflicts: dict[str, dict[str, Any]] = {}
    tables_out: dict[str, Any] = {}
    auto_merge: dict[str, list[dict[str, Any]]] = {}

    for tname, rd_map in diffs.items():
        rows_out = []
        auto_rows = []
        for key, rd in sorted(rd_map.items()):
            rows_out.append(_row_payload(rd))
            if rd.auto_row is not None:
                auto_rows.append(rd.auto_row)
            if rd.kind is RowKind.FIELD_CONFLICT:
                for col, fd in rd.fields.items():
                    if fd.status is FieldStatus.FIELD_CONFLICT:
                        g = _field_conflict_group(rd, col)
                        conflicts[g["id"]] = g
            elif rd.kind is RowKind.DELETE_VS_MODIFY:
                g = _delete_vs_modify_group(rd)
                conflicts[g["id"]] = g
            elif rd.kind is RowKind.INSERT_COLLISION:
                g = _insert_collision_group(rd)
                conflicts[g["id"]] = g
        tables_out[tname] = {"columns": schema.tables[tname].columns, "rows": rows_out}
        auto_merge[tname] = auto_rows

    _build_unique_groups(schema, diffs, conflicts)
    _build_fk_groups(schema, diffs, conflicts)

    conflict_list = sorted(conflicts.values(), key=lambda g: g["id"])
    plan = {
        "format_version": FORMAT_VERSION,
        "tool": "merge3",
        "inputs": {
            "base": {"path": base_path, "sha256": file_sha256(base_path)},
            "left": {"path": left_path, "sha256": file_sha256(left_path)},
            "right": {"path": right_path, "sha256": file_sha256(right_path)},
            "schema_sql": {"path": schema_path, "sha256": file_sha256(schema_path)},
        },
        "tables": tables_out,
        "auto_merge": auto_merge,
        "conflicts": conflict_list,
        "stats": _stats(diffs, conflict_list),
    }
    plan["plan_digest"] = digest_of({k: v for k, v in plan.items() if k != "plan_digest"})
    return plan


def _stats(diffs, conflict_list) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for rd_map in diffs.values():
        for rd in rd_map.values():
            counts[rd.kind.value] = counts.get(rd.kind.value, 0) + 1
    return {
        "row_kinds": dict(sorted(counts.items())),
        "conflict_groups": len(conflict_list),
        "conflict_types": dict(
            sorted({t: sum(1 for g in conflict_list if g["type"] == t)
                    for t in {g["type"] for g in conflict_list}}.items())
        ),
    }


def _build_unique_groups(
    schema: Schema, diffs: dict[str, dict[str, RowDiff]],
    conflicts: dict[str, dict[str, Any]],
) -> None:
    """检测自动合并结果内部的跨行唯一约束碰撞（如两个 code 都变成 COMMON）。"""
    for tname, ts in schema.tables.items():
        value_cols = set(ts.value_columns())
        for ucols in ts.unique:
            if len(ucols) != 1 or ucols[0] not in value_cols:
                continue
            col = ucols[0]
            buckets: dict[Any, list[str]] = {}
            for key, rd in diffs[tname].items():
                if rd.auto_row is None:
                    continue
                val = rd.auto_row[col]
                if val is None:  # SQL 唯一约束中 NULL 互不相等
                    continue
                buckets.setdefault(val, []).append(key)
            for val, keys in buckets.items():
                if len(keys) < 2:
                    continue
                members = []
                for key in keys:
                    rd = diffs[tname][key]
                    members.append(
                        {
                            "table": tname,
                            "key": key,
                            "column": col,
                            "values": _field_values(rd, col),
                            "options": _side_options(_field_values(rd, col)),
                        }
                    )
                gid = f"unique:{tname}:{col}:{val}"
                conflicts[gid] = {
                    "id": gid,
                    "type": "unique_conflict",
                    "title": f"{tname} 的唯一列 {col} 出现重复值 {val!r}",
                    "constraint": f"{tname}.{col} 上有 UNIQUE 约束，合并后最多一行可取 {val!r}",
                    "affected": [{"table": tname, "key": k, "column": col} for k in keys],
                    "column": col,
                    "colliding_value": val,
                    "members": members,
                    "options": [
                        {
                            "decision": {
                                "action": "unique_pick",
                                "members": [
                                    {"table": tname, "key": k, "column": col, "side": "left|right|base"}
                                    for k in keys
                                ],
                            },
                            "description": "为每个成员行分别选择取值侧；裁决后再次检查唯一性",
                        }
                    ],
                    "dependencies": [],
                }


def _build_fk_groups(
    schema: Schema, diffs: dict[str, dict[str, RowDiff]],
    conflicts: dict[str, dict[str, Any]],
) -> None:
    """外键组：自动存活的订单引用了被一侧删除的客户。

    * 客户 DVM + 存活订单 -> 合并成一个 fk_delete_reference 组（取代单独 DVM 组）；
    * 客户两侧皆删 + 存活订单 -> 同组，选项为显式恢复 base 客户或显式丢弃订单；
    * 订单本身仍在冲突中（如 DVM）-> 只登记依赖，矛盾裁决在 resolve 阶段派生新组。
    """
    cust_diffs = diffs["customers"]
    order_diffs = diffs["orders"]

    # 客户主键 -> 引用它的订单差异分类
    referencing: dict[str, list[RowDiff]] = {}
    for rd in order_diffs.values():
        ref = _order_ref_customer(rd)
        if ref is not None:
            referencing.setdefault(ref, []).append(rd)

    for ckey, crd in sorted(cust_diffs.items()):
        refs = referencing.get(ckey, [])
        auto_refs = [r for r in refs if r.auto_row is not None]
        if not auto_refs:
            if refs:
                _link_dependency(crd, refs, conflicts)
            continue
        if crd.kind not in (RowKind.DELETE_VS_MODIFY, RowKind.DELETED):
            continue

        gid = f"fk:customers:{ckey}"
        deleted_both = crd.kind is RowKind.DELETED
        order_members = []
        for ord_rd in sorted(auto_refs, key=lambda r: r.key):
            order_members.append(
                {
                    "table": "orders",
                    "key": ord_rd.key,
                    "present": ord_rd.present,
                    "values": {
                        "base": _full_side(ord_rd, "base"),
                        "left": _full_side(ord_rd, "left"),
                        "right": _full_side(ord_rd, "right"),
                    },
                    "inserted_by": ord_rd.inserted_by,
                }
            )

        if deleted_both:
            options = [
                {
                    "decision": {"action": "restore_customer", "source": "base"},
                    "description": f"显式恢复 base 中的客户 {ckey}，引用订单保留",
                },
                {
                    "decision": {
                        "action": "delete_customer_discard_orders",
                        "order_keys": [r.key for r in auto_refs],
                    },
                    "description": "维持删除，并由用户显式丢弃列出的引用订单（不会自动删订单）",
                },
            ]
            title = f"客户 {ckey} 两侧均删除，但仍有订单引用它"
        else:
            keep_side = crd.modifying_side
            options = [
                {
                    "decision": {"action": "keep_customer", "side": keep_side},
                    "description": f"保留 {keep_side} 侧客户 {ckey}，引用订单自然成立",
                },
                {
                    "decision": {
                        "action": "delete_customer_discard_orders",
                        "order_keys": [r.key for r in auto_refs],
                    },
                    "description": "执行删除，并由用户显式丢弃列出的引用订单（不会自动删订单）",
                },
            ]
            title = (
                f"客户 {ckey}：{crd.deleting_side} 侧删除，而另一侧的订单仍引用它"
            )

        conflicts.pop(f"dvm:customers:{ckey}", None)
        conflicts[gid] = {
            "id": gid,
            "type": "fk_delete_reference",
            "title": title,
            "constraint": "orders.customer_id 外键引用 customers.customer_id，客户不存在则订单孤立",
            "affected": [{"table": "customers", "key": ckey}]
            + [{"table": "orders", "key": r.key} for r in auto_refs],
            "customer": {
                "key": ckey,
                "kind": crd.kind.value,
                "deleting_side": crd.deleting_side,
                "modifying_side": crd.modifying_side,
                "values": {
                    "base": _full_side(crd, "base"),
                    "left": _full_side(crd, "left"),
                    "right": _full_side(crd, "right"),
                },
            },
            "orders": order_members,
            "options": options,
            "dependencies": [],
        }


def _order_ref_customer(rd: RowDiff) -> str | None:
    """取订单引用的客户主键；两侧都不存在时返回 None。"""
    for side in ("left", "right", "base"):
        payload = _full_side(rd, side)
        if payload is not None:
            return payload["customer_id"]
    return None


def _link_dependency(
    crd: RowDiff, refs: list[RowDiff], conflicts: dict[str, dict[str, Any]]
) -> None:
    """订单自身仍在冲突中时登记双向依赖，resolve 阶段负责派生级联组。"""
    cgroup = conflicts.get(f"dvm:customers:{crd.key}") or conflicts.get(
        f"fk:customers:{crd.key}"
    )
    for ord_rd in refs:
        ogroup = conflicts.get(f"dvm:orders:{ord_rd.key}") or conflicts.get(
            f"insert:orders:{ord_rd.key}"
        )
        if cgroup is not None and ogroup is not None:
            note = (
                f"若 orders.{ord_rd.key} 最终保留，则 customers.{crd.key} 必须保留"
                "（外键 orders.customer_id -> customers.customer_id）"
            )
            cgroup["dependencies"].append({"group": ogroup["id"], "note": note})
            ogroup["dependencies"].append(
                {
                    "group": cgroup["id"],
                    "note": f"保留本订单要求客户 {crd.key} 存活，否则将派生新的外键裁决组",
                }
            )


def write_plan(plan: dict[str, Any], out_path: str) -> None:
    write_json(out_path, plan)
