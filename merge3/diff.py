"""三方差异分类。

以 base 为基线识别 left / right 的真实变更。关键区分：
  * None（SQL NULL）、""（空字符串）是真实值，用 MISSING 哨兵表示整行不存在；
  * 两边都没改 -> unchanged；
  * 各改不同字段 -> independent，可自动合并；
  * 同字段改成不同值 -> field_conflict，不能把“两边不同”一律叫冲突，
    只有相对 base 的同字段异值才算；
  * 同主键新增且内容不同 -> insert_collision；完全相同则等价；
  * 一边删行另一边改行 -> delete_vs_modify，不静默决定。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .canonical import MISSING
from .schema import Schema, TableSchema


class FieldStatus(str, Enum):
    UNCHANGED = "unchanged"                    # 两侧均未动
    LEFT_ONLY = "left_only"                    # 仅 left 改，自动取 left
    RIGHT_ONLY = "right_only"                  # 仅 right 改，自动取 right
    BOTH_SAME = "both_same"                    # 两侧改成同一值，自动取该值
    FIELD_CONFLICT = "field_conflict"          # 同字段改成不同值
    LEFT_SET = "left_set"                      # 新增行：仅 left 提供该值
    RIGHT_SET = "right_set"                    # 新增行：仅 right 提供该值
    BOTH_SET_SAME = "both_set_same"            # 新增行：两侧同值
    BOTH_SET_DIFF = "both_set_diff"            # 新增行：两侧异值（碰撞）


class RowKind(str, Enum):
    UNCHANGED = "unchanged"
    AUTO_MERGED = "auto_merged"                # 字段级自动合并后的存活行
    FIELD_CONFLICT = "field_conflict"          # 含至少一个字段冲突
    INSERT_AUTO = "insert_auto"                # 新增且无碰撞
    INSERT_COLLISION = "insert_collision"      # 同主键新增、内容不同
    DELETE_VS_MODIFY = "delete_vs_modify"      # 一删一改
    DELETED = "deleted"                        # 两侧都删（自动删除）
    DELETED_ONE_SIDE = "deleted_one_side"      # 仅一侧删除、另一侧未动 -> 待决组


@dataclass
class FieldDiff:
    name: str
    base: Any
    left: Any
    right: Any
    status: FieldStatus
    auto_value: Any = None                     # 可自动合并时的取值，否则 None
    sources: list[str] = field(default_factory=list)  # 取值来源 side 名


@dataclass
class RowDiff:
    table: str
    key: str
    kind: RowKind
    fields: dict[str, FieldDiff]
    present: dict[str, bool]                   # {"base":..,"left":..,"right":..}
    deleting_side: str | None = None           # delete_vs_modify 时的删除方
    modifying_side: str | None = None
    auto_row: dict[str, Any] | None = None     # 可直接落库的整行（含主键）
    # 新增行的来源；普通行为 None
    inserted_by: list[str] = field(default_factory=list)


def _read_table(db_path: str, ts: TableSchema) -> dict[str, dict[str, Any]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cols = ", ".join(ts.columns)
        rows = con.execute(f"SELECT {cols} FROM {ts.name}").fetchall()
    finally:
        con.close()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        rec = dict(zip(ts.columns, row))
        out[str(rec[ts.pk_col])] = rec
    return out


def _side_value(side_row: dict[str, Any] | None, col: str) -> Any:
    if side_row is None:
        return MISSING
    return side_row[col]


def _classify_existing(
    table: str, key: str, base: dict[str, Any], left: dict[str, Any] | None,
    right: dict[str, Any] | None, ts: TableSchema,
) -> RowDiff:
    """base 中存在该主键时的分类。"""
    fields: dict[str, FieldDiff] = {}
    has_conflict = False
    auto: dict[str, Any] = {ts.pk_col: key}
    present = {"base": True, "left": left is not None, "right": right is not None}

    if left is None and right is None:
        # 双删行同样保留 base 字段视图，供外键引用分析使用。
        for col in ts.value_columns():
            fields[col] = FieldDiff(col, base[col], MISSING, MISSING, FieldStatus.UNCHANGED)
        return RowDiff(table, key, RowKind.DELETED, fields, present, auto_row=None)

    if left is None or right is None:
        deleting = "left" if left is None else "right"
        modifying = "right" if left is None else "left"
        keep = right if left is None else left
        for col in ts.value_columns():
            bv, mv = base[col], keep[col]
            if bv == mv:
                status = FieldStatus.UNCHANGED
            else:
                status = FieldStatus.LEFT_ONLY if modifying == "left" else FieldStatus.RIGHT_ONLY
            fd = FieldDiff(col, bv, MISSING, mv, status) if deleting == "left" else FieldDiff(
                col, bv, mv, MISSING, status
            )
            fields[col] = fd
        return RowDiff(
            table, key, RowKind.DELETE_VS_MODIFY, fields, present,
            deleting_side=deleting, modifying_side=modifying, auto_row=None,
        )

    # 三侧都存在：逐字段三方比较。
    any_change = False
    for col in ts.value_columns():
        bv, lv, rv = base[col], left[col], right[col]
        l_changed, r_changed = lv != bv, rv != bv
        if not l_changed and not r_changed:
            status, auto_val, sources = FieldStatus.UNCHANGED, bv, []
        elif l_changed and not r_changed:
            status, auto_val, sources = FieldStatus.LEFT_ONLY, lv, ["left"]
            any_change = True
        elif r_changed and not l_changed:
            status, auto_val, sources = FieldStatus.RIGHT_ONLY, rv, ["right"]
            any_change = True
        elif lv == rv:
            status, auto_val, sources = FieldStatus.BOTH_SAME, lv, ["left", "right"]
            any_change = True
        else:
            status, auto_val, sources = FieldStatus.FIELD_CONFLICT, None, []
            has_conflict = True
            any_change = True
        fields[col] = FieldDiff(col, bv, lv, rv, status, auto_val, sources)
        # 注意：不能用 auto_val is not None 判定——NULL 是合法合并值，整列必须保留。
        if status == FieldStatus.UNCHANGED:
            auto[col] = bv
        elif status != FieldStatus.FIELD_CONFLICT:
            auto[col] = auto_val

    if has_conflict:
        kind = RowKind.FIELD_CONFLICT
        auto_row = None
    elif any_change:
        kind = RowKind.AUTO_MERGED
        auto_row = auto
    else:
        kind = RowKind.UNCHANGED
        auto_row = auto
    return RowDiff(table, key, kind, fields, present, auto_row=auto_row)


def _classify_inserted(
    table: str, key: str, left: dict[str, Any] | None, right: dict[str, Any] | None,
    ts: TableSchema,
) -> RowDiff:
    """base 中不存在该主键：纯新增场景。"""
    fields: dict[str, FieldDiff] = {}
    present = {"base": False, "left": left is not None, "right": right is not None}
    inserted_by = [s for s, row in (("left", left), ("right", right)) if row is not None]

    if left is None or right is None:
        keep = left if left is not None else right
        side = "left" if left is not None else "right"
        auto = {ts.pk_col: key}
        for col in ts.value_columns():
            status = FieldStatus.LEFT_SET if side == "left" else FieldStatus.RIGHT_SET
            fields[col] = FieldDiff(
                col, MISSING, _side_value(left, col), _side_value(right, col),
                status, keep[col], [side],
            )
            auto[col] = keep[col]
        return RowDiff(table, key, RowKind.INSERT_AUTO, fields, present,
                       auto_row=auto, inserted_by=inserted_by)

    # 两侧都新增了同主键：逐字段比较，任何一列不同即为碰撞。
    collision = any(left[c] != right[c] for c in ts.value_columns())
    auto = {ts.pk_col: key}
    for col in ts.value_columns():
        lv, rv = left[col], right[col]
        if lv == rv:
            status = FieldStatus.BOTH_SET_SAME
            fields[col] = FieldDiff(col, MISSING, lv, rv, status, lv, ["left", "right"])
            auto[col] = lv
        else:
            status = FieldStatus.BOTH_SET_DIFF
            fields[col] = FieldDiff(col, MISSING, lv, rv, status)
    if collision:
        return RowDiff(table, key, RowKind.INSERT_COLLISION, fields, present,
                       auto_row=None, inserted_by=inserted_by)
    return RowDiff(table, key, RowKind.INSERT_AUTO, fields, present,
                   auto_row=auto, inserted_by=inserted_by)


def diff_database(
    base_path: str, left_path: str, right_path: str, schema: Schema,
) -> dict[str, dict[str, RowDiff]]:
    result: dict[str, dict[str, RowDiff]] = {}
    for tname, ts in schema.tables.items():
        base_rows = _read_table(base_path, ts)
        left_rows = _read_table(left_path, ts)
        right_rows = _read_table(right_path, ts)
        all_keys = sorted(set(base_rows) | set(left_rows) | set(right_rows))
        diffs: dict[str, RowDiff] = {}
        for key in all_keys:
            if key in base_rows:
                diffs[key] = _classify_existing(
                    tname, key, base_rows[key], left_rows.get(key), right_rows.get(key), ts
                )
            else:
                diffs[key] = _classify_inserted(
                    tname, key, left_rows.get(key), right_rows.get(key), ts
                )
        result[tname] = diffs
    return result
