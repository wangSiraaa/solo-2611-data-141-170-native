"""从 SQLite 内省固定两表的 schema 与约束。

不做 schema 迁移：只认识 fixtures/schema.sql 定义的 customers / orders，
表结构（列、主键、外键、唯一、非负 CHECK）必须与预期一致，否则拒绝执行。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from .errors import Merge3Error

TABLES = ("customers", "orders")

# 固定 schema 的基线定义，用于拒绝悄悄漂移的输入库。
EXPECTED = {
    "customers": {
        "columns": ["customer_id", "code", "name", "email"],
        "pk": ["customer_id"],
        "not_null": ["customer_id", "code", "name"],
        "unique": [["customer_id"], ["code"]],
        "fks": [],
        "checks": [],
    },
    "orders": {
        "columns": ["order_id", "customer_id", "amount_cents", "note"],
        "pk": ["order_id"],
        "not_null": ["order_id", "customer_id", "amount_cents"],
        "unique": [["order_id"]],
        "fks": [
            {"columns": ["customer_id"], "ref_table": "customers", "ref_columns": ["customer_id"]}
        ],
        "checks": [{"columns": ["amount_cents"], "kind": "nonnegative"}],
    },
}


@dataclass
class FKSpec:
    columns: list[str]
    ref_table: str
    ref_columns: list[str]


@dataclass
class TableSchema:
    name: str
    columns: list[str]
    pk: list[str]
    not_null: list[str]
    unique: list[list[str]]
    fks: list[FKSpec] = field(default_factory=list)
    # 每项形如 {"columns": [...], "kind": "nonnegative"}
    checks: list[dict] = field(default_factory=list)

    @property
    def pk_col(self) -> str:
        assert len(self.pk) == 1
        return self.pk[0]

    def value_columns(self) -> list[str]:
        return [c for c in self.columns if c not in self.pk]


@dataclass
class Schema:
    tables: dict[str, TableSchema]

    def check_sql(self) -> str:
        return "PRAGMA foreign_keys=ON;\n" + "\n".join(
            [
                "CREATE TABLE customers(customer_id TEXT PRIMARY KEY, code TEXT NOT NULL UNIQUE, "
                "name TEXT NOT NULL, email TEXT);",
                "CREATE TABLE orders(order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL "
                "REFERENCES customers(customer_id), amount_cents INTEGER NOT NULL "
                "CHECK(amount_cents>=0), note TEXT);",
            ]
        )


def _introspect(con: sqlite3.Connection, table: str) -> TableSchema:
    info = con.execute(f"PRAGMA table_info({table})").fetchall()
    if not info:
        raise Merge3Error(f"输入库缺少固定表 {table}")
    columns = [r[1] for r in info]
    pk = [r[1] for r in info if r[5] > 0]
    pk.sort(key=lambda c: next(r[5] for r in info if r[1] == c))
    # 单列 INTEGER/TEXT PRIMARY KEY 在 table_info 中 notnull 标志为 0，但主键隐式 NOT NULL。
    not_null = sorted(set([r[1] for r in info if r[3] == 1]) | set(pk))

    unique: list[list[str]] = []
    for il in con.execute(f"PRAGMA index_list({table})").fetchall():
        # il: (seq, name, unique, origin, partial)
        if not il[2]:
            continue
        idx_cols = [r[2] for r in con.execute(f"PRAGMA index_info({il[1]})").fetchall()]
        idx_cols.sort()
        if idx_cols not in unique:
            unique.append(idx_cols)
    unique.sort()

    fks: list[FKSpec] = []
    for fk in con.execute(f"PRAGMA foreign_key_list({table})").fetchall():
        # fk: (id, seq, table, from, to, on_update, on_delete, match)
        existing = next((f for f in fks if f.ref_table == fk[2]), None)
        if existing is None:
            existing = FKSpec(columns=[], ref_table=fk[2], ref_columns=[])
            fks.append(existing)
        existing.columns.append(fk[3])
        existing.ref_columns.append(fk[4])
    fks.sort(key=lambda f: f.ref_table)

    checks: list[dict] = []
    ddl = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()[0]
    # 固定工具只需要识别 amount_cents>=0 这一类 CHECK。
    tail = ddl[ddl.upper().find("CHECK") :].upper()
    if "AMOUNT_CENTS" in tail and ">=" in tail and "0" in tail:
        checks.append({"columns": ["amount_cents"], "kind": "nonnegative"})

    return TableSchema(table, columns, pk, not_null, unique, fks, checks)


def load_schema(db_path: str) -> Schema:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = {t: _introspect(con, t) for t in TABLES}
    finally:
        con.close()

    for name, ts in tables.items():
        exp = EXPECTED[name]
        actual = {
            "columns": ts.columns,
            "pk": ts.pk,
            "not_null": sorted(ts.not_null),
            "unique": sorted(ts.unique),
            "fks": [
                {"columns": f.columns, "ref_table": f.ref_table, "ref_columns": f.ref_columns}
                for f in ts.fks
            ],
            "checks": ts.checks,
        }
        expected = {
            "columns": exp["columns"],
            "pk": exp["pk"],
            "not_null": sorted(exp["not_null"]),
            "unique": sorted(exp["unique"]),
            "fks": exp["fks"],
            "checks": exp["checks"],
        }
        if actual != expected:
            raise Merge3Error(
                f"{db_path} 的 {name} 表结构与固定 schema 不一致：\n"
                f"实际 {actual}\n预期 {expected}"
            )
    return Schema(tables)
