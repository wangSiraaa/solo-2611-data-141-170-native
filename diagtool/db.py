"""只读 SQLite 访问与模式解析。

数据库以 URI mode=ro 打开，并额外设置 query_only，保证诊断过程绝不写入。
"""

import sqlite3
from dataclasses import dataclass

from .errors import PlanError, UnsupportedSyntaxError


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    notnull: bool
    pk: int


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo]
    pk_columns: list[str]

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def has_column(self, name: str) -> bool:
        low = name.lower()
        return any(c.name.lower() == low for c in self.columns)

    def canonical_column(self, name: str) -> str:
        low = name.lower()
        for c in self.columns:
            if c.name.lower() == low:
                return c.name
        raise PlanError(f"表 {self.name} 不存在列 {name}")


class Database:
    def __init__(self, path: str):
        self.path = path
        try:
            self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        except sqlite3.Error as exc:  # pragma: no cover - 环境相关
            raise PlanError(f"无法以只读方式打开数据库 {path}: {exc}") from exc
        self.conn.execute("PRAGMA query_only=ON")
        self._table_cache: dict[str, TableInfo] = {}
        self._row_cache: dict[str, list[dict]] = {}

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- 模式 ----
    def table(self, name: str) -> TableInfo:
        low = name.lower()
        if low in self._table_cache:
            return self._table_cache[low]
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND lower(name)=?",
            (low,),
        ).fetchone()
        if row is None:
            raise PlanError(f"查询引用的表 {name} 不存在于数据库中")
        actual = row[0]
        cols = []
        for cid, cname, ctype, notnull, _dflt, pk in self.conn.execute(
            f"PRAGMA table_info({quote_ident(actual)})"
        ):
            cols.append(ColumnInfo(name=cname, type=ctype or "", notnull=bool(notnull),
                                   pk=int(pk)))
        pk_cols = [c.name for c in sorted(cols, key=lambda c: c.pk) if c.pk > 0]
        info = TableInfo(name=actual, columns=cols, pk_columns=pk_cols)
        self._table_cache[low] = info
        return info

    # ---- 数据 ----
    def rows(self, table_name: str) -> list[dict]:
        info = self.table(table_name)
        key = info.name.lower()
        if key in self._row_cache:
            return self._row_cache[key]
        order_cols = info.pk_columns or info.column_names
        order_sql = ", ".join(quote_ident(c) for c in order_cols)
        cur = self.conn.execute(
            f"SELECT * FROM {quote_ident(info.name)} ORDER BY {order_sql}"
        )
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, r)) for r in cur.fetchall()]
        self._row_cache[key] = rows
        return rows

    def execute(self, sql: str, bindings: dict) -> list[tuple]:
        try:
            cur = self.conn.execute(sql, bindings)
        except sqlite3.Error as exc:
            raise UnsupportedSyntaxError(f"SQLite 执行失败: {exc}\nSQL:\n{sql}") from exc
        return cur.fetchall()

    # ---- 列解析 ----
    def resolve(
        self,
        column,
        alias_map: dict[str, TableInfo],
        context: str,
    ) -> tuple[str, TableInfo, str]:
        """把 ColumnRef 解析为 (alias, TableInfo, 规范列名)。"""
        if column.table_alias is not None:
            alias = column.table_alias
            if alias not in alias_map:
                raise UnsupportedSyntaxError(
                    f"{context}: 列 {alias}.{column.name} 引用了未知别名 {alias}"
                )
            info = alias_map[alias]
            return alias, info, info.canonical_column(column.name)
        candidates = [(a, info) for a, info in alias_map.items()
                      if info.has_column(column.name)]
        if not candidates:
            raise UnsupportedSyntaxError(
                f"{context}: 找不到列 {column.name} 所属的表"
            )
        if len(candidates) > 1:
            raise UnsupportedSyntaxError(
                f"{context}: 列 {column.name} 在多个表中存在，必须加表别名限定"
            )
        alias, info = candidates[0]
        return alias, info, info.canonical_column(column.name)

    @staticmethod
    def row_identity(table: TableInfo, values: dict) -> dict:
        if table.pk_columns:
            pk_values = [{"column": c, "value": values.get(c)} for c in table.pk_columns]
            kind = "primary_key"
        else:
            pk_values = []
            kind = "no_primary_key"
        return {
            "table": table.name,
            "identity_kind": kind,
            "primary_key": pk_values,
            "values": {k: values.get(k) for k in table.column_names},
        }
