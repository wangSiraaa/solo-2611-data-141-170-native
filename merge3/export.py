"""export 入口：写出完整 SQLite 与逐变更来源，并重新打开校验。

输出库包含固定两表（与 schema.sql 完全一致）外加 merge_provenance 审计表。
写入后关闭再以 foreign_keys=ON 重新打开，执行：
  * PRAGMA foreign_key_check
  * 全部 UNIQUE 列的重复值检查
  * NOT NULL 与 amount_cents 非负整数检查
任何失败都删除输出文件并报错，绝不留下半套结果。输入库全程只读打开。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .canonical import read_json
from .errors import Merge3Error
from .resolve import verify_inputs
from .schema import TABLES, load_schema

PROVENANCE_DDL = (
    "CREATE TABLE merge_provenance("
    "table_name TEXT NOT NULL, row_key TEXT NOT NULL, column_name TEXT NOT NULL, "
    "change_kind TEXT NOT NULL, sources TEXT NOT NULL, value_json TEXT, note TEXT)"
)


def export(resolved_path: str, out_db_path: str) -> dict[str, Any]:
    resolved = read_json(resolved_path)
    verify_inputs(resolved)
    schema = load_schema(resolved["inputs"]["base"]["path"])

    out_path = Path(out_db_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    con = sqlite3.connect(out_path)
    try:
        con.execute("PRAGMA foreign_keys=ON")
        for stmt in (
            "CREATE TABLE customers(customer_id TEXT PRIMARY KEY, code TEXT NOT NULL UNIQUE, "
            "name TEXT NOT NULL, email TEXT)",
            "CREATE TABLE orders(order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL "
            "REFERENCES customers(customer_id), amount_cents INTEGER NOT NULL "
            "CHECK(amount_cents>=0), note TEXT)",
            PROVENANCE_DDL,
        ):
            con.execute(stmt)

        # 父表先行，保证外键在写入过程中也成立。
        for tname in TABLES:
            ts = schema.tables[tname]
            placeholders = ", ".join("?" for _ in ts.columns)
            cols = ", ".join(ts.columns)
            con.executemany(
                f"INSERT INTO {tname}({cols}) VALUES ({placeholders})",
                [[row[c] for c in ts.columns] for row in resolved["merged"][tname]],
            )

        con.executemany(
            "INSERT INTO merge_provenance(table_name,row_key,column_name,change_kind,"
            "sources,value_json,note) VALUES (?,?,?,?,?,?,?)",
            [
                (
                    p["table"], p["key"], p["column"], p["change_kind"],
                    ",".join(p.get("sources") or []),
                    _json_scalar(p.get("value")), p.get("note"),
                )
                for p in resolved["provenance"]
            ],
        )
        con.commit()
    except sqlite3.Error as exc:
        con.close()
        out_path.unlink(missing_ok=True)
        raise Merge3Error(f"写入输出库时被 SQLite 约束拒绝：{exc}") from exc
    except Exception:
        con.close()
        out_path.unlink(missing_ok=True)
        raise
    finally:
        try:
            con.close()
        except Exception:
            pass

    verify_output(out_db_path, resolved, schema)
    return {
        "output_db": str(out_path),
        "resolved_digest": resolved["resolved_digest"],
        "rows": {t: len(resolved["merged"][t]) for t in TABLES},
    }


def _json_scalar(value) -> str | None:
    if value is None:
        return None
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def verify_output(path: str, resolved: dict[str, Any], schema) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute("PRAGMA foreign_keys=ON")
        problems: list[str] = []

        fk_violations = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk_violations:
            problems.append(f"foreign_key_check 失败：{fk_violations}")

        for tname, ts in schema.tables.items():
            for ucols in ts.unique:
                cols = ", ".join(ucols)
                dupes = con.execute(
                    f"SELECT {cols}, COUNT(*) c FROM {tname} "
                    f"GROUP BY {cols} HAVING c > 1"
                ).fetchall()
                if dupes:
                    problems.append(f"{tname} UNIQUE({cols}) 重复：{dupes}")
            for col in ts.not_null:
                bad = con.execute(
                    f"SELECT COUNT(*) FROM {tname} WHERE {col} IS NULL"
                ).fetchone()[0]
                if bad:
                    problems.append(f"{tname}.{col} 存在 {bad} 个 NULL，违反 NOT NULL")
            for chk in ts.checks:
                if chk["kind"] == "nonnegative":
                    col = chk["columns"][0]
                    bad = con.execute(
                        f"SELECT COUNT(*) FROM {tname} "
                        f"WHERE typeof({col}) != 'integer' OR {col} < 0"
                    ).fetchone()[0]
                    if bad:
                        problems.append(f"{tname}.{col} 存在 {bad} 个非法非负整数值")

            # 导出内容必须与 resolved 文档逐行一致。
            expected = {row[ts.pk_col]: row for row in resolved["merged"][tname]}
            actual = {
                r[0]: dict(zip(ts.columns, r))
                for r in con.execute(f"SELECT * FROM {tname}").fetchall()
            }
            if actual != expected:
                problems.append(f"{tname} 导出内容与裁决结果不一致")
    finally:
        con.close()

    if problems:
        Path(path).unlink(missing_ok=True)
        raise Merge3Error("导出库校验失败：" + "；".join(problems))
