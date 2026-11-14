"""端到端演示：用 fixtures 分支跑 plan -> resolve -> export 全流程。

运行：python3 tests/demo_fixtures.py
产物全部落在 work/demo/ 下，输入库只读访问、不修改。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from merge3.plan import build_plan, write_plan  # noqa: E402

OUT = ROOT / "work" / "demo"
FIX = ROOT / "fixtures"


def cli(*args: str) -> dict:
    r = subprocess.run(
        [sys.executable, "-m", "merge3", *args],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return json.loads(r.stdout)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    plan = build_plan(
        str(FIX / "base.sqlite"), str(FIX / "left.sqlite"),
        str(FIX / "right.sqlite"), str(FIX / "schema.sql"),
    )
    plan_path = OUT / "plan.json"
    write_plan(plan, str(plan_path))

    print("=" * 70)
    print("1) PLAN：以 base 识别两侧真实变更，共", len(plan["conflicts"]), "个冲突组")
    print("=" * 70)
    for g in plan["conflicts"]:
        print(f"  [{g['type']}] {g['id']}")
        print(f"      {g['title']}")
        print(f"      约束：{g['constraint']}")
        affected = ", ".join(f"{a['table']}.{a['key']}" for a in g["affected"])
        print(f"      受影响：{affected}")
    print("  自动合并（无冲突）：")
    for tname, rows in plan["auto_merge"].items():
        pk_name = "customer_id" if tname == "customers" else "order_id"
        for row in rows:
            print(f"      {tname}.{row[pk_name]} -> {row}")

    decisions = {
        "plan_digest": plan["plan_digest"],
        "decisions": [
            # 同字段冲突：C2.name 取 left
            {"group_id": "field:customers:C2:name",
             "decision": {"action": "pick", "side": "left"}},
            # 删除/改动：C3 两侧分歧，选择保留 right 版本
            {"group_id": "dvm:customers:C3",
             "decision": {"action": "keep_row", "side": "right"}},
            # O3 同样保留 right（金额 3000 -> 3100 的修改不丢失）
            {"group_id": "dvm:orders:O3",
             "decision": {"action": "keep_row", "side": "right"}},
            # 引用冲突：left 删 C4、right 新增 O4 引用 C4 -> 保留 right 客户
            {"group_id": "fk:customers:C4",
             "decision": {"action": "keep_customer", "side": "right"}},
            # 新增碰撞 O9：按字段混合，金额取 right、备注取 left
            {"group_id": "insert:orders:O9",
             "decision": {"action": "field_mix",
                          "fields": {"amount_cents": "right", "note": "left"}}},
            # 跨行唯一冲突：C5 保留 COMMON(left)，C6 回退 ZETA(base)
            {"group_id": "unique:customers:code:COMMON",
             "decision": {"action": "unique_pick", "members": [
                 {"table": "customers", "key": "C5", "column": "code", "side": "left"},
                 {"table": "customers", "key": "C6", "column": "code", "side": "base"},
             ]}},
        ],
    }
    dec_path = OUT / "decisions.json"
    dec_path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 70)
    print("2) RESOLVE：用户裁决文件显式点名 plan_digest，全量复查约束")
    print("=" * 70)
    res = cli("resolve", "--plan", str(plan_path),
              "--decisions", str(dec_path), "--out", str(OUT / "resolved.json"))
    print(" ", json.dumps(res, ensure_ascii=False))

    print()
    print("=" * 70)
    print("3) EXPORT：写完整 SQLite + merge_provenance，重新打开校验")
    print("=" * 70)
    exp = cli("export", "--resolved", str(OUT / "resolved.json"),
              "--out", str(OUT / "merged.sqlite"))
    print(" ", json.dumps(exp, ensure_ascii=False))

    import sqlite3
    con = sqlite3.connect(OUT / "merged.sqlite")
    con.execute("PRAGMA foreign_keys=ON")
    print("\n最终 customers：")
    for r in con.execute("SELECT * FROM customers ORDER BY customer_id"):
        print(" ", r)
    print("最终 orders：")
    for r in con.execute("SELECT * FROM orders ORDER BY order_id"):
        print(" ", r)
    print("foreign_key_check：", con.execute("PRAGMA foreign_key_check").fetchall())
    print("逐变更来源示例（C1 独立字段合并 / O9 混合裁决）：")
    for r in con.execute(
        "SELECT table_name,row_key,column_name,sources,change_kind FROM merge_provenance "
        "WHERE (table_name='customers' AND row_key='C1') OR "
        "(table_name='orders' AND row_key='O9') ORDER BY table_name,column_name"
    ):
        print(" ", r)
    con.close()
    print("\n演示产物目录：", OUT)


if __name__ == "__main__":
    main()
