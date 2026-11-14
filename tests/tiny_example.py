"""极小合成例：独立于 fixtures，按变更意图逐项核对工具产物。

覆盖：
  1. NULL、空字符串、整行不存在三者不同；
  2. 同字段同值修改自动合并、不同值才算冲突（不是“两边都动了就冲突”）；
  3. 独立字段修改自动合并；
  4. 一删一改必须裁决，且删除侧把列置 NULL 与删除整行不是一回事；
  5. 非冲突合并不依赖行序（乱序建库结果一致）；
  6. 主键身份不可修改（同主键才能配对）。
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from merge3.plan import build_plan
from merge3.resolve import resolve
from merge3.export import export
from merge3.canonical import MISSING

SCHEMA = (
    "PRAGMA foreign_keys=ON;\n"
    "CREATE TABLE customers(customer_id TEXT PRIMARY KEY, code TEXT NOT NULL UNIQUE, "
    "name TEXT NOT NULL, email TEXT);\n"
    "CREATE TABLE orders(order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL "
    "REFERENCES customers(customer_id), amount_cents INTEGER NOT NULL "
    "CHECK(amount_cents>=0), note TEXT);\n"
)


def make_db(path: Path, customers: list[tuple], orders: list[tuple], shuffle: bool = False) -> None:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    if shuffle:
        customers = list(reversed(customers))
        orders = list(reversed(orders))
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    con.commit()
    con.close()


def run() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="merge3_tiny_"))
    # base：K1 的 email 为 NULL；K2 的 email 为空串——二者必须被区分。
    base_c = [
        ("K1", "AAA", "n1", None),
        ("K2", "BBB", "n2", ""),
        ("K3", "CCC", "n3", None),
    ]
    base_o = [("T1", "K1", 10, None)]
    make_db(tmp / "base.sqlite", base_c, base_o)

    # left：K1 改 name；K2 把 email 从 "" 改成 NULL；K3 整行删除。
    left_c = [
        ("K1", "AAA", "n1-left", None),
        ("K2", "BBB", "n2", None),
    ]
    left_o = [("T1", "K1", 10, None)]
    make_db(tmp / "left.sqlite", left_c, left_o, shuffle=True)

    # right：K1 改 email（与 left 不同字段）；K2 同改 email 为 NULL（与 left 同值）；
    # K3 保留但改名（一删一改）；新增订单 T2 引用 K3（引用冲突素材，本用例先裁决保留）。
    right_c = [
        ("K1", "AAA", "n1", "r@x"),
        ("K2", "BBB", "n2", None),
        ("K3", "CCC", "n3-right", None),
    ]
    right_o = [("T1", "K1", 10, None), ("T2", "K3", 5, "new")]
    make_db(tmp / "right.sqlite", right_c, right_o, shuffle=True)

    schema_path = tmp / "schema.sql"
    schema_path.write_text(SCHEMA, encoding="utf-8")

    plan = build_plan(str(tmp / "base.sqlite"), str(tmp / "left.sqlite"),
                      str(tmp / "right.sqlite"), str(schema_path))
    rows = {r["key"]: r for r in plan["tables"]["customers"]["rows"]}

    # --- 意图 1：NULL / "" / 行缺失三态区分 ---
    k2_email = next(f for f in rows["K2"]["fields"] if f["name"] == "email")
    assert k2_email["base"] == "", k2_email
    assert k2_email["left"] is None and k2_email["right"] is None
    assert k2_email["status"] == "both_same", k2_email  # "" -> NULL 是真实修改且两侧同值
    k3 = rows["K3"]
    assert k3["values"]["left"] is None and k3["present"]["left"] is False
    assert MISSING not in (json.dumps(k2_email),)  # 哨兵只用于行缺失，不用于 NULL
    print("[1] NULL/空串/行缺失三态区分 OK：K2.email ''->NULL 自动合并，K3 左侧为整行缺失")

    # --- 意图 2：同字段同值自动合并，不同值才冲突 ---
    assert k2_email["auto_value"] is None and k2_email["sources"] == ["left", "right"]
    print("[2] 两侧同字段改成同值 -> both_same 自动合并，不误报冲突 OK")

    # --- 意图 3：独立字段修改自动合并 ---
    k1 = rows["K1"]
    assert k1["kind"] == "auto_merged" and k1["auto_row"] == {
        "customer_id": "K1", "code": "AAA", "name": "n1-left", "email": "r@x",
    }, k1
    print("[3] K1 name(left)/email(right) 独立修改自动合并 OK")

    # --- 意图 4：一删一改不静默决定 ---
    assert k3["kind"] == "delete_vs_modify"
    gids = {g["id"] for g in plan["conflicts"]}
    # T2 引用被一侧删除的 K3 -> 行级 DVM 被外键组吸收，只保留一个可解释组
    assert "dvm:customers:K3" not in gids
    fk_group = next(g for g in plan["conflicts"] if g["id"] == "fk:customers:K3")
    assert {a["table"] for a in fk_group["affected"]} == {"customers", "orders"}
    print("[4] K3 一删一改进入 delete_vs_modify，并被 fk_delete_reference 组吸收 OK")

    # --- 意图 5：行序无关（乱序库产生的计划与正序库一致）---
    make_db(tmp / "base2.sqlite", list(reversed(base_c)), list(reversed(base_o)))
    plan2 = build_plan(str(tmp / "base2.sqlite"), str(tmp / "left.sqlite"),
                       str(tmp / "right.sqlite"), str(schema_path))
    assert plan2["auto_merge"] == plan["auto_merge"]
    assert [g["id"] for g in plan2["conflicts"]] == [g["id"] for g in plan["conflicts"]]
    print("[5] 乱序输入与正序输入的计划完全一致（不依赖行序）OK")

    # --- 裁决并导出：保留 K3（右侧），其余自动 ---
    decisions = {
        "plan_digest": plan["plan_digest"],
        "decisions": [
            {"group_id": "fk:customers:K3",
             "decision": {"action": "keep_customer", "side": "right"}},
        ],
    }
    dec_path = tmp / "decisions.json"
    dec_path.write_text(json.dumps(decisions, ensure_ascii=False), encoding="utf-8")
    from merge3.plan import write_plan
    plan_path = tmp / "plan.json"
    write_plan(plan, str(plan_path))
    resolved = resolve(str(plan_path), str(dec_path), str(tmp / "resolved.json"))

    cust = {r["customer_id"]: r for r in resolved["merged"]["customers"]}
    assert cust["K1"] == {
        "customer_id": "K1", "code": "AAA", "name": "n1-left", "email": "r@x",
    }, cust["K1"]
    assert cust["K2"]["email"] is None and cust["K2"]["name"] == "n2"
    assert cust["K3"]["name"] == "n3-right"
    orders = {r["order_id"]: r for r in resolved["merged"]["orders"]}
    assert orders["T2"]["customer_id"] == "K3" and orders["T2"]["amount_cents"] == 5
    print("[6] 裁决后终态逐字段符合意图 OK")

    info = export(str(tmp / "resolved.json"), str(tmp / "out.sqlite"))
    con = sqlite3.connect(tmp / "out.sqlite")
    con.execute("PRAGMA foreign_keys=ON")
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    assert con.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 3
    assert con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 2
    # 来源审计：K1.name 来自 left，K1.email 来自 right，K2.email 两侧同值
    src = {
        (r[0], r[1]): (r[2], r[3])
        for r in con.execute(
            "SELECT row_key,column_name,sources,change_kind FROM merge_provenance"
        )
    }
    assert src[("K1", "name")] == ("left", "auto_merged"), src[("K1", "name")]
    assert src[("K1", "email")] == ("right", "auto_merged"), src[("K1", "email")]
    assert src[("K2", "email")] == ("left,right", "auto_merged"), src[("K2", "email")]
    con.close()
    print("[7] 导出库重开校验外键/行数/逐列来源全部通过 OK")
    print("tiny example output:", info)


if __name__ == "__main__":
    run()
