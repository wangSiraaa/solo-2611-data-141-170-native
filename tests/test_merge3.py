"""merge3 全量回归（标准库 unittest，无第三方依赖）。

覆盖需求逐条：
  三方分类（未改/独立改/同字段冲突/新增碰撞/删除对修改）、
  NULL vs 空串 vs 行缺失、跨行唯一冲突、外键引用冲突、
  裁决绑定摘要与源变化拒绝、派生冲突不发布半成品、
  导出重开校验、左右对称、行序无关、输入不可变。
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from merge3.canonical import MISSING, read_json
from merge3.errors import DecisionError, InputChangedError, UnresolvedConflictsError
from merge3.export import export
from merge3.plan import build_plan, write_plan
from merge3.resolve import resolve

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures"
SCHEMA_SQL = (FIX / "schema.sql").read_text(encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_db(path: Path, customers: list[tuple], orders: list[tuple]) -> None:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA_SQL)
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    con.commit()
    con.close()


class FixturePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_plan(
            str(FIX / "base.sqlite"), str(FIX / "left.sqlite"),
            str(FIX / "right.sqlite"), str(FIX / "schema.sql"),
        )
        cls.groups = {g["id"]: g for g in cls.plan["conflicts"]}
        cls.rows = {
            t: {r["key"]: r for r in p["rows"]}
            for t, p in cls.plan["tables"].items()
        }

    def test_six_expected_conflict_groups(self):
        self.assertEqual(
            sorted(self.groups),
            sorted([
                "field:customers:C2:name",
                "dvm:customers:C3",
                "dvm:orders:O3",
                "fk:customers:C4",
                "insert:orders:O9",
                "unique:customers:code:COMMON",
            ]),
        )

    def test_independent_field_changes_auto_merge(self):
        c1 = self.rows["customers"]["C1"]
        self.assertEqual(c1["kind"], "auto_merged")
        name_f = next(f for f in c1["fields"] if f["name"] == "name")
        email_f = next(f for f in c1["fields"] if f["name"] == "email")
        self.assertEqual(name_f["status"], "left_only")
        self.assertEqual(email_f["status"], "right_only")
        self.assertEqual(c1["auto_row"]["name"], "甲-改名")
        self.assertEqual(c1["auto_row"]["email"], "new-a@example.invalid")

    def test_same_field_different_values_is_conflict(self):
        c2 = self.rows["customers"]["C2"]
        self.assertEqual(c2["kind"], "field_conflict")
        name_f = next(f for f in c2["fields"] if f["name"] == "name")
        self.assertEqual(name_f["status"], "field_conflict")
        self.assertIsNone(name_f["auto_value"])
        self.assertEqual(name_f["base"], "乙")
        self.assertEqual(name_f["left"], "乙-左")
        self.assertEqual(name_f["right"], "乙-右")

    def test_unchanged_row_not_flagged(self):
        self.assertEqual(self.rows["orders"]["O1"]["kind"], "unchanged")
        self.assertEqual(self.rows["orders"]["O2"]["kind"], "unchanged")

    def test_delete_vs_modify_on_customer_and_order(self):
        self.assertEqual(self.rows["customers"]["C3"]["kind"], "delete_vs_modify")
        self.assertEqual(self.rows["customers"]["C3"]["deleting_side"], "left")
        self.assertEqual(self.rows["customers"]["C3"]["modifying_side"], "right")
        self.assertEqual(self.rows["orders"]["O3"]["kind"], "delete_vs_modify")

    def test_insert_collision_o9(self):
        o9 = self.rows["orders"]["O9"]
        self.assertEqual(o9["kind"], "insert_collision")
        self.assertEqual(sorted(o9["inserted_by"]), ["left", "right"])
        amount = next(f for f in o9["fields"] if f["name"] == "amount_cents")
        self.assertEqual(amount["status"], "both_set_diff")
        self.assertEqual(amount["left"], 900)
        self.assertEqual(amount["right"], 950)

    def test_unique_conflict_cross_row_group(self):
        g = self.groups["unique:customers:code:COMMON"]
        affected = sorted((a["key"]) for a in g["affected"])
        self.assertEqual(affected, ["C5", "C6"])
        self.assertIn("UNIQUE", g["constraint"])

    def test_fk_reference_group(self):
        g = self.groups["fk:customers:C4"]
        self.assertEqual(g["type"], "fk_delete_reference")
        self.assertEqual(
            sorted(a["table"] + ":" + a["key"] for a in g["affected"]),
            ["customers:C4", "orders:O4"],
        )
        actions = {o["decision"]["action"] for o in g["options"]}
        self.assertEqual(actions, {"keep_customer", "delete_customer_discard_orders"})

    def test_cross_group_dependency_linked(self):
        c3 = self.groups["dvm:customers:C3"]
        o3 = self.groups["dvm:orders:O3"]
        self.assertTrue(any(d["group"] == "dvm:orders:O3" for d in c3["dependencies"]))
        self.assertTrue(any(d["group"] == "dvm:customers:C3" for d in o3["dependencies"]))

    def test_plan_retains_three_side_values(self):
        c3 = self.rows["customers"]["C3"]
        self.assertIsNone(c3["values"]["left"])           # 整行缺失
        self.assertIsNotNone(c3["values"]["right"])
        self.assertIsNotNone(c3["values"]["base"])


class TriStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        _make_db(self.tmp / "b.sqlite",
                 [("K1", "A", "n", None), ("K2", "B", "n", "")], [])
        _make_db(self.tmp / "l.sqlite",
                 [("K1", "A", "n", ""), ("K2", "B", "n", None)], [])
        _make_db(self.tmp / "r.sqlite",
                 [("K1", "A", "n", None), ("K2", "B", "n", None)], [])
        sp = self.tmp / "s.sql"
        sp.write_text(SCHEMA_SQL)
        self.plan = build_plan(str(self.tmp / "b.sqlite"), str(self.tmp / "l.sqlite"),
                               str(self.tmp / "r.sqlite"), str(sp))
        self.rows = {r["key"]: r for r in self.plan["tables"]["customers"]["rows"]}

    def test_null_empty_string_missing_are_distinct(self):
        k1 = next(f for f in self.rows["K1"]["fields"] if f["name"] == "email")
        # base NULL -> left "" 是真实修改；right 未动 -> 自动取 left 的空串
        self.assertEqual(k1["status"], "left_only")
        self.assertEqual(k1["auto_value"], "")
        k2 = next(f for f in self.rows["K2"]["fields"] if f["name"] == "email")
        # base "" -> 两侧都改 NULL：同值自动合并，NULL 不是缺失
        self.assertEqual(k2["status"], "both_same")
        self.assertIsNone(k2["auto_value"])
        self.assertEqual(self.rows["K2"]["auto_row"]["email"], None)

    def test_missing_sentinel_only_for_absent_rows(self):
        # 所有真实字段值里不应出现 MISSING 哨兵
        for row in self.plan["tables"]["customers"]["rows"]:
            for f in row["fields"]:
                for side in ("base", "left", "right"):
                    val = f[side]
                    if val is MISSING:
                        self.fail("真实字段值不得使用 MISSING 哨兵")


class ResolveAndExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.plan = build_plan(
            str(FIX / "base.sqlite"), str(FIX / "left.sqlite"),
            str(FIX / "right.sqlite"), str(FIX / "schema.sql"),
        )
        cls.plan_path = cls.tmp / "plan.json"
        write_plan(cls.plan, str(cls.plan_path))

    def _decisions(self, **overrides):
        d = {
            "plan_digest": self.plan["plan_digest"],
            "decisions": [
                {"group_id": "field:customers:C2:name",
                 "decision": {"action": "pick", "side": "left"}},
                {"group_id": "dvm:customers:C3",
                 "decision": {"action": "keep_row", "side": "right"}},
                {"group_id": "dvm:orders:O3",
                 "decision": {"action": "keep_row", "side": "right"}},
                {"group_id": "fk:customers:C4",
                 "decision": {"action": "keep_customer", "side": "right"}},
                {"group_id": "insert:orders:O9",
                 "decision": {"action": "keep_row", "side": "left"}},
                {"group_id": "unique:customers:code:COMMON",
                 "decision": {"action": "unique_pick", "members": [
                     {"table": "customers", "key": "C5", "column": "code", "side": "left"},
                     {"table": "customers", "key": "C6", "column": "code", "side": "base"},
                 ]}},
            ],
        }
        by_id = {x["group_id"]: x for x in d["decisions"]}
        for gid, dec in overrides.items():
            if dec is None:
                d["decisions"] = [x for x in d["decisions"] if x["group_id"] != gid]
            else:
                by_id[gid]["decision"] = dec
        return d

    def _resolve(self, doc, name="r.json"):
        p = self.tmp / name
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        return resolve(str(self.plan_path), str(p), str(self.tmp / ("out_" + name)))

    def test_full_success_and_export(self):
        resolved = self._resolve(self._decisions(), "ok.json")
        cust = {r["customer_id"]: r for r in resolved["merged"]["customers"]}
        self.assertEqual(cust["C5"]["code"], "COMMON")
        self.assertEqual(cust["C6"]["code"], "ZETA")
        info = export(str(self.tmp / "out_ok.json"), str(self.tmp / "merged.sqlite"))
        self.assertEqual(info["rows"], {"customers": 6, "orders": 5})
        con = sqlite3.connect(self.tmp / "merged.sqlite")
        con.execute("PRAGMA foreign_keys=ON")
        self.assertEqual(con.execute("PRAGMA foreign_key_check").fetchall(), [])
        self.assertEqual(
            con.execute("SELECT code, COUNT(*) FROM customers GROUP BY code HAVING COUNT(*)>1").fetchall(),
            [],
        )
        self.assertEqual(
            con.execute("SELECT COUNT(*) FROM orders WHERE amount_cents<0").fetchone()[0], 0
        )
        # 审计表存在且含逐列来源
        self.assertTrue(con.execute(
            "SELECT 1 FROM merge_provenance WHERE table_name='customers' AND row_key='C1' "
            "AND column_name='email' AND sources='right'"
        ).fetchone())
        con.close()

    def test_field_mix_on_insert_collision(self):
        d = self._decisions(**{
            "insert:orders:O9": {"action": "field_mix",
                                  "fields": {"amount_cents": "right", "note": "left"}}
        })
        resolved = self._resolve(d, "mix.json")
        o9 = next(r for r in resolved["merged"]["orders"] if r["order_id"] == "O9")
        self.assertEqual(o9["amount_cents"], 950)   # right
        self.assertEqual(o9["note"], "left new")    # left

    def test_field_mix_must_cover_all_differing_columns(self):
        d = self._decisions(**{
            "insert:orders:O9": {"action": "field_mix", "fields": {"note": "left"}}
        })
        with self.assertRaises(DecisionError):
            self._resolve(d, "mix_bad.json")

    def test_unique_collision_rechecked_returns_residual(self):
        d = self._decisions(**{
            "unique:customers:code:COMMON": {"action": "unique_pick", "members": [
                {"table": "customers", "key": "C5", "column": "code", "side": "left"},
                {"table": "customers", "key": "C6", "column": "code", "side": "right"},
            ]}
        })
        with self.assertRaises(UnresolvedConflictsError) as ctx:
            self._resolve(d, "uniq.json")
        self.assertEqual(ctx.exception.residual[0]["id"], "unique:customers:code:COMMON")
        self.assertFalse((self.tmp / "out_uniq.json").exists())

    def test_fk_derived_then_second_pass_resolves(self):
        d = self._decisions(**{
            "dvm:customers:C3": {"action": "delete_row"},
        })
        with self.assertRaises(UnresolvedConflictsError) as ctx:
            self._resolve(d, "pass1.json")
        residual = ctx.exception.residual
        self.assertEqual(residual[0]["id"], "fk:customers:C3")
        d["decisions"].append(
            {"group_id": "fk:customers:C3",
             "decision": {"action": "restore_customer", "source": "base"}}
        )
        resolved = self._resolve(d, "pass2.json")
        cust = {r["customer_id"]: r for r in resolved["merged"]["customers"]}
        self.assertEqual(cust["C3"]["code"], "GAMMA")  # base 恢复值

    def test_missing_decision_rejected(self):
        d = self._decisions(**{"insert:orders:O9": None})
        with self.assertRaises(DecisionError):
            self._resolve(d, "missing.json")

    def test_stale_plan_digest_rejected(self):
        d = self._decisions()
        d["plan_digest"] = "0" * 64
        with self.assertRaises(DecisionError):
            self._resolve(d, "stale.json")

    def test_changed_input_rejected(self):
        modified = self.tmp / "base_mod.sqlite"
        shutil.copy(FIX / "base.sqlite", modified)
        con = sqlite3.connect(modified)
        con.execute("UPDATE customers SET name='x' WHERE customer_id='C1'")
        con.commit()
        con.close()
        plan2 = copy.deepcopy(self.plan)
        plan2["inputs"]["base"]["path"] = str(modified)
        p2 = self.tmp / "plan2.json"
        write_plan(plan2, str(p2))
        d = self._decisions()
        dp = self.tmp / "d.json"
        dp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(InputChangedError):
            resolve(str(p2), str(dp), str(self.tmp / "never.json"))

    def test_unknown_group_and_duplicate_rejected(self):
        d = self._decisions()
        d["decisions"].append({"group_id": "field:no:such",
                               "decision": {"action": "pick", "side": "left"}})
        with self.assertRaises(DecisionError):
            self._resolve(d, "unknown.json")
        d2 = self._decisions()
        d2["decisions"].append(d2["decisions"][0])
        with self.assertRaises(DecisionError):
            self._resolve(d2, "dup.json")

    def test_inputs_never_modified(self):
        before = {n: _sha(FIX / n) for n in ("base.sqlite", "left.sqlite", "right.sqlite")}
        self._resolve(self._decisions(), "imm.json")
        after = {n: _sha(FIX / n) for n in before}
        self.assertEqual(before, after)


class SymmetryAndOrderTests(unittest.TestCase):
    def test_swap_left_right_symmetric_end_state(self):
        p_lr = build_plan(str(FIX / "base.sqlite"), str(FIX / "left.sqlite"),
                          str(FIX / "right.sqlite"), str(FIX / "schema.sql"))
        p_rl = build_plan(str(FIX / "base.sqlite"), str(FIX / "right.sqlite"),
                          str(FIX / "left.sqlite"), str(FIX / "schema.sql"))
        self.assertEqual(p_lr["auto_merge"], p_rl["auto_merge"])
        self.assertEqual(
            sorted(g["type"] for g in p_lr["conflicts"]),
            sorted(g["type"] for g in p_rl["conflicts"]),
        )

    def test_row_order_independence(self):
        tmp = Path(tempfile.mkdtemp())
        for name in ("base", "left", "right"):
            src = sqlite3.connect(FIX / f"{name}.sqlite")
            rows_c = src.execute("SELECT * FROM customers ORDER BY RANDOM()").fetchall()
            rows_o = src.execute("SELECT * FROM orders ORDER BY RANDOM()").fetchall()
            src.close()
            _make_db(tmp / f"{name}.sqlite", rows_c, rows_o)
        p1 = build_plan(str(FIX / "base.sqlite"), str(FIX / "left.sqlite"),
                        str(FIX / "right.sqlite"), str(FIX / "schema.sql"))
        p2 = build_plan(str(tmp / "base.sqlite"), str(tmp / "left.sqlite"),
                        str(tmp / "right.sqlite"), str(FIX / "schema.sql"))
        self.assertEqual(p1["auto_merge"], p2["auto_merge"])
        self.assertEqual(
            [g["id"] for g in p1["conflicts"]], [g["id"] for g in p2["conflicts"]]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
