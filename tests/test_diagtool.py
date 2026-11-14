"""诊断工具的测试：样例覆盖、语义正确性、最小性、行序不变性、只读性、语法拒绝。

运行：python3 -m unittest discover -s tests -v
"""

import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from diagtool.config import Plan, load_allowed_changes, load_plan
from diagtool.db import Database
from diagtool.errors import DiagError, PlanError, UnsupportedSyntaxError
from diagtool.parser import parse_sql
from diagtool.report import run_diagnosis
from diagtool.solutions import identity_index
from diagtool.runner import execute_rows
from diagtool.analysis import AnalysisContext
from diagtool.rewrite import rewrite_remove_atoms

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures"


def make_sqlite(tmpdir: Path, seed: str) -> Path:
    path = tmpdir / "report.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(seed)
    return path


def make_seed_db(tmpdir: Path) -> Path:
    return make_sqlite(tmpdir, (FIX / "seed.sql").read_text(encoding="utf-8"))


QUERY = (FIX / "query.sql").read_text(encoding="utf-8")


class BaseCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.dbpath = make_seed_db(self.tmp)
        self.plan = load_plan(FIX / "plan.json")
        self.allowed = load_allowed_changes(FIX / "allowed_changes.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_report(self, targets=None, allowed=None, query=QUERY, dbpath=None):
        plan = self.plan
        if targets is not None:
            raw = dict(plan.raw)
            raw["targets"] = targets
            plan = Plan(parameters=plan.parameters, target_table=plan.target_table,
                        target_pk=plan.target_pk, targets=targets,
                        result_identity_column=plan.result_identity_column, raw=raw)
        return run_diagnosis(dbpath or self.dbpath, query, plan,
                             allowed or self.allowed)


class TestSemantics(unittest.TestCase):
    """Python 三值/亲和性求值必须与 SQLite 真实执行逐项一致。"""

    def test_affinity_matches_sqlite(self):
        from diagtool.semantics import affinity_of_type, sql_compare
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE t(tv TEXT, iv INTEGER, nv NUMERIC, rv REAL)")
        db.executemany("INSERT INTO t VALUES(?,?,?,?)", [
            ("10", 10, 10, 1.5), (" 10 ", 10, 10, 2.0), ("abc", 5, 5, 1e6),
            (None, None, None, None), ("3.0", 3, 3.0, 123456789012345678),
            ("100", 100, 100, 1e-5)])
        rows = db.execute("SELECT * FROM t").fetchall()
        types = {"tv": ("TEXT", 0), "iv": ("INTEGER", 1),
                 "nv": ("NUMERIC", 2), "rv": ("REAL", 3)}
        operands = [10, 3, 3.0, "10", "abc", "3", None, "3.0", 100, 1.5,
                    123456789012345678, 1e6]
        for col, (aff, ix) in types.items():
            for op in ("=", "<>", "<", ">", ">=", "<="):
                for rv in operands:
                    sqlite_rows = db.execute(
                        f"SELECT {col} {op} ? FROM t", (rv,)).fetchall()
                    for i, (g, ) in enumerate(sqlite_rows):
                        val = rows[i][ix]
                        py = sql_compare(val, affinity_of_type(aff), rv,
                                         "NONE", op)
                        py = None if py is None else int(py)
                        sg = None if g is None else int(g)
                        self.assertEqual(
                            py, sg,
                            f"{col} {op} {val!r} {rv!r}")

    def test_tri_logic(self):
        from diagtool.semantics import tri_and, tri_or
        self.assertEqual(tri_and(True, None), None)
        self.assertEqual(tri_and(False, None), False)
        self.assertEqual(tri_or(True, None), True)
        self.assertEqual(tri_or(False, None), None)
        self.assertEqual(tri_or(None, None), None)


class TestParser(unittest.TestCase):
    def test_structure(self):
        s = parse_sql(QUERY)
        self.assertEqual(s.base_table.name, "orders")
        self.assertEqual([j.join_type for j in s.joins], ["INNER", "LEFT"])
        self.assertEqual(len(s.atom_by_index), 4)
        # 原子顺序与源码一致，且区段精确
        for i, atom in s.atom_by_index.items():
            self.assertEqual(QUERY[atom.start:atom.end], atom.sql_text)

    def test_left_join_order_independent_on(self):
        s = parse_sql("SELECT o.order_id FROM orders o JOIN payments p "
                      "ON p.order_id = o.order_id")
        self.assertEqual(s.joins[0].join_type, "INNER")

    def test_reject_unknown(self):
        bad = [
            "SELECT COUNT(*) FROM orders",
            "SELECT o.order_id FROM orders o WHERE o.amount_cents > (SELECT 1)",
            "SELECT o.order_id FROM orders o WHERE o.priority IN ('rush')",
            "SELECT o.order_id FROM orders o WHERE o.amount_cents BETWEEN 1 AND 2",
            "SELECT o.order_id FROM orders o WHERE o.priority LIKE 'r%'",
            "SELECT o.order_id FROM orders o WHERE NOT o.amount_cents = 1",
            "SELECT o.order_id FROM orders o ORDER BY 1",
            "SELECT o.order_id FROM orders o JOIN customers c "
            "ON o.customer_id=c.customer_id JOIN payments p ON p.order_id=o.order_id "
            "JOIN payments p2 ON p2.order_id=o.order_id",
            "SELECT o.order_id FROM orders o WHERE o.amount_cents > 1 UNION SELECT 1",
        ]
        for sql in bad:
            with self.subTest(sql=sql):
                with self.assertRaises(UnsupportedSyntaxError):
                    parse_sql(sql)

    def test_reject_positional_param(self):
        with self.assertRaises(UnsupportedSyntaxError):
            parse_sql("SELECT o.order_id FROM orders o WHERE o.amount_cents > ?")

    def test_preserves_parentheses_structure(self):
        s = parse_sql(QUERY)
        # WHERE 顶层是 AND，左子含 OR（括号未被拍平）
        self.assertEqual(s.where.operator, "AND")
        left = s.where.left
        self.assertEqual(left.operator, "AND")
        self.assertEqual(left.right.operator, "OR")


class TestDiagnosis(BaseCase):
    def test_categories(self):
        r = self.run_report()
        cats = {d["target_pk"]: d["category"] for d in r["target_diagnoses"]}
        self.assertEqual(cats["O2"], "where_false")
        self.assertEqual(cats["O3"], "where_false")   # FALSE 主导，含 UNKNOWN 原子
        self.assertEqual(cats["O4"], "where_unknown")  # LEFT NULL 扩展
        self.assertEqual(cats["O5"], "inner_join_no_match")  # 无客户

    def test_no_payment_uses_left_null(self):
        r = self.run_report(targets=["O4"])
        d = r["target_diagnoses"][0]
        cand = d["candidate_rows"][0]
        self.assertEqual(cand["null_extended_aliases"], ["p"])
        self.assertEqual(cand["atom_states"][3], "UNKNOWN")
        # OR 分支：amount FALSE 但 priority TRUE -> OR TRUE
        tree = cand["where_path"]["tree"]
        self.assertEqual(tree["result"], "UNKNOWN")
        self.assertEqual(tree["left"]["right"]["operator"], "OR")
        self.assertEqual(tree["left"]["right"]["result"], "TRUE")

    def test_multiple_payments_each_has_evidence(self):
        r = self.run_report(targets=["O1"])
        d = r["target_diagnoses"][0]
        self.assertTrue(d["visible_in_original"])
        self.assertEqual(d["visible_row_count"], 2)
        # 两条候选行各有独立证据
        self.assertEqual(len(d["candidate_rows"]), 2)
        pks = {c["join_rows"]["p"]["primary_key"]["payment_id"]
               for c in d["candidate_rows"]}
        self.assertEqual(pks, {"P11", "P12"})

    def test_unknown_status(self):
        r = self.run_report(targets=["O3"])
        cand = r["target_diagnoses"][0]["candidate_rows"][0]
        self.assertEqual(cand["atom_states"][3], "UNKNOWN")
        self.assertEqual(cand["where_result"], "FALSE")

    def test_no_customer(self):
        r = self.run_report(targets=["O5"])
        d = r["target_diagnoses"][0]
        self.assertEqual(d["category"], "inner_join_no_match")
        step = [s for s in d["join_trace"] if s["operation"] == "inner_join"][0]
        self.assertEqual(step["matched_row_count"], 0)

    def test_base_row_missing(self):
        r = self.run_report(targets=["O99"])
        self.assertEqual(r["target_diagnoses"][0]["category"], "base_row_missing")
        self.assertFalse(r["solutions"]["per_target"][0]["feasible"])


class TestSolutions(BaseCase):
    def test_minimum_costs(self):
        r = self.run_report()
        got = {t["target"]: (t["feasible"], t["minimum_cost"])
               for t in r["solutions"]["per_target"]}
        self.assertEqual(got["O2"], (True, 3))
        self.assertEqual(got["O3"], (True, 2))
        self.assertEqual(got["O4"], (True, 1))
        self.assertFalse(got["O5"][0])

    def test_no_solution_does_not_invent_rows(self):
        r = self.run_report(targets=["O5"])
        t = r["solutions"]["per_target"][0]
        self.assertFalse(t["feasible"])
        self.assertEqual(t["verified_solutions"], [])
        self.assertIn("INNER", t["rejected_reason"])

    def test_every_solution_revalidated_by_sql(self):
        r = self.run_report()
        for t in r["solutions"]["per_target"]:
            for sol in t["verified_solutions"]:
                self.assertTrue(sol["revalidation"]["target_visible"][t["target"]])
                self.assertFalse(sol["applied"])

    def test_impact_multiset_counts(self):
        # O4: 删 payment_status 后，应新增 O4(NULL 付款) 与 O6/P61 各 1 行。
        r = self.run_report(targets=["O4"])
        sol = r["solutions"]["per_target"][0]["verified_solutions"][0]
        imp = sol["revalidation"]["impact"]
        added = {(a["row"]["order_id"], a["row"].get("payment_id")): a["count"]
                 for a in imp["added"]}
        self.assertEqual(added.get(("O4", None)), 1)
        self.assertEqual(added.get(("O6", "P61")), 1)

    def test_parallel_minimum(self):
        allowed = load_allowed_changes(FIX / "allowed_changes.json")
        # 增加 paid_status 备选，制造 O2 的并列最小方案
        data = json.loads((FIX / "allowed_changes.json").read_text())
        data["parameter_alternatives"]["paid_status"] = ["pending"]
        alt_path = self.tmp / "allowed_alt.json"
        alt_path.write_text(json.dumps(data))
        alt = load_allowed_changes(alt_path)
        r = self.run_report(targets=["O2"], allowed=alt)
        t = r["solutions"]["per_target"][0]
        self.assertTrue(t["feasible"])
        self.assertEqual(t["minimum_cost"], 3)
        self.assertGreaterEqual(t["parallel_minimum_count"], 2)
        # 并列方案的操作集合不同
        sigs = {tuple(sorted((op.get("parameter") or op.get("predicate_id"))
                             for op in s["operations"]))
                for s in t["verified_solutions"]}
        self.assertGreaterEqual(len(sigs), 2)


class TestInvariants(BaseCase):
    def test_row_order_invariance(self):
        # 构造物理行序反转的等价库
        alt = self.tmp / "shuffled.sqlite"
        src = sqlite3.connect(self.dbpath)
        dst = sqlite3.connect(alt)
        dst.executescript((FIX / "seed.sql").read_text())
        for table in ("customers", "orders", "payments"):
            rows = src.execute(f"SELECT * FROM {table}").fetchall()
            dst.execute(f"DELETE FROM {table}")
            for row in reversed(rows):
                dst.execute(f"INSERT INTO {table} VALUES "
                            f"({','.join('?' * len(row))})", row)
        dst.commit()
        src.close()
        dst.close()

        a = self.run_report()
        b = self.run_report(dbpath=alt)
        # 诊断类别与证据原子状态应按内容一致（与物理行序无关）。
        for da, db_ in zip(a["target_diagnoses"], b["target_diagnoses"]):
            self.assertEqual(da["category"], db_["category"])
            sa = sorted(tuple(sorted(c["atom_states"].items()))
                        for c in da["candidate_rows"])
            sb = sorted(tuple(sorted(c["atom_states"].items()))
                        for c in db_["candidate_rows"])
            self.assertEqual(sa, sb)

        def summary(r):
            return [(t["target"], t["feasible"], t["minimum_cost"],
                     t["parallel_minimum_count"])
                    for t in r["solutions"]["per_target"]]
        self.assertEqual(summary(a), summary(b))

    def test_read_only(self):
        before = self.dbpath.read_bytes()
        self.run_report()
        self.assertEqual(before, self.dbpath.read_bytes())

    def test_db_opened_query_only(self):
        with Database(str(self.dbpath)) as db:
            with self.assertRaises(sqlite3.OperationalError):
                db.conn.execute("CREATE TABLE hack(x)")
            with self.assertRaises(sqlite3.OperationalError):
                db.conn.execute("DELETE FROM orders")


class TestRewrite(unittest.TestCase):
    def test_remove_replaces_with_true(self):
        s = parse_sql(QUERY)
        atom = s.atom_by_index[3]
        out = rewrite_remove_atoms(QUERY, [atom])
        self.assertIn("AND TRUE", out)
        self.assertNotIn("p.status = :paid_status", out)
        # 其余谓词与括号结构保留
        s2 = parse_sql(out)
        self.assertEqual(len(s2.atom_by_index), 3)
        self.assertEqual(s2.where.left.right.operator, "OR")

    def test_remove_inside_or_preserves_parens(self):
        sql = ("SELECT o.order_id FROM orders o "
               "WHERE o.amount_cents >= :min_amount AND "
               "(o.priority = 'rush' OR o.amount_cents >= :min_amount)")
        s = parse_sql(sql)
        # 删除括号内的第二个 OR 分支
        target = s.atom_by_index[2]
        out = rewrite_remove_atoms(sql, [target])
        self.assertIn("(o.priority = 'rush' OR TRUE)", out)
        s2 = parse_sql(out)
        self.assertEqual(len(s2.atom_by_index), 2)


if __name__ == "__main__":
    unittest.main()
