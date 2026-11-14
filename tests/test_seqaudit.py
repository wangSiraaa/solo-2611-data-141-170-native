"""seqaudit 单元测试（Python 3.11 标准库 unittest）。

运行：python -m unittest discover -s tests -v
"""
from __future__ import annotations

import contextlib
import io
import json
import unittest
from dataclasses import replace
from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path

from seqaudit.audit import run_audit
from seqaudit.cli import main as cli_main
from seqaudit.eligibility import check_eligibility
from seqaudit.evidence import (
    INF,
    evidence_ge,
    evidence_value,
    evidence_walk_product_form,
    mixture_weights,
)
from seqaudit.io import ObsRow, load_lot, load_observations, load_plan
from seqaudit.order import derive_order
from seqaudit.tiny import enumerate_tiny
from seqaudit.verify import verify_report

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures"
CASES = ROOT / "cases"


def load_base():
    plan = load_plan(FIX / "plan.json")
    lot = load_lot(FIX / "lot.csv")
    return plan, lot


def meta():
    return {
        k: {"path": f"<{k}>", "sha256": f"hash-{k}"}
        for k in ("plan", "lot", "observations")
    }


def rows_from_defects(order, defect_positions):
    """按派生顺序构造 m=len(order) 条观察，defect_positions 为缺陷位（0 基）。"""
    return [
        ObsRow(i + 2, str(i + 1), item, "1" if i in defect_positions else "0")
        for i, item in enumerate(order)
    ]


class TestDrawOrder(unittest.TestCase):
    def test_order_reproduces_observations(self):
        plan, lot = load_base()
        order = derive_order(lot, plan.seed)
        obs = load_observations(FIX / "observations.csv")
        self.assertEqual([r.item_id for r in obs], order[: len(obs)])
        self.assertEqual(order[:3], ["part-034", "part-037", "part-054"])


class TestEligibility(unittest.TestCase):
    def setUp(self):
        self.plan, self.lot = load_base()
        self.order = derive_order(self.lot, self.plan.seed)

    def ok_report(self, rows):
        return check_eligibility(self.plan, self.lot, rows)

    def test_base_observations_eligible(self):
        obs = load_observations(FIX / "observations.csv")
        self.assertTrue(self.ok_report(obs).ok)

    def test_duplicate_item_blocked(self):
        obs = load_observations(CASES / "duplicate.csv")
        elig = self.ok_report(obs)
        self.assertFalse(elig.ok)
        names = {c.name for c in elig.checks if not c.ok}
        self.assertIn("items_unique", names)
        self.assertTrue(any("part-017" in d for d in elig.diagnostics))

    def test_reordered_blocked(self):
        rows = rows_from_defects(self.order[:10], set())
        rows[3], rows[4] = rows[4], rows[3]  # 对调第 4、5 抽
        rows = [ObsRow(r.line, str(i + 1), r.item_id, r.defect_raw) for i, r in enumerate(rows)]
        elig = self.ok_report(rows)
        self.assertFalse(elig.ok)
        names = {c.name for c in elig.checks if not c.ok}
        self.assertIn("prefix_matches_plan_order", names)

    def test_unknown_item_blocked(self):
        rows = rows_from_defects(self.order[:5], set())
        rows[2] = ObsRow(4, "3", "part-999", "0")
        elig = self.ok_report(rows)
        self.assertFalse(elig.ok)
        names = {c.name for c in elig.checks if not c.ok}
        self.assertIn("items_known", names)

    def test_gap_in_draw_no_blocked(self):
        rows = rows_from_defects(self.order[:5], set())
        rows[3] = ObsRow(5, "7", rows[3].item_id, "0")  # 漏号：1,2,3,7,5
        elig = self.ok_report(rows)
        self.assertFalse(elig.ok)
        names = {c.name for c in elig.checks if not c.ok}
        self.assertIn("draw_sequence_contiguous", names)

    def test_non_binary_defect_blocked(self):
        rows = rows_from_defects(self.order[:5], set())
        rows[1] = ObsRow(3, "2", rows[1].item_id, "2")
        elig = self.ok_report(rows)
        self.assertFalse(elig.ok)
        names = {c.name for c in elig.checks if not c.ok}
        self.assertIn("record_format", names)

    def test_duplicate_lot_roster_blocked(self):
        bad_lot = list(self.lot)
        bad_lot[1] = bad_lot[0]
        elig = check_eligibility(self.plan, bad_lot, [])
        self.assertFalse(elig.ok)
        names = {c.name for c in elig.checks if not c.ok}
        self.assertIn("lot_roster_matches_plan", names)


class TestEvidenceMath(unittest.TestCase):
    def test_initial_value_one(self):
        self.assertEqual(evidence_value(60, 7, 0, 0), Fraction(1))

    def test_hand_computed_tiny(self):
        # N=6, d0=2, t=3, k=0: 手算混合似然比 = 15/4
        self.assertEqual(evidence_value(6, 2, 3, 0), Fraction(15, 4))
        # t=4, k=0: 组合形式 = (1/2)(1 + (C(2,1)/C(6,1))/(C(2,2)/C(6,2)))... 直接核对数值
        self.assertEqual(evidence_value(6, 2, 4, 0), Fraction(10))

    def test_extension_null_certain_gives_zero(self):
        # k = d0+1：已见缺陷数本身证明零假设为真
        self.assertEqual(evidence_value(60, 7, 20, 8), Fraction(0))

    def test_extension_null_refuted_gives_inf(self):
        # 剩余 5 件，还需 6 个缺陷才到 d0：不可能
        self.assertEqual(evidence_value(60, 7, 55, 1), INF)

    def test_full_inspection_extremes(self):
        # 查到末件：D<d0 -> inf；D>=d0 -> 0
        self.assertEqual(evidence_value(60, 7, 60, 6), INF)
        self.assertEqual(evidence_value(60, 7, 60, 7), Fraction(0))

    def test_weights_frozen_uniform(self):
        self.assertEqual(mixture_weights(7), tuple(Fraction(1, 7) for _ in range(7)))

    def test_cross_implementation_on_all_tiny_paths(self):
        # 两种独立实现（组合数形式 vs 逐步条件乘积形式）在全部 tiny 路径上一致
        n, d0 = 6, 2
        for size in range(0, n + 1):
            for config in combinations(range(n), size):
                cfg = frozenset(config)
                for perm in permutations(range(n)):
                    defects = [1 if it in cfg else 0 for it in perm]
                    walk = evidence_walk_product_form(n, d0, defects)
                    k = 0
                    for t, x in enumerate(defects, start=1):
                        k += x
                        self.assertEqual(walk[t - 1], evidence_value(n, d0, t, k))


class TestAuditDecisions(unittest.TestCase):
    def setUp(self):
        self.plan, self.lot = load_base()
        self.order = derive_order(self.lot, self.plan.seed)

    def audit(self, rows, plan=None):
        return run_audit(plan or self.plan, self.lot, rows, meta())

    def test_base_accepts_at_30(self):
        obs = load_observations(FIX / "observations.csv")
        report, code = self.audit(obs)
        self.assertEqual(code, 0)
        self.assertTrue(report["certificate_valid"])
        a = report["audit"]
        self.assertEqual(a["decision"], "ACCEPT")
        self.assertEqual(a["first_accept_draw"], 30)
        self.assertEqual(a["first_evidence_crossing_draw"], 26)
        looks = {l["draw"]: l for l in a["looks"]}
        self.assertEqual(set(looks), {10, 20, 30, 40})
        self.assertFalse(looks[10]["accept_allowed"])
        self.assertFalse(looks[20]["accept_allowed"])
        self.assertTrue(looks[30]["accept_allowed"])
        self.assertEqual(looks[30]["defects"], 0)
        self.assertEqual(looks[30]["remaining"], 30)
        self.assertEqual(looks[30]["threshold"], "20")

    def test_defects_case_never_accepts(self):
        obs = load_observations(CASES / "defects.csv")
        report, code = self.audit(obs)
        self.assertEqual(code, 0)
        a = report["audit"]
        self.assertEqual(a["decision"], "CONTINUE_INSUFFICIENT_EVIDENCE")
        self.assertIsNone(a["first_accept_draw"])
        looks = {l["draw"]: l for l in a["looks"]}
        self.assertEqual(looks[30]["defects"], 7)
        self.assertEqual(looks[30]["evidence_exact"], "0")
        self.assertTrue(all(not l["accept_allowed"] for l in looks.values()))

    def test_duplicate_case_no_valid_certificate(self):
        obs = load_observations(CASES / "duplicate.csv")
        report, code = self.audit(obs)
        self.assertEqual(code, 2)
        self.assertFalse(report["certificate_valid"])
        self.assertIsNone(report["audit"])
        self.assertFalse(report["eligibility"]["ok"])
        self.assertTrue(report["eligibility"]["diagnostics"])

    def test_batching_does_not_change_prefix_evidence(self):
        obs = load_observations(FIX / "observations.csv")
        base_report, _ = self.audit(obs)
        alt_plan = replace(self.plan, batches=(5, 15, 25, 33, 40))
        alt_report, _ = self.audit(obs, plan=alt_plan)
        base_path = [p["evidence_exact"] for p in base_report["audit"]["path"]]
        alt_path = [p["evidence_exact"] for p in alt_report["audit"]["path"]]
        self.assertEqual(base_path, alt_path)
        # 查看点取值等于同前缀证据
        for l in alt_report["audit"]["looks"]:
            self.assertEqual(l["evidence_exact"], base_path[l["draw"] - 1])

    def test_full_inspection_good_lot_accepted_with_inf(self):
        defects = set(self.order[:6])  # 前 6 件缺陷，共 D=6 < 7
        rows = rows_from_defects(self.order, {i for i, it in enumerate(self.order) if it in defects})
        report, code = self.audit(rows)
        self.assertEqual(code, 0)
        a = report["audit"]
        self.assertEqual(a["decision"], "ACCEPT")
        self.assertEqual(a["full_inspection"]["true_total_defects"], 6)
        self.assertTrue(a["full_inspection"]["decision_matches_truth"])
        self.assertEqual(a["path"][-1]["evidence_exact"], "inf")

    def test_full_inspection_bad_lot_not_accepted(self):
        rows = rows_from_defects(self.order, set(range(7)))  # D=7 >= 7
        report, code = self.audit(rows)
        self.assertEqual(code, 0)
        a = report["audit"]
        self.assertEqual(a["decision"], "DO_NOT_ACCEPT")
        self.assertEqual(a["full_inspection"]["true_total_defects"], 7)
        self.assertTrue(a["full_inspection"]["decision_matches_truth"])
        self.assertEqual(a["path"][-1]["evidence_exact"], "0")

    def test_evidence_never_counts_unobserved_as_conforming(self):
        # 只查 1 件合格品时证据必须接近 1（不能把未观察件当合格提前接受）
        rows = rows_from_defects(self.order[:1], set())
        report, _ = self.audit(rows)
        ev = Fraction(report["audit"]["path"][0]["evidence_exact"])
        self.assertLess(ev, Fraction(6, 5))
        self.assertGreater(ev, Fraction(1))
        self.assertEqual(report["audit"]["decision"], "CONTINUE_INSUFFICIENT_EVIDENCE")


class TestVerify(unittest.TestCase):
    def setUp(self):
        self.plan, self.lot = load_base()

    def test_verify_base_certificate_passes(self):
        obs = load_observations(FIX / "observations.csv")
        report, _ = run_audit(self.plan, self.lot, obs, meta())
        record = verify_report(self.plan, self.lot, obs, report, meta())
        self.assertEqual(record["overall"], "PASS")

    def test_verify_detects_tampered_evidence(self):
        obs = load_observations(FIX / "observations.csv")
        report, _ = run_audit(self.plan, self.lot, obs, meta())
        tampered = json.loads(json.dumps(report))
        tampered["audit"]["looks"][2]["evidence_exact"] = "999999"
        record = verify_report(self.plan, self.lot, obs, tampered, meta())
        self.assertEqual(record["overall"], "FAIL")

    def test_verify_detects_tampered_decision(self):
        obs = load_observations(CASES / "defects.csv")
        report, _ = run_audit(self.plan, self.lot, obs, meta())
        tampered = json.loads(json.dumps(report))
        tampered["audit"]["decision"] = "ACCEPT"
        record = verify_report(self.plan, self.lot, obs, tampered, meta())
        self.assertEqual(record["overall"], "FAIL")

    def test_verify_ineligible_report_must_be_invalid(self):
        obs = load_observations(CASES / "duplicate.csv")
        report, code = run_audit(self.plan, self.lot, obs, meta())
        self.assertEqual(code, 2)
        record = verify_report(self.plan, self.lot, obs, report, meta())
        self.assertEqual(record["overall"], "PASS")
        # 若资格失败却给出有效证书，必须 FAIL
        forged = json.loads(json.dumps(report))
        forged["certificate_valid"] = True
        record2 = verify_report(self.plan, self.lot, obs, forged, meta())
        self.assertEqual(record2["overall"], "FAIL")


class TestMainPlanExactRisk(unittest.TestCase):
    """主计划的精确风险核查：在 (t,k) 状态空间上独立 DP（超几何转移），
    计算整条路径上曾接受的概率，核对 <= alpha；不复用审计的接受标记。"""

    @staticmethod
    def ever_accept_prob(D, n=60, d0=7, threshold=Fraction(20)):
        weights = mixture_weights(d0)
        dp = {(0, 0): Fraction(1)}
        accepted = Fraction(0)
        for t in range(n):
            nxt = {}
            for (tt, k), p in dp.items():
                if tt != t:
                    continue
                rem, slots = D - k, n - t
                for x in (0, 1):
                    px = Fraction(rem, slots) if x == 1 else Fraction(slots - rem, slots)
                    if px == 0:
                        continue
                    k2 = k + x
                    if evidence_ge(evidence_value(n, d0, t + 1, k2, weights), threshold):
                        accepted += p * px
                    else:
                        nxt[(t + 1, k2)] = nxt.get((t + 1, k2), Fraction(0)) + p * px
            dp = nxt
        return accepted

    def test_risk_bound_holds_for_every_null_lot(self):
        worst = max(self.ever_accept_prob(D) for D in range(7, 61))
        self.assertLessEqual(worst, Fraction(1, 20))
        self.assertEqual(worst, self.ever_accept_prob(7))  # 边界零假设最不利

    def test_good_lot_accepted_with_probability_one(self):
        self.assertEqual(self.ever_accept_prob(6), Fraction(1))


class TestTinyEnumeration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = enumerate_tiny(6, 2, Fraction(1, 3))

    def test_risk_bound_respected_on_whole_path(self):
        r = self.record
        self.assertTrue(r["risk_bound_respected"])
        self.assertEqual(r["max_false_accept_probability_exact"], "1/5")
        self.assertLessEqual(
            Fraction(r["max_false_accept_probability_exact"]), Fraction(1, 3)
        )

    def test_every_null_configuration_enumerated(self):
        rows = self.record["null_configurations"]
        self.assertEqual(len(rows), 57)  # C(6,2)+...+C(6,6)
        by_d = {}
        for row in rows:
            by_d.setdefault(row["defects_total"], set()).add(row["probability_exact"])
        self.assertEqual(by_d[2], {"1/5"})
        self.assertEqual(by_d[3], {"1/20"})
        self.assertEqual(by_d[4], {"0"})
        self.assertEqual(by_d[6], {"0"})

    def test_good_configurations_always_accepted_by_full_inspection(self):
        self.assertTrue(
            self.record["all_good_configurations_accepted_by_full_inspection"]
        )


class TestCli(unittest.TestCase):
    def run_cli(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli_main(argv)
        return code, buf.getvalue()

    def test_cli_audit_base(self):
        code, out = self.run_cli([
            "audit", "--plan", str(FIX / "plan.json"), "--lot",
            str(FIX / "lot.csv"), "--observations", str(FIX / "observations.csv"),
        ])
        self.assertEqual(code, 0)
        self.assertIn("ACCEPT", out)

    def test_cli_audit_duplicate_exit_2(self):
        code, out = self.run_cli([
            "audit", "--plan", str(FIX / "plan.json"), "--lot",
            str(FIX / "lot.csv"), "--observations", str(CASES / "duplicate.csv"),
        ])
        self.assertEqual(code, 2)
        self.assertIn("不出具有效审计证书", out)

    def test_cli_tiny(self):
        code, out = self.run_cli(["verify", "--tiny", str(FIX / "tiny.json")])
        self.assertEqual(code, 0)
        self.assertIn("1/5", out)


if __name__ == "__main__":
    unittest.main()
