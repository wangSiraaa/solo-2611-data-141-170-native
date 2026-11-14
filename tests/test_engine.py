"""Engine semantics: path-wise eligibility, unknown consistency, states, reports."""

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from survey_eligibility.engine import (
    ASKED_MISSING,
    ASKED_REFUSED,
    CONDITIONAL_ANSWER,
    MUST,
    MAY,
    NEVER,
    STRAY_ANSWER,
    STRUCTURALLY_NOT_ASKED,
    VALID_ANSWER,
    Engine,
)
from survey_eligibility.exhaustive import brute_force, compare_with_evaluation
from survey_eligibility.loader import load_inputs
from survey_eligibility.questionnaire import load_questionnaire, parse_questionnaire
from survey_eligibility.report import render_long_table, render_summary, write_outputs

from tests.helpers import EDGE, FIXTURES, ROOT, base_tiny_questionnaire, write_jsonl


class EngineFixturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = load_questionnaire(str(FIXTURES / "questionnaire.json"))
        cls.data = load_inputs(
            str(FIXTURES / "households.jsonl"),
            str(FIXTURES / "roster.jsonl"),
            str(FIXTURES / "responses.jsonl"),
            cls.q,
        )
        cls.engine = Engine(cls.q, cls.data)
        cls.ev = cls.engine.evaluate()

    def member(self, hid, mid):
        for m in self.ev.members:
            if (m.household_id, m.member_id) == (hid, mid):
                return m
        raise AssertionError(f"member {(hid, mid)} not evaluated")

    def test_adult_employee_is_must_everywhere(self):
        m = self.member("H1", "M1")
        self.assertEqual(m.complete_paths, 1)
        self.assertEqual(m.questions["age"].eligibility, MUST)
        self.assertEqual(m.questions["hours"].eligibility, MUST)
        self.assertEqual(m.questions["hours"].answer_state, VALID_ANSWER)
        # zero hours is a real answer, not missing
        self.assertEqual(m.questions["hours"].response_value, 0)
        self.assertEqual(m.questions["schooling"].eligibility, NEVER)
        self.assertEqual(m.questions["pension"].eligibility, NEVER)
        self.assertEqual(m.questions["pension"].answer_state, STRUCTURALLY_NOT_ASKED)

    def test_unknown_age_keeps_both_branches_open_and_consistent(self):
        m = self.member("H1", "M2")
        self.assertEqual(m.complete_paths, 3)  # school path, working path, non-working path
        self.assertEqual(m.questions["age"].eligibility, MUST)
        self.assertEqual(m.questions["age"].answer_state, ASKED_REFUSED)
        self.assertEqual(m.questions["schooling"].eligibility, MAY)
        self.assertEqual(m.questions["employment"].eligibility, MAY)
        self.assertEqual(m.questions["pension"].eligibility, MAY)
        self.assertEqual(m.questions["health"].eligibility, MUST)
        self.assertEqual(m.questions["hours"].eligibility, NEVER)

    def test_refused_consent_makes_member_questions_conditional_or_never(self):
        m = self.member("H2", "M1")
        # consent unknown: 1 path ends immediately + age=70 gives two member
        # routes (working / non-working, both reaching pension at 65+).
        self.assertEqual(m.complete_paths, 3)
        self.assertEqual(m.questions["age"].eligibility, MAY)
        self.assertEqual(m.questions["employment"].eligibility, MAY)
        self.assertEqual(m.questions["schooling"].eligibility, NEVER)
        self.assertEqual(m.questions["hours"].eligibility, MAY)
        self.assertEqual(m.questions["pension"].eligibility, MAY)
        self.assertEqual(m.questions["health"].eligibility, MAY)
        # Answers exist but only apply on routes that actually ask.
        self.assertEqual(m.questions["age"].answer_state, CONDITIONAL_ANSWER)
        self.assertEqual(m.questions["pension"].answer_state, CONDITIONAL_ANSWER)

    def test_consent_no_never_asks_anything(self):
        m = self.member("H3", "M1")
        self.assertEqual(m.complete_paths, 1)
        for question in ("age", "employment", "hours", "schooling", "pension", "health"):
            self.assertEqual(m.questions[question].eligibility, NEVER)
        self.assertEqual(m.questions["age"].answer_state, STRAY_ANSWER)
        self.assertEqual(m.questions["employment"].answer_state, STRAY_ANSWER)

    def test_stray_pension_never_infers_route(self):
        m = self.member("H1", "M3")  # child, pension=yes is residue
        self.assertFalse(m.questions["pension"].eligibility == MUST)
        self.assertEqual(m.questions["pension"].eligibility, NEVER)
        self.assertEqual(m.questions["pension"].answer_state, STRAY_ANSWER)
        self.assertEqual(m.questions["schooling"].eligibility, MUST)

    def test_path_evidence_includes_support_and_counterexample(self):
        m = self.member("H1", "M2")
        qe = m.questions["schooling"]
        self.assertIsNotNone(qe.supporting_path)
        self.assertIsNotNone(qe.counterexample_path)
        self.assertIn("q_schooling", qe.supporting_path["nodes"])
        self.assertNotIn("q_schooling", qe.counterexample_path["nodes"])
        # Unknown age is sampled once and reused; it must satisfy both tests.
        witness = qe.supporting_path["witness"]["member.age"]
        self.assertIn(witness["value"], range(0, 101))
        self.assertIn("domain_sampled", witness["value_source"])

    def test_unknown_member_issue_isolated_not_driving_eligibility(self):
        self.assertEqual(len(self.ev.members), 5)
        self.assertIn("unknown_member", {i.kind for i in self.data.issues})
        self.assertEqual(self.engine.responses.get(("H1", "M9", "age"), None), None)

    def test_brute_force_matches_engine(self):
        brute = brute_force(self.q, self.data)
        self.assertEqual(compare_with_evaluation(brute, self.ev), [])


class EngineTinyTest(unittest.TestCase):
    """Exhaustive check on a minimal domain: age 0..3, three questions."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.q = parse_questionnaire(base_tiny_questionnaire())
        households = [
            {"household_id": "H1", "consent": {"status": "answered", "value": "yes"}},
            {"household_id": "H2", "consent": {"status": "refused"}},
            {"household_id": "H3", "consent": {"status": "answered", "value": "no"}},
        ]
        roster = [
            {"household_id": h, "member_id": "M1"} for h in ("H1", "H2", "H3")
        ]
        responses = [
            {"household_id": "H1", "member_id": "M1", "question": "age", "status": "refused"},
            {"household_id": "H1", "member_id": "M1", "question": "schooling", "status": "answered", "value": "yes"},
            {"household_id": "H1", "member_id": "M1", "question": "pension", "status": "answered", "value": "yes"},
            {"household_id": "H3", "member_id": "M1", "question": "age", "status": "answered", "value": 2},
        ]
        cls.data = load_inputs(
            str(write_jsonl(d / "h.jsonl", households)),
            str(write_jsonl(d / "r.jsonl", roster)),
            str(write_jsonl(d / "p.jsonl", responses)),
            cls.q,
        )
        cls.engine = Engine(cls.q, cls.data)
        cls.ev = cls.engine.evaluate()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def member(self, hid, mid="M1"):
        for m in self.ev.members:
            if (m.household_id, m.member_id) == (hid, mid):
                return m
        raise AssertionError("missing member")

    def test_brute_force_all_instances(self):
        brute = brute_force(self.q, self.data)
        self.assertEqual(compare_with_evaluation(brute, self.ev), [])

    def test_unknown_age_consistent_between_ge2_and_ge3(self):
        m = self.member("H1")
        # age 0..3 unknown -> school path (age<2): 0,1; adult skips school
        # (2), pension (3).  Routes: age 0,1 -> schooling; age 2 -> no
        # schooling, no pension; age 3 -> pension.
        self.assertEqual(m.complete_paths, 3)
        self.assertEqual(m.questions["age"].eligibility, MUST)
        self.assertEqual(m.questions["age"].answer_state, ASKED_REFUSED)
        self.assertEqual(m.questions["schooling"].eligibility, MAY)
        self.assertEqual(m.questions["pension"].eligibility, MAY)
        self.assertEqual(m.questions["pension"].answer_state, CONDITIONAL_ANSWER)

    def test_consent_refused(self):
        m = self.member("H2")
        self.assertEqual(m.complete_paths, 4)  # no + 3 member routes
        for question in self.q.member_questions:
            self.assertIn(m.questions[question].eligibility, (MAY, NEVER))
        self.assertEqual(m.questions["schooling"].eligibility, MAY)

    def test_consent_no_leaves_answers_as_residue(self):
        m = self.member("H3")
        self.assertEqual(m.complete_paths, 1)
        self.assertEqual(m.questions["age"].eligibility, NEVER)
        self.assertEqual(m.questions["age"].answer_state, STRAY_ANSWER)

    def test_missing_and_refused_are_distinct_states(self):
        self.assertEqual(self.member("H1").questions["age"].answer_state, ASKED_REFUSED)

    def test_must_denominator_and_may_separation(self):
        summary = render_summary(self.q, self.data, self.ev, self.engine)
        age = summary["per_question"]["age"]
        # H1 age refused (MUST), H2 consent refused -> age MAY, H3 consent no -> NEVER.
        self.assertEqual(age["eligibility_counts"], {MUST: 1, MAY: 1, NEVER: 1})
        self.assertEqual(age["denominators"]["asked_at_least_some_paths"], 2)
        self.assertEqual(age["response_rates"]["valid_answer_among_must"]["denominator"], 1)
        self.assertEqual(age["response_rates"]["valid_answer_among_must"]["numerator"], 0)
        schooling = summary["per_question"]["schooling"]
        # H1 age unknown -> MAY; H2 consent refused -> MAY; H3 consent no -> NEVER.
        self.assertEqual(schooling["eligibility_counts"], {MUST: 0, MAY: 2, NEVER: 1})
        self.assertEqual(schooling["response_rates"]["conditional_answer_among_may"]["denominator"], 2)


class ConditionalReferenceTest(unittest.TestCase):
    """A branch may test a question asked on only some incoming routes.

    On routes that never asked it the value is unknown (full domain) and must
    stay internally consistent; it is never borrowed from another member.
    """

    def test_conditional_reference_matches_brute_force(self):
        raw = base_tiny_questionnaire()
        raw["nodes"]["post"] = {
            "kind": "branch",
            "test": {"ref": "member.schooling", "op": "eq", "value": "yes"},
            "true": "end",
            "false": "end",
        }
        raw["nodes"]["q_pension"]["next"] = "post"
        q = parse_questionnaire(raw)
        self.assertTrue(any("conditional reference" in w for w in q.warnings), q.warnings)

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            data = load_inputs(
                str(write_jsonl(d / "h.jsonl", [
                    {"household_id": "H1", "consent": {"status": "answered", "value": "yes"}},
                ])),
                str(write_jsonl(d / "r.jsonl", [{"household_id": "H1", "member_id": "M1"}])),
                str(write_jsonl(d / "p.jsonl", [
                    {"household_id": "H1", "member_id": "M1", "question": "age", "status": "refused"},
                ])),
                q,
            )
            ev = Engine(q, data).evaluate()
            member = ev.members[0]
            # age 0,1 -> ask schooling; 2 -> skip; 3 -> skip + pension.
            self.assertEqual(member.complete_paths, 4)
            self.assertEqual(member.questions["schooling"].eligibility, MAY)
            brute = brute_force(q, data)
            self.assertEqual(compare_with_evaluation(brute, ev), [])


class OutputContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = load_questionnaire(str(FIXTURES / "questionnaire.json"))
        cls.data = load_inputs(
            str(FIXTURES / "households.jsonl"),
            str(FIXTURES / "roster.jsonl"),
            str(FIXTURES / "responses.jsonl"),
            cls.q,
        )
        cls.ev = Engine(cls.q, cls.data).evaluate()

    def test_long_table_has_contract_key_and_states(self):
        text, rows = render_long_table(self.ev, self.q)
        header = next(csv.reader(io.StringIO(text)))
        for column in ("household_id", "member_id", "question"):
            self.assertIn(column, header)
        states = {row["answer_state"] for row in rows}
        self.assertTrue(states <= {
            VALID_ANSWER, ASKED_MISSING, ASKED_REFUSED,
            CONDITIONAL_ANSWER, STRUCTURALLY_NOT_ASKED, STRAY_ANSWER,
        })
        for row in rows:
            self.assertIn(row["eligibility"], (MUST, MAY, NEVER))
            self.assertIn("denominator", row["denominator_note"])

    def test_summary_separates_must_may_never(self):
        summary = render_summary(self.q, self.data, self.ev, Engine(self.q, self.data))
        for question, block in summary["per_question"].items():
            counts = block["eligibility_counts"]
            self.assertEqual(sum(counts.values()), len(self.ev.members))
            self.assertIn("denominators", block)
            self.assertIn("response_rates", block)

    def test_write_outputs_creates_three_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_outputs(tmp, self.q, self.data, self.ev, Engine(self.q, self.data))
            self.assertEqual(set(paths), {"long_table", "path_evidence", "summary"})
            for path in paths.values():
                self.assertTrue(Path(path).exists(), path)
            summary = json.loads(Path(paths["summary"]).read_text())
            self.assertEqual(summary["issues"]["count"], 1)
            lines = Path(paths["path_evidence"]).read_text().splitlines()
            self.assertEqual(len(lines), len(self.ev.members) * len(self.q.member_questions))

    def test_original_inputs_untouched(self):
        # The tool must not rewrite fixtures; assert byte-identical content.
        original = (FIXTURES / "responses.jsonl").read_bytes()
        self.q2 = load_questionnaire(str(FIXTURES / "questionnaire.json"))
        load_inputs(
            str(FIXTURES / "households.jsonl"),
            str(FIXTURES / "roster.jsonl"),
            str(FIXTURES / "responses.jsonl"),
            self.q2,
        )
        self.assertEqual((FIXTURES / "responses.jsonl").read_bytes(), original)


class EdgeCasesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = load_questionnaire(str(FIXTURES / "questionnaire.json"))
        cls.data = load_inputs(
            str(EDGE / "households.jsonl"),
            str(EDGE / "roster.jsonl"),
            str(EDGE / "responses.jsonl"),
            cls.q,
        )
        cls.ev = Engine(cls.q, cls.data).evaluate()

    def member(self, hid, mid="M1"):
        for m in self.ev.members:
            if (m.household_id, m.member_id) == (hid, mid):
                return m
        raise AssertionError("missing member")

    def test_unknown_age_and_refused_consent(self):
        self.assertEqual(self.member("E1").questions["age"].answer_state, ASKED_REFUSED)
        self.assertEqual(self.member("E1").questions["age"].eligibility, MUST)
        e2 = self.member("E2")
        self.assertEqual(e2.complete_paths, 6)
        self.assertEqual(e2.questions["age"].eligibility, MAY)
        e3 = self.member("E3")   # consent missing behaves like unknown
        self.assertEqual(e3.complete_paths, 6)
        self.assertEqual(e3.questions["age"].eligibility, MAY)

    def test_cross_branch_convergence_stays_must(self):
        m = self.member("E1")
        # health is reached from both the schooling and the employment branch.
        self.assertEqual(m.questions["health"].eligibility, MUST)
        self.assertEqual(m.questions["health"].answer_state, VALID_ANSWER)

    def test_residue_pension_for_child(self):
        m = self.member("E4", "M2")
        self.assertEqual(m.questions["pension"].eligibility, NEVER)
        self.assertEqual(m.questions["pension"].answer_state, STRAY_ANSWER)
        self.assertEqual(m.questions["schooling"].eligibility, MUST)

    def test_same_member_id_two_households_distinct(self):
        m1 = self.member("E4", "M1")
        m2 = self.member("E4", "M2")
        self.assertEqual(m1.questions["age"].response_value, 70)
        self.assertEqual(m2.questions["age"].response_value, 12)
        self.assertEqual(m1.questions["pension"].eligibility, MUST)
        self.assertEqual(m2.questions["pension"].eligibility, NEVER)

    def test_brute_force_matches(self):
        brute = brute_force(self.q, self.data)
        self.assertEqual(compare_with_evaluation(brute, self.ev), [])


if __name__ == "__main__":
    unittest.main()
