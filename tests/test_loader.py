"""Loader: error isolation, duplicate resolution, instance identity."""

import tempfile
import unittest
from pathlib import Path

from survey_eligibility.loader import load_inputs
from survey_eligibility.questionnaire import parse_questionnaire

from tests.helpers import FIXTURES, base_tiny_questionnaire, write_jsonl, write_json


class LoaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.q = parse_questionnaire(base_tiny_questionnaire())

    def tearDown(self):
        self.tmp.cleanup()

    def _load(self, households, roster, responses):
        hp = write_jsonl(self.dir / "h.jsonl", households)
        rp = write_jsonl(self.dir / "r.jsonl", roster)
        pp = write_jsonl(self.dir / "p.jsonl", responses)
        return load_inputs(str(hp), str(rp), str(pp), self.q)

    def test_unknown_member_response_is_isolated(self):
        data = self._load(
            [{"household_id": "H1", "consent": {"status": "answered", "value": "yes"}}],
            [{"household_id": "H1", "member_id": "M1"}],
            [{"household_id": "H1", "member_id": "M9", "question": "age", "status": "answered", "value": 2}],
        )
        kinds = {issue.kind for issue in data.issues}
        self.assertIn("unknown_member", kinds)
        self.assertEqual(len(data.responses), 1, "unknown member row stays visible, not deleted")
        self.assertIn(1, data.excluded_response_lines)

    def test_out_of_domain_value_isolated(self):
        data = self._load(
            [{"household_id": "H1", "consent": {"status": "answered", "value": "yes"}}],
            [{"household_id": "H1", "member_id": "M1"}],
            [{"household_id": "H1", "member_id": "M1", "question": "age", "status": "answered", "value": 999}],
        )
        self.assertEqual([i.kind for i in data.issues if i.kind == "out_of_domain"], ["out_of_domain"])
        self.assertFalse(data.responses[0].usable)
        self.assertIn(1, data.excluded_response_lines)

    def test_wrong_question_scope_isolated(self):
        data = self._load(
            [{"household_id": "H1", "consent": {"status": "answered", "value": "yes"}}],
            [{"household_id": "H1", "member_id": "M1"}],
            [{"household_id": "H1", "member_id": "M1", "question": "consent", "status": "answered", "value": "yes"}],
        )
        self.assertIn("wrong_question", {i.kind for i in data.issues})

    def test_conflicting_duplicates_isolate_whole_key(self):
        data = self._load(
            [{"household_id": "H1", "consent": {"status": "answered", "value": "yes"}}],
            [{"household_id": "H1", "member_id": "M1"}],
            [
                {"household_id": "H1", "member_id": "M1", "question": "age", "status": "answered", "value": 1},
                {"household_id": "H1", "member_id": "M1", "question": "age", "status": "answered", "value": 2},
            ],
        )
        conflict = [i for i in data.issues if i.kind == "conflicting_duplicate_answer"]
        self.assertEqual(len(conflict), 1)
        self.assertEqual(conflict[0].line, 1)
        # Neither value is allowed to drive eligibility.
        self.assertTrue(all(r.line in data.excluded_response_lines for r in data.responses))

    def test_identical_duplicates_collapse(self):
        data = self._load(
            [{"household_id": "H1", "consent": {"status": "answered", "value": "yes"}}],
            [{"household_id": "H1", "member_id": "M1"}],
            [
                {"household_id": "H1", "member_id": "M1", "question": "age", "status": "answered", "value": 2},
                {"household_id": "H1", "member_id": "M1", "question": "age", "status": "answered", "value": 2},
            ],
        )
        usable = [r for r in data.responses if r.line not in data.excluded_response_lines]
        self.assertEqual(len(usable), 1)
        self.assertEqual(usable[0].value, 2)
        self.assertEqual([i for i in data.issues if i.kind == "conflicting_duplicate_answer"], [])

    def test_duplicate_roster_member_isolated(self):
        data = self._load(
            [{"household_id": "H1", "consent": {"status": "answered", "value": "yes"}}],
            [
                {"household_id": "H1", "member_id": "M1"},
                {"household_id": "H1", "member_id": "M1"},
            ],
            [],
        )
        self.assertIn("duplicate_roster_member", {i.kind for i in data.issues})
        self.assertEqual(len(data.members), 1)

    def test_same_member_id_across_households_is_distinct(self):
        data = self._load(
            [
                {"household_id": "H1", "consent": {"status": "answered", "value": "yes"}},
                {"household_id": "H2", "consent": {"status": "answered", "value": "no"}},
            ],
            [
                {"household_id": "H1", "member_id": "M1"},
                {"household_id": "H2", "member_id": "M1"},
            ],
            [
                {"household_id": "H1", "member_id": "M1", "question": "age", "status": "answered", "value": 1},
                {"household_id": "H2", "member_id": "M1", "question": "age", "status": "answered", "value": 3},
            ],
        )
        self.assertEqual(len(data.members), 2)
        self.assertEqual(len([r for r in data.responses if r.line not in data.excluded_response_lines]), 2)

    def test_malformed_response_status_isolated(self):
        data = self._load(
            [{"household_id": "H1", "consent": {"status": "answered", "value": "yes"}}],
            [{"household_id": "H1", "member_id": "M1"}],
            [{"household_id": "H1", "member_id": "M1", "question": "age", "status": "weird", "value": 1}],
        )
        self.assertIn("malformed_response", {i.kind for i in data.issues})

    def test_missing_or_refused_with_value_is_flagged_but_value_ignored(self):
        data = self._load(
            [{"household_id": "H1", "consent": {"status": "answered", "value": "yes"}}],
            [{"household_id": "H1", "member_id": "M1"}],
            [{"household_id": "H1", "member_id": "M1", "question": "age", "status": "refused", "value": 2}],
        )
        self.assertIn("malformed_response", {i.kind for i in data.issues})
        usable = [r for r in data.responses if r.usable]
        self.assertEqual(usable[0].value, None)


if __name__ == "__main__":
    unittest.main()
