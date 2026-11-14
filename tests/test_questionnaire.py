"""Questionnaire structure: types, reference scope, jump targets, cycles."""

import json
import unittest
from pathlib import Path

from survey_eligibility.questionnaire import (
    QuestionnaireValidationError,
    load_questionnaire,
    parse_questionnaire,
)

from tests.helpers import FIXTURES, ROOT, base_tiny_questionnaire


def _parse(raw):
    return parse_questionnaire(raw)


class FrozenQuestionnaireTest(unittest.TestCase):
    def test_frozen_questionnaire_loads_and_reports_questions(self):
        q = _parse(json.loads((FIXTURES / "questionnaire.json").read_text()))
        self.assertEqual(q.start, "consent_gate")
        self.assertEqual(
            q.member_asked_questions(),
            ["age", "employment", "hours", "schooling", "pension", "health"],
        )
        self.assertEqual(q.warnings, [])

    def test_cyclic_questionnaire_rejected(self):
        raw = json.loads((FIXTURES / "cases" / "cyclic-questionnaire.json").read_text())
        with self.assertRaises(QuestionnaireValidationError) as ctx:
            _parse(raw)
        self.assertIn("cycle", " ".join(ctx.exception.problems))


class StructuralValidationTest(unittest.TestCase):
    def test_dangling_true_target_rejected(self):
        raw = base_tiny_questionnaire()
        raw["nodes"]["gate"]["true"] = "does_not_exist"
        with self.assertRaises(QuestionnaireValidationError) as ctx:
            _parse(raw)
        self.assertIn("undefined", " ".join(ctx.exception.problems))

    def test_dangling_ask_next_rejected(self):
        raw = base_tiny_questionnaire()
        raw["nodes"]["q_age"]["next"] = "missing_node"
        with self.assertRaises(QuestionnaireValidationError) as ctx:
            _parse(raw)
        self.assertIn("undefined", " ".join(ctx.exception.problems))

    def test_bad_start_rejected(self):
        raw = base_tiny_questionnaire()
        raw["start"] = "nope"
        with self.assertRaises(QuestionnaireValidationError):
            _parse(raw)

    def test_member_reference_to_undeclared_question_rejected(self):
        raw = base_tiny_questionnaire()
        raw["nodes"]["adult"]["test"]["ref"] = "member.hours"
        with self.assertRaises(QuestionnaireValidationError) as ctx:
            _parse(raw)
        self.assertIn("not a declared member question", " ".join(ctx.exception.problems))

    def test_household_reference_to_member_question_rejected(self):
        raw = base_tiny_questionnaire()
        raw["nodes"]["gate"]["test"]["ref"] = "household.age"
        with self.assertRaises(QuestionnaireValidationError) as ctx:
            _parse(raw)
        self.assertIn("not a declared field", " ".join(ctx.exception.problems))

    def test_reference_to_question_never_asked_before_rejected(self):
        raw = base_tiny_questionnaire()
        # gate runs before age is ever asked, so it cannot test member.age.
        raw["nodes"]["gate"]["test"]["ref"] = "member.age"
        with self.assertRaises(QuestionnaireValidationError) as ctx:
            _parse(raw)
        self.assertIn("never asked", " ".join(ctx.exception.problems))

    def test_wrong_operator_for_integer_rejected(self):
        raw = base_tiny_questionnaire()
        raw["nodes"]["adult"]["test"]["op"] = "eq"
        with self.assertRaises(QuestionnaireValidationError) as ctx:
            _parse(raw)
        self.assertIn("only ge/lt", " ".join(ctx.exception.problems))

    def test_test_value_outside_domain_rejected(self):
        raw = base_tiny_questionnaire()
        raw["nodes"]["adult"]["test"]["value"] = 99
        with self.assertRaises(QuestionnaireValidationError) as ctx:
            _parse(raw)
        self.assertIn("outside the declared domain", " ".join(ctx.exception.problems))

    def test_enum_domain_must_be_nonempty_and_typed(self):
        raw = base_tiny_questionnaire()
        raw["member_questions"]["schooling"]["values"] = []
        with self.assertRaises(QuestionnaireValidationError):
            _parse(raw)
        raw = base_tiny_questionnaire()
        raw["member_questions"]["schooling"]["values"] = ["yes", 1]
        with self.assertRaises(QuestionnaireValidationError):
            _parse(raw)

    def test_integer_bounds_must_be_consistent(self):
        raw = base_tiny_questionnaire()
        raw["member_questions"]["age"]["minimum"] = 5
        with self.assertRaises(QuestionnaireValidationError):
            _parse(raw)

    def test_unknown_kind_rejected(self):
        raw = base_tiny_questionnaire()
        raw["nodes"]["q_age"]["kind"] = "loop"
        with self.assertRaises(QuestionnaireValidationError) as ctx:
            _parse(raw)
        self.assertIn("unsupported kind", " ".join(ctx.exception.problems))

    def test_asking_household_field_as_member_question_rejected(self):
        raw = base_tiny_questionnaire()
        raw["nodes"]["q_pension"]["question"] = "consent"
        with self.assertRaises(QuestionnaireValidationError):
            _parse(raw)

    def test_valid_tiny_questionnaire_has_declared_domain_sizes(self):
        q = _parse(base_tiny_questionnaire())
        self.assertEqual(q.member_questions["age"].domain_size, 4)
        self.assertEqual(q.member_questions["schooling"].domain(), ["yes", "no"])

    def test_conditional_reference_is_warning_not_error(self):
        # Branch references a question asked on only one of two incoming paths.
        raw = base_tiny_questionnaire()
        raw["nodes"]["post"] = {
            "kind": "branch",
            "test": {"ref": "member.schooling", "op": "eq", "value": "yes"},
            "true": "end",
            "false": "end",
        }
        # old -> q_pension (after school or adult) and post, so schooling may
        # not have been asked (adult branch skips it).
        raw["nodes"]["q_pension"]["next"] = "post"
        q = _parse(raw)
        self.assertTrue(any("conditional reference" in w for w in q.warnings), q.warnings)


if __name__ == "__main__":
    unittest.main()
