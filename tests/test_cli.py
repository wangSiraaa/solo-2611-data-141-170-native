"""CLI behaviour: structural rejection must happen before any answer is read."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from survey_eligibility import cli
from survey_eligibility import loader as loader_module

from tests.helpers import FIXTURES


class CliTest(unittest.TestCase):
    def test_cyclic_questionnaire_rejected_before_reading_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                loader_module, "load_inputs", side_effect=AssertionError("answers were read")
            ):
                code = cli.main(
                    [
                        "--questionnaire", str(FIXTURES / "cases" / "cyclic-questionnaire.json"),
                        "--households", str(FIXTURES / "households.jsonl"),
                        "--roster", str(FIXTURES / "roster.jsonl"),
                        "--responses", str(FIXTURES / "responses.jsonl"),
                        "--output-dir", tmp,
                    ]
                )
            self.assertEqual(code, 2)
            self.assertFalse((Path(tmp) / "long_table.csv").exists())

    def test_valid_run_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = cli.main(
                [
                    "--questionnaire", str(FIXTURES / "questionnaire.json"),
                    "--households", str(FIXTURES / "households.jsonl"),
                    "--roster", str(FIXTURES / "roster.jsonl"),
                    "--responses", str(FIXTURES / "responses.jsonl"),
                    "--output-dir", tmp,
                ]
            )
            self.assertEqual(code, 0)
            for name in ("long_table.csv", "path_evidence.jsonl", "summary.json"):
                self.assertTrue((Path(tmp) / name).exists(), name)
            summary = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertEqual(summary["instances"]["roster_members_evaluated"], 5)

    def test_missing_questionnaire_file_reports_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = cli.main(
                [
                    "--questionnaire", "/no/such/questionnaire.json",
                    "--households", str(FIXTURES / "households.jsonl"),
                    "--roster", str(FIXTURES / "roster.jsonl"),
                    "--responses", str(FIXTURES / "responses.jsonl"),
                    "--output-dir", tmp,
                ]
            )
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
