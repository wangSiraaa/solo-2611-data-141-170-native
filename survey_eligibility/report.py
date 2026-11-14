"""Output renderers: long table (CSV), path evidence (JSONL) and summary JSON.

Every rate carries an explicit denominator.  MAY instances are never merged
silently into either "eligible" or "not eligible": MUST, MAY and NEVER counts
are reported side by side and each rate names its own base.
"""
from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List, Tuple

from .engine import (
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
    Evaluation,
    MemberEvaluation,
)
from .loader import InputData
from .questionnaire import Questionnaire

LONG_TABLE_COLUMNS = [
    "household_id",
    "member_id",
    "roster_line",
    "question",
    "eligibility",
    "complete_paths",
    "paths_visiting_question",
    "paths_avoiding_question",
    "answer_state",
    "observed_status",
    "has_usable_answer",
    "response_line",
    "response_value",
    "answer_state_definition",
    "admissible_path_exists",
    "denominator_note",
]

ANSWER_STATE_DEFINITIONS = {
    VALID_ANSWER: "answer present for a question asked on every admissible path (MUST)",
    ASKED_MISSING: "question is visited on at least one admissible path but no usable answer exists",
    ASKED_REFUSED: "question is visited on at least one admissible path and the respondent refused",
    CONDITIONAL_ANSWER: (
        "answer present for a MAY question: used only on paths that actually reach it"
    ),
    STRUCTURALLY_NOT_ASKED: "no admissible path reaches the question and no answer exists",
    STRAY_ANSWER: (
        "no admissible path reaches the question, yet an answer exists; the residual "
        "answer is preserved but never used to infer eligibility"
    ),
}

ELIGIBILITY_DEFINITIONS = {
    MUST: "visited by every admissible complete path",
    MAY: "visited by some admissible complete paths and not by others",
    NEVER: "visited by no admissible complete path",
}


def _long_row(member: MemberEvaluation, qe) -> Dict[str, Any]:
    return {
        "household_id": member.household_id,
        "member_id": member.member_id,
        "roster_line": member.roster_line,
        "question": qe.question,
        "eligibility": qe.eligibility,
        "complete_paths": qe.complete_paths,
        "paths_visiting_question": qe.visiting_paths,
        "paths_avoiding_question": qe.avoiding_paths,
        "answer_state": qe.answer_state,
        "observed_status": qe.observed_status,
        "has_usable_answer": qe.has_usable_answer,
        "response_line": "" if qe.response_line is None else qe.response_line,
        "response_value": "" if qe.response_value is None else qe.response_value,
        "answer_state_definition": ANSWER_STATE_DEFINITIONS[qe.answer_state],
        "admissible_path_exists": qe.complete_paths > 0,
        "denominator_note": (
            f"eligibility denominator = {qe.complete_paths} admissible complete path(s); "
            f"paths visiting this question = {qe.visiting_paths}; "
            f"paths avoiding it = {qe.avoiding_paths}"
        ),
    }


def render_long_table(evaluation: Evaluation, questionnaire: Questionnaire) -> Tuple[str, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for member in evaluation.members:
        for question in questionnaire.member_questions:
            qe = member.questions[question]
            rows.append(_long_row(member, qe))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=LONG_TABLE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue(), rows


def render_path_evidence(evaluation: Evaluation, questionnaire: Questionnaire) -> str:
    lines: List[str] = []
    for member in evaluation.members:
        for question in questionnaire.member_questions:
            qe = member.questions[question]
            record = {
                "household_id": member.household_id,
                "member_id": member.member_id,
                "roster_line": member.roster_line,
                "question": question,
                "eligibility": qe.eligibility,
                "eligibility_definition": ELIGIBILITY_DEFINITIONS[qe.eligibility],
                "complete_paths": qe.complete_paths,
                "supporting_path": qe.supporting_path,
                "counterexample_path": qe.counterexample_path,
            }
            lines.append(json.dumps(record, ensure_ascii=False, sort_keys=False))
    return "\n".join(lines) + ("\n" if lines else "")


def _rate(numerator: int, denominator: int) -> Dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": None if denominator == 0 else round(numerator / denominator, 6),
    }


def render_summary(
    questionnaire: Questionnaire,
    data: InputData,
    evaluation: Evaluation,
    engine: Engine,
) -> Dict[str, Any]:
    questions = list(questionnaire.member_questions)
    per_question: Dict[str, Any] = {}
    for question in questions:
        must_instances: List[MemberEvaluation] = []
        may_instances: List[MemberEvaluation] = []
        never_instances: List[MemberEvaluation] = []
        state_counts = {state: 0 for state in ANSWER_STATE_DEFINITIONS}
        observed_counts: Dict[str, int] = {}
        for member in evaluation.members:
            qe = member.questions[question]
            if qe.eligibility == MUST:
                must_instances.append(member)
            elif qe.eligibility == MAY:
                may_instances.append(member)
            else:
                never_instances.append(member)
            state_counts[qe.answer_state] += 1
            observed_counts[qe.observed_status] = observed_counts.get(qe.observed_status, 0) + 1

        must_n = len(must_instances)
        may_n = len(may_instances)
        never_n = len(never_instances)
        # A usable answer for a MUST question is a valid answer; for a MAY
        # question it is conditional and cannot enter the MUST response rate.
        must_valid = state_counts[VALID_ANSWER]
        may_conditional = state_counts[CONDITIONAL_ANSWER]
        # "Answered at all" among MAY counts conditional answers plus any
        # residual/closed answer states? Only conditional answers are legitimate
        # MAY answers; stray answers are counted separately below.
        per_question[question] = {
            "eligibility_counts": {MUST: must_n, MAY: may_n, NEVER: never_n},
            "evaluated_instances": len(evaluation.members),
            "instances_without_admissible_path": [
                f"{m.household_id}/{m.member_id}"
                for m in evaluation.members
                if m.no_admissible_path
            ],
            "answer_states": state_counts,
            "observed_status_counts": observed_counts,
            "denominators": {
                "must": must_n,
                "may": may_n,
                "never": never_n,
                "asked_at_least_some_paths": must_n + may_n,
                "all_instances": len(evaluation.members),
            },
            "response_rates": {
                "valid_answer_among_must": _rate(must_valid, must_n),
                "conditional_answer_among_may": _rate(may_conditional, may_n),
                "valid_or_conditional_among_asked_at_least_some_paths": _rate(
                    must_valid + may_conditional, must_n + may_n
                ),
                "note": (
                    "MAY instances are reported in their own rate and are neither added "
                    "to nor removed from the MUST denominator."
                ),
            },
            "members": {
                MUST: [f"{m.household_id}/{m.member_id}" for m in must_instances],
                MAY: [f"{m.household_id}/{m.member_id}" for m in may_instances],
                NEVER: [f"{m.household_id}/{m.member_id}" for m in never_instances],
            },
        }

    issues = [issue.to_dict() for issue in data.issues]
    issue_counts: Dict[str, int] = {}
    for issue in issues:
        issue_counts[issue["kind"]] = issue_counts.get(issue["kind"], 0) + 1

    return {
        "questionnaire": {
            "version": questionnaire.version,
            "start": questionnaire.start,
            "member_questions": questions,
            "household_fields": list(questionnaire.household_fields),
            "eligibility_definitions": ELIGIBILITY_DEFINITIONS,
            "answer_state_definitions": ANSWER_STATE_DEFINITIONS,
            "warnings": questionnaire.warnings,
        },
        "instances": {
            "households": len(data.households),
            "roster_members_evaluated": len(evaluation.members),
            "roster_members_skipped": len(evaluation.skipped_members),
            "skipped": evaluation.skipped_members,
        },
        "per_question": per_question,
        "issues": {
            "count": len(issues),
            "by_kind": issue_counts,
            "items": issues,
        },
        "isolation_notes": [
            "Every input file is read-only; no answer is imputed and no questionnaire "
            "is edited.",
            "Instance identity is (household_id, member_id); the same member_id in "
            "different households is a different instance.",
            "An answer is consulted only on paths whose walk reaches its ask node.",
            "Unknown values stay inside their declared domain; a value constrained by "
            "several tests on one path is a single consistent value.",
            "Out-of-domain values, unknown roster members, wrong-scope questions and "
            "conflicting duplicates are isolated with source line references and never "
            "used to derive eligibility.",
        ],
    }


def write_outputs(
    out_dir: str,
    questionnaire: Questionnaire,
    data: InputData,
    evaluation: Evaluation,
    engine: Engine,
) -> Dict[str, str]:
    import os

    os.makedirs(out_dir, exist_ok=True)
    long_csv, _ = render_long_table(evaluation, questionnaire)
    paths_jsonl = render_path_evidence(evaluation, questionnaire)
    summary = render_summary(questionnaire, data, evaluation, engine)
    paths = {
        "long_table": os.path.join(out_dir, "long_table.csv"),
        "path_evidence": os.path.join(out_dir, "path_evidence.jsonl"),
        "summary": os.path.join(out_dir, "summary.json"),
    }
    with open(paths["long_table"], "w", encoding="utf-8", newline="") as fh:
        fh.write(long_csv)
    with open(paths["path_evidence"], "w", encoding="utf-8") as fh:
        fh.write(paths_jsonl)
    with open(paths["summary"], "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return paths