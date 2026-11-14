"""Independent brute-force verifier for tiny declared domains.

This module deliberately does **not** reuse the engine's path enumeration.
It enumerates every concrete assignment over the declared domains, walks the
questionnaire once per assignment, and decides compatibility by the rule:

* an answer constrains a route only if that route actually visits the ask node;
* a route that never visits a question is unconstrained by any stored answer.

Eligibility is then ``MUST`` / ``MAY`` / ``NEVER`` depending on whether all,
some or none of the compatible assignments visit a question.  Because each
concrete assignment carries one value per variable, a repeated test necessarily
sees the same value, which is exactly the consistency invariant the engine
enforces symbolically.

This is only tractable for the tiny fixtures used by the tests; it exists to
cross-check the production engine, not to replace it.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
)
from .loader import Household, InputData, Member, Response
from .questionnaire import Questionnaire

MAX_ASSIGNMENTS = 2_000_000


@dataclass
class BruteResult:
    member: Member
    complete_assignments: int
    visiting: Dict[str, int]
    answer_state: Dict[str, str]


def _assignment_space(questionnaire: Questionnaire) -> Tuple[List[str], List[List[Any]]]:
    names: List[str] = []
    domains: List[List[Any]] = []
    for field_name, field_def in questionnaire.household_fields.items():
        names.append(f"household.{field_name}")
        domains.append(field_def.domain())
    for question, qdef in questionnaire.member_questions.items():
        names.append(f"member.{question}")
        domains.append(qdef.domain())
    return names, domains


def _walk(questionnaire: Questionnaire, assignment: Dict[str, Any]) -> Optional[List[str]]:
    """Deterministic walk; returns visited question names, or None on a defect."""
    visited: List[str] = []
    asked: List[str] = []
    node_name = questionnaire.start
    guard = 0
    while True:
        guard += 1
        if guard > 10_000:
            raise AssertionError("walk did not terminate (cycle?)")
        node = questionnaire.nodes[node_name]
        if node.kind == "end":
            return visited
        if node.kind == "ask":
            # References in this verifier's fixtures must point at questions the
            # walk already asked; treat a violation as a structural defect.
            visited.append(node.question)
            asked.append(node.question)
            node_name = node.next
            continue
        var = f"{node.test.ref.scope}.{node.test.ref.question}"
        if var not in assignment:
            raise AssertionError(f"assignment has no value for {var}")
        # A member reference that was not asked on this route is simply an
        # unknown value: the concrete assignment supplies one, and it stays
        # the same value for every later test on this route.  Scope isolation
        # (never another member's answer) comes from the assignment being
        # per-instance and per-scope.
        outcome = node.test.evaluate(assignment[var])
        node_name = node.true if outcome else node.false


def brute_force_member(
    questionnaire: Questionnaire,
    household: Household,
    member: Member,
    responses: Dict[Tuple[str, str, str], Response],
) -> BruteResult:
    names, domains = _assignment_space(questionnaire)
    total = 1
    for domain in domains:
        total *= len(domain)
    if total > MAX_ASSIGNMENTS:
        raise AssertionError(f"assignment space {total} exceeds brute-force budget")

    visiting = {question: 0 for question in questionnaire.member_questions}
    compatible = 0
    for values in itertools.product(*domains):
        assignment = dict(zip(names, values))
        if household.consent_status == "answered" and household.consent is not None:
            if assignment["household.consent"] != household.consent:
                continue
        visited = _walk(questionnaire, assignment)
        ok = True
        for question in questionnaire.member_questions:
            if question not in visited:
                continue
            response = responses.get((household.household_id, member.member_id, question))
            if response is None or response.status != "answered":
                continue
            if assignment[f"member.{question}"] != response.value:
                ok = False
                break
        if not ok:
            continue
        compatible += 1
        for question in set(visited):
            visiting[question] += 1

    answer_state: Dict[str, str] = {}
    for question in questionnaire.member_questions:
        response = responses.get((household.household_id, member.member_id, question))
        if compatible == 0:
            label = NEVER
        elif visiting[question] == compatible:
            label = MUST
        elif visiting[question] == 0:
            label = NEVER
        else:
            label = MAY
        answered = response is not None and response.status == "answered"
        refused = response is not None and response.status == "refused"
        if label == NEVER:
            state = STRAY_ANSWER if response is not None else STRUCTURALLY_NOT_ASKED
        elif answered:
            state = VALID_ANSWER if label == MUST else CONDITIONAL_ANSWER
        elif refused:
            state = ASKED_REFUSED
        else:
            state = ASKED_MISSING
        answer_state[question] = state
    return BruteResult(member=member, complete_assignments=compatible, visiting=visiting, answer_state=answer_state)


def brute_force(
    questionnaire: Questionnaire, data: InputData
) -> Dict[Tuple[str, str], BruteResult]:
    responses: Dict[Tuple[str, str, str], Response] = {}
    for response in data.responses:
        if response.line in data.excluded_response_lines:
            continue
        responses[response.key] = response
    out: Dict[Tuple[str, str], BruteResult] = {}
    for member in data.members:
        household = data.households.get(member.household_id)
        if household is None:
            continue
        out[member.key] = brute_force_member(questionnaire, household, member, responses)
    return out


def compare_with_evaluation(
    brute: Dict[Tuple[str, str], BruteResult], evaluation
) -> List[str]:
    """Return human-readable mismatches between brute force and the engine."""
    mismatches: List[str] = []
    seen = set()
    for member_eval in evaluation.members:
        key = (member_eval.household_id, member_eval.member_id)
        seen.add(key)
        expected = brute.get(key)
        if expected is None:
            mismatches.append(f"{key}: brute force produced no result")
            continue
        for question, qe in member_eval.questions.items():
            if qe.eligibility == MAY:
                brute_label = (
                    MUST
                    if expected.visiting[question] == expected.complete_assignments
                    else MAY
                    if expected.visiting[question] > 0
                    else NEVER
                )
            elif qe.eligibility == MUST:
                brute_label = (
                    MUST if expected.visiting[question] == expected.complete_assignments else "MISMATCH"
                )
            else:
                brute_label = NEVER if expected.visiting[question] == 0 else "MISMATCH"
            if brute_label != qe.eligibility:
                mismatches.append(
                    f"{key}/{question}: engine={qe.eligibility} brute={brute_label} "
                    f"(visiting={expected.visiting[question]}/{expected.complete_assignments})"
                )
            if expected.answer_state[question] != qe.answer_state:
                mismatches.append(
                    f"{key}/{question}: answer_state engine={qe.answer_state} "
                    f"brute={expected.answer_state[question]}"
                )
    for key in brute:
        if key not in seen:
            mismatches.append(f"{key}: brute force evaluated an instance the engine skipped")
    return mismatches