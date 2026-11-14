"""Input loading with per-row error isolation.

Responses are never discarded silently.  A response that cannot be used for
*eligibility* is still preserved as a record and surfaced as an isolated
error with its source line, so unknown members, out-of-domain values, wrong
questions and conflicting duplicates stay visible without contaminating the
eligible set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .questionnaire import Questionnaire


@dataclass(frozen=True)
class Household:
    household_id: str
    consent: Optional[str]      # None means missing or refused -> whole domain
    consent_status: str         # answered / missing / refused
    line: int


@dataclass(frozen=True)
class Member:
    household_id: str
    member_id: str
    line: int

    @property
    def key(self) -> Tuple[str, str]:
        return (self.household_id, self.member_id)


@dataclass(frozen=True)
class Response:
    household_id: str
    member_id: str
    question: str
    status: str                 # answered / missing / refused
    value: Any                  # only meaningful when status == answered
    line: int
    raw: Dict[str, Any] = field(default_factory=dict, compare=False)
    usable: bool = True         # False when the value failed domain validation

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.household_id, self.member_id, self.question)


@dataclass(frozen=True)
class Issue:
    kind: str
    message: str
    source: str                 # file path
    line: int
    record: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "source": self.source,
            "line": self.line,
            "record": self.record,
        }


@dataclass
class InputData:
    households: Dict[str, Household] = field(default_factory=dict)
    members: List[Member] = field(default_factory=list)
    responses: List[Response] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    # response keys that failed validation and must not drive eligibility
    excluded_response_lines: List[int] = field(default_factory=list)

    def issues_by_kind(self) -> Dict[str, List[Issue]]:
        out: Dict[str, List[Issue]] = {}
        for issue in self.issues:
            out.setdefault(issue.kind, []).append(issue)
        return out


RESPONSE_STATUSES = ("answered", "missing", "refused")


def _iter_jsonl(path: str, source: str, issues: List[Issue]):
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                issues.append(
                    Issue(
                        kind="malformed_json",
                        message=f"line is not valid JSON: {exc.msg}",
                        source=source,
                        line=line_no,
                        record={"raw": stripped},
                    )
                )
                continue
            if not isinstance(record, dict):
                issues.append(
                    Issue(
                        kind="malformed_json",
                        message="line is not a JSON object",
                        source=source,
                        line=line_no,
                        record={"raw": stripped},
                    )
                )
                continue
            yield line_no, record


def load_households(path: str, questionnaire: Questionnaire) -> Tuple[Dict[str, Household], List[Issue]]:
    issues: List[Issue] = []
    households: Dict[str, Household] = {}
    allowed = set(questionnaire.household_fields["consent"].values) if "consent" in questionnaire.household_fields else set()
    for line_no, record in _iter_jsonl(path, path, issues):
        hid = record.get("household_id")
        if not isinstance(hid, str) or not hid:
            issues.append(
                Issue("unknown_member", "household row lacks a valid household_id", path, line_no, record)
            )
            continue
        if hid in households:
            issues.append(
                Issue("duplicate_household", f"household {hid!r} declared more than once", path, line_no, record)
            )
            continue
        consent = record.get("consent")
        if not isinstance(consent, dict):
            issues.append(
                Issue("malformed_household", f"household {hid!r} lacks a consent object", path, line_no, record)
            )
            continue
        status = consent.get("status")
        if status not in RESPONSE_STATUSES:
            issues.append(
                Issue(
                    "malformed_household",
                    f"household {hid!r} consent status {status!r} is not one of {list(RESPONSE_STATUSES)}",
                    path,
                    line_no,
                    record,
                )
            )
            continue
        value: Optional[str] = None
        if status == "answered":
            value = consent.get("value")
            if value not in allowed:
                issues.append(
                    Issue(
                        "out_of_domain",
                        f"household {hid!r} consent value {value!r} is outside domain {sorted(allowed)}; "
                        "consent is treated as unknown, not as an answer",
                        path,
                        line_no,
                        record,
                    )
                )
                value = None
                status = "missing"
        households[hid] = Household(hid, value, status, line_no)
    return households, issues


def load_roster(path: str) -> Tuple[List[Member], List[Issue]]:
    issues: List[Issue] = []
    members: List[Member] = []
    seen = set()
    for line_no, record in _iter_jsonl(path, path, issues):
        hid, mid = record.get("household_id"), record.get("member_id")
        if not isinstance(hid, str) or not hid or not isinstance(mid, str) or not mid:
            issues.append(
                Issue("malformed_roster", "roster row needs string household_id and member_id", path, line_no, record)
            )
            continue
        key = (hid, mid)
        if key in seen:
            issues.append(
                Issue(
                    "duplicate_roster_member",
                    f"roster member ({hid!r}, {mid!r}) declared more than once",
                    path,
                    line_no,
                    record,
                )
            )
            continue
        seen.add(key)
        members.append(Member(hid, mid, line_no))
    return members, issues


def load_responses(path: str, questionnaire: Questionnaire) -> Tuple[List[Response], List[Issue]]:
    issues: List[Issue] = []
    responses: List[Response] = []
    questions = questionnaire.member_questions
    for line_no, record in _iter_jsonl(path, path, issues):
        hid, mid = record.get("household_id"), record.get("member_id")
        question = record.get("question")
        if not isinstance(hid, str) or not isinstance(mid, str) or not isinstance(question, str):
            issues.append(
                Issue(
                    "malformed_response",
                    "response needs string household_id, member_id and question",
                    path,
                    line_no,
                    record,
                )
            )
            continue
        status = record.get("status")
        if status not in RESPONSE_STATUSES:
            issues.append(
                Issue(
                    "malformed_response",
                    f"response status {status!r} is not one of {list(RESPONSE_STATUSES)}",
                    path,
                    line_no,
                    record,
                )
            )
            continue
        if question not in questions:
            # Could be a household-only field or a name that does not exist.
            scope = "household" if question in questionnaire.household_fields else "unknown"
            issues.append(
                Issue(
                    "wrong_question",
                    f"question {question!r} is not a member question ({scope} scope); "
                    "it cannot be a per-member answer",
                    path,
                    line_no,
                    record,
                )
            )
            continue
        value = record.get("value")
        usable = True
        if status == "answered":
            if "value" not in record:
                issues.append(
                    Issue("malformed_response", "answered response has no value", path, line_no, record)
                )
                value, usable = None, False
            elif not questions[question].in_domain(value):
                issues.append(
                    Issue(
                        "out_of_domain",
                        f"value {value!r} for member question {question!r} is outside its "
                        "declared domain; the value is isolated and not used for eligibility",
                        path,
                        line_no,
                        record,
                    )
                )
                value, usable = None, False
        else:
            if "value" in record:
                issues.append(
                    Issue(
                        "malformed_response",
                        f"status {status!r} response for {question!r} carries a value; "
                        "value is ignored (domain stays open)",
                        path,
                        line_no,
                        record,
                    )
                )
            value = None
        responses.append(
            Response(hid, mid, question, status, value, line_no, record, usable=usable)
        )
    return responses, issues


def resolve_duplicates(responses: List[Response], source: str) -> Tuple[List[Response], List[Issue], List[int]]:
    """Collapse responses to one per (household, member, question).

    Identical duplicates are harmless; any disagreement is a conflict.  Rows
    whose value already failed domain validation are unusable no matter what
    else sits on the key (they were reported when loaded).  A conflicting key
    is excluded from the usable responses entirely -- the tool never picks a
    winner -- and every involved line is reported.
    """
    issues: List[Issue] = []
    grouped: Dict[Tuple[str, str, str], List[Response]] = {}
    excluded_lines: List[int] = []
    for response in responses:
        if not response.usable:
            excluded_lines.append(response.line)
            continue
        grouped.setdefault(response.key, []).append(response)

    usable: List[Response] = []
    for key, group in grouped.items():
        if len({_signature(r) for r in group}) == 1:
            usable.append(group[0])
            # Exact duplicates are harmless for eligibility but are still
            # recorded as non-primary so the long table points at one line.
            excluded_lines.extend(r.line for r in group[1:])
            continue
        for response in group:
            excluded_lines.append(response.line)
        issues.append(
            Issue(
                "conflicting_duplicate_answer",
                f"conflicting duplicate answers for ({key[0]}, {key[1]}, {key[2]}) on lines "
                f"{[r.line for r in group]}; the whole key is isolated and not used",
                source,
                group[0].line,
                {"key": list(key), "records": [r.raw for r in group]},
            )
        )
    return usable, issues, excluded_lines


def _hashable(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _signature(response: Response) -> Tuple[str, Any]:
    """Identity of a response for duplicate comparison.

    Domain-failed rows are unusable no matter what else sits on the key, so
    only their status participates; otherwise a valid duplicate must agree on
    both status and value.
    """
    if not response.usable:
        return ("invalid", response.status)
    return (response.status, _hashable(response.value))


def load_inputs(
    households_path: str,
    roster_path: str,
    responses_path: str,
    questionnaire: Questionnaire,
) -> InputData:
    households, h_issues = load_households(households_path, questionnaire)
    members, r_issues = load_roster(roster_path)
    responses, resp_issues = load_responses(responses_path, questionnaire)
    usable, dup_issues, excluded = resolve_duplicates(responses, responses_path)

    data = InputData(
        households=households,
        members=members,
        responses=responses,
        issues=h_issues + r_issues + resp_issues + dup_issues,
        excluded_response_lines=excluded,
    )
    _flag_orphan_records(data, households_path, roster_path, responses_path)
    return data


def _flag_orphan_records(
    data: InputData, households_path: str, roster_path: str, responses_path: str
) -> None:
    member_keys = {m.key for m in data.members}
    for response in data.responses:
        if response.line in data.excluded_response_lines:
            continue
        if (response.household_id, response.member_id) not in member_keys:
            data.issues.append(
                Issue(
                    "unknown_member",
                    f"response for ({response.household_id!r}, {response.member_id!r}) "
                    "does not match any roster member; isolated and not used",
                    responses_path,
                    response.line,
                    response.raw,
                )
            )
            data.excluded_response_lines.append(response.line)
        elif response.household_id not in data.households:
            data.issues.append(
                Issue(
                    "unknown_household",
                    f"response for unknown household {response.household_id!r}; isolated and not used",
                    responses_path,
                    response.line,
                    response.raw,
                )
            )
            data.excluded_response_lines.append(response.line)
    for member in data.members:
        if member.household_id not in data.households:
            data.issues.append(
                Issue(
                    "unknown_household",
                    f"roster member ({member.household_id!r}, {member.member_id!r}) "
                    "belongs to an undeclared household; the instance cannot be evaluated",
                    roster_path,
                    member.line,
                    {"household_id": member.household_id, "member_id": member.member_id},
                )
            )
