"""Eligibility engine.

For each roster instance the engine enumerates every *complete* path through
the acyclic jump graph that is compatible with the known answers.  Unknown
values (missing, refused, or no record at all) are represented as the full
declared domain and are refined only when a branch actually tests them, so a
single latent value stays consistent along a path (age cannot be both < 18
and >= 65).

Only paths that physically pass an ask node consult that question's answer.
An answer that exists but whose ask node is never reached on a path is
residue: it is reported as a stray answer and never used to justify routing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from .loader import Household, InputData, Member, Response
from .questionnaire import Questionnaire

# Eligibility labels
MUST = "MUST"
MAY = "MAY"
NEVER = "NEVER"

# Answer states
VALID_ANSWER = "valid_answer"
ASKED_MISSING = "asked_missing"
ASKED_REFUSED = "asked_refused"
CONDITIONAL_ANSWER = "conditional_answer"
STRUCTURALLY_NOT_ASKED = "structurally_not_asked"
STRAY_ANSWER = "stray_answer"

CONSENT_VAR = "household.consent"

MAX_PATHS_PER_INSTANCE = 250_000


class EngineError(RuntimeError):
    pass


@dataclass
class Path:
    nodes: List[str]
    steps: List[Dict[str, Any]]
    visited_questions: List[str]
    domains: Dict[str, FrozenSet[Any]]
    consulted: FrozenSet[str]

    def evidence(self, engine: "Engine", household: "Household", member: "Member") -> Dict[str, Any]:
        seen: List[str] = []
        for q in self.visited_questions:
            if q not in seen:
                seen.append(q)
        witness = {}
        visited_set = set(self.visited_questions)
        for var in sorted(self.consulted):
            value = engine.representative(var, self.domains[var])
            witness[var] = {
                "value": value,
                "value_source": engine.describe_value_source(
                    var, household, member, visited_set
                ),
            }
        return {
            "nodes": list(self.nodes),
            "visited_questions": seen,
            "steps": [dict(step) for step in self.steps],
            "witness": witness,
            "note": (
                "One concrete assignment consistent with this route. Values described as "
                "domain_sampled stand in for values that were unknown (refused/missing/"
                "unrecorded); they stay identical across every test on the route."
            ),
        }


@dataclass
class QuestionEvaluation:
    question: str
    eligibility: str
    observed_status: str            # answered / missing / refused / absent
    has_usable_answer: bool
    answer_state: str
    response_line: Optional[int]
    response_value: Any
    complete_paths: int
    visiting_paths: int
    avoiding_paths: int
    supporting_path: Optional[Dict[str, Any]]
    counterexample_path: Optional[Dict[str, Any]]


@dataclass
class MemberEvaluation:
    household_id: str
    member_id: str
    roster_line: int
    complete_paths: int
    no_admissible_path: bool
    questions: Dict[str, QuestionEvaluation] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class Evaluation:
    members: List[MemberEvaluation]
    skipped_members: List[Dict[str, Any]] = field(default_factory=list)


class Engine:
    def __init__(self, questionnaire: Questionnaire, data: InputData):
        self.q = questionnaire
        self.data = data
        self.var_order: Dict[str, List[Any]] = {}
        self.var_def: Dict[str, Any] = {}
        self._build_variables()

        # One usable response per (household, member, question).  Loader has
        # already dropped invalid/conflicting rows.
        self.responses: Dict[Tuple[str, str, str], Response] = {}
        for response in data.responses:
            if response.line in data.excluded_response_lines:
                continue
            self.responses[response.key] = response

        self.issues: List[Dict[str, Any]] = []

    # -- setup -----------------------------------------------------------
    def _build_variables(self) -> None:
        consent = self.q.household_fields.get("consent")
        if consent is not None:
            self.var_order[CONSENT_VAR] = consent.domain()
            self.var_def[CONSENT_VAR] = consent
        for name, qdef in self.q.member_questions.items():
            var = f"member.{name}"
            self.var_order[var] = qdef.domain()
            self.var_def[var] = qdef

    def describe_value_source(
        self, var: str, household: Household, member: Member, visited_questions: set
    ) -> str:
        if var == CONSENT_VAR:
            if household.consent_status == "answered" and household.consent is not None:
                return "answer"
            return f"domain_sampled ({household.consent_status})"
        question = var.split(".", 1)[1]
        response = self._response(household.household_id, member.member_id, question)
        if question not in visited_questions:
            # Mirrors the semantics rule: an answer counts only on paths that
            # actually reach its ask node.  Any value here is a hypothetical
            # domain sample, never the stored answer.
            if response is not None and response.status == "answered":
                return "domain_sampled (stored answer exists but this route never asks it)"
            return "domain_sampled (not asked on this route)"
        if response is not None and response.status == "answered":
            return "answer"
        if response is not None:
            return f"domain_sampled ({response.status})"
        return "domain_sampled (no record)"

    def representative(self, var: str, domain: FrozenSet[Any]) -> Any:
        for value in self.var_order[var]:
            if value in domain:
                return value
        raise EngineError(f"empty domain for {var}")

    # -- per-instance inputs --------------------------------------------
    def _response(self, household_id: str, member_id: str, question: str) -> Optional[Response]:
        response = self.responses.get((household_id, member_id, question))
        if response is None or not response.usable:
            return None
        return response

    def _raw_response(self, household_id: str, member_id: str, question: str) -> Optional[Response]:
        """Any record on the key, including one isolated as unusable."""
        key = (household_id, member_id, question)
        response = self.responses.get(key)
        if response is not None:
            return response
        for candidate in self.data.responses:
            if candidate.key == key:
                return candidate
        return None

    def _initial_domains(self, household: Household) -> Dict[str, FrozenSet[Any]]:
        domains: Dict[str, FrozenSet[Any]] = {}
        for var, values in self.var_order.items():
            domains[var] = frozenset(values)
        if CONSENT_VAR in domains:
            if household.consent_status == "answered" and household.consent is not None:
                domains[CONSENT_VAR] = frozenset({household.consent})
        return domains

    # -- path enumeration ----------------------------------------------
    def enumerate_paths(self, household: Household, member: Member) -> List[Path]:
        initial = self._initial_domains(household)
        paths: List[Path] = []
        budget = [MAX_PATHS_PER_INSTANCE]

        def dfs(
            node_name: str,
            domains: Dict[str, FrozenSet[Any]],
            visited: List[str],
            steps: List[Dict[str, Any]],
            trace: List[str],
            consulted: FrozenSet[str],
        ) -> None:
            if budget[0] <= 0:
                raise EngineError(
                    f"path enumeration exceeded {MAX_PATHS_PER_INSTANCE} complete paths "
                    f"for ({household.household_id}, {member.member_id}); refusing to truncate"
                )
            node = self.q.nodes[node_name]
            trace = trace + [node_name]
            if node.kind == "end":
                budget[0] -= 1
                paths.append(
                    Path(
                        nodes=trace,
                        steps=list(steps),
                        visited_questions=list(visited),
                        domains=dict(domains),
                        consulted=consulted,
                    )
                )
                return
            if node.kind == "ask":
                var = f"member.{node.question}"
                new_domains = dict(domains)
                response = self._response(household.household_id, member.member_id, node.question)
                if response is not None and response.status == "answered":
                    narrowed = new_domains[var] & frozenset({response.value})
                    if not narrowed:
                        return   # answered value contradicts the route: infeasible
                    new_domains[var] = narrowed
                    source = "answer"
                else:
                    source = "domain_open"
                step = {
                    "node": node_name,
                    "kind": "ask",
                    "question": node.question,
                    "answer_source": source,
                }
                dfs(
                    node.next,
                    new_domains,
                    visited + [node.question],
                    steps + [step],
                    trace,
                    consulted | {var},
                )
                return
            # branch
            test = node.test
            var = f"{test.ref.scope}.{test.ref.question}"
            if var not in domains:
                raise EngineError(f"branch {node_name} references undeclared variable {var}")
            domain = domains[var]
            true_values = frozenset(v for v in domain if test.evaluate(v))
            false_values = domain - true_values
            test_text = f"{var} {test.op} {test.value!r}"
            if true_values:
                new_domains = dict(domains)
                new_domains[var] = true_values
                step = {
                    "node": node_name,
                    "kind": "branch",
                    "test": test_text,
                    "outcome": True,
                }
                dfs(node.true, new_domains, visited, steps + [step], trace, consulted | {var})
            if false_values:
                new_domains = dict(domains)
                new_domains[var] = false_values
                step = {
                    "node": node_name,
                    "kind": "branch",
                    "test": test_text,
                    "outcome": False,
                }
                dfs(node.false, new_domains, visited, steps + [step], trace, consulted | {var})

        dfs(self.q.start, initial, [], [], [], frozenset())
        return paths

    # -- classification -------------------------------------------------
    def evaluate_member(self, household: Household, member: Member) -> MemberEvaluation:
        paths = self.enumerate_paths(household, member)
        no_path = not paths
        if no_path:
            self.issues.append(
                {
                    "kind": "no_admissible_complete_path",
                    "message": (
                        f"member ({household.household_id}, {member.member_id}) has no complete "
                        "path compatible with the known answers"
                    ),
                    "household_id": household.household_id,
                    "member_id": member.member_id,
                }
            )

        result = MemberEvaluation(
            household_id=household.household_id,
            member_id=member.member_id,
            roster_line=member.line,
            complete_paths=len(paths),
            no_admissible_path=no_path,
        )

        for question in self.q.member_questions:
            visiting = [p for p in paths if question in p.visited_questions]
            avoiding = [p for p in paths if question not in p.visited_questions]
            present, absent = bool(visiting), bool(avoiding)
            if present and not absent:
                eligibility = MUST
            elif present and absent:
                eligibility = MAY
            else:
                eligibility = NEVER

            response = self._response(household.household_id, member.member_id, question)
            raw = self._raw_response(household.household_id, member.member_id, question)
            if response is None:
                if raw is not None:
                    observed = f"{raw.status} (isolated)"   # out-of-domain / conflict / invalid
                else:
                    observed = "absent"
                usable_answer = False
                response_line = raw.line if raw is not None else None
                response_value = None
            else:
                observed = response.status
                usable_answer = response.status == "answered"
                response_line = response.line
                response_value = response.value

            if eligibility == NEVER:
                answer_state = STRAY_ANSWER if (raw is not None) else STRUCTURALLY_NOT_ASKED
            elif usable_answer:
                answer_state = VALID_ANSWER if eligibility == MUST else CONDITIONAL_ANSWER
            elif observed == "refused":
                answer_state = ASKED_REFUSED
            else:
                # missing, absent, or an answer isolated as invalid/conflicting
                answer_state = ASKED_MISSING

            if no_path:
                result.notes.append(
                    f"{question}: no admissible complete paths; MUST/MAY/NEVER not meaningful"
                )

            support = visiting[0].evidence(self, household, member) if visiting else None
            counter = avoiding[0].evidence(self, household, member) if avoiding else None
            result.questions[question] = QuestionEvaluation(
                question=question,
                eligibility=eligibility,
                observed_status=observed,
                has_usable_answer=usable_answer,
                answer_state=answer_state,
                response_line=response_line,
                response_value=response_value,
                complete_paths=len(paths),
                visiting_paths=len(visiting),
                avoiding_paths=len(avoiding),
                supporting_path=support,
                counterexample_path=counter,
            )
        return result

    def evaluate(self) -> Evaluation:
        evaluation = Evaluation(members=[])
        roster_keys = {m.key for m in self.data.members}
        for member in self.data.members:
            household = self.data.households.get(member.household_id)
            if household is None:
                evaluation.skipped_members.append(
                    {
                        "household_id": member.household_id,
                        "member_id": member.member_id,
                        "roster_line": member.line,
                        "reason": "household is not declared; instance cannot be evaluated",
                    }
                )
                continue
            evaluation.members.append(self.evaluate_member(household, member))
        # Responses for roster members that were isolated (unknown member etc.)
        # are already surfaced by the loader; keep a count here for the summary.
        evaluation.skipped_members.extend(
            {
                "household_id": r.household_id,
                "member_id": r.member_id,
                "response_line": r.line,
                "reason": "response key was isolated and is not part of eligibility",
            }
            for r in self.data.responses
            if r.key[:2] not in roster_keys
        )
        return evaluation
