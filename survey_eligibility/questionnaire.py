"""Questionnaire model and structural validation.

Loading a questionnaire never touches answers.  A structurally invalid
questionnaire (cycle, dangling jump, bad reference scope, untyped value, ...)
is rejected before any response file is read.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

HOUSEHOLD = "household"
MEMBER = "member"

# Operators.  ``eq``/``ne`` accept enum values, ``ge``/``lt`` integers.
COMPARISON_OPS = ("eq", "ne", "ge", "lt")
SUPPORTED_KINDS = ("branch", "ask", "end")
SUPPORTED_TYPES = ("enum", "integer")


class QuestionnaireValidationError(ValueError):
    """Raised when the frozen questionnaire is structurally unusable."""

    def __init__(self, problems: List[str]):
        self.problems = problems
        super().__init__("questionnaire rejected:\n  - " + "\n  - ".join(problems))


@dataclass(frozen=True)
class QuestionDef:
    name: str
    scope: str                     # HOUSEHOLD or MEMBER
    qtype: str                     # enum or integer
    values: tuple = ()             # enum domain, input order
    minimum: int | None = None
    maximum: int | None = None

    @property
    def domain_size(self) -> int:
        if self.qtype == "enum":
            return len(self.values)
        return int(self.maximum) - int(self.minimum) + 1

    def domain(self) -> List[Any]:
        """Enumerated domain in declaration order (bounded integers by value)."""
        if self.qtype == "enum":
            return list(self.values)
        return list(range(self.minimum, self.maximum + 1))

    def in_domain(self, value: Any) -> bool:
        if self.qtype == "enum":
            return isinstance(value, str) and value in self.values
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True)
class Ref:
    scope: str                     # household or member
    question: str                  # field / question name


@dataclass(frozen=True)
class Test:
    ref: Ref
    op: str
    value: Any

    def evaluate(self, value: Any) -> bool:
        if self.op == "eq":
            return value == self.value
        if self.op == "ne":
            return value != self.value
        if self.op == "ge":
            return value >= self.value
        if self.op == "lt":
            return value < self.value
        raise AssertionError(f"unhandled operator {self.op!r}")


@dataclass(frozen=True)
class Node:
    name: str
    kind: str                      # branch, ask, end
    # branch
    test: Test | None = None
    true: str | None = None
    false: str | None = None
    # ask
    question: str | None = None
    next: str | None = None


@dataclass
class Questionnaire:
    version: str
    start: str
    household_fields: Dict[str, QuestionDef]
    member_questions: Dict[str, QuestionDef]
    nodes: Dict[str, Node]
    warnings: List[str] = field(default_factory=list)

    # -- convenience -----------------------------------------------------
    def question(self, scope: str, name: str) -> QuestionDef:
        if scope == HOUSEHOLD:
            return self.household_fields[name]
        return self.member_questions[name]

    def asks(self) -> List[str]:
        """Names of nodes that ask a member question, in document order."""
        return [n.name for n in self.nodes.values() if n.kind == "ask"]

    def member_asked_questions(self) -> List[str]:
        seen: List[str] = []
        for n in self.nodes.values():
            if n.kind == "ask" and n.question not in seen:
                seen.append(n.question)
        return seen

    def question_scopes(self) -> Dict[str, Set[str]]:
        scopes: Dict[str, Set[str]] = {}
        for n in self.nodes.values():
            if n.kind == "ask":
                scope = MEMBER if n.question in self.member_questions else HOUSEHOLD
                scopes.setdefault(n.question, set()).add(scope)
        return scopes

    def reachable_nodes(self) -> Set[str]:
        seen: Set[str] = set()
        stack = [self.start]
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            node = self.nodes[name]
            if node.kind == "branch":
                stack.extend([node.true, node.false])
            elif node.kind == "ask":
                stack.append(node.next)
        return seen


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_questionnaire(path: str) -> Questionnaire:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return parse_questionnaire(raw)


def parse_questionnaire(raw: Dict[str, Any]) -> Questionnaire:
    problems: List[str] = []
    warnings: List[str] = []

    if not isinstance(raw, dict):
        raise QuestionnaireValidationError(["questionnaire root must be an object"])

    version = raw.get("version")
    if not isinstance(version, str) or not version:
        problems.append("version must be a non-empty string")

    household_fields = _parse_defs(raw.get("household_fields"), "household_fields", problems)
    member_questions = _parse_defs(raw.get("member_questions"), "member_questions", problems)

    nodes_raw = raw.get("nodes")
    nodes: Dict[str, Node] = {}
    if not isinstance(nodes_raw, dict) or not nodes_raw:
        problems.append("nodes must be a non-empty object")
        nodes_raw = {}
    for name, spec in nodes_raw.items():
        node = _parse_node(name, spec, problems)
        if node is not None:
            nodes[name] = node
    start = raw.get("start")
    if not isinstance(start, str):
        problems.append("start must name a node")
    elif start not in nodes:
        problems.append(f"start {start!r} is not defined in nodes")

    # -- jump targets ----------------------------------------------------
    for node in nodes.values():
        if node.kind == "branch":
            for label, target in (("true", node.true), ("false", node.false)):
                if target not in nodes:
                    problems.append(f"node {node.name!r}: {label} target {target!r} is undefined")
        elif node.kind == "ask":
            if node.next not in nodes:
                problems.append(f"node {node.name!r}: next target {node.next!r} is undefined")

    # -- references ------------------------------------------------------
    for node in nodes.values():
        if node.kind != "branch":
            continue
        ref = node.test.ref
        if ref.scope == MEMBER:
            if ref.question not in member_questions:
                problems.append(
                    f"node {node.name!r}: member ref {ref.question!r} is not a declared member question"
                )
                continue
            qdef = member_questions[ref.question]
            if qdef.qtype == "integer" and node.test.op not in ("ge", "lt"):
                problems.append(
                    f"node {node.name!r}: integer question {ref.question!r} "
                    f"supports only ge/lt, got {node.test.op!r}"
                )
            if qdef.qtype == "enum" and node.test.op not in ("eq", "ne"):
                problems.append(
                    f"node {node.name!r}: enum question {ref.question!r} "
                    f"supports only eq/ne, got {node.test.op!r}"
                )
            _check_literal(node.name, ref, qdef, node.test.value, "test value", problems)
        elif ref.scope == HOUSEHOLD:
            if ref.question not in household_fields:
                problems.append(
                    f"node {node.name!r}: household ref {ref.question!r} is not a declared field"
                )
                continue
            qdef = household_fields[ref.question]
            if qdef.qtype == "enum" and node.test.op not in ("eq", "ne"):
                problems.append(
                    f"node {node.name!r}: household field {ref.question!r} "
                    f"supports only eq/ne, got {node.test.op!r}"
                )
            if qdef.qtype == "integer" and node.test.op not in ("ge", "lt"):
                problems.append(
                    f"node {node.name!r}: integer household field {ref.question!r} "
                    f"supports only ge/lt, got {node.test.op!r}"
                )
            _check_literal(node.name, ref, qdef, node.test.value, "test value", problems)

    # -- ask nodes: question scope and declared question ---------------
    for node in nodes.values():
        if node.kind != "ask":
            continue
        if node.question in member_questions and node.question in household_fields:
            problems.append(
                f"node {node.name!r}: question {node.question!r} is declared at both household and member scope"
            )
        elif node.question not in member_questions and node.question not in household_fields:
            problems.append(f"node {node.name!r}: question {node.question!r} is not declared")
        elif node.question in household_fields:
            # Household questions are household context, never re-asked per
            # member in a member-level walk.  Allowing it here would make
            # "already asked" checks ambiguous.
            problems.append(
                f"node {node.name!r}: asking household field {node.question!r} inside the "
                "member questionnaire is not supported (household fields are context only)"
            )
        else:
            # Every member reference in the questionnaire must resolve to a
            # question that this member has already been asked on *all* paths
            # reaching the referencing branch.  Checked below globally.
            pass

    # A member reference must name a member question that can have been asked
    # before the branch on at least one path.  If it is asked on *every* path
    # the value is always available; if only on some paths the reference is
    # conditional -- on the paths that never asked it the value is simply
    # unknown (full domain), never taken from another member's answer.
    if start in nodes:
        must_before = must_asked_before(start, nodes)
        may_before = may_asked_before(start, nodes)
        reachable_now = _reachable(start, nodes)
        for node in nodes.values():
            if node.kind != "branch" or node.test is None:
                continue
            if node.name not in reachable_now:
                continue
            ref = node.test.ref
            if ref.scope != MEMBER:
                continue
            if ref.question not in may_before.get(node.name, set()):
                problems.append(
                    f"node {node.name!r}: references member.{ref.question}, which is never "
                    "asked on any path reaching this branch"
                )
            elif ref.question not in must_before.get(node.name, set()):
                warnings.append(
                    f"node {node.name!r}: conditional reference to member.{ref.question}; "
                    "paths that do not ask it treat the value as unknown (full domain)"
                )

    # -- cycles ------------------------------------------------------------
    cycle = _find_cycle(start, nodes) if start in nodes else None
    if cycle:
        problems.append("cycle detected (structural, not fixed-point evaluable): " + " -> ".join(cycle))

    reachable = _reachable(start, nodes) if start in nodes else set()
    for name in nodes:
        if name not in reachable:
            warnings.append(f"node {name!r} is unreachable from start")
    if "end" not in {nodes[n].kind for n in reachable if n in nodes}:
        warnings.append("no reachable end node bound on every path (possible infinite walk)")

    if problems:
        raise QuestionnaireValidationError(problems)

    return Questionnaire(
        version=version,
        start=start,
        household_fields=household_fields,
        member_questions=member_questions,
        nodes=nodes,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_defs(raw: Any, label: str, problems: List[str]) -> Dict[str, QuestionDef]:
    out: Dict[str, QuestionDef] = {}
    if raw is None:
        return out
    if not isinstance(raw, dict):
        problems.append(f"{label} must be an object")
        return out
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            problems.append(f"{label}.{name}: definition must be an object")
            continue
        qtype = spec.get("type")
        if qtype not in SUPPORTED_TYPES:
            problems.append(f"{label}.{name}: unsupported type {qtype!r}")
            continue
        if len(spec) > 1:
            for key in spec:
                if key != "type" and key not in ("values", "minimum", "maximum"):
                    problems.append(f"{label}.{name}: unexpected key {key!r}")
        if qtype == "enum":
            values = spec.get("values")
            if not isinstance(values, list) or not values:
                problems.append(f"{label}.{name}: enum requires a non-empty values list")
                continue
            bad = [v for v in values if not isinstance(v, str)]
            if bad:
                problems.append(f"{label}.{name}: enum values must be strings, got {bad!r}")
                continue
            if len(set(values)) != len(values):
                problems.append(f"{label}.{name}: enum values must be unique")
                continue
            extra = [k for k in ("minimum", "maximum") if k in spec]
            if extra:
                problems.append(f"{label}.{name}: enum must not declare {extra!r}")
            out[name] = QuestionDef(
                name=name,
                scope=HOUSEHOLD if label == "household_fields" else MEMBER,
                qtype="enum",
                values=tuple(values),
            )
        else:
            lo, hi = spec.get("minimum"), spec.get("maximum")
            if isinstance(lo, bool) or not isinstance(lo, int):
                problems.append(f"{label}.{name}: integer minimum must be an int")
                continue
            if isinstance(hi, bool) or not isinstance(hi, int):
                problems.append(f"{label}.{name}: integer maximum must be an int")
                continue
            if lo > hi:
                problems.append(f"{label}.{name}: minimum {lo} exceeds maximum {hi}")
                continue
            if "values" in spec:
                problems.append(f"{label}.{name}: integer must not declare values")
            out[name] = QuestionDef(
                name=name,
                scope=HOUSEHOLD if label == "household_fields" else MEMBER,
                qtype="integer",
                minimum=lo,
                maximum=hi,
            )
    return out


def _parse_node(name: str, spec: Any, problems: List[str]) -> Node | None:
    if not isinstance(spec, dict):
        problems.append(f"node {name!r}: must be an object")
        return None
    kind = spec.get("kind")
    if kind not in SUPPORTED_KINDS:
        problems.append(f"node {name!r}: unsupported kind {kind!r}")
        return None
    if kind == "end":
        extra = set(spec) - {"kind"}
        if extra:
            problems.append(f"node {name!r}: end nodes take no keys besides kind, got {sorted(extra)}")
        return Node(name=name, kind="end")
    if kind == "ask":
        extra = set(spec) - {"kind", "question", "next"}
        if extra:
            problems.append(f"node {name!r}: unexpected keys {sorted(extra)}")
        question, nxt = spec.get("question"), spec.get("next")
        if not isinstance(question, str):
            problems.append(f"node {name!r}: ask requires a question name")
            question = ""
        if not isinstance(nxt, str):
            problems.append(f"node {name!r}: ask requires a next node")
            nxt = ""
        return Node(name=name, kind="ask", question=question, next=nxt)
    # branch
    extra = set(spec) - {"kind", "test", "true", "false"}
    if extra:
        problems.append(f"node {name!r}: unexpected keys {sorted(extra)}")
    test_spec = spec.get("test")
    test = None
    if not isinstance(test_spec, dict):
        problems.append(f"node {name!r}: branch requires a test object")
    else:
        ref_spec = test_spec.get("ref")
        ref = _parse_ref(name, ref_spec, problems)
        op = test_spec.get("op")
        if op not in COMPARISON_OPS:
            problems.append(f"node {name!r}: unsupported operator {op!r}")
        if "value" not in test_spec:
            problems.append(f"node {name!r}: test requires a value")
        if ref is not None and op in COMPARISON_OPS and "value" in test_spec:
            test = Test(ref=ref, op=op, value=test_spec["value"])
        test_extra = set(test_spec) - {"ref", "op", "value"}
        if test_extra:
            problems.append(f"node {name!r}: unexpected test keys {sorted(test_extra)}")
    true_t, false_t = spec.get("true"), spec.get("false")
    if not isinstance(true_t, str):
        problems.append(f"node {name!r}: branch requires a true target")
        true_t = ""
    if not isinstance(false_t, str):
        problems.append(f"node {name!r}: branch requires a false target")
        false_t = ""
    return Node(name=name, kind="branch", test=test, true=true_t, false=false_t)


def _parse_ref(node_name: str, raw: Any, problems: List[str]) -> Ref | None:
    if not isinstance(raw, str):
        problems.append(f"node {node_name!r}: test ref must be a 'scope.name' string")
        return None
    if "." not in raw:
        problems.append(f"node {node_name!r}: test ref {raw!r} must be 'household.<field>' or 'member.<question>'")
        return None
    scope, field_name = raw.split(".", 1)
    if scope not in (HOUSEHOLD, MEMBER):
        problems.append(f"node {node_name!r}: test ref scope {scope!r} must be household or member")
        return None
    if not field_name or "." in field_name:
        problems.append(f"node {node_name!r}: malformed test ref {raw!r}")
        return None
    return Ref(scope=scope, question=field_name)


def _check_literal(
    node_name: str, ref: Ref, qdef: QuestionDef, value: Any, label: str, problems: List[str]
) -> None:
    if not qdef.in_domain(value):
        problems.append(
            f"node {node_name!r}: {label} {value!r} is outside the declared domain of "
            f"{ref.scope}.{ref.question}"
        )


def must_asked_before(start: str, nodes: Dict[str, Node]):
    """Questions asked on *every* path from ``start`` to each node.

    Standard forward data-flow: the entry node has an empty guaranteed set and
    each node's guaranteed set is the intersection of its predecessors'
    post-sets (with the node's own question added after an ask).  Iterated to
    a fixpoint, so it also terminates on (about to be rejected) cyclic graphs.
    """
    preds: Dict[str, List[str]] = {name: [] for name in nodes}
    for node in nodes.values():
        for target in _successors(node):
            if target in nodes:
                preds[target].append(node.name)

    in_sets: Dict[str, Set[str]] = {name: set() for name in nodes}
    # Seed: start is entered without any question asked.
    changed = True
    while changed:
        changed = False
        for name, node in nodes.items():
            if name == start:
                incoming = set()
            elif preds[name]:
                incoming = set.intersection(
                    *[_post_set(nodes[p], in_sets[p]) for p in preds[name]]
                )
            else:
                incoming = set()          # unreachable
            if incoming != in_sets[name]:
                in_sets[name] = incoming
                changed = True
    return in_sets


def may_asked_before(start: str, nodes: Dict[str, Node]):
    """Questions asked on *at least one* path from ``start`` to each node."""
    preds: Dict[str, List[str]] = {name: [] for name in nodes}
    for node in nodes.values():
        for target in _successors(node):
            if target in nodes:
                preds[target].append(node.name)

    in_sets: Dict[str, Set[str]] = {name: set() for name in nodes}
    changed = True
    while changed:
        changed = False
        for name, node in nodes.items():
            if name == start:
                incoming = set()
            elif preds[name]:
                incoming = set().union(*[_post_set(nodes[p], in_sets[p]) for p in preds[name]])
            else:
                incoming = set()
            if incoming != in_sets[name]:
                in_sets[name] = incoming
                changed = True
    return in_sets


def _successors(node: Node) -> List[str]:
    if node.kind == "branch":
        return [node.true, node.false]
    if node.kind == "ask":
        return [node.next]
    return []


def _post_set(node: Node, in_set: Set[str]) -> Set[str]:
    if node.kind == "ask":
        return set(in_set) | {node.question}
    return set(in_set)
    seen: Set[str] = set()
    stack = [start]
    while stack:
        name = stack.pop()
        if name in seen or name not in nodes:
            continue
        seen.add(name)
        node = nodes[name]
        if node.kind == "branch":
            stack.extend([node.true, node.false])
        elif node.kind == "ask":
            stack.append(node.next)
    return seen


def _reachable(start: str, nodes: Dict[str, Node]) -> Set[str]:
    seen: Set[str] = set()
    stack = [start]
    while stack:
        name = stack.pop()
        if name in seen or name not in nodes:
            continue
        seen.add(name)
        stack.extend(_successors(nodes[name]))
    return seen


def _find_cycle(start: str, nodes: Dict[str, Node]) -> List[str] | None:
    """Return the node sequence of a reachable cycle, or None."""
    color: Dict[str, int] = {}
    parent: Dict[str, str] = {}

    def successors(name: str) -> List[str]:
        node = nodes[name]
        if node.kind == "branch":
            return [node.true, node.false]
        if node.kind == "ask":
            return [node.next]
        return []

    def dfs(name: str) -> List[str] | None:
        color[name] = 1
        for nxt in successors(name):
            if nxt not in nodes:
                continue
            if color.get(nxt) == 1:
                cycle = [nxt]
                cur = name
                while cur != nxt:
                    cycle.append(cur)
                    cur = parent[cur]
                cycle.append(nxt)
                return list(reversed(cycle))
            if color.get(nxt, 0) == 0:
                parent[nxt] = name
                found = dfs(nxt)
                if found:
                    return found
        color[name] = 2
        return None

    if start in nodes:
        return dfs(start)
    return None

