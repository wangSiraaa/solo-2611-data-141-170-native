"""Model loading, exact rational parsing, and input validation.

Design choices that matter for the task:

* All arithmetic is :class:`fractions.Fraction` from the original text
  literals.  No floats and no tolerances are used anywhere in decisions.
* A history is represented as ``(bought test set, observed values)``, so
  purchase *order* is irrelevant and equivalent sub-problems merge.
* A joint table row with omitted symbol combinations is treated as an
  implicit zero probability; a state whose listed rows do not sum to
  exactly one is an input error, never silently renormalised.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence



class ModelError(ValueError):
    """Invalid or inconsistent model input.  Message names the concrete issue."""


class ZeroProbabilityError(ModelError):
    """An observation history has probability zero; no posterior is fabricated."""


# --------------------------------------------------------------------------
# Exact scalar parsing
# --------------------------------------------------------------------------

def parse_fraction(value: Any, where: str) -> Fraction:
    """Parse an int / decimal / rational string exactly.

    Accepted: int, Decimal, Fraction, finite float (via ``str``), strings such
    as ``"3/10"``, ``"0.3"``, ``"1e-3"``, ``" 2 "``.  Anything else, NaN,
    infinity, or a malformed literal raises :class:`ModelError` naming
    ``where``.
    """
    if isinstance(value, bool):  # bool is an int subclass; reject it
        raise ModelError(f"{where}: boolean {value!r} is not a number")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Decimal):
        try:
            f = Fraction(value)
        except (ValueError, ArithmeticError) as exc:
            raise ModelError(f"{where}: invalid decimal {value!r}: {exc}") from exc
        return f
    if isinstance(value, float):
        try:
            dec = Decimal(str(value))
        except InvalidOperation as exc:
            raise ModelError(f"{where}: invalid number {value!r}: {exc}") from exc
        return parse_fraction(dec, where)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ModelError(f"{where}: empty numeric value")
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError) as exc:
            raise ModelError(f"{where}: cannot parse exact number {value!r}: {exc}") from exc
    raise ModelError(f"{where}: unsupported numeric type {type(value).__name__}: {value!r}")


def parse_probability(value: Any, where: str) -> Fraction:
    p = parse_fraction(value, where)
    if p < 0 or p > 1:
        raise ModelError(f"{where}: probability {frac_str(p)} is outside [0, 1]")
    return p


def frac_str(f: Fraction) -> str:
    """Canonical exact text: ``"8"`` for integers, ``"35/3"`` otherwise."""
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def frac_float(f: Fraction) -> float:
    """Approximate display value.  Never used for decisions or ties."""
    return float(f.numerator) / float(f.denominator)



# --------------------------------------------------------------------------
# Outcome vectors
# --------------------------------------------------------------------------

def split_outcomes(text: str, where: str, expected: int | None = None) -> tuple[str, ...]:
    """Split an outcome string into one symbol per test, in ``test_order``.

    A comma-separated form (``"1,0,1,1"`` or ``"red,blue"``) is always split
    on commas.  Without commas each character is one result -- unless the
    model defines exactly one test, in which case the whole string is that
    test's single symbol (so ``"positive"`` is not mistaken for 8 results).
    """
    if not isinstance(text, str):
        raise ModelError(f"{where}: outcomes must be a string, got {type(text).__name__}")
    stripped = text.strip()
    if "," in stripped:
        parts = tuple(part.strip() for part in stripped.split(","))
        if any(part == "" for part in parts):
            raise ModelError(f"{where}: empty symbol in comma-separated outcomes {text!r}")
        return parts
    if stripped == "":
        return ()
    if expected == 1:
        return (stripped,)
    return tuple(stripped)


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Test:
    id: str
    cost: Fraction
    requires: tuple[str, ...]
    position: int


@dataclass(frozen=True)
class Action:
    id: str
    losses: Mapping[str, Fraction]


@dataclass
class Model:
    states: tuple[str, ...]
    priors: Mapping[str, Fraction]
    tests: tuple[Test, ...]                 # ordered by test_order / position
    actions: tuple[Action, ...]             # canonical order (sorted ids)
    joint: Mapping[str, Mapping[tuple[str, ...], Fraction]]  # non-zero rows
    alphabets: tuple[tuple[str, ...], ...]  # symbols observed per position
    budgets: tuple[Fraction, ...] | None
    objective: str
    tie_note: str
    source: str = "<memory>"
    warnings: tuple[str, ...] = ()

    # ---- basic accessors -------------------------------------------------

    @property
    def n_tests(self) -> int:
        return len(self.tests)

    def test_by_id(self, test_id: str) -> Test:
        for t in self.tests:
            if t.id == test_id:
                return t
        raise ModelError(f"unknown test id {test_id!r}")

    def action_by_id(self, action_id: str) -> Action:
        for a in self.actions:
            if a.id == action_id:
                return a
        raise ModelError(f"unknown action id {action_id!r}")

    def spent(self, bought: frozenset[str]) -> Fraction:
        total = Fraction(0)
        for t in self.tests:
            if t.id in bought:
                total += t.cost
        return total

    def total_test_cost(self) -> Fraction:
        return sum((t.cost for t in self.tests), Fraction(0))

    # ---- exact distribution queries -------------------------------------

    def joint_prob(self, state: str, outcomes: tuple[str, ...]) -> Fraction:
        return self.joint[state].get(outcomes, Fraction(0))

    def likelihood(self, state: str, bought: frozenset[str],
                   obs: Mapping[str, str]) -> Fraction:
        """P(observed results on ``bought`` | state), exactly.

        The joint is frozen: we sum the listed joint rows matching every
        observed coordinate.  Unlisted coordinates are marginalised out.
        """
        if not bought:
            return Fraction(1)
        rows = self.joint[state]
        constraints = [(t.position, obs[t.id])
                       for t in self.tests if t.id in bought]
        result = Fraction(0)
        for vector, prob in rows.items():
            if all(vector[pos] == value for pos, value in constraints):
                result += prob
        return result

    def evidence(self, bought: frozenset[str], obs: Mapping[str, str]) -> Fraction:
        return sum((self.priors[s] * self.likelihood(s, bought, obs)
                    for s in self.states), Fraction(0))

    def posterior(self, bought: frozenset[str],
                  obs: Mapping[str, str]) -> dict[str, Fraction]:
        evidence = self.evidence(bought, obs)
        if evidence <= 0:
            # Never invent a posterior for a zero-probability observation.
            known = ", ".join(f"{obs[t.id]}@{t.id}" for t in self.tests
                              if t.id in bought) or "<none>"
            raise ZeroProbabilityError(
                f"observation history has probability 0 (known results: {known}); "
                "refusing to fabricate a posterior"
            )
        return {s: self.priors[s] * self.likelihood(s, bought, obs) / evidence
                for s in self.states}

    def outcome_marginal(self, bought: frozenset[str], obs: Mapping[str, str],
                         test_id: str, value: str) -> Fraction:
        """P(test_id=value | current history), exact conditional probability."""
        node_ev = self.evidence(bought, obs)
        if node_ev <= 0:
            raise ZeroProbabilityError("cannot condition on a zero-probability history")
        t = self.test_by_id(test_id)
        child_obs = dict(obs)
        child_obs[test_id] = value
        child_bought = bought | frozenset((test_id,))
        return self.evidence(child_bought, child_obs) / node_ev

    def action_expectation(self, action: Action,
                           post: Mapping[str, Fraction]) -> Fraction:
        return sum((post[s] * action.losses[s] for s in self.states), Fraction(0))


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------

def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise ModelError(f"input file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ModelError(f"invalid JSON in {path}: {exc}") from exc


def _check_requires_cycles(tests: Sequence[Test]) -> None:
    graph = {t.id: tuple(t.requires) for t in tests}
    known = set(graph)
    for t in tests:
        for req in t.requires:
            if req not in known:
                raise ModelError(
                    f"test {t.id!r} requires unknown test {req!r}"
                )
            if req == t.id:
                raise ModelError(f"test {t.id!r} requires itself")
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for dep in graph[node]:
            if state.get(dep) == 1:
                cycle_start = stack.index(dep)
                cycle = stack[cycle_start:] + [dep]
                raise ModelError(
                    "cyclic test prerequisites: " + " -> ".join(cycle)
                )
            if state.get(dep) == 0 or dep not in state:
                visit(dep)
        stack.pop()
        state[node] = 2

    for node in graph:
        if state.get(node, 0) == 0:
            visit(node)


def _parse_tests(raw: Any, test_order: Any, where: str) -> tuple[Test, ...]:
    if not isinstance(raw, list) or not raw:
        # A model with no tests is legal; an empty/missing list is accepted,
        # but the joint then needs one empty result per state.
        raw = raw or []
        if not isinstance(raw, list):
            raise ModelError(f"{where}: 'tests' must be a list")
    ids: list[str] = []
    requires_raw: dict[str, list[str]] = {}
    costs: dict[str, Fraction] = {}
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ModelError(f"{where}: tests[{idx}] must be an object")
        test_id = item.get("id")
        if not isinstance(test_id, str) or not test_id.strip():
            raise ModelError(f"{where}: tests[{idx}] has no non-empty string id")
        test_id = test_id.strip()
        if test_id in costs:
            raise ModelError(f"{where}: duplicate test id {test_id!r}")
        costs[test_id] = parse_fraction(item.get("cost", 0), f"{where}: test {test_id!r} cost")
        if costs[test_id] < 0:
            raise ModelError(f"{where}: test {test_id!r} has negative cost {frac_str(costs[test_id])}")
        req = item.get("requires", [])
        if not isinstance(req, list):
            raise ModelError(f"{where}: test {test_id!r} 'requires' must be a list")
        for r in req:
            if not isinstance(r, str) or not r.strip():
                raise ModelError(f"{where}: test {test_id!r} has invalid requirement {r!r}")
        requires_raw[test_id] = [r.strip() for r in req]
        ids.append(test_id)

    if test_order is not None:
        if not isinstance(test_order, list):
            raise ModelError(f"{where}: 'test_order' must be a list")
        order_ids: list[str] = []
        for idx, tid in enumerate(test_order):
            if not isinstance(tid, str) or not tid.strip():
                raise ModelError(f"{where}: test_order[{idx}] is not a non-empty string")
            tid = tid.strip()
            if tid in order_ids:
                raise ModelError(f"{where}: test_order repeats test {tid!r}")
            if tid not in costs:
                raise ModelError(f"{where}: test_order references unknown test {tid!r}")
            order_ids.append(tid)
        missing = [tid for tid in ids if tid not in order_ids]
        if missing:
            raise ModelError(
                f"{where}: tests missing from test_order: {', '.join(missing)}"
            )
        ids = order_ids
    # Without test_order the declaration order defines outcome positions.

    tests = tuple(
        Test(id=tid, cost=costs[tid],
             requires=tuple(requires_raw[tid]), position=pos)
        for pos, tid in enumerate(ids)
    )
    _check_requires_cycles(tests)
    return tests


def _parse_prior(raw: Any, where: str) -> tuple[tuple[str, ...], dict[str, Fraction]]:
    if not isinstance(raw, dict) or not raw:
        raise ModelError(f"{where}: 'states' must be a non-empty object of priors")
    states: list[str] = []
    priors: dict[str, Fraction] = {}
    for state, value in raw.items():
        if not isinstance(state, str) or not state.strip():
            raise ModelError(f"{where}: invalid state name {state!r}")
        state = state.strip()
        if state in priors:
            raise ModelError(f"{where}: duplicate state {state!r}")
        priors[state] = parse_probability(value, f"{where}: prior for state {state!r}")
        states.append(state)
    total = sum(priors.values(), Fraction(0))
    if total != 1:
        raise ModelError(
            f"{where}: state priors sum to {frac_str(total)} instead of exactly 1"
        )
    return tuple(states), priors


def _parse_joint_rows(rows: Iterable[tuple[int, Mapping[str, Any]]], states: Sequence[str],
                      tests: Sequence[Test], where: str
                      ) -> tuple[dict[str, dict[tuple[str, ...], Fraction]],
                                 tuple[tuple[str, ...], ...]]:
    n = len(tests)
    parsed: dict[str, dict[tuple[str, ...], Fraction]] = {s: {} for s in states}
    symbols: list[set[str]] = [set() for _ in range(n)]
    line_of: dict[tuple[str, tuple[str, ...]], int] = {}
    for line, row in rows:
        missing = [k for k in ("state", "outcomes", "probability") if k not in row or row[k] is None]
        if missing:
            raise ModelError(f"{where} line {line}: missing column(s) {', '.join(missing)}")
        state = str(row["state"]).strip()
        if state not in parsed:
            raise ModelError(f"{where} line {line}: unknown state {state!r}")
        vector = split_outcomes(row["outcomes"], f"{where} line {line}", expected=n)
        if len(vector) != n:
            raise ModelError(
                f"{where} line {line}: state {state!r} has {len(vector)} result symbol(s), "
                f"expected {n} (one per test in test_order)"
            )
        prob = parse_probability(row["probability"], f"{where} line {line}: probability")
        key = (state, vector)
        if key in line_of:
            raise ModelError(
                f"{where} line {line}: duplicate row for state {state!r} and outcomes "
                f"{''.join(vector)!r} (first seen on line {line_of[key]})"
            )
        line_of[key] = line
        if prob != 0:
            parsed[state][vector] = prob
        for pos, sym in enumerate(vector):
            symbols[pos].add(sym)
    for s in states:
        if not parsed[s]:
            raise ModelError(
                f"{where}: no joint rows for state {s!r}; result coverage is incomplete"
            )
        total = sum(parsed[s].values(), Fraction(0))
        if total != 1:
            raise ModelError(
                f"{where}: joint probabilities for state {s!r} sum to {frac_str(total)} "
                "instead of exactly 1 (complete joint distribution required)"
            )
    if n > 0:
        for pos in range(n):
            if not symbols[pos]:
                raise ModelError(
                    f"{where}: no result symbols observed for test position {pos} "
                    f"({tests[pos].id!r}); result coverage is incomplete"
                )
    alphabets = tuple(tuple(sorted(s)) for s in symbols)
    # Sanity: all listed vectors only use declared alphabets (true by build).
    return parsed, alphabets


def _load_joint_csv(path: str, states: Sequence[str], tests: Sequence[Test]
                    ) -> tuple[dict[str, dict[tuple[str, ...], Fraction]],
                               tuple[tuple[str, ...], ...]]:
    try:
        handle = open(path, "r", encoding="utf-8", newline="")
    except FileNotFoundError:
        raise ModelError(f"joint file not found: {path}") from None
    with handle:
        reader = csv.DictReader(handle)
        expected = {"state", "outcomes", "probability"}
        if reader.fieldnames is None:
            raise ModelError(f"{path}: joint CSV is empty")
        fields = {f.strip() for f in reader.fieldnames if f is not None}
        missing = expected - fields
        if missing:
            raise ModelError(f"{path}: joint CSV missing column(s): {', '.join(sorted(missing))}")
        rows = [(idx + 2, row) for idx, row in enumerate(reader)]
    return _parse_joint_rows(rows, states, tests, path)


def _parse_actions_csv(path: str, states: Sequence[str]) -> tuple[Action, ...]:
    try:
        handle = open(path, "r", encoding="utf-8", newline="")
    except FileNotFoundError:
        raise ModelError(f"actions file not found: {path}") from None
    with handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ModelError(f"{path}: actions CSV is empty")
        header = [f.strip() if f else "" for f in reader.fieldnames]
        if header[0] != "action":
            raise ModelError(
                f"{path}: first column must be 'action', got {header[0]!r}"
            )
        state_cols = header[1:]
        unknown = [c for c in state_cols if c not in states]
        if unknown:
            raise ModelError(f"{path}: unknown state column(s): {', '.join(unknown)}")
        missing = [s for s in states if s not in state_cols]
        if missing:
            raise ModelError(f"{path}: missing loss column(s) for state(s): {', '.join(missing)}")
        actions: dict[str, dict[str, Fraction]] = {}
        for line, row in enumerate(reader, start=2):
            action_id = row.get("action")
            if action_id is None or not str(action_id).strip():
                raise ModelError(f"{path} line {line}: empty action id")
            action_id = str(action_id).strip()
            if action_id in actions:
                raise ModelError(f"{path} line {line}: duplicate action {action_id!r}")
            losses: dict[str, Fraction] = {}
            for state in states:
                raw = row.get(state)
                if raw is None:
                    raise ModelError(
                        f"{path} line {line}: action {action_id!r} is missing loss for state {state!r}"
                    )
                losses[state] = parse_fraction(raw, f"{path} line {line}: action {action_id!r} loss for {state!r}")
            actions[action_id] = losses
    if not actions:
        raise ModelError(f"{path}: no terminal actions defined")
    # Canonical, input-row-order-independent ordering (also used for ties).
    return tuple(Action(id=aid, losses=actions[aid]) for aid in sorted(actions))


def _parse_actions_json(raw: Any, states: Sequence[str], where: str) -> tuple[Action, ...]:
    if not isinstance(raw, dict) or not raw:
        raise ModelError(f"{where}: 'actions' must be a non-empty object")
    actions: dict[str, dict[str, Fraction]] = {}
    for action_id, losses in raw.items():
        if not isinstance(action_id, str) or not action_id.strip():
            raise ModelError(f"{where}: invalid action id {action_id!r}")
        action_id = action_id.strip()
        if not isinstance(losses, dict):
            raise ModelError(f"{where}: action {action_id!r} must map states to losses")
        unknown = set(losses) - set(states)
        if unknown:
            raise ModelError(f"{where}: action {action_id!r} has unknown state(s): {', '.join(sorted(unknown))}")
        missing = [s for s in states if s not in losses]
        if missing:
            raise ModelError(f"{where}: action {action_id!r} is missing state(s): {', '.join(missing)}")
        actions[action_id] = {
            s: parse_fraction(losses[s], f"{where}: action {action_id!r} loss for {s!r}")
            for s in states
        }
    return tuple(Action(id=aid, losses=actions[aid]) for aid in sorted(actions))


def _parse_budgets(raw: Any, where: str) -> tuple[Fraction, ...]:
    if isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    budgets = []
    for idx, value in enumerate(items):
        b = parse_fraction(value, f"{where}: budget[{idx}]")
        if b < 0:
            raise ModelError(f"{where}: negative budget {frac_str(b)}")
        budgets.append(b)
    if not budgets:
        raise ModelError(f"{where}: no budgets given")
    return tuple(sorted(set(budgets)))


# --------------------------------------------------------------------------
# Public loaders
# --------------------------------------------------------------------------

def _resolve_aux(path: str, base: str) -> str:
    """Honour an explicit path as given; fall back to the model's directory."""
    if os.path.isabs(path) or os.path.exists(path):
        return os.path.abspath(path)
    return os.path.join(base, path)


def load_model(model_path: str, actions_path: str | None = None,
               joint_path: str | None = None, plans_path: str | None = None,
               budgets: Sequence[Any] | None = None) -> Model:
    """Load the fixture format (model.json + joint.csv + actions.csv + plans.json).

    A single self-contained JSON (like ``tiny.json``) is also accepted when it
    embeds ``joint`` and ``actions``.  Explicit CLI paths override the
    ``joint_file`` relative to the model file.
    """
    raw = _read_json(model_path)
    if not isinstance(raw, dict):
        raise ModelError(f"{model_path}: top-level JSON must be an object")
    base = os.path.dirname(os.path.abspath(model_path))
    warnings: list[str] = []

    if "states" not in raw:
        raise ModelError(f"{model_path}: missing 'states' priors")
    states, priors = _parse_prior(raw["states"], model_path)
    tests = _parse_tests(raw.get("tests", []), raw.get("test_order"), model_path)

    if "joint" in raw:
        joint_raw = raw["joint"]
        if not isinstance(joint_raw, list):
            raise ModelError(f"{model_path}: embedded 'joint' must be a list of rows")
        rows = []
        for idx, row in enumerate(joint_raw):
            if not isinstance(row, dict):
                raise ModelError(f"{model_path}: joint[{idx}] must be an object")
            rows.append((idx + 1, row))
        joint, alphabets = _parse_joint_rows(rows, states, tests, f"{model_path}: embedded joint")
    else:
        jp = joint_path or raw.get("joint_file")
        if not jp:
            raise ModelError(
                f"{model_path}: no joint distribution provided "
                "(expected 'joint_file', embedded 'joint', or --joint)"
            )
        jp = _resolve_aux(jp, base)
        joint, alphabets = _load_joint_csv(jp, states, tests)

    if "actions" in raw:
        actions = _parse_actions_json(raw["actions"], states, f"{model_path}: embedded actions")
    else:
        ap = actions_path or raw.get("actions_file")
        if not ap:
            raise ModelError(
                f"{model_path}: no terminal actions provided "
                "(expected 'actions_file', embedded 'actions', or --actions)"
            )
        ap = _resolve_aux(ap, base)
        actions = _parse_actions_csv(ap, states)

    if budgets is not None:
        model_budgets = _parse_budgets(list(budgets), "command line")
    elif "budget" in raw:
        model_budgets = _parse_budgets(raw["budget"], f"{model_path}: budget")
    elif plans_path:
        plan_raw = _read_json(plans_path)
        if not isinstance(plan_raw, dict) or "budgets" not in plan_raw:
            raise ModelError(f"{plans_path}: 'budgets' missing")
        model_budgets = _parse_budgets(plan_raw["budgets"], plans_path)
    elif "budgets" in raw:
        model_budgets = _parse_budgets(raw["budgets"], model_path)
    else:
        model_budgets = None

    objective = raw.get("objective") or (
        "minimise expected (inspection fees actually paid + terminal action loss); "
        "budget limits cumulative inspection fees per branch"
    )
    tie_note = raw.get("ties")
    plans_tie = None
    plans_obj = None
    if plans_path:
        plan_raw = _read_json(plans_path)
        if isinstance(plan_raw, dict):
            plans_obj = plan_raw.get("objective")
            plans_tie = plan_raw.get("ties")
    if plans_obj:
        objective = plans_obj
    if plans_tie:
        tie_note = plans_tie

    test_sum = sum((t.cost for t in tests), Fraction(0))
    if model_budgets is not None:
        for b in model_budgets:
            if b > test_sum:
                warnings.append(
                    f"budget {frac_str(b)} exceeds total test cost {frac_str(test_sum)}; "
                    "additional budget cannot be spent"
                )

    # Documented supported scope: <=4 states, <=4 binary tests, <=3 actions.
    # Exceeding it is not fatal (the exact DP still runs), but it is worth
    # flagging because the guarantees and runtime profile are for that scope.
    if len(states) > 4:
        warnings.append(f"model has {len(states)} states; documented scope is <=4")
    if len(tests) > 4:
        warnings.append(f"model has {len(tests)} tests; documented scope is <=4")
    if len(actions) > 3:
        warnings.append(f"model has {len(actions)} terminal actions; documented scope is <=3")
    for t in tests:
        if len(alphabets[t.position]) > 2:
            warnings.append(
                f"test {t.id!r} has {len(alphabets[t.position])} result symbols; "
                "documented scope is binary tests"
            )

    return Model(
        states=states,
        priors=priors,
        tests=tests,
        actions=actions,
        joint=joint,
        alphabets=alphabets,
        budgets=model_budgets,
        objective=str(objective),
        tie_note=str(tie_note) if tie_note else "",
        source=os.path.abspath(model_path),
        warnings=tuple(warnings),
    )


def model_summary(model: Model) -> dict[str, Any]:
    """JSON-safe structural summary used by reports and invariance checks."""
    return {
        "states": [{"state": s, "prior": frac_str(model.priors[s])} for s in model.states],
        "tests": [
            {
                "id": t.id,
                "position": t.position,
                "cost": frac_str(t.cost),
                "requires": list(t.requires),
            }
            for t in model.tests
        ],
        "actions": [
            {"id": a.id, "losses": {s: frac_str(a.losses[s]) for s in model.states}}
            for a in model.actions
        ],
        "alphabets": {str(pos): list(sym) for pos, sym in enumerate(model.alphabets)},
        "budgets": [frac_str(b) for b in (model.budgets or ())],
        "objective": model.objective,
        "tie_note": model.tie_note,
        "warnings": list(model.warnings),
        "total_test_cost": frac_str(model.total_test_cost()),
    }
