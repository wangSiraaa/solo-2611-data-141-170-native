"""Exact dynamic programme over reachable observation histories.

The state of knowledge is fully captured by

    ( set of tests already bought , results observed for exactly those tests )

and *not* by the order in which the tests were bought.  Two purchase orders
that buy the same set and see the same results lead to the same posterior
and the same remaining budget, so they are one and the same sub-problem;
the memo key below enforces that merge.  Because the joint distribution is
frozen ("原始检测结果一次抽定，重复读取不产生新结果"), reordering a
purchase cannot change the distribution of results either.

At every reachable history the solver compares, exactly:

  * stopping now with each terminal action, and
  * buying each currently purchasable test (prerequisites met and fee within
    the branch's remaining budget), paying its fee and continuing optimally.

It never pre-orders tests and never ranks by information gain alone.

Tie-breaking is by exact rational equality (no tolerance): a candidate is tied
only when its expected value is *identically* equal as a Fraction.  The public
stable rule for the primary policy is:

  1. If a terminal action attains the node value, stop wins over buying a test
     (a test whose fee is exactly offset by its information is not bought).
  2. Among tying terminal actions, the lexicographically smallest action id.
  3. Otherwise the tying test with the smallest ``test_order`` position
     (then id).

All tying candidates are still reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .model import (
    Action,
    Model,
    ModelError,
    ZeroProbabilityError,
    frac_str,
    parse_fraction,
)


TIE_RULE = (
    "Exact equality of expected cost only (no tolerance). Primary policy: "
    "(1) stopping beats buying a test on an exact tie; "
    "(2) among tying terminal actions, smallest action id lexicographically; "
    "(3) otherwise among tying tests, smallest test_order position, then id."
)


# --------------------------------------------------------------------------
# Options exposed at every node
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionOption:
    action_id: str
    expected_loss: Fraction
    losses: dict[str, Fraction]

    def to_dict(self, value: Fraction) -> dict[str, Any]:
        return {
            "action": self.action_id,
            "expected_loss": frac_str(self.expected_loss),
            "expected_loss_float": float(self.expected_loss),
            "is_optimal": self.expected_loss == value,
        }


@dataclass
class TestOption:
    test_id: str
    position: int
    cost: Fraction
    requires: tuple[str, ...]
    missing_requires: tuple[str, ...]
    budget_ok: bool
    remaining_budget: Fraction
    expected_total: Fraction | None = None   # cost + optimal continuation
    continuation: Fraction | None = None
    branch_probs: dict[str, Fraction] = field(default_factory=dict)

    @property
    def buyable(self) -> bool:
        return self.expected_total is not None

    def to_dict(self, value: Fraction, selected: bool) -> dict[str, Any]:
        item: dict[str, Any] = {
            "test": self.test_id,
            "position": self.position,
            "cost": frac_str(self.cost),
            "requires": list(self.requires),
            "missing_requires": list(self.missing_requires),
            "budget_ok": self.budget_ok,
            "remaining_budget": frac_str(self.remaining_budget),
            "status": "available" if self.buyable else "blocked",
        }
        if self.expected_total is None:
            reasons = []
            if self.missing_requires:
                reasons.append("unsatisfied prerequisite(s): "
                               + ", ".join(self.missing_requires))
            if not self.budget_ok:
                reasons.append(
                    f"fee {frac_str(self.cost)} exceeds remaining budget "
                    f"{frac_str(self.remaining_budget)}"
                )
            item["blocked_reasons"] = reasons
            item["expected_total"] = None
            item["selected"] = False
        else:
            item["expected_total"] = frac_str(self.expected_total)
            item["expected_total_float"] = float(self.expected_total)
            item["continuation_value"] = frac_str(self.continuation)
            item["is_optimal"] = self.expected_total == value
            item["selected"] = selected
        return item


@dataclass
class Node:
    key: tuple[Any, ...]
    bought: frozenset[str]
    obs: dict[str, str]                      # canonical: test_order order
    reach_probability: Fraction              # P(this history), unconditional
    posterior: dict[str, Fraction]
    spent: Fraction
    budget: Fraction
    remaining: Fraction
    actions: list[ActionOption]
    tests: list[TestOption]
    value: Fraction
    decision_kind: str                        # "stop" | "test"
    decision_id: str
    ties: list[tuple[str, str]]               # ("action"|"test", id)
    stop_explanation: dict[str, Any] | None
    children: dict[str, "Node | None"]        # primary test only
    branch_probs: dict[str, Fraction]         # primary test only
    depth: int

    @property
    def ordered_obs(self) -> list[tuple[str, str]]:
        return list(self.obs.items())

    def action_option(self, action_id: str) -> ActionOption:
        for option in self.actions:
            if option.action_id == action_id:
                return option
        raise KeyError(action_id)

    def test_option(self, test_id: str) -> TestOption:
        for option in self.tests:
            if option.test_id == test_id:
                return option
        raise KeyError(test_id)


def _ordered_bought(model: Model, bought: frozenset[str]) -> tuple[str, ...]:
    return tuple(t.id for t in model.tests if t.id in bought)


def history_key(model: Model, bought: frozenset[str], obs: dict[str, str]) -> tuple[Any, ...]:
    """Order-insensitive canonical key ``(bought in test_order, observed values)``."""
    bought_order = _ordered_bought(model, bought)
    return (bought_order, tuple((tid, obs[tid]) for tid in bought_order))


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------

class Solver:
    def __init__(self, model: Model, budget: Fraction | None = None):
        if budget is None:
            # No externally supplied budget: the cap is the sum of all fees,
            # which can never bind because each test is bought at most once.
            budget = model.total_test_cost()
            self.unlimited = True
        else:
            budget = parse_fraction(budget, "budget")
            if budget < 0:
                raise ModelError(f"negative budget {frac_str(budget)}")
            self.unlimited = False
        self.model = model
        self.budget = budget
        self._cache: dict[tuple[Any, ...], Node] = {}
        self.memo_hits = 0
        self._build_calls = 0
        self.root = self._build(frozenset(), {}, depth=0)

    # ---- core recursion --------------------------------------------------

    def _build(self, bought: frozenset[str], obs: dict[str, str], depth: int = 0) -> Node:
        key = history_key(self.model, bought, obs)
        existing = self._cache.get(key)
        if existing is not None:
            self.memo_hits += 1
            return existing
        self._build_calls += 1

        model = self.model
        try:
            posterior = model.posterior(bought, obs)
        except ZeroProbabilityError:
            raise  # never fabricate a posterior
        reach = model.evidence(bought, obs)
        if reach <= 0:  # defensive: should be unreachable via positive branches
            raise ZeroProbabilityError(
                f"history {key!r} has zero probability; refusing to solve it"
            )
        spent = model.spent(bought)
        remaining = self.budget - spent
        if remaining < 0:
            raise ModelError(
                f"internal budget violation: spent {frac_str(spent)} exceeds "
                f"budget {frac_str(self.budget)}"
            )

        actions = [
            ActionOption(
                action_id=a.id,
                expected_loss=model.action_expectation(a, posterior),
                losses=dict(a.losses),
            )
            for a in model.actions
        ]
        best_stop = min(actions, key=lambda o: (o.expected_loss, o.action_id))

        tests: list[TestOption] = []
        for test in model.tests:
            if test.id in bought:
                continue
            missing = tuple(r for r in test.requires if r not in bought)
            budget_ok = test.cost <= remaining
            option = TestOption(
                test_id=test.id,
                position=test.position,
                cost=test.cost,
                requires=test.requires,
                missing_requires=missing,
                budget_ok=budget_ok,
                remaining_budget=remaining,
            )
            if not missing and budget_ok:
                continuation = Fraction(0)
                probs: dict[str, Fraction] = {}
                for symbol in model.alphabets[test.position]:
                    branch = model.outcome_marginal(bought, obs, test.id, symbol)
                    probs[symbol] = branch
                    if branch > 0:
                        child_obs = dict(obs)
                        child_obs[test.id] = symbol
                        child = self._build(bought | {test.id}, child_obs, depth + 1)
                        continuation += branch * child.value
                total_branch = sum(probs.values(), Fraction(0))
                if total_branch != 1:
                    raise ModelError(
                        f"internal inconsistency: conditional probabilities for test "
                        f"{test.id!r} at history {key!r} sum to {frac_str(total_branch)}, not 1"
                    )
                option.branch_probs = probs
                option.continuation = continuation
                option.expected_total = test.cost + continuation
            tests.append(option)

        available = [o for o in tests if o.buyable]
        best_test = min(available, key=lambda o: (o.expected_total, o.position)) if available else None

        # Exact tie: stopping is preferred over buying a test.
        if best_test is None or best_test.expected_total >= best_stop.expected_loss:
            value = best_stop.expected_loss
            decision_kind, decision_id = "stop", best_stop.action_id
        else:
            value = best_test.expected_total
            decision_kind, decision_id = "test", best_test.test_id

        ties_actions = sorted(
            (a.action_id for a in actions if a.expected_loss == value)
        )
        ties_tests = sorted(
            (o.test_id for o in available if o.expected_total == value),
            key=lambda tid: model.test_by_id(tid).position,
        )
        ties = [("action", aid) for aid in ties_actions] + [("test", tid) for tid in ties_tests]

        stop_explanation = None
        if decision_kind == "stop":
            stop_explanation = _stop_explanation(
                tests, available, best_test, best_stop, value, remaining,
                total_tests=len(model.tests),
            )

        children: dict[str, Node | None] = {}
        branch_probs: dict[str, Fraction] = {}
        if decision_kind == "test":
            chosen = next(o for o in tests if o.test_id == decision_id)
            branch_probs = dict(chosen.branch_probs)
            for symbol, branch in branch_probs.items():
                if branch > 0:
                    child_obs = dict(obs)
                    child_obs[chosen.test_id] = symbol
                    children[symbol] = self._build(
                        bought | {chosen.test_id}, child_obs, depth + 1
                    )
                else:
                    # Zero-probability continuation: reported, but no posterior
                    # is invented and it contributes nothing to the expectation.
                    children[symbol] = None

        node = Node(
            key=key,
            bought=bought,
            obs=dict(obs),
            reach_probability=reach,
            posterior=posterior,
            spent=spent,
            budget=self.budget,
            remaining=remaining,
            actions=actions,
            tests=tests,
            value=value,
            decision_kind=decision_kind,
            decision_id=decision_id,
            ties=ties,
            stop_explanation=stop_explanation,
            children=children,
            branch_probs=branch_probs,
            depth=depth,
        )
        self._cache[key] = node
        return node

    # ---- lookup ----------------------------------------------------------

    def node_for(self, bought: frozenset[str], obs: dict[str, str]) -> Node:
        key = history_key(self.model, bought, obs)
        node = self._cache.get(key)
        if node is None:
            node = self._build(bought, dict(obs))
        return node

    @property
    def node_count(self) -> int:
        return len(self._cache)


def _stop_explanation(tests: list[TestOption], available: list[TestOption],
                      best_test: TestOption | None, best_stop: ActionOption,
                      value: Fraction, remaining: Fraction,
                      total_tests: int = 0) -> dict[str, Any]:
    unbought = len(tests)
    blocked = [o for o in tests if not o.buyable]
    budget_limited = any(
        o.budget_ok is False for o in blocked
    )
    prereq_limited = any(o.missing_requires for o in blocked)
    info_better = best_test is not None and best_test.expected_total < value
    margin = None
    if best_test is not None:
        margin = best_test.expected_total - value  # >= 0 when stopping
    if total_tests == 0:
        code = "no_tests_in_model"
    elif unbought == 0:
        code = "all_tests_already_bought"
    elif not available:
        code = "constraints_block_all_remaining_tests"
    elif best_test is not None and best_test.expected_total == value:
        code = "additional_information_exactly_offsets_fee"
    else:
        code = "additional_information_not_worth_its_fee"
    return {
        "reason_code": code,
        "buyable_tests": [o.test_id for o in available],
        "blocked_tests": [o.test_id for o in blocked],
        "best_buyable_test": best_test.test_id if best_test else None,
        "best_buyable_test_expected_total": (
            frac_str(best_test.expected_total) if best_test else None
        ),
        "stop_expected_cost": frac_str(value),
        "margin_over_stop": frac_str(margin) if margin is not None else None,
        "extra_information_would_pay": info_better,
        "budget_limited_at_this_node": budget_limited,
        "prerequisites_limited_at_this_node": prereq_limited,
        "remaining_budget": frac_str(remaining),
    }


# --------------------------------------------------------------------------
# Solution object and cost accounting
# --------------------------------------------------------------------------

@dataclass
class Solution:
    model: Model
    budget: Fraction
    unlimited: bool
    root: Node
    node_count: int
    memo_hits: int
    build_calls: int
    leaves: int
    max_depth: int
    expected_fees: Fraction
    expected_terminal_loss: Fraction
    no_detection_cost: Fraction
    no_detection_actions: list[tuple[str, Fraction]]
    gross_information_value: Fraction
    net_information_value: Fraction

    @property
    def value(self) -> Fraction:
        return self.root.value

    def cost_composition(self) -> dict[str, Any]:
        return {
            "budget": frac_str(self.budget),
            "budget_unlimited": self.unlimited,
            "no_detection": {
                "expected_terminal_loss": frac_str(self.no_detection_cost),
                "optimal_terminal_actions": [
                    {"action": aid, "expected_loss": frac_str(loss)}
                    for aid, loss in self.no_detection_actions
                ],
            },
            "full_adaptive_strategy": {
                "expected_inspection_fees": frac_str(self.expected_fees),
                "expected_terminal_loss": frac_str(self.expected_terminal_loss),
                "total_expected_cost": frac_str(self.value),
                "check_fees_plus_loss": frac_str(
                    self.expected_fees + self.expected_terminal_loss
                ),
            },
            "information_value": {
                "gross_reduction_in_terminal_loss": frac_str(self.gross_information_value),
                "inspection_fees_paid": frac_str(self.expected_fees),
                "net_value_after_fees": frac_str(self.net_information_value),
                "definition": (
                    "gross = no-detection terminal loss - full-strategy terminal loss; "
                    "net = gross - inspection fees = total saving versus no detection. "
                    "Information value is never merged with inspection fees."
                ),
            },
        }


def solve(model: Model, budget: Fraction | None = None) -> Solution:
    """Solve for an adaptive policy under ``budget`` (``None`` = never binding).

    If ``budget`` is omitted and the model carries exactly one budget, that
    budget is used; with several, the caller must choose explicitly.
    """
    if budget is None and model.budgets:
        if len(model.budgets) == 1:
            budget = model.budgets[0]
        else:
            raise ModelError(
                "several budgets are configured; pass one explicitly "
                f"(choices: {', '.join(frac_str(b) for b in model.budgets)})"
            )
    solver = Solver(model, budget)

    leaves = 0
    max_depth = 0
    expected_fees = Fraction(0)
    expected_loss = Fraction(0)
    reach_sum = Fraction(0)
    stack = [solver.root]
    while stack:
        node = stack.pop()
        if node.decision_kind == "stop":
            leaves += 1
            max_depth = max(max_depth, node.depth)
            expected_fees += node.reach_probability * node.spent
            expected_loss += node.reach_probability * node.action_option(node.decision_id).expected_loss
            reach_sum += node.reach_probability
        else:
            for symbol, child in node.children.items():
                if child is None:
                    continue
                stack.append(child)
    if reach_sum != 1:
        raise ModelError(
            f"internal error: primary strategy path probabilities sum to "
            f"{frac_str(reach_sum)}, not 1"
        )
    value = expected_fees + expected_loss
    if value != solver.root.value:
        raise ModelError(
            "internal error: strategy leaf accounting "
            f"({frac_str(value)}) disagrees with root value "
            f"({frac_str(solver.root.value)})"
        )

    prior = model.priors
    no_detection_options = sorted(
        ((a.id, model.action_expectation(a, prior)) for a in model.actions),
        key=lambda item: (item[1], item[0]),
    )
    no_detection = no_detection_options[0][1]
    no_detection_ties = [(aid, val) for aid, val in no_detection_options if val == no_detection]

    gross = no_detection - expected_loss
    net = no_detection - solver.root.value
    if gross < 0 or net < 0:
        raise ModelError(
            "internal error: information worsened expected loss "
            f"(gross={frac_str(gross)}, net={frac_str(net)})"
        )
    if gross - expected_fees != net:
        raise ModelError("internal error: information value decomposition does not balance")

    return Solution(
        model=model,
        budget=solver.budget,
        unlimited=solver.unlimited,
        root=solver.root,
        node_count=solver.node_count,
        memo_hits=solver.memo_hits,
        build_calls=solver._build_calls,
        leaves=leaves,
        max_depth=max_depth,
        expected_fees=expected_fees,
        expected_terminal_loss=expected_loss,
        no_detection_cost=no_detection,
        no_detection_actions=no_detection_ties,
        gross_information_value=gross,
        net_information_value=net,
    )


def solve_model(model_path: str, budget: Fraction | None = None,
                actions_path: str | None = None, joint_path: str | None = None,
                plans_path: str | None = None) -> Solution:
    """Convenience: load a model from disk and solve it in one call."""
    from .model import load_model
    model = load_model(model_path, actions_path=actions_path,
                       joint_path=joint_path, plans_path=plans_path)
    return solve(model, budget)


def solution_summary(solution: Solution) -> dict[str, Any]:
    """JSON-safe summary (full tree is produced by :mod:`policy_tool.report`)."""
    return {
        "budget": frac_str(solution.budget),
        "budget_unlimited": solution.unlimited,
        "value": frac_str(solution.value),
        "value_float": float(solution.value),
        "tie_rule": TIE_RULE,
        "solver": {
            "distinct_histories_solved": solution.node_count,
            "history_build_calls": solution.build_calls,
            "memo_cache_hits_merged": solution.memo_hits,
            "strategy_leaves": solution.leaves,
            "max_depth": solution.max_depth,
            "note": (
                "A history is keyed by (bought test set, observed results); purchase "
                "order is ignored, so orders that reach the same information are "
                "merged into one DP sub-problem."
            ),
        },
        "cost_composition": solution.cost_composition(),
        "root_decision": {
            "kind": solution.root.decision_kind,
            "id": solution.root.decision_id,
            "value": frac_str(solution.root.value),
            "tied_candidates": [
                {"kind": kind, "id": cid} for kind, cid in solution.root.ties
            ],
        },
    }
