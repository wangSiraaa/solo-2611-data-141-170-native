"""Replay a solved policy against a concrete state and full result vector.

Replay does **not** read the solver's stored total.  It walks the primary
strategy tree:

* it validates the supplied state and result vector against the model,
* at every node it reads the tree's recorded choice, and *independently*
  re-scores the terminal actions with the node's posterior to confirm the
  recorded action is still optimal (and reports all equal-cost actions),
* it accumulates the fees actually paid along the path and the real terminal
  loss, and
* it re-averages those path costs over the *original* prior/joint distribution
  and compares that with the tree value, to prove the tree's branch
  probabilities reproduce the frozen joint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .model import Model, ModelError, frac_str
from .solver import Node, Solution, solve


class ReplayError(ModelError):
    """Replay could not be performed with the given inputs."""


@dataclass
class ReplayStep:
    depth: int
    bought_before: list[str]
    observed: dict[str, str]
    posterior: dict[str, Fraction]
    decision_kind: str
    decision_id: str
    reason: str
    fee_paid: Fraction | None
    tied_actions_on_this_node: list[str] = field(default_factory=list)


@dataclass
class PathSummary:
    """One reachable leaf of the primary policy, independently re-derived."""
    state: str
    outcomes: dict[str, str]
    probability: Fraction          # P(state, vector), from the frozen joint
    fees: Fraction
    terminal_action: str
    terminal_loss: Fraction
    total_cost: Fraction


@dataclass
class ReplayResult:
    budget: Fraction
    state: str
    outcomes: dict[str, str]
    steps: list[ReplayStep]
    fees_paid: Fraction
    terminal_action: str
    terminal_loss: Fraction
    total_path_cost: Fraction
    path_probability: Fraction
    expected_reconstructed: Fraction
    tree_value: Fraction
    consistent: bool
    distribution_check: dict[str, Any]
    all_paths: list[PathSummary]
    summary: dict[str, Any]


def _resolve_outcome_vector(model: Model, outcomes: Any, results: Any,
                            ) -> dict[str, str]:
    """Accept a full vector (str/list/dict) or partial named ``results``."""
    resolved: dict[str, str] = {}
    if outcomes is not None:
        if isinstance(outcomes, dict):
            resolved.update({str(k).strip(): str(v).strip() for k, v in outcomes.items()})
        else:
            if isinstance(outcomes, (list, tuple)):
                vector = tuple(str(x).strip() for x in outcomes)
            elif isinstance(outcomes, str):
                text = outcomes.strip()
                if "," in text:
                    vector = tuple(p.strip() for p in text.split(","))
                elif model.n_tests == 1:
                    vector = (text,)
                else:
                    vector = tuple(text)
            else:
                raise ReplayError(
                    f"outcomes must be a string, list, or object; got "
                    f"{type(outcomes).__name__}"
                )
            if len(vector) != model.n_tests:
                raise ReplayError(
                    f"full result vector has {len(vector)} symbol(s) but the model "
                    f"defines {model.n_tests} test(s) in test_order "
                    f"({', '.join(t.id for t in model.tests)})"
                )
            for test, symbol in zip(model.tests, vector):
                resolved[test.id] = symbol
    if results:
        for key, value in dict(results).items():
            key = str(key).strip()
            resolved[key] = str(value).strip()
    unknown = sorted(set(resolved) - {t.id for t in model.tests})
    if unknown:
        raise ReplayError(f"unknown test id(s) in results: {', '.join(unknown)}")
    return resolved


def replay(solution: Solution, state: str,
           outcomes: Any = None, results: Any = None) -> ReplayResult:
    """Walk the primary strategy for a concrete state/full result vector."""
    model = solution.model
    if state not in model.states:
        raise ReplayError(
            f"unknown state {state!r}; expected one of {', '.join(model.states)}"
        )
    observed = _resolve_outcome_vector(model, outcomes, results)

    # The fixed full vector must actually be possible under the true state
    # (not merely under the posterior) or the replay is meaningless.
    full = {t.id: observed.get(t.id) for t in model.tests}
    if any(v is None for v in full.values()):
        missing = [t.id for t in model.tests if t.id not in observed]
        raise ReplayError(
            f"missing result(s) for test(s): {', '.join(missing)}"
        )
    vector = tuple(full[t.id] for t in model.tests)
    for test, symbol in zip(model.tests, vector):
        if symbol not in model.alphabets[test.position]:
            raise ReplayError(
                f"result {symbol!r} for test {test.id!r} is not in the observed "
                f"alphabet {list(model.alphabets[test.position])}"
            )
    joint_prob = model.joint_prob(state, vector)
    if joint_prob <= 0:
        raise ReplayError(
            f"result vector {''.join(vector)!r} has probability 0 under state "
            f"{state!r}; this path cannot occur"
        )

    node: Node = solution.root
    steps: list[ReplayStep] = []
    fees = Fraction(0)
    while node.decision_kind == "test":
        test_id = node.decision_id
        chosen = node.test_option(test_id)
        symbol = observed[test_id]
        if symbol not in chosen.branch_probs:
            raise ReplayError(
                f"test {test_id!r} has no branch {symbol!r} at depth {node.depth}"
            )
        if chosen.branch_probs[symbol] <= 0:
            raise ReplayError(
                f"branch {test_id}={symbol!r} has probability 0 at depth "
                f"{node.depth}; the strategy never reaches a posterior here"
            )
        # Re-score terminal actions from the recorded posterior independently,
        # then report every candidate that exactly ties the node value.
        rescored = {a.id: model.action_expectation(a, node.posterior) for a in model.actions}
        ties_at_node = [("action", aid) for aid in sorted(
            aid for aid, val in rescored.items() if val == node.value
        )] + [("test", tid) for tid in sorted(
            (o.test_id for o in node.tests
             if o.expected_total is not None and o.expected_total == node.value),
            key=lambda tid: model.test_by_id(tid).position,
        )]
        steps.append(ReplayStep(
            depth=node.depth,
            bought_before=list(t.id for t in model.tests if t.id in node.bought),
            observed=dict(node.obs),
            posterior=dict(node.posterior),
            decision_kind="test",
            decision_id=test_id,
            reason=f"buy test {test_id} for fee {frac_str(chosen.cost)}",
            fee_paid=chosen.cost,
            tied_actions_on_this_node=[f"{k}:{i}" for k, i in ties_at_node],
        ))
        fees += chosen.cost
        child = node.children.get(symbol)
        if child is None:
            raise ReplayError(
                f"internal: missing child for {test_id}={symbol!r}"
            )
        node = child

    action = node.action_option(node.decision_id)
    rescored_final = {a.id: model.action_expectation(a, node.posterior) for a in model.actions}
    best_final = min(rescored_final.values())
    if rescored_final[node.decision_id] != best_final:
        raise ReplayError(
            f"internal: recorded terminal action {node.decision_id!r} is not optimal "
            f"at depth {node.depth} ({frac_str(rescored_final[node.decision_id])} vs "
            f"{frac_str(best_final)})"
        )
    terminal_ties = sorted(aid for aid, val in rescored_final.items() if val == best_final)
    steps.append(ReplayStep(
        depth=node.depth,
        bought_before=list(t.id for t in model.tests if t.id in node.bought),
        observed=dict(node.obs),
        posterior=dict(node.posterior),
        decision_kind="stop",
        decision_id=node.decision_id,
        reason=_reason_text(node),
        fee_paid=None,
        tied_actions_on_this_node=[f"action:{a}" for a in terminal_ties],
    ))

    # Real loss under the supplied true state, not an expectation.
    loss = action.losses[state]
    paid = model.spent(node.bought)
    if paid != fees:
        raise ReplayError(
            f"internal: summed fees {frac_str(fees)} != branch spent {frac_str(paid)}"
        )
    if fees > solution.budget:
        raise ReplayError(
            f"branch fees {frac_str(fees)} exceed budget {frac_str(solution.budget)}"
        )
    total = fees + loss

    # Re-average path costs over the frozen distribution, enumerating all paths.
    all_paths = _all_paths(solution)
    agg = _replay_expectation(solution, all_paths)
    expected = agg["expected_total"]
    consistent = expected == solution.value
    if consistent is not True:
        raise ReplayError(
            f"strategy tree re-averaged expectation {frac_str(expected)} disagrees "
            f"with solver value {frac_str(solution.value)}; distribution is not reproduced"
        )

    return ReplayResult(
        budget=solution.budget,
        state=state,
        outcomes={t.id: observed[t.id] for t in model.tests},
        steps=steps,
        fees_paid=fees,
        terminal_action=node.decision_id,
        terminal_loss=loss,
        total_path_cost=total,
        path_probability=model.priors[state] * joint_prob,
        expected_reconstructed=expected,
        tree_value=solution.value,
        consistent=bool(consistent),
        distribution_check=agg,
        all_paths=all_paths,
        summary={},
    )


def _reason_text(node: Node) -> str:
    exp = node.stop_explanation or {}
    code = exp.get("reason_code")
    mapping = {
        "no_tests_in_model": "the model defines no tests to buy",
        "all_tests_already_bought": "all available tests have been bought",
        "constraints_block_all_remaining_tests":
            "remaining tests blocked by budget/prerequisites",
        "additional_information_exactly_offsets_fee":
            "extra information exactly offsets its fee (tie -> stop)",
        "additional_information_not_worth_its_fee":
            "extra information is not worth its fee",
    }
    return mapping.get(code, f"stop: {code}")


def _path_for_vector(node: Node, model: Model, vector: dict[str, str]
                     ) -> tuple[Fraction, str, Fraction]:
    """Follow the recorded policy for one full vector; no stored probs used.

    Returns ``(fees, terminal_action, path_probability_conditional)`` where the
    probability is the exact probability of the *observed prefix decisions*
    given the model, recomputed here rather than read from the tree.
    """
    prob = Fraction(1)
    while node.decision_kind == "test":
        test_id = node.decision_id
        symbol = vector[test_id]
        # Recompute P(symbol | history) from the frozen joint.
        branch = model.outcome_marginal(node.bought, node.obs, test_id, symbol)
        if branch <= 0:
            raise ReplayError(
                f"vector assigns {test_id}={symbol!r} but that branch has "
                "probability 0 under this history"
            )
        prob *= branch
        child = node.children.get(symbol)
        if child is None:
            raise ReplayError(f"policy has no branch for {test_id}={symbol!r}")
        node = child
    return model.spent(node.bought), node.decision_id, prob


def _all_paths(solution: Solution) -> list[PathSummary]:
    """Enumerate every (state, full vector) leaf and its real path cost.

    Values are obtained by walking the recorded policy and using the *true*
    state's loss, then weighting by the frozen joint probability
    ``P(state) * P(vector | state)``.  Nothing here reads a stored total.
    """
    model = solution.model
    paths: list[PathSummary] = []
    for state in model.states:
        for vector_tuple, joint_prob in model.joint[state].items():
            if joint_prob <= 0:
                continue
            vector = {t.id: vector_tuple[t.position] for t in model.tests}
            fees, action_id, _ = _path_for_vector(solution.root, model, vector)
            loss = model.action_by_id(action_id).losses[state]
            paths.append(PathSummary(
                state=state,
                outcomes=vector,
                probability=model.priors[state] * joint_prob,
                fees=fees,
                terminal_action=action_id,
                terminal_loss=loss,
                total_cost=fees + loss,
            ))
    return paths


def _replay_expectation(solution: Solution,
                        paths: list[PathSummary] | None = None) -> dict[str, Any]:
    """Re-derive the expectation from the *original* prior/joint distribution.

    Uses ``P(state, vector) * path_cost`` summed over every non-zero joint row.
    Stored tree totals and stored branch probabilities are not consulted.
    """
    model = solution.model
    if paths is None:
        paths = _all_paths(solution)
    expected_fees = Fraction(0)
    expected_loss = Fraction(0)
    path_prob_sum = Fraction(0)
    per_state: dict[str, Fraction] = {s: Fraction(0) for s in model.states}

    for path in paths:
        row_prob = path.probability
        if row_prob <= 0:
            continue
        expected_fees += row_prob * path.fees
        expected_loss += row_prob * path.terminal_loss
        path_prob_sum += row_prob
        per_state[path.state] += row_prob * path.total_cost

    if path_prob_sum != 1:
        raise ReplayError(
            f"re-assembled joint probabilities sum to {frac_str(path_prob_sum)}, not 1"
        )
    total = expected_fees + expected_loss
    return {
        "expected_inspection_fees": frac_str(expected_fees),
        "expected_terminal_loss": frac_str(expected_loss),
        "expected_total": total,
        "expected_total_text": frac_str(total),
        "per_state_expected_cost": {s: frac_str(v) for s, v in per_state.items()},
        "branch_probability_sum": frac_str(path_prob_sum),
        "path_count": len(paths),
        "distribution_source": (
            "frozen prior + joint rows re-summed exactly; policy re-walked per "
            "full result vector without reading stored branch probabilities"
        ),
    }


def replay_summary(result: ReplayResult) -> dict[str, Any]:
    terminal_action_counts: dict[str, int] = {}
    for path in result.all_paths:
        key = path.terminal_action
        terminal_action_counts[key] = terminal_action_counts.get(key, 0) + 1
    return {
        "budget": frac_str(result.budget),
        "state": result.state,
        # ``outcomes`` is built in test_order, so joining its values is the
        # full result vector exactly as recorded in joint.csv.
        "full_result_vector": "".join(result.outcomes.values()),
        "outcomes": dict(result.outcomes),
        "fees_paid": frac_str(result.fees_paid),
        "terminal_action": result.terminal_action,
        "terminal_loss": frac_str(result.terminal_loss),
        "total_path_cost": frac_str(result.total_path_cost),
        "path_probability": frac_str(result.path_probability),
        "expected_reconstructed": frac_str(result.expected_reconstructed),
        "tree_value": frac_str(result.tree_value),
        "consistent": result.consistent,
        "distribution_check": result.distribution_check,
        "terminal_action_counts_over_all_paths": terminal_action_counts,
        "all_paths": [
            {
                "state": p.state,
                "outcomes": dict(p.outcomes),
                "probability": frac_str(p.probability),
                "fees": frac_str(p.fees),
                "terminal_action": p.terminal_action,
                "terminal_loss": frac_str(p.terminal_loss),
                "total_cost": frac_str(p.total_cost),
            }
            for p in result.all_paths
        ],
        "steps": [
            {
                "depth": s.depth,
                "known_results": s.observed,
                "posterior": {k: frac_str(v) for k, v in s.posterior.items()},
                "decision": {"kind": s.decision_kind, "id": s.decision_id},
                "reason": s.reason,
                "fee_paid": frac_str(s.fee_paid) if s.fee_paid is not None else None,
                "ties": s.tied_actions_on_this_node,
            }
            for s in result.steps
        ],
    }


def replay_model(model: Model, budget: Fraction | None, state: str,
                 outcomes: Any = None, results: Any = None) -> ReplayResult:
    solution = solve(model, budget)
    return replay(solution, state, outcomes=outcomes, results=results)
