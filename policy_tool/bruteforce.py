"""Independent brute-force cross-check (used only by the test suite).

Enumerates every *permutation* of every subset of tests (all observation
histories with purchase order retained) and computes the optimal adaptive
policy value by naive recursion with **no** memoisation and **no** ordering
canonicalisation.  If the DP's order-free merging is correct, the two values
must agree exactly.  Also re-derives the expected cost by walking the primary
policy tree and summing ``P(history) * cost`` directly.
"""

from __future__ import annotations

from fractions import Fraction
from .model import Model


def brute_force_value(model: Model, budget: Fraction) -> Fraction:
    """Optimal expected cost by exhaustive permutation recursion (no memo)."""

    def rec(bought: tuple[str, ...], obs: dict[str, str]) -> Fraction:
        bought_set = frozenset(bought)
        spent = model.spent(bought_set)
        remaining = budget - spent
        post = model.posterior(bought_set, obs)
        best = min(model.action_expectation(a, post) for a in model.actions)
        for test in model.tests:
            if test.id in bought_set:
                continue
            if any(r not in bought_set for r in test.requires):
                continue
            if test.cost > remaining:
                continue
            cont = Fraction(0)
            for symbol in model.alphabets[test.position]:
                p = model.outcome_marginal(bought_set, obs, test.id, symbol)
                if p == 0:
                    continue
                child_obs = dict(obs)
                child_obs[test.id] = symbol
                cont += p * rec(bought + (test.id,), child_obs)
            best = min(best, test.cost + cont)
        return best

    return rec((), {})


def walk_policy_value(solution) -> Fraction:
    """Re-walk the primary tree; sum P(leaf) * (fees + terminal loss)."""
    model = solution.model
    total = Fraction(0)

    def walk(node, prob: Fraction) -> None:
        nonlocal total
        if node.decision_kind == "stop":
            total += prob * (node.spent + node.action_option(node.decision_id).expected_loss)
            return
        for symbol, child in node.children.items():
            if child is not None:
                walk(child, prob * node.branch_probs[symbol])

    walk(solution.root, Fraction(1))
    return total
