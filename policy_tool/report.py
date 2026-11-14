"""Static, human-readable and JSON/Markdown reports for a solved policy."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from .model import Model, frac_str, frac_float
from .solver import TIE_RULE, Node, Solution, solution_summary


def _pct(f: Fraction) -> str:
    return f"{frac_float(f) * 100:.4f}%"


def _num(f: Fraction) -> str:
    text = frac_str(f)
    if f.denominator == 1:
        return text
    return f"{text} ({frac_float(f):.6f})"


def node_to_dict(node: Node, model: Model) -> dict[str, Any]:
    """Structured node: known results, posterior, budget, options, costs."""
    result: dict[str, Any] = {
        "depth": node.depth,
        "known_results": dict(node.obs),
        "bought_tests": [t.id for t in model.tests if t.id in node.bought],
        "reach_probability": frac_str(node.reach_probability),
        "posterior": {s: frac_str(r) for s, r in node.posterior.items()},
        "budget": {
            "total": frac_str(node.budget),
            "spent": frac_str(node.spent),
            "remaining": frac_str(node.remaining),
        },
        "terminal_actions": [a.to_dict(node.value) for a in node.actions],
        "tests": [
            t.to_dict(node.value, t.test_id == node.decision_id)
            for t in node.tests
        ],
        "value": frac_str(node.value),
        "decision": {"kind": node.decision_kind, "id": node.decision_id},
        "ties": [{"kind": k, "id": i} for k, i in node.ties],
        "stop_explanation": node.stop_explanation,
    }
    if node.decision_kind == "test":
        result["children"] = {
            symbol: (node_to_dict(child, model) if child is not None else {
                "unreachable": True,
                "reason": "zero-probability branch; no posterior fabricated",
                "branch_probability": "0",
            })
            for symbol, child in node.children.items()
        }
        result["branch_probabilities"] = {s: frac_str(p) for s, p in node.branch_probs.items()}
    return result


def strategy_tree(solution: Solution) -> dict[str, Any]:
    return {
        "budget": frac_str(solution.budget),
        "budget_unlimited": solution.unlimited,
        "tie_rule": TIE_RULE,
        "expected_total": frac_str(solution.value),
        "tree": node_to_dict(solution.root, solution.model),
    }


def _render_node(node: Node, model: Model, budget: Fraction, lines: list[str],
                 indent: str = "") -> None:
    def emit(text: str = "") -> None:
        lines.append(f"{indent}{text}")

    hist = ", ".join(f"{tid}={val}" for tid, val in node.obs.items()) or "none"
    emit(f"[depth {node.depth}] history: {hist}")
    emit(f"  reach probability: {_num(node.reach_probability)}  ({_pct(node.reach_probability)})")
    post = "  ".join(f"{s}={frac_str(v)}" for s, v in node.posterior.items())
    emit(f"  posterior: {post}")
    emit(f"  budget: {frac_str(budget)}  spent: {frac_str(node.spent)}  "
         f"remaining: {frac_str(node.remaining)}")
    emit("  terminal actions (expected loss under posterior):")
    for a in sorted(node.actions, key=lambda o: (o.expected_loss, o.action_id)):
        mark = "*" if a.expected_loss == node.value and node.decision_kind == "stop" else " "
        tie = " [tie]" if a.expected_loss == node.value and a.action_id != node.decision_id else ""
        emit(f"    {mark} {a.action_id:<16} {_num(a.expected_loss)}{tie}")
    emit("  tests (expected cost = fee + optimal continuation):")
    blocked_any = False
    for t in node.tests:
        if t.buyable:
            mark = "*" if node.decision_kind == "test" and t.test_id == node.decision_id else " "
            tie = " [tie]" if t.expected_total == node.value and t.test_id != node.decision_id else ""
            emit(f"    {mark} {t.test_id:<12} fee {frac_str(t.cost):<6} "
                 f"requires {list(t.requires)}  expected total {_num(t.expected_total)}{tie}")
        else:
            blocked_any = True
            why = []
            if t.missing_requires:
                why.append("missing prerequisites " + ", ".join(t.missing_requires))
            if not t.budget_ok:
                why.append(f"fee {frac_str(t.cost)} > remaining {frac_str(t.remaining_budget)}")
            emit(f"      {t.test_id:<12} BLOCKED: {'; '.join(why)}")
    if node.decision_kind == "test":
        chosen = node.test_option(node.decision_id)
        emit(f"  => BUY {node.decision_id} (expected total {_num(node.value)})")
        emit(f"     branch probabilities: "
             + ", ".join(f"{s}:{frac_str(p)}" for s, p in chosen.branch_probs.items()))
        for symbol, child in node.children.items():
            emit()
            prob = chosen.branch_probs[symbol]
            emit(f"  -- branch {node.decision_id}={symbol} "
                 f"(prob {_num(prob)}) --")
            if child is None:
                emit("    zero-probability branch: no posterior, contributes 0")
            else:
                _render_node(child, model, budget, lines, indent + "    ")
    else:
        exp = node.stop_explanation or {}
        if exp:
            emit(f"  => STOP: {exp.get('reason_code')}")
            if exp.get("best_buyable_test") is not None:
                emit(f"     best buyable test {exp['best_buyable_test']} has expected "
                     f"total {exp['best_buyable_test_expected_total']} vs stop "
                     f"{exp['stop_expected_cost']} (margin {exp['margin_over_stop']})")
            if exp.get("blocked_tests"):
                emit(f"     blocked by budget/prerequisites: "
                     f"{', '.join(exp['blocked_tests'])}")
        else:
            emit(f"  => STOP (terminal action {node.decision_id})")


def render_tree(solution: Solution) -> str:
    lines: list[str] = []
    model = solution.model
    lines.append("ADAPTIVE DETECTION POLICY TREE (exact rational values)")
    lines.append("=" * 72)
    lines.append(f"budget per branch: {frac_str(solution.budget)}"
                 + (" (unlimited/diagnostic)" if solution.unlimited else ""))
    lines.append(f"primary tie rule: {TIE_RULE}")
    lines.append(f"expected total cost: {_num(solution.value)}")
    lines.append("")
    _render_node(solution.root, model, solution.budget, lines)
    return "\n".join(lines)


def render_report(solution: Solution) -> str:
    model = solution.model
    summary = solution_summary(solution)
    comp = summary["cost_composition"]
    lines: list[str] = []
    lines.append("# Static strategy report")
    lines.append("")
    lines.append(f"Model: `{model.source}`")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(model.objective)
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append("States and priors (exact):")
    for s in model.states:
        lines.append(f"- `{s}`: {frac_str(model.priors[s])}")
    lines.append("")
    lines.append("Tests (declaration = outcome position in `test_order`):")
    for t in model.tests:
        lines.append(f"- `{t.id}`: cost {frac_str(t.cost)}, "
                     f"requires {list(t.requires)}, result symbols "
                     f"{list(model.alphabets[t.position])}")
    lines.append("")
    lines.append("Terminal actions (loss by true state):")
    for a in model.actions:
        losses = ", ".join(f"{s}={frac_str(a.losses[s])}" for s in model.states)
        lines.append(f"- `{a.id}`: {losses}")
    if model.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in model.warnings:
            lines.append(f"- {w}")
    lines.append("")
    lines.append("## Cost composition")
    lines.append("")
    nd = comp["no_detection"]
    lines.append(f"- No-detection plan: expected terminal loss **{nd['expected_terminal_loss']}** "
                 f"with {', '.join(a['action'] for a in nd['optimal_terminal_actions'])} "
                 "(ties shown; primary = smallest action id).")
    fs = comp["full_adaptive_strategy"]
    lines.append(f"- Full adaptive strategy: expected inspection fees **{fs['expected_inspection_fees']}** "
                 f"+ expected terminal loss **{fs['expected_terminal_loss']}** = "
                 f"**{fs['total_expected_cost']}**.")
    iv = comp["information_value"]
    lines.append(f"- Information value: gross loss reduction **{iv['gross_reduction_in_terminal_loss']}**, "
                 f"minus fees **{iv['inspection_fees_paid']}**, equals net saving "
                 f"**{iv['net_value_after_fees']}**. Fees are not netted into the information value itself.")
    lines.append("")
    lines.append("## Solver accounting")
    lines.append("")
    solver_info = summary["solver"]
    lines.append(f"- Distinct histories solved: {solver_info['distinct_histories_solved']}")
    lines.append(f"- DP build calls: {solver_info['history_build_calls']}, "
                 f"memo merges (equivalent sub-problems): {solver_info['memo_cache_hits_merged']}")
    lines.append(f"- Strategy leaves: {solver_info['strategy_leaves']}, "
                 f"max depth: {solver_info['max_depth']}")
    lines.append(f"- {solver_info['note']}")
    lines.append("")
    lines.append("## Tie handling")
    lines.append("")
    lines.append(TIE_RULE)
    lines.append("")
    lines.append("## Strategy tree")
    lines.append("")
    lines.append("```")
    lines.append(render_tree(solution))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def static_report(solution: Solution, fmt: str = "markdown") -> str:
    if fmt in ("markdown", "md", "text"):
        return render_report(solution)
    if fmt == "tree":
        return render_tree(solution)
    raise ValueError(f"unknown report format {fmt!r}")
