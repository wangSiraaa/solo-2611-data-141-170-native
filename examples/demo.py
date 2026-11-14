#!/usr/bin/env python3
"""End-to-end demonstration of the offline policy analyser.

Produces, under ``out/``:

* ``tiny_handcheck.txt``      -- hand-derivable tiny.json solution
* ``model/b*.{json,md,txt}``  -- every plan budget: solution, report, tree
* ``correlated.txt``          -- correlated-diagnostic case and why the joint matters
* ``replay_examples.txt``     -- replay of concrete state/vector pairs
* ``self_check.txt``          -- brute-force cross-check + invariance results

Run:  python3 examples/demo.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from policy_tool.bruteforce import brute_force_value, walk_policy_value
from policy_tool.model import load_model
from policy_tool.replay import replay, replay_summary
from policy_tool.report import render_report, render_tree, strategy_tree
from policy_tool.solver import solve, solution_summary

OUT = os.path.join(ROOT, "out")
FIX = os.path.join(ROOT, "fixtures")


def write(rel: str, text: str) -> None:
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")
    print(f"wrote {os.path.relpath(path, ROOT)}")


def tiny_demo() -> None:
    model = load_model(os.path.join(FIX, "tiny.json"))
    sol = solve(model, Fraction(1))
    lines = [
        "TINY HAND CHECK (tiny.json, budget 1)",
        "=" * 60,
        "Prior: fault=1/3, ok=2/3.",
        "lamp perfectly reveals the state; fee 1.",
        "Losses: repair(fault=2, ok=4), release(fault=20, ok=0).",
        "",
        "No detection:",
        "  repair : (1/3)*2 + (2/3)*4 = 10/3",
        "  release: (1/3)*20 + (2/3)*0 = 20/3",
        "  best   = 10/3  (repair)",
        "",
        "Buy lamp (fee 1), then act with certainty:",
        "  = 1 + (1/3)*repair_fault + (2/3)*release_ok",
        "  = 1 + (1/3)*2 + (2/3)*0 = 1 + 2/3 = 5/3",
        "",
        f"Solver value           : {sol.value}  (= 5/3: correct)",
        f"Expected fees          : {sol.expected_fees}",
        f"Expected terminal loss : {sol.expected_terminal_loss}",
        f"No-detection cost      : {sol.no_detection_cost}",
        f"Gross information value: {sol.gross_information_value}",
        f"Net value after fees   : {sol.net_information_value}",
        "",
        render_tree(sol),
    ]
    write("tiny_handcheck.txt", "\n".join(lines))
    write("tiny_solution.json", json.dumps({
        "model": "tiny.json",
        "solution": solution_summary(sol),
        "strategy_tree": strategy_tree(sol),
    }, ensure_ascii=False, indent=2))


def main_budgets() -> None:
    model = load_model(
        os.path.join(FIX, "model.json"),
        actions_path=os.path.join(FIX, "actions.csv"),
        plans_path=os.path.join(FIX, "plans.json"),
    )
    for budget in model.budgets:
        sol = solve(model, budget)
        label = str(budget)
        payload = {
            "solution": solution_summary(sol),
            "strategy_tree": strategy_tree(sol),
        }
        write(f"model/solution_b{label}.json",
              json.dumps(payload, ensure_ascii=False, indent=2))
        write(f"model/report_b{label}.md", render_report(sol))
        write(f"model/tree_b{label}.txt", render_tree(sol))


def correlated_demo() -> None:
    model = load_model(os.path.join(FIX, "correlated.json"))
    sol1 = solve(model, Fraction(1))
    sol2 = solve(model, Fraction(2))
    lines = [
        "CORRELATED DIAGNOSTICS (fixtures/correlated.json)",
        "=" * 60,
        "States broken/fine each 1/2.  Results are perfectly correlated with",
        "the state: broken -> (a=1,b=1), fine -> (a=0,b=0).",
        "Marginal P(a=1)=1/2 and P(b=1)=1/2 for both states, and",
        "P(a=1)*P(b=1)=1/4 while the true joint P(a=1,b=1)=1/2.",
        "Multiplying marginal likelihoods would therefore be wrong; the solver",
        "conditions on the full joint, so buying a already reveals the state",
        "and a second, correlated test adds no information.",
        "",
        "Actions: scrap(10,10), keep(30,0), inspect(12,12).",
        "",
        f"Budget 1 value: {sol1.value}, root buys {sol1.root.decision_id}",
        f"Budget 2 value: {sol2.value}, root buys {sol2.root.decision_id}",
        "  (extra budget does not help: the second test is redundant given",
        "   perfect correlation, and its fee would only add cost.)",
        "",
        render_tree(sol1),
    ]
    write("correlated.txt", "\n".join(lines))


def replay_demo() -> None:
    model = load_model(
        os.path.join(FIX, "model.json"),
        actions_path=os.path.join(FIX, "actions.csv"),
    )
    sol = solve(model, Fraction(5))
    examples = [
        ("bearing", "1011"),
        ("power", "0110"),
        ("healthy", "0000"),
        ("healthy", "1111"),
        ("bearing", "0010"),
    ]
    chunks = ["REPLAY EXAMPLES (model.json, budget 5)", "=" * 60]
    for state, vector in examples:
        r = replay(sol, state, outcomes=vector)
        s = replay_summary(r)
        chunks.append("")
        chunks.append(f"state={state} vector={vector}")
        chunks.append(f"  decisions: "
                      + " -> ".join(f"{st.decision_kind}:{st.decision_id}"
                                    for st in r.steps))
        chunks.append(f"  fees={s['fees_paid']} terminal={s['terminal_action']} "
                      f"loss={s['terminal_loss']} path_total={s['total_path_cost']}")
        chunks.append(f"  reconstructed expectation={s['expected_reconstructed']} "
                      f"(tree value {s['tree_value']}, consistent={s['consistent']})")
    write("replay_examples.txt", "\n".join(chunks))


def self_check() -> None:
    model = load_model(
        os.path.join(FIX, "model.json"),
        actions_path=os.path.join(FIX, "actions.csv"),
        plans_path=os.path.join(FIX, "plans.json"),
    )
    lines = ["SELF CHECK", "=" * 60,
             "DP vs exhaustive permutation brute force vs tree re-walk:", ""]
    for budget in model.budgets:
        sol = solve(model, budget)
        bf = brute_force_value(model, budget)
        wk = walk_policy_value(sol)
        lines.append(
            f"  budget {budget}: dp={sol.value} brute={bf} walk={wk} "
            f"match={sol.value == bf == wk}"
        )

    # Row-order invariance
    with open(os.path.join(FIX, "joint.csv")) as fh:
        rows = fh.read().strip().splitlines()
    header, body = rows[0], rows[1:]
    random.Random(7).shuffle(body)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    tmp.write("\n".join([header] + body) + "\n")
    tmp.close()
    try:
        shuffled = load_model(os.path.join(FIX, "model.json"),
                              actions_path=os.path.join(FIX, "actions.csv"),
                              joint_path=tmp.name)
        invariant = all(
            solve(model, b).value == solve(shuffled, b).value
            and strategy_tree(solve(model, b)) == strategy_tree(solve(shuffled, b))
            for b in model.budgets
        )
    finally:
        os.unlink(tmp.name)
    lines += ["", f"Row-order shuffle preserves value and full tree: {invariant}"]

    # Replay re-averaging over every joint row equals the solver value.
    sol = solve(model, Fraction(5))
    total = Fraction(0)
    for state in model.states:
        for vector, prob in model.joint[state].items():
            r = replay(sol, state, outcomes="".join(vector))
            total += model.priors[state] * prob * r.total_path_cost
    lines += ["",
              f"Replay re-average over all joint rows = {total} "
              f"(solver {sol.value}, equal={total == sol.value})"]
    write("self_check.txt", "\n".join(lines))


def main() -> None:
    tiny_demo()
    main_budgets()
    correlated_demo()
    replay_demo()
    self_check()
    print("\nDone. Inspect out/ for artifacts.")


if __name__ == "__main__":
    main()
