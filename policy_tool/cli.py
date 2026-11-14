"""Command-line entry points: ``solve``, ``replay`` and ``validate``.

Examples
--------
    python -m policy_tool solve \
        --model fixtures/model.json --actions fixtures/actions.csv \
        --plans fixtures/plans.json --budget 5 --json-out out/b5.json

    python -m policy_tool solve --model fixtures/model.json --all-budgets \
        --out-dir out/model

    python -m policy_tool replay --model fixtures/model.json --budget 5 \
        --state bearing --outcomes 1011

    python -m policy_tool replay --model fixtures/tiny.json --state fault \
        --outcomes 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction
from typing import Any, Sequence

from .model import Model, ModelError, frac_str, load_model, model_summary, parse_fraction
from .replay import ReplayError, replay, replay_model, replay_summary
from .report import render_report, render_tree, strategy_tree
from .solver import TIE_RULE, Solution, solve, solution_summary


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="model JSON (model.json or tiny.json)")
    parser.add_argument("--actions", help="terminal actions CSV (overrides model field)")
    parser.add_argument("--joint", help="joint distribution CSV (overrides model field)")
    parser.add_argument("--plans", help="plans JSON supplying budgets/objective/tie note")
    parser.add_argument("--budget", help="single budget (exact number or fraction)")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")


def _load(args: argparse.Namespace, use_cli_budget: bool = True) -> Model:
    return load_model(
        args.model,
        actions_path=args.actions,
        joint_path=args.joint,
        plans_path=args.plans,
        budgets=([args.budget] if args.budget is not None and use_cli_budget
                 else None),
    )


def _write(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _solution_payload(model: Model, solution: Solution) -> dict[str, Any]:
    return {
        "model": model_summary(model),
        "solution": solution_summary(solution),
        "strategy_tree": strategy_tree(solution),
    }


def cmd_solve(args: argparse.Namespace) -> int:
    multiple = bool(args.all_budgets or args.calibrate)
    model = _load(args, use_cli_budget=not multiple)
    budgets: list[Fraction | None]
    if multiple:
        if not model.budgets:
            raise ModelError("--all-budgets/--calibrate needs budgets in the model or plans")
        budgets = list(model.budgets)
        if args.budget is not None:
            requested = parse_fraction(args.budget, "--budget")
            if requested not in budgets:
                budgets.append(requested)
            budgets = sorted(set(budgets))
    elif args.budget is not None:
        budgets = [parse_fraction(args.budget, "--budget")]
    elif model.budgets and len(model.budgets) == 1:
        budgets = [model.budgets[0]]
    elif model.budgets:
        raise ModelError(
            "several budgets are configured; pass --budget or --all-budgets. Choices: "
            + ", ".join(frac_str(b) for b in model.budgets)
        )
    else:
        budgets = [None]  # unconstrained diagnostic

    out_dir = args.out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for budget in budgets:
        solution = solve(model, budget)
        label = "unbounded" if solution.unlimited else frac_str(budget)
        file_label = label.replace("/", "_").replace("-", "neg")
        payload = _solution_payload(model, solution)

        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            if len(budgets) > 1:
                print("=" * 72)
                print(f"BUDGET {label}")
                print("=" * 72)
            print(render_tree(solution))
            print()
            comp = solution.cost_composition()
            print("COST COMPOSITION")
            print(f"  no detection               : "
                  f"{comp['no_detection']['expected_terminal_loss']}")
            fs = comp["full_adaptive_strategy"]
            print(f"  expected inspection fees    : {fs['expected_inspection_fees']}")
            print(f"  expected terminal loss      : {fs['expected_terminal_loss']}")
            print(f"  total expected cost         : {fs['total_expected_cost']}")
            iv = comp["information_value"]
            print(f"  gross information value     : {iv['gross_reduction_in_terminal_loss']}")
            print(f"  net value after fees        : {iv['net_value_after_fees']}")
            print(f"  solver: {solution.node_count} distinct histories, "
                  f"{solution.memo_hits} memo merges, "
                  f"{solution.leaves} leaves, depth {solution.max_depth}")

        if args.json_out:
            target = args.json_out if len(budgets) == 1 else os.path.join(
                out_dir or os.path.dirname(args.json_out) or ".",
                f"solution_b{file_label}.json",
            )
            _write(target, json.dumps(payload, ensure_ascii=False, indent=2))
        if args.report_out:
            target = args.report_out if len(budgets) == 1 else os.path.join(
                out_dir or os.path.dirname(args.report_out) or ".",
                f"report_b{file_label}.md",
            )
            _write(target, render_report(solution))
        if out_dir:
            _write(os.path.join(out_dir, f"solution_b{file_label}.json"),
                   json.dumps(payload, ensure_ascii=False, indent=2))
            _write(os.path.join(out_dir, f"report_b{file_label}.md"), render_report(solution))
            _write(os.path.join(out_dir, f"tree_b{file_label}.txt"), render_tree(solution))

    return 0


def _parse_results(items: Sequence[str] | None) -> dict[str, str] | None:
    if not items:
        return None
    result = {}
    for item in items:
        if "=" not in item:
            raise ModelError(f"--result expects test=value, got {item!r}")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def cmd_replay(args: argparse.Namespace) -> int:
    model = _load(args)
    budget = parse_fraction(args.budget, "--budget") if args.budget is not None else None
    if budget is None and model.budgets and len(model.budgets) > 1:
        raise ModelError("several budgets configured; pass --budget for replay")
    if budget is None and model.budgets and len(model.budgets) == 1:
        budget = model.budgets[0]
    result = replay_model(
        model, budget, args.state,
        outcomes=args.outcomes, results=_parse_results(args.result),
    )
    payload = replay_summary(result)
    payload["tie_rule"] = TIE_RULE
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("REPLAY")
    print(f"  budget          : {payload['budget']}")
    print(f"  true state      : {result.state}")
    print(f"  result vector   : {payload['full_result_vector']} "
          f"({payload['outcomes']})")
    print(f"  path probability: {payload['path_probability']}")
    print()
    print("  decisions along the path:")
    for step in result.steps:
        known = ", ".join(f"{k}={v}" for k, v in step.observed.items()) or "none"
        post = ", ".join(f"{k}={frac_str(v)}" for k, v in step.posterior.items())
        fee = frac_str(step.fee_paid) if step.fee_paid is not None else "-"
        print(f"    depth {step.depth}: known [{known}] posterior [{post}]")
        print(f"             {step.decision_kind.upper():<4} {step.decision_id:<16} "
              f"fee {fee}")
        if step.tied_actions_on_this_node:
            print(f"             ties: {', '.join(step.tied_actions_on_this_node)}")
    print()
    print(f"  fees paid       : {payload['fees_paid']}")
    print(f"  terminal action : {payload['terminal_action']}")
    print(f"  terminal loss   : {payload['terminal_loss']} (true state {result.state})")
    print(f"  path total cost : {payload['total_path_cost']}")
    print()
    print("  re-averaged over the frozen distribution:")
    dc = result.distribution_check
    print(f"    expected fees   : {dc['expected_inspection_fees']}")
    print(f"    expected loss   : {dc['expected_terminal_loss']}")
    print(f"    expected total  : {dc['expected_total']}")
    print(f"    tree value      : {payload['tree_value']}")
    print(f"    consistent      : {result.consistent}")
    print(f"    per-state expected cost: {dc['per_state_expected_cost']}")
    print()
    print(f"  terminal action over all {len(result.all_paths)} joint paths: "
          f"{payload['terminal_action_counts_over_all_paths']}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    model = load_model(args.model, actions_path=args.actions,
                       joint_path=args.joint, plans_path=args.plans)
    summary = model_summary(model)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"OK: {model.source}")
        print(f"  states : {', '.join(model.states)}")
        print(f"  tests  : {', '.join(t.id for t in model.tests)}")
        print(f"  actions: {', '.join(a.id for a in model.actions)}")
        print(f"  budgets: {', '.join(frac_str(b) for b in (model.budgets or ())) or '(none)'}")
        for w in model.warnings:
            print(f"  warning: {w}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="policy_tool",
        description="Offline adaptive inspection-policy analyser (exact arithmetic).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    solve_p = sub.add_parser("solve", help="solve and emit the static strategy report")
    _add_model_args(solve_p)
    solve_p.add_argument("--all-budgets", action="store_true",
                         help="solve every budget in plans/model")
    solve_p.add_argument("--calibrate", action="store_true",
                         help="alias of --all-budgets (budget sensitivity)")
    solve_p.add_argument("--out-dir", help="write solution JSON + markdown + tree per budget")
    solve_p.add_argument("--json-out", help="write the solution JSON here")
    solve_p.add_argument("--report-out", help="write the markdown report here")
    solve_p.set_defaults(func=cmd_solve)

    replay_p = sub.add_parser("replay", help="replay the policy for one state/full results")
    _add_model_args(replay_p)
    replay_p.add_argument("--state", required=True, help="true state name")
    replay_p.add_argument("--outcomes", help="full vector, e.g. 1011 or 1,0,1,1")
    replay_p.add_argument("--result", action="append",
                          help="named result test=value (repeatable)")
    replay_p.set_defaults(func=cmd_replay)

    val_p = sub.add_parser("validate", help="validate inputs without solving")
    _add_model_args(val_p)
    val_p.set_defaults(func=cmd_validate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ModelError, ReplayError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
