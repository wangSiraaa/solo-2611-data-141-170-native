"""Command line interface: ``python -m satflow analyze ...``."""

from __future__ import annotations

import argparse
import os
import sys

from .analysis import analyze_events
from .errors import SatflowError, StructureError
from .fcsio import load_fcs
from .plan import load_plan
from .reporting import (
    write_diagnostics,
    write_event_status_csv,
    write_gate_counts,
    write_witnesses,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="satflow",
        description=(
            "Feasibility-aware gate analysis for saturated four-channel FCS 3.1 data. "
            "No clustering, no logicle transform, no cell-type diagnosis, no matrix fitting."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze", help="build per-event joint feasible sets and classify gates"
    )
    analyze.add_argument("--fcs", required=True, help="path to the FCS 3.1 file")
    analyze.add_argument("--plan", required=True, help="path to the plan JSON file")
    analyze.add_argument("--outdir", required=True, help="directory for the four deliverables")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "analyze":  # pragma: no cover - argparse enforces this
        return 2

    try:
        plan = load_plan(args.plan)
        data, channel_names, fcs_text = load_fcs(args.fcs, plan.detectors)
        results = analyze_events(data, plan)
    except StructureError as exc:
        print(f"STRUCTURE ERROR: {exc}", file=sys.stderr)
        return 2
    except SatflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "event-status.csv")
    counts_path = os.path.join(args.outdir, "gate-counts.json")
    witnesses_path = os.path.join(args.outdir, "witnesses.json")
    diagnostics_path = os.path.join(args.outdir, "diagnostics.json")

    write_event_status_csv(csv_path, results, plan)
    counts = write_gate_counts(counts_path, results, plan)
    write_witnesses(witnesses_path, results, plan)
    diagnostics = write_diagnostics(
        diagnostics_path, results, plan, fcs_text, args.fcs, args.plan
    )

    totals = counts["totals"]
    print("satflow analyze complete")
    print(f"  events total={totals['events_in_fcs']} "
          f"feasible={totals['feasible_events']} "
          f"infeasible={totals['infeasible_events']} "
          f"solver_error={totals['solver_error_events']}")
    for gate_id, gate in counts["gates"].items():
        lo, hi = gate["proportion_lower_bound"], gate["proportion_upper_bound"]
        print(
            f"  gate {gate_id}: count in [{gate['count_determined_lower_bound']}, "
            f"{gate['count_possible_upper_bound']}] of {totals['feasible_events']} feasible "
            f"-> proportion in [{lo:.4f}, {hi:.4f}]"
        )
    if not diagnostics["accounting_identity_holds"] or not diagnostics["witness_checks"][
        "all_present_witnesses_pass"
    ]:
        print("  WARNING: self-check failure, see diagnostics.json", file=sys.stderr)
        return 1
    print(f"  outputs: {csv_path}, {counts_path}, {witnesses_path}, {diagnostics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
