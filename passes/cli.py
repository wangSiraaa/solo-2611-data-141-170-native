"""``passes`` command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import InputError, PassesError, PropagationError
from .io import load_horizon, load_plan, load_station, load_tle
from .output import render_summary, write_artifacts
from .solver import SolverConfig, solve

_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passes",
        description=(
            "Compute geometric communication windows for one synthetic TLE and one "
            "ground station, including sub-windows cut by a raised azimuth horizon."
        ),
    )
    parser.add_argument("--tle", type=Path, default=_ROOT / "fixtures" / "synthetic.tle")
    parser.add_argument("--station", type=Path, default=_ROOT / "fixtures" / "station.json")
    parser.add_argument("--horizon", type=Path, default=_ROOT / "fixtures" / "horizon.csv")
    parser.add_argument("--plan", type=Path, default=_ROOT / "fixtures" / "plan.json")
    parser.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    parser.add_argument("--search-step", type=float, default=None, help="coarse search step [s]")
    parser.add_argument("--verify-step", type=float, default=None, help="verification step [s]")
    parser.add_argument("--audit-step", type=float, default=None, help="independent audit step [s]")
    parser.add_argument("--node-probe-step", type=float, default=None, help="mask-node probe step [s]")
    parser.add_argument("--root-xtol", type=float, default=None, help="Brent xtol [s]")
    parser.add_argument("--observation-step", type=float, default=None, help="observation table step [s]")
    parser.add_argument("--quiet", action="store_true", help="do not print the summary")
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="also print the full passes.json payload to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        tle = load_tle(args.tle)
        station = load_station(args.station)
        mask = load_horizon(args.horizon)
        plan = load_plan(args.plan)
    except InputError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2

    config = SolverConfig()
    overrides = {
        "search_step_s": args.search_step,
        "verify_step_s": args.verify_step,
        "audit_step_s": args.audit_step,
        "node_probe_step_s": args.node_probe_step,
        "root_xtol_s": args.root_xtol,
        "observation_step_s": args.observation_step,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, float(value))
    config.boundary_tolerance_s = plan.boundary_tolerance_s

    try:
        result = solve(tle, station, mask, plan, config)
    except PropagationError as exc:
        failure = {
            "status": "propagation_error",
            "message": str(exc),
            "failures": exc.failures,
        }
        out = args.out
        out.mkdir(parents=True, exist_ok=True)
        (out / "propagation_error.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        print(f"propagation error: {exc.summary()}", file=sys.stderr)
        print(f"located failure details: {out / 'propagation_error.json'}", file=sys.stderr)
        return 3
    except PassesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    paths = write_artifacts(args.out, result)
    if not args.quiet:
        print(render_summary(result))
        print("")
        print("artifacts:")
        for name, path in paths.items():
            print(f"  {name:18s} {path}")
    if args.json_stdout:
        from .output import _json_ready

        print(json.dumps(_json_ready(result.payload), indent=2, ensure_ascii=False))
    return 0 if result.verification["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
