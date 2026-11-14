"""Command-line driver.

Usage
-----
    python -m price_link --quotes fixtures/quotes.csv \
        --products fixtures/products.csv --plan fixtures/plan.json \
        --out outputs

    # also run the single-bridge and disconnected cases, and the micro checks
    python -m price_link --run-all --out outputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import load_inputs
from .estimator import fit
from .explain import explain, result_summary, write_reports
from .verify import run_micro_checks


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quotes", default="fixtures/quotes.csv")
    p.add_argument("--products", default="fixtures/products.csv")
    p.add_argument("--plan", default="fixtures/plan.json")
    p.add_argument("--out", default="outputs")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=20250101)
    p.add_argument("--min-valid-fraction", type=float, default=0.80)
    p.add_argument("--min-valid-replicates", type=int, default=20)
    p.add_argument("--no-robust", action="store_true", help="disable Huber downweighting")
    p.add_argument("--run-all", action="store_true",
                   help="run base, single_bridge, disconnected cases and micro checks")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _run_case(name: str, quotes: str, products: str, plan: str, out: str, args) -> dict:
    inputs = load_inputs(quotes, products, plan)
    result = fit(
        inputs=inputs,
        n_boot=args.n_boot,
        seed=args.seed,
        min_valid_fraction=args.min_valid_fraction,
        min_valid_replicates=args.min_valid_replicates,
        robust=not args.no_robust,
    )
    paths = write_reports(result, out, name)
    summary = result_summary(result)
    summary["paths"] = paths
    if not args.quiet:
        print(explain(result))
        print(f"[{name}] reports written: {paths}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}

    if args.run_all:
        cases = {
            "base": ("fixtures/quotes.csv", "fixtures/products.csv", "fixtures/plan.json"),
            "single_bridge": ("cases/single_bridge.csv", "fixtures/products.csv", "fixtures/plan.json"),
            "disconnected": ("cases/disconnected.csv", "fixtures/products.csv", "fixtures/plan.json"),
        }
        for name, (q, pr, pl) in cases.items():
            summaries[name] = _run_case(name, q, pr, pl, args.out, args)
        summaries["micro_checks"] = run_micro_checks(out / "micro_checks.json")
    else:
        summaries["base"] = _run_case(
            "base", args.quotes, args.products, args.plan, args.out, args
        )

    comparison_path = out / "case_comparison.json"
    comparison_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    # Compact cross-case table for the base and two alternative cases.
    lines = ["# Case comparison: base vs single bridge vs disconnected\n"]
    lines.append("| case | " + " | ".join(summaries.get("base", {}).get("months", [])) + " |")
    lines.append("|---|" + "---:|" * len(summaries.get("base", {}).get("months", [])))
    for name in ("base", "single_bridge", "disconnected"):
        if name not in summaries:
            continue
        mi = summaries[name]["month_index"]
        months = summaries[name]["months"]
        cells = []
        for m in months:
            v = mi[m]
            if v["status"] == "insufficient_support":
                cells.append("NOT IDENTIFIED")
            elif v["index_base100"] is None:
                cells.append(v["status"])
            elif v["interval_reportable"]:
                lo, hi = v["ci_ratio"]
                cells.append(f"{v['index_base100']:.2f} [{100*lo:.2f},{100*hi:.2f}]")
            else:
                cells.append(f"{v['index_base100']:.2f} (interval withheld)")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    (out / "case_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    if not args.quiet:
        print("\n".join(lines))
    print(f"[done] comparison written to {comparison_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
