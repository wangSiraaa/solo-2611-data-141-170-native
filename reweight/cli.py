"""Command-line interface: fit / estimate / verify with static reports.

The three stages share one deterministic pipeline; each focuses its console
output on its own artefact and always writes the same static report.

Examples
--------
python -m reweight fit --out out/base
python -m reweight fit --margins cases/infeasible_margins.csv --out out/infeasible
python -m reweight fit --replicate cases/replicate_no_older.csv --out out/no_older
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import report, tool


def _paths(args):
    return {
        "respondents": args.respondents or "fixtures/respondents.csv",
        "margins": args.margins or "fixtures/margins.csv",
        "plan": args.plan or "fixtures/plan.json",
        "replicate": args.replicate or "fixtures/replicate_weights.csv",
    }


def _run_pipeline(paths):
    inputs = tool.load_inputs(paths["respondents"], paths["margins"],
                              paths["plan"], paths["replicate"])
    A = tool.build_constraint_matrix(inputs.respondents, inputs.margins)
    d = np.array([r.base_weight for r in inputs.respondents], dtype=float)

    base_fit = tool.fit(inputs, A, d)
    reps: list[tool.ReplicateResult] = []
    var = None
    vreport = None
    if base_fit.status == "feasible":
        reps = tool.replicate_estimates(inputs, A)
        var = tool.replicate_variance(inputs, tool.estimate(inputs, base_fit.weights), reps)
        vreport = tool.verify(inputs, A, d, base_fit, reps, var)
    return inputs, A, d, base_fit, reps, var, vreport


def run(args) -> int:
    inputs, A, d, base_fit, reps, var, vreport = _run_pipeline(_paths(args))
    outdir = args.out

    if base_fit.status != "feasible":
        # infeasible or solver-level failure: no weights, no SE; diagnostics.
        fe = tool.check_feasibility(inputs, A, d, inputs.plan.relative_lower,
                                    inputs.plan.relative_upper)
        paths = report.write_infeasible_artefacts(inputs, A, d, base_fit, fe,
                                                  outdir)
        label = "INFEASIBLE" if base_fit.status == "infeasible" else "FAILED"
        print(f"status: {label} -> {paths['report']}")
        for msg in base_fit.diagnostics:
            print(" -", msg)
        return 2 if base_fit.status == "infeasible" else 1

    paths = report.write_artefacts(inputs, A, d, base_fit, var, reps,
                                   vreport, outdir)
    w = base_fit.weights
    print(f"status: {base_fit.status}")
    print(f"objective: {base_fit.objective:.10g}  "
          f"max_residual: {base_fit.max_residual:.3e}  "
          f"optimality: {'ok' if base_fit.optimality_ok else 'FAIL'}")
    print(f"verification: {'PASS' if vreport.ok else 'FAIL'}")
    if args.command == "estimate":
        for i, r in enumerate(inputs.respondents):
            print(f"  {r.respondent_id}\tbase={d[i]:.6g}\tw={w[i]:.8g}")
    elif args.command == "verify":
        print("constraint checks:")
        for c in vreport.constraint_checks:
            print(f"  {c['constraint_id']:<20} lhs={c['lhs']:.8g} "
                  f"rhs={c['rhs']:.8g} resid={c['residual']:.2e} "
                  f"{'ok' if c['ok'] else 'FAIL'}")
        print(f"  objective gap={vreport.objective_check['gap']:.2e} "
              f"kkt={vreport.optimality_check}")
    for dom in inputs.plan.domains:
        did = dom["id"]
        print(f"  {did}: estimate={var.estimates[did]:.8g} "
              f"se={var.standard_errors[did]:.6g} "
              f"complete={var.complete[did]}")
    for note in var.notes:
        print(f"  note: {note}")
    print(f"artefacts: {outdir}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "estimate", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--respondents")
        p.add_argument("--margins")
        p.add_argument("--plan")
        p.add_argument("--replicate")
        p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
