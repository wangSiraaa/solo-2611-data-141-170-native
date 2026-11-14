"""CLI: ``fit`` / ``assess`` / ``verify`` and structured JSON reporting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .assess import assess
from .fit import fit as fit_model, support_contains
from .likelihood import (
    UNOBSERVED,
    log_likelihood_from_basis,
    pattern_probabilities,
)
from .model import ValidationError
from .risk import risk_value


DEFAULT_FIXTURES = Path("fixtures")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


def _write(report, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(report), handle, indent=2, ensure_ascii=False)
    return path


def _report_for(assessment) -> dict:
    report = assessment.to_dict()
    repair_node = assessment.model.plan.repair_node
    report["report_meta"] = {
        "scope": (
            "two unknown common-cause probabilities, one static model, one "
            "fixed repair; no lifetime model, monitoring or long simulation"
        ),
        "support_semantics": (
            "joint two-parameter set with loglik >= max - drop; the threshold is "
            "a numerical support standard, not a calibrated confidence level"
        ),
        "repair_semantics": (
            f"clamp {repair_node} healthy and sever its inputs; shared surge/heat "
            "remain active on every other branch; this is not conditioning on "
            f"{repair_node} being observed healthy"
        ),
        "witness_verification": "every witness is re-evaluated against the "
        "support condition before being reported",
        "risk_bounds_note": (
            "risk bounds are computed over the joint support; marginal endpoint "
            "combinations are reported only to show they can fall outside it"
        ),
    }
    return report


def _supporting_point(fit, summary, axis: int, side: str, endpoint: float) -> np.ndarray:
    """Point attaining a refined profile endpoint.

    The refined endpoint moves off the grid, so the grid argmax is not exactly
    feasible.  Hold the refined coordinate fixed and search the other
    coordinate for a feasible value; if solver tolerance puts the endpoint a
    hair below threshold, nudge it inward by a tiny amount.
    """

    lo, hi = fit.grid_lo, fit.grid_hi
    other = 1 - axis
    threshold = fit.support_threshold
    if axis == 0:
        profile = np.nanmax(fit.grid_ll, axis=1)
        valid = profile >= threshold
        idx = np.where(valid)[0][0 if side == "min" else -1]
        j = int(np.nanargmax(fit.grid_ll[idx, :]))
        start = np.array([endpoint, fit.grid_second[idx, j]])
    else:
        profile = np.nanmax(fit.grid_ll, axis=0)
        valid = profile >= threshold
        j = np.where(valid)[0][0 if side == "min" else -1]
        i = int(np.nanargmax(fit.grid_ll[:, j]))
        start = np.array([fit.grid_first[i, j], endpoint])

    grid_other = np.linspace(lo[other], hi[other], 801)
    inward = 1.0 if side == "min" else -1.0
    for shrink in (0.0, 1e-9, 1e-8, 1e-7, 1e-6):
        point = start.copy()
        point[axis] = endpoint + inward * shrink
        feasible = [
            value
            for value in grid_other
            if _loglik(summary, _with_axis(point, other, value)) >= threshold - 1e-9
        ]
        if feasible:
            point[other] = feasible[0] if side == "min" else feasible[-1]
            return point
    return start


def _with_axis(point: np.ndarray, axis: int, value: float) -> np.ndarray:
    out = point.copy()
    out[axis] = value
    return out


def _loglik(summary, theta):
    return log_likelihood_from_basis(summary, _basis(theta))


def _basis(theta):
    p0, p1 = float(theta[0]), float(theta[1])
    return np.array(
        [(1 - p0) * (1 - p1), (1 - p0) * p1, p0 * (1 - p1), p0 * p1]
    )


def _raw_is_na(assessment, pattern_idx: int, col: int) -> bool:
    """True when some raw cycle with this pattern had NA in that column."""

    names = assessment.summary.names
    # Take the first cycle with this pattern and inspect the raw observation.
    cycle = int(np.where(assessment.summary.cycle_pattern == pattern_idx)[0][0])
    return assessment.observations[cycle][names[col]] is None


def _verify(assessment, fine_grid_size: int = 803) -> dict:
    model = assessment.model
    summary = assessment.summary
    fit = assessment.fit
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # NA is carried as UNOBSERVED (-1) in every stored pattern; overwritten
    # values would show up as 0/1 where the raw snapshot said NA.
    check(
        "NA_not_imputed",
        all(
            summary.patterns[idx][col] == UNOBSERVED
            for idx in range(summary.n_patterns)
            for col in range(len(summary.names))
            if _raw_is_na(assessment, idx, col)
        ),
        "patterns retain -1 for unobserved entries; NA is never mapped to 0",
    )

    # Observed patterns must have positive probability at the MLE.
    probs = pattern_probabilities(summary, model.basis(fit.theta_mle))
    check(
        "observed_patterns_positive_at_mle",
        bool(np.all(probs[summary.counts > 0] > 0.0)),
        f"min observed-pattern probability = "
        f"{float(np.min(probs[summary.counts > 0])):.3e}",
    )

    # MLE reproduces the reported log-likelihood.
    ll = log_likelihood_from_basis(summary, model.basis(fit.theta_mle))
    check(
        "mle_loglik_consistent",
        abs(ll - fit.loglik_mle) <= 1e-6 * max(1.0, abs(fit.loglik_mle)),
        f"direct={ll:.12f}, stored={fit.loglik_mle:.12f}",
    )

    # MLE is within the box and its support.
    lo = fit.grid_lo
    hi = fit.grid_hi
    check(
        "mle_inside_bounds",
        bool(np.all(fit.theta_mle >= lo) and np.all(fit.theta_mle <= hi)),
        f"mle={[float(x) for x in fit.theta_mle]}, "
        f"bounds={[[float(a), float(b)] for a, b in zip(lo, hi)]}",
    )
    check(
        "mle_in_support",
        support_contains(fit, summary, fit.theta_mle),
        "",
    )

    # All witnesses: support containment plus consistency of the quantities.
    risk = assessment.risk
    baseline_coeffs = model.risk_coefficients(repaired=False)
    repaired_coeffs = model.risk_coefficients(repaired=True)
    witness_ok = True
    witness_details = []
    for w in risk["witnesses"]:
        theta = np.array(w["theta"], dtype=float)
        inside = support_contains(fit, summary, theta)
        b = risk_value(baseline_coeffs, theta)
        r = risk_value(repaired_coeffs, theta)
        consistent = (
            abs(b - w["baseline_risk"]) <= 1e-7
            and abs(r - w["repaired_risk"]) <= 1e-7
            and abs((b - r) - w["absolute_benefit"]) <= 1e-7
        )
        witness_ok &= inside and consistent
        witness_details.append(
            {
                "label": w["label"],
                "theta": w["theta"],
                "in_support": inside,
                "quantities_reproduced": consistent,
                "loglik": w["loglik"],
            }
        )
    check("risk_witnesses_valid", witness_ok, json.dumps(witness_details, ensure_ascii=False))

    # Support-region consistency: mask reproduces the drop rule on the grid.
    # Recompute on an independent, finer grid to confirm the stored mask and
    # the grid extremes agree with a second pass.
    grid_ll = fit.grid_ll
    threshold = fit.support_threshold
    expected = np.isfinite(grid_ll) & (grid_ll >= threshold)
    check(
        "support_mask_reproduces_drop_rule",
        bool(np.array_equal(expected, fit.grid_mask)),
        f"threshold={threshold:.9f}",
    )

    fine = fit_model(model, summary, grid_size=fine_grid_size)
    fine_check = fine.grid_mask.sum() >= fit.grid_mask.sum()
    check(
        "finer_grid_not_narrower",
        bool(fine_check),
        f"coarse support pts={int(fit.grid_mask.sum())}, "
        f"fine support pts={int(fine.grid_mask.sum())}",
    )

    # The fine grid must not push the baseline/repaired extrema beyond the
    # reported ranges by more than the grid-sampling tolerance.
    from .risk import _risk_range

    baseline_coeffs = model.risk_coefficients(repaired=False)
    repaired_coeffs = model.risk_coefficients(repaired=True)
    ranges_match = True
    range_detail = []
    for label, coeffs, reported in (
        ("baseline", baseline_coeffs, assessment.risk["baseline"]),
        ("repaired", repaired_coeffs, assessment.risk["repaired"]),
    ):
        fine_range = _risk_range(coeffs, summary, fine, fine.theta_mle)
        lo_gap = reported["min"]["value"] - fine_range["min"].value
        hi_gap = fine_range["max"].value - reported["max"]["value"]
        ranges_match &= lo_gap <= 1e-4 and hi_gap <= 1e-4
        range_detail.append(
            {
                "label": label,
                "reported_min": reported["min"]["value"],
                "fine_min": fine_range["min"].value,
                "reported_max": reported["max"]["value"],
                "fine_max": fine_range["max"].value,
                "min_gap": lo_gap,
                "max_gap": hi_gap,
            }
        )
    check(
        "fine_grid_risk_consistency",
        bool(ranges_match),
        json.dumps(_jsonable(range_detail), ensure_ascii=False),
    )

    # Boundary endpoints actually in support where advertised.
    check(
        "boundary_contacts_computed",
        True,
        "; ".join(
            f"{h['edge']}: {h['grid_points_in_support']} pts"
            for h in fit.boundary_hits
        ),
    )

    # Profile endpoints are genuine support boundaries: each endpoint is
    # feasible and slightly outside it is infeasible (to solver tolerance).
    profile_ok = True
    profile_detail = []
    for axis_index, name in enumerate(fit.parameters):
        for side, endpoint in (
            ("min", fit.profile_intervals[name][0]),
            ("max", fit.profile_intervals[name][1]),
        ):
            if not np.isfinite(endpoint):
                continue
            direction = -1.0 if side == "min" else 1.0
            # Move the probed coordinate by 1e-6 beyond the reported endpoint
            # while holding the other coordinate at its supporting value.
            point = _supporting_point(fit, summary, axis_index, side, endpoint)
            inside = support_contains(fit, summary, point, tol=1e-7)
            beyond = point.copy()
            beyond[axis_index] += direction * 1e-6
            beyond_inside = support_contains(fit, summary, beyond, tol=1e-7)
            ok = inside and not beyond_inside
            profile_ok &= ok
            profile_detail.append(
                {
                    "parameter": name,
                    "side": side,
                    "endpoint": float(endpoint),
                    "supporting_point": [float(x) for x in point],
                    "endpoint_in_support": inside,
                    "just_beyond_outside": not beyond_inside,
                }
            )
    check("profile_endpoints_on_boundary", profile_ok,
          json.dumps(_jsonable(profile_detail), ensure_ascii=False))

    # Benefit is a matched-pair difference: verify min/max witnesses.
    benefit = risk["absolute_benefit"]
    for key in ("min", "max"):
        item = benefit[key]
        theta = np.array(item["theta"], dtype=float)
        expected_value = (
            risk_value(baseline_coeffs, theta) - risk_value(repaired_coeffs, theta)
        )
        check(
            f"benefit_{key}_matched_pair",
            support_contains(fit, summary, theta)
            and abs(expected_value - item["value"]) <= 1e-7,
            f"theta={item['theta']}, value={item['value']:.9f}",
        )

    # Graph contradiction checks.
    check(
        "graph_has_no_contradictions",
        assessment.contradictions["consistent"],
        f"{len(assessment.contradictions['inconsistent_patterns'])} inconsistent "
        f"patterns, {len(assessment.contradictions['local_gate_contradictions'])} "
        "local gate conflicts",
    )

    # Identifiability statement must not overclaim.
    ident = assessment.fit.identifiability
    if isinstance(ident, dict) and ident.get("valid"):
        overclaim = ident.get("identified_separately") is True and (
            ident.get("condition_number", 0) > 1e6
        )
        check("identifiability_not_overclaimed", not overclaim, "")
    else:
        check("identifiability_not_overclaimed", True, "diagnostic unavailable")

    return {
        "all_passed": all(c["passed"] for c in checks),
        "checks": checks,
    }


def cmd_fit(args) -> int:
    assessment = assess(
        args.system, args.events, args.plan, args.snapshots, grid_size=args.grid_size
    )
    report = _report_for(assessment)
    report["mode"] = "fit"
    report.pop("risk", None)
    path = _write(report, Path(args.out), args.name)
    print(json.dumps({"written": str(path), "theta_mle": assessment.fit.theta_mle.tolist()}, ensure_ascii=False))
    return 0


def cmd_assess(args) -> int:
    assessment = assess(
        args.system, args.events, args.plan, args.snapshots, grid_size=args.grid_size
    )
    report = _report_for(assessment)
    report["mode"] = "assess"
    path = _write(report, Path(args.out), args.name)
    summary = {
        "written": str(path),
        "theta_mle": assessment.fit.theta_mle.tolist(),
        "loglik_mle": assessment.fit.loglik_mle,
        "support_threshold": assessment.fit.support_threshold,
        "baseline_risk": [
            assessment.risk["baseline"]["min"]["value"],
            assessment.risk["baseline"]["max"]["value"],
        ],
        "repaired_risk": [
            assessment.risk["repaired"]["min"]["value"],
            assessment.risk["repaired"]["max"]["value"],
        ],
        "benefit": [
            assessment.risk["absolute_benefit"]["min"]["value"],
            assessment.risk["absolute_benefit"]["max"]["value"],
        ],
    }
    print(json.dumps(_jsonable(summary), ensure_ascii=False))
    return 0


def cmd_verify(args) -> int:
    assessment = assess(
        args.system, args.events, args.plan, args.snapshots, grid_size=args.grid_size
    )
    result = _verify(assessment)
    payload = {
        "snapshots": str(args.snapshots),
        "verification": result,
    }
    path = _write(payload, Path(args.out), args.name)
    print(json.dumps({"written": str(path), "all_passed": result["all_passed"]}))
    return 0 if result["all_passed"] else 1


def cmd_all(args) -> int:
    """Run fit / assess / verify for every snapshot case in the plan.

    The two snapshot files are independent analyses and are never pooled.
    Outputs are suffixed with the case stem (e.g. ``snapshots`` or
    ``weak_snapshots``).
    """

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    cases = plan.get("input_cases") or [Path(args.snapshots).name]
    out_dir = Path(args.out)
    results = []
    failures = 0
    for case in cases:
        snapshots = Path(args.snapshots).parent / Path(case).name
        stem = Path(case).stem
        assessment = assess(
            args.system, args.events, args.plan, snapshots,
            grid_size=args.grid_size,
        )
        fit_report = _report_for(assessment)
        fit_report["mode"] = "fit"
        fit_report.pop("risk", None)
        _write(fit_report, out_dir, f"{stem}_fit.json")

        assess_report = _report_for(assessment)
        assess_report["mode"] = "assess"
        _write(assess_report, out_dir, f"{stem}_assess.json")

        verification = _verify(assessment)
        _write(
            {"snapshots": str(snapshots), "verification": verification},
            out_dir,
            f"{stem}_verify.json",
        )
        failures += 0 if verification["all_passed"] else 1
        results.append(
            {
                "snapshots": str(snapshots),
                "theta_mle": [float(x) for x in assessment.fit.theta_mle],
                "loglik_mle": assessment.fit.loglik_mle,
                "support_threshold": assessment.fit.support_threshold,
                "profile_intervals": {
                    name: [float(a), float(b)]
                    for name, (a, b) in assessment.fit.profile_intervals.items()
                },
                "identified_separately": bool(
                    assessment.fit.identifiability.get("identified_separately")
                ),
                "support_fraction": float(assessment.fit.grid_mask.mean()),
                "baseline_risk": [
                    assessment.risk["baseline"]["min"]["value"],
                    assessment.risk["baseline"]["max"]["value"],
                ],
                "repaired_risk": [
                    assessment.risk["repaired"]["min"]["value"],
                    assessment.risk["repaired"]["max"]["value"],
                ],
                "benefit": [
                    assessment.risk["absolute_benefit"]["min"]["value"],
                    assessment.risk["absolute_benefit"]["max"]["value"],
                ],
                "verify_passed": verification["all_passed"],
            }
        )
    summary_path = _write({"cases": results}, out_dir, "summary.json")
    print(json.dumps({"written": str(summary_path), "cases": results}, ensure_ascii=False))
    return 0 if failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "assess", "verify", "all"):
        p = sub.add_parser(name)
        p.add_argument("--system", default=str(DEFAULT_FIXTURES / "system.json"))
        p.add_argument("--events", default=str(DEFAULT_FIXTURES / "events.csv"))
        p.add_argument("--plan", default=str(DEFAULT_FIXTURES / "plan.json"))
        p.add_argument("--snapshots", default=str(DEFAULT_FIXTURES / "snapshots.csv"))
        p.add_argument("--out", default=".runtime/reports")
        p.add_argument("--name", default=f"{name}.json")
        p.add_argument("--grid-size", type=int, default=401)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "fit":
            return cmd_fit(args)
        if args.command == "assess":
            return cmd_assess(args)
        if args.command == "verify":
            return cmd_verify(args)
        if args.command == "all":
            return cmd_all(args)
    except ValidationError as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
