"""Top-level ``fit`` / ``assess`` semantics shared by the CLI and tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fit import FitResult, fit
from .likelihood import (
    PatternSummary,
    contradiction_report,
    design_rank,
    pattern_table,
    summarise_patterns,
)
from .model import ReliabilityModel, ValidationError, build_model, load_snapshots
from .risk import risk_scenario


@dataclass
class Assessment:
    model: ReliabilityModel
    summary: PatternSummary
    fit: FitResult
    risk: dict
    diagnostics: dict
    snapshot_path: str
    contradictions: dict
    design: dict
    pattern_rows: list
    fit_check: dict
    observations: list[dict] = field(default_factory=list)
    cycle_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "input": {
                "snapshots": self.snapshot_path,
                "cycles": int(len(self.summary.cycle_pattern)),
            },
            "parameters": {
                "names": list(self.fit.parameters),
                "bounds": [list(b) for b in self.fit.bounds],
            },
            "fit": fit_to_dict(self.fit),
            "identifiability": self.fit.identifiability,
            "design_local_information": self.design,
            "observations": {
                "distinct_patterns": self.summary.n_patterns,
                "patterns": [
                    {
                        "pattern": row.pattern,
                        "cycles": row.count,
                        "probability_at_mle": row.probability,
                        "log_probability_at_mle": row.log_probability,
                    }
                    for row in self.pattern_rows
                ],
            },
            "contradictions": self.contradictions,
            "fit_checks": self.fit_check,
            "risk": self.risk,
            "diagnostics": self.diagnostics,
        }


def _fit_checks(
    model: ReliabilityModel,
    summary: PatternSummary,
    fit_result: FitResult,
) -> dict:
    values = []
    for name, theta in zip(model.parameters, fit_result.theta_mle):
        lo, hi = model.plan.bounds[name]
        if theta <= lo + 1e-9 or theta >= hi - 1e-9:
            values.append(f"{name}={theta:.12g} is on the {lo}-{hi} box edge")
    return {
        "mle_is_interior": not values,
        "boundary_flags": values,
        "support_reaches_box_edge": [
            hit
            for hit in fit_result.boundary_hits
            if hit["grid_points_in_support"] > 0
        ],
        "profile_intervals": {
            name: list(fit_result.profile_intervals[name])
            for name in model.parameters
        },
        "support_threshold": fit_result.support_threshold,
        "support_loglik_drop": fit_result.support_drop,
        "grid_resolution": {
            "size": fit_result.grid_size,
            "axis_spacing": list(fit_result.axis_spacing),
        },
        "note": (
            "support region is the set where loglik >= max loglik - drop; this is "
            "a numerical support criterion, not a calibrated confidence region"
        ),
    }


def assess(
    system_path: str | Path,
    events_path: str | Path,
    plan_path: str | Path,
    snapshots_path: str | Path,
    grid_size: int = 401,
) -> Assessment:
    model = build_model(system_path, events_path, plan_path)
    system = model.system
    observations, cycle_ids = load_snapshots(snapshots_path, system["observables"])

    summary = summarise_patterns(model, observations)
    contradictions = contradiction_report(model, summary)
    if contradictions["inconsistent_patterns"]:
        detail = "; ".join(
            f"{row['pattern']} (x{row['cycles']})"
            for row in contradictions["inconsistent_patterns"]
        )
        raise ValidationError(
            "snapshot contradicts the graph: no primitive-event configuration "
            f"can produce these observed states: {detail}"
        )
    fit_result = fit(model, summary, grid_size=grid_size)
    design = design_rank(model, summary)
    risk = risk_scenario(model, summary, fit_result)
    patterns = pattern_table(model, summary, fit_result.theta_mle)
    fit_check = _fit_checks(model, summary, fit_result)

    diagnostics = {
        "graph_validation": model.validate_graph(),
        "cycles": len(cycle_ids),
        "distinct_patterns": summary.n_patterns,
        "missing_values": int(sum(
            1
            for obs in observations
            for v in obs.values()
            if v is None
        )),
        "missing_policy": "NA kept as unobserved; never imputed as healthy",
        "no_pooling": (
            "cycles are independent; components observed in the same cycle are "
            "not treated as independent samples"
        ),
        "mle_at_boundary": fit_result.at_boundary,
        "support_reaches_box_edge": fit_result.boundary_hits,
    }
    return Assessment(
        model=model,
        summary=summary,
        fit=fit_result,
        risk=risk,
        diagnostics=diagnostics,
        snapshot_path=str(snapshots_path),
        contradictions=contradictions,
        design=design,
        pattern_rows=patterns,
        fit_check=fit_check,
        observations=observations,
        cycle_ids=cycle_ids,
    )


def _support_slices(fit_result: FitResult, n_slices: int = 21) -> list[dict]:
    """Conditional support intervals: for sampled p_first, the p_second range.

    This is a compact, reproducible rendering of the joint support boundary.
    """

    grid = fit_result.grid_mask
    first_axis = fit_result.grid_first[:, 0]
    second_axis = fit_result.grid_second[0, :]
    indices = np.unique(
        np.linspace(0, first_axis.size - 1, n_slices).round().astype(int)
    )
    rows: list[dict] = []
    for i in indices:
        cols = np.where(grid[i, :])[0]
        rows.append(
            {
                "first_value": float(first_axis[i]),
                "second_min": float(second_axis[cols[0]]) if cols.size else None,
                "second_max": float(second_axis[cols[-1]]) if cols.size else None,
                "covered": bool(cols.size),
            }
        )
    return rows


def fit_to_dict(fit_result: FitResult) -> dict:
    """Serialisable fit evidence (grid arrays are summarised, not dumped)."""

    names = list(fit_result.parameters)
    return {
        "theta_mle": fit_result.theta_mle.tolist(),
        "loglik_mle": fit_result.loglik_mle,
        "loglik_drop": fit_result.support_drop,
        "support_threshold": fit_result.support_threshold,
        "optimizer": {
            "starts": fit_result.starts_used,
            "message": fit_result.optimizer_message,
            "at_boundary": fit_result.at_boundary,
        },
        "local_maxima": fit_result.local_maxima,
        "support_grid": {
            "size": fit_result.grid_size,
            "axis_spacing": list(fit_result.axis_spacing),
            "points_in_support": int(fit_result.grid_mask.sum()),
            "fraction_in_support": float(fit_result.grid_mask.mean()),
        },
        "support_slices": _support_slices(fit_result),
        "marginal_projection_intervals": {
            names[0]: list(fit_result.marginal_intervals[names[0]]),
            names[1]: list(fit_result.marginal_intervals[names[1]]),
        },
        "profile_intervals": {
            names[0]: list(fit_result.profile_intervals[names[0]]),
            names[1]: list(fit_result.profile_intervals[names[1]]),
        },
        "profile_curves": {
            names[0]: {
                "theta": fit_result.profile_curves["axis0"]["grid_value"].tolist(),
                "profile_loglik": fit_result.profile_curves["axis0"]["profile_loglik"].tolist(),
            },
            names[1]: {
                "theta": fit_result.profile_curves["axis1"]["grid_value"].tolist(),
                "profile_loglik": fit_result.profile_curves["axis1"]["profile_loglik"].tolist(),
            },
        },
        "boundary_contact": fit_result.boundary_hits,
    }
