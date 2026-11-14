"""End-to-end analysis orchestration and aggregate counting.

Every FCS row is processed exactly once and accounted for in exactly one of
three buckets: ``FEASIBLE`` events (with per-gate verdicts), ``INFEASIBLE``
events (empty joint feasible set, listed separately) and ``SOLVER_ERROR``
events (backend failure, never silently dropped).  Gate proportions use the
count of feasible events as the denominator -- never an uncertain-parent
count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .errors import SolverError
from .plan import Plan
from .solver import (
    EventConstraint,
    GateDecision,
    build_constraints,
    check_witness,
    classify_gate,
    diagnose_infeasibility,
    feasibility_lp,
)

VERDICTS = ("CERTAIN_IN", "CERTAIN_OUT", "UNCERTAIN")


@dataclass
class EventResult:
    event_index: int
    observation: list[float]
    saturated: list[bool]
    status: str  # FEASIBLE | INFEASIBLE | SOLVER_ERROR
    infeasibility_reason: str | None
    constraints: list[EventConstraint]
    representative_point: list[float] | None
    representative_check: dict | None
    gate_decisions: dict[str, GateDecision] = field(default_factory=dict)
    solver_message: str | None = None


def analyze_events(data: np.ndarray, plan: Plan) -> list[EventResult]:
    n_markers = len(plan.markers)
    eps = plan.feasibility_tolerance
    sat_value = plan.saturation_value
    results: list[EventResult] = []

    for event_index, observation in enumerate(data):
        a_ub, b_ub, constraints = build_constraints(observation, plan)
        saturated = [bool(float(v) >= sat_value) for v in observation]

        try:
            outcome = feasibility_lp(a_ub, b_ub, n_markers)
        except SolverError as exc:
            results.append(
                EventResult(
                    event_index=event_index,
                    observation=[float(v) for v in observation],
                    saturated=saturated,
                    status="SOLVER_ERROR",
                    infeasibility_reason=None,
                    constraints=constraints,
                    representative_point=None,
                    representative_check=None,
                    solver_message=str(exc),
                )
            )
            continue

        if not outcome.feasible:
            reason = diagnose_infeasibility(a_ub, b_ub, constraints, plan)
            results.append(
                EventResult(
                    event_index=event_index,
                    observation=[float(v) for v in observation],
                    saturated=saturated,
                    status="INFEASIBLE",
                    infeasibility_reason=reason,
                    constraints=constraints,
                    representative_point=None,
                    representative_check=None,
                    solver_message=outcome.message,
                )
            )
            continue

        representative = outcome.point
        rep_check = check_witness(
            representative.tolist(), a_ub, b_ub,
            np.zeros((0, n_markers)), np.zeros(0), eps,
        )
        decisions: dict[str, GateDecision] = {}
        for gate_id, gate in plan.gates.items():
            decisions[gate_id] = classify_gate(gate, plan, a_ub, b_ub, eps)

        results.append(
            EventResult(
                event_index=event_index,
                observation=[float(v) for v in observation],
                saturated=saturated,
                status="FEASIBLE",
                infeasibility_reason=None,
                constraints=constraints,
                representative_point=representative.tolist(),
                representative_check=rep_check,
                gate_decisions=decisions,
                solver_message=outcome.message,
            )
        )

    return results


def summarize_gates(results: list[EventResult], plan: Plan) -> dict:
    feasible = [r for r in results if r.status == "FEASIBLE"]
    infeasible = [r for r in results if r.status == "INFEASIBLE"]
    solver_errors = [r for r in results if r.status == "SOLVER_ERROR"]
    denominator = len(feasible)

    gates_summary = {}
    for gate_id, gate in plan.gates.items():
        verdicts = {v: 0 for v in VERDICTS}
        for result in feasible:
            verdicts[result.gate_decisions[gate_id].verdict] += 1
        certain_in = verdicts["CERTAIN_IN"]
        certain_out = verdicts["CERTAIN_OUT"]
        uncertain = verdicts["UNCERTAIN"]
        # Deterministic lower bound and possible upper bound on the true count.
        count_lower = certain_in
        count_upper = certain_in + uncertain
        gates_summary[gate_id] = {
            "parent": gate.parent,
            "parent_chain": list(gate.chain),
            "effective_closed_bounds": {
                marker: list(bounds) for marker, bounds in gate.bounds.items()
            },
            "certain_in": certain_in,
            "certain_out": certain_out,
            "uncertain": uncertain,
            "count_determined_lower_bound": count_lower,
            "count_possible_upper_bound": count_upper,
            "proportion_lower_bound": (count_lower / denominator) if denominator else None,
            "proportion_upper_bound": (count_upper / denominator) if denominator else None,
        }

    return {
        "totals": {
            "events_in_fcs": len(results),
            "feasible_events": denominator,
            "infeasible_events": len(infeasible),
            "solver_error_events": len(solver_errors),
        },
        "proportion_denominator": "all feasible events (infeasible and solver-error events are excluded, never relabelled as out-of-gate)",
        "gates": gates_summary,
        "infeasible_event_indices": [r.event_index for r in infeasible],
        "solver_error_event_indices": [r.event_index for r in solver_errors],
    }
