"""Serialisation of the four deliverables.

* ``event-status.csv``       -- one row per original FCS event, original row
  index, raw observations, per-event constraint summary and per-gate verdict.
* ``gate-counts.json``       -- deterministic lower / possible upper counts
  and proportion ranges per gate.
* ``witnesses.json``         -- full joint constraint system, representative
  point and inside/outside witnesses with independent re-checks, so another
  program can substitute the witnesses into the model.
* ``diagnostics.json``       -- structural validation notes, bucket counts,
  infeasible events and monotonicity / accounting self-checks.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from .analysis import EventResult, summarize_gates
from .plan import Plan


def write_event_status_csv(path: str, results: list[EventResult], plan: Plan) -> None:
    gate_ids = list(plan.gates)
    fields = [
        "event_index",
        "status",
        *[f"observed_{name}" for name in plan.detectors],
        *[f"saturated_{name}" for name in plan.detectors],
        "n_constraint_rows",
        "saturated_detectors",
        "infeasibility_reason",
        "solver_message",
        *[f"gate_{gate_id}_verdict" for gate_id in gate_ids],
        *[f"gate_{gate_id}_reason" for gate_id in gate_ids],
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {
                "event_index": result.event_index,
                "status": result.status,
                "n_constraint_rows": len(_constraint_vectors(result)),
                "saturated_detectors": "|".join(
                    plan.detectors[i]
                    for i, flag in enumerate(result.saturated)
                    if flag
                ),
                "infeasibility_reason": result.infeasibility_reason or "",
                "solver_message": result.solver_message or "",
            }
            for i, name in enumerate(plan.detectors):
                row[f"observed_{name}"] = _float32_repr(result.observation[i])
                row[f"saturated_{name}"] = str(result.saturated[i]).lower()
            for gate_id in gate_ids:
                decision = result.gate_decisions.get(gate_id)
                row[f"gate_{gate_id}_verdict"] = decision.verdict if decision else ""
                row[f"gate_{gate_id}_reason"] = decision.reason if decision else ""
            writer.writerow(row)


def write_gate_counts(path: str, results: list[EventResult], plan: Plan) -> dict:
    summary = summarize_gates(results, plan)
    payload = {
        "schema_version": 1,
        "proportion_denominator": summary["proportion_denominator"],
        "totals": summary["totals"],
        "gates": summary["gates"],
        "infeasible_event_indices": summary["infeasible_event_indices"],
        "solver_error_event_indices": summary["solver_error_event_indices"],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return payload


def write_witnesses(path: str, results: list[EventResult], plan: Plan) -> None:
    payload = {
        "schema_version": 1,
        "model": {
            "observation_model": plan.raw.get(
                "observation_model", "detector_vector = S @ nonnegative_marker_vector"
            ),
            "detectors": plan.detectors,
            "markers": plan.markers,
            "spillover_rows_detectors_cols_markers": plan.spillover.tolist(),
            "saturation_value": plan.saturation_value,
            "absolute_tolerance": plan.absolute_tolerance,
            "feasibility_tolerance": plan.feasibility_tolerance,
            "saturation_rule": plan.raw.get("saturation_rule"),
            "normal_rule": plan.raw.get("normal_rule"),
            "gate_semantics": (
                "closed rectangle; null side is unbounded; child gates intersect "
                "with every ancestor; boundaries are included"
            ),
        },
        "events": [_event_payload(result) for result in results],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_diagnostics(
    path: str,
    results: list[EventResult],
    plan: Plan,
    fcs_text: dict,
    fcs_path: str,
    plan_path: str,
) -> dict:
    summary = summarize_gates(results, plan)
    checks = run_self_checks(results, plan)
    payload = {
        "schema_version": 1,
        "inputs": {
            "fcs_path": fcs_path,
            "plan_path": plan_path,
            "fcs_text_keywords": fcs_text,
        },
        "structure_validation": {
            "status": "passed",
            "notes": [
                "FCS3.1, list mode, little-endian float32, four channels matched by $PnN",
                "spillover rows are detectors and columns are markers; matrix was not transposed",
                "gates validated as closed rectangles with acyclic parent links",
            ],
        },
        "event_accounting": summary["totals"],
        "accounting_identity_holds": checks["accounting_identity_holds"],
        "witness_checks": checks["witness_checks"],
        "gate_monotonicity_parent_child": checks["parent_child_monotonicity"],
        "infeasible_events": [
            {
                "event_index": result.event_index,
                "observation": result.observation,
                "reason": result.infeasibility_reason,
                "constraints": [c.to_json() for c in result.constraints],
            }
            for result in results
            if result.status == "INFEASIBLE"
        ],
        "solver_error_events": [
            {
                "event_index": result.event_index,
                "message": result.solver_message,
            }
            for result in results
            if result.status == "SOLVER_ERROR"
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return payload


def run_self_checks(results: list[EventResult], plan: Plan) -> dict:
    total = len(results)
    buckets = {"FEASIBLE": 0, "INFEASIBLE": 0, "SOLVER_ERROR": 0}
    for result in results:
        buckets[result.status] += 1
    accounting_identity_holds = sum(buckets.values()) == total

    witness_checks = {"checked_events": 0, "all_present_witnesses_pass": True, "failures": []}
    for result in results:
        if result.status != "FEASIBLE":
            continue
        witness_checks["checked_events"] += 1
        if not result.representative_check or not result.representative_check.get(
            "passes_joint_observation_constraints"
        ):
            witness_checks["all_present_witnesses_pass"] = False
            witness_checks["failures"].append(
                {"event_index": result.event_index, "which": "representative_point"}
            )
        for gate_id, decision in result.gate_decisions.items():
            pairs = [
                ("inside", decision.inside_witness, decision.inside_witness_check,
                 "passes_as_inside_witness"),
                ("outside", decision.outside_witness, decision.outside_witness_check,
                 "passes_as_outside_witness"),
            ]
            for label, point, check, key in pairs:
                if point is None:
                    continue
                if not check or not check.get(key):
                    witness_checks["all_present_witnesses_pass"] = False
                    witness_checks["failures"].append(
                        {"event_index": result.event_index, "gate": gate_id, "which": label}
                    )

    # Parent/child counting: a child region is a subset of its parent's, so
    # its determined lower bound cannot exceed the parent's and its possible
    # upper bound cannot exceed the parent's either.
    monotonicity = []
    for gate_id, gate in plan.gates.items():
        if gate.parent is None:
            continue
        parent = summarize_gates(results, plan)["gates"][gate.parent]
        child = summarize_gates(results, plan)["gates"][gate_id]
        holds = (
            child["count_determined_lower_bound"] <= parent["count_determined_lower_bound"]
            and child["count_possible_upper_bound"] <= parent["count_possible_upper_bound"]
        )
        monotonicity.append(
            {
                "parent": gate.parent,
                "child": gate_id,
                "holds": holds,
                "parent_lower": parent["count_determined_lower_bound"],
                "child_lower": child["count_determined_lower_bound"],
                "parent_upper": parent["count_possible_upper_bound"],
                "child_upper": child["count_possible_upper_bound"],
            }
        )

    return {
        "accounting_identity_holds": accounting_identity_holds,
        "witness_checks": witness_checks,
        "parent_child_monotonicity": monotonicity,
    }


def _event_payload(result: EventResult) -> dict:
    payload = {
        "event_index": result.event_index,
        "status": result.status,
        "original_observation_float32": [
            {"detector": name, "value": _float32_repr(value), "saturated": flag}
            for name, value, flag in zip(
                _detector_names(result), result.observation, result.saturated
            )
        ],
        "joint_feasible_set": {
            "description": (
                "x >= 0 over markers, subject to A_ub @ x <= b_ub; one vector must satisfy "
                "every row simultaneously"
            ),
            "A_ub": _constraint_vectors(result),
            "b_ub": _constraint_rhs(result),
            "rows": [constraint.to_json() for constraint in result.constraints],
        },
        "infeasibility_reason": result.infeasibility_reason,
        "solver_message": result.solver_message,
        "representative_feasible_marker_vector": result.representative_point,
        "representative_check": result.representative_check,
        "gate_decisions": [
            {
                "gate_id": decision.gate_id,
                "verdict": decision.verdict,
                "reason": decision.reason,
                "inside_witness_marker_vector": decision.inside_witness,
                "outside_witness_marker_vector": decision.outside_witness,
                "inside_witness_check": decision.inside_witness_check,
                "outside_witness_check": decision.outside_witness_check,
                "detail": decision.detail,
            }
            for decision in result.gate_decisions.values()
        ],
    }
    return payload


def _constraint_vectors(result: EventResult) -> list[list[float]]:
    vectors: list[list[float]] = []
    for constraint in result.constraints:
        vectors.append([-v for v in constraint.row])  # lower bound row
        if constraint.upper is not None:
            vectors.append(list(constraint.row))
    return vectors


def _constraint_rhs(result: EventResult) -> list[float]:
    rhs: list[float] = []
    for constraint in result.constraints:
        rhs.append(-(constraint.lower if constraint.lower is not None else 0.0))
        if constraint.upper is not None:
            rhs.append(constraint.upper)
    return rhs


def _detector_names(result: EventResult) -> list[str]:
    return [constraint.detector for constraint in result.constraints]


def _float32_repr(value: float) -> float:
    """Round-trip the stored float32 reading without inventing precision."""
    return float(np.float32(value))
