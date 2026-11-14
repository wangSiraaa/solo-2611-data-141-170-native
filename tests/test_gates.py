"""Gate semantics: parent inheritance, closed boundaries, enlargement."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import plan_from
from satflow.analysis import analyze_events, summarize_gates
from satflow.plan import parse_plan
from satflow.errors import StructureError
from satflow.solver import gate_rows


def _three_event_plan(gates):
    # Identity 2x2, three events: (10,10), (20,20), (30,30).
    return plan_from(
        ["D1", "D2"], ["M1", "M2"], [[1.0, 0.0], [0.0, 1.0]],
        gates=gates,
        absolute_tolerance=0.5,
    )


EVENTS = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float32)


def test_child_inherits_parent_bounds():
    plan = _three_event_plan([
        {"id": "parent", "parent": None, "closed_bounds": {"M1": [15, None]}},
        {"id": "child", "parent": "parent", "closed_bounds": {"M2": [None, 25]}},
    ])
    child = plan.gates["child"]
    assert child.bounds == {"M1": (15.0, None), "M2": (None, 25.0)}
    assert child.chain == ("parent", "child")
    results = analyze_events(EVENTS, plan)
    # Event (20,20) is inside both; (10,10) fails the parent M1 bound;
    # (30,30) fails the child M2 bound.
    verdicts = [r.gate_decisions["child"].verdict for r in results]
    assert verdicts == ["CERTAIN_OUT", "CERTAIN_IN", "CERTAIN_OUT"]


def test_enlarging_a_gate_never_reduces_counts():
    narrow = _three_event_plan([
        {"id": "g", "parent": None, "closed_bounds": {"M1": [25, None], "M2": [25, None]}},
    ])
    wide = _three_event_plan([
        {"id": "g", "parent": None, "closed_bounds": {"M1": [15, None], "M2": [15, None]}},
    ])
    narrow_summary = summarize_gates(analyze_events(EVENTS, narrow), narrow)["gates"]["g"]
    wide_summary = summarize_gates(analyze_events(EVENTS, wide), wide)["gates"]["g"]
    assert wide_summary["count_determined_lower_bound"] >= narrow_summary["count_determined_lower_bound"]
    assert wide_summary["count_possible_upper_bound"] >= narrow_summary["count_possible_upper_bound"]
    # Narrow admits only (30,30); wide admits (20,20) and (30,30).
    assert narrow_summary["certain_in"] == 1
    assert wide_summary["certain_in"] == 2


def test_enlargement_holds_with_uncertain_events():
    # Band [9.5, 10.5]: a gate at 10.5 cuts through it (UNCERTAIN), while a
    # gate at 9.5 contains the whole closed band (CERTAIN_IN).
    plan = plan_from(
        ["D1"], ["M1"], [[1.0]],
        gates=[{"id": "narrow", "parent": None, "closed_bounds": {"M1": [10.5, None]}}],
        absolute_tolerance=0.5,
    )
    plan_wide = plan_from(
        ["D1"], ["M1"], [[1.0]],
        gates=[{"id": "narrow", "parent": None, "closed_bounds": {"M1": [9.5 - 1e-3, None]}}],
        absolute_tolerance=0.5,
        feasibility_tolerance=1e-7,
    )
    obs = np.array([[10.0]], dtype=np.float32)
    narrow = analyze_events(obs, plan)[0].gate_decisions["narrow"]
    wide = analyze_events(obs, plan_wide)[0].gate_decisions["narrow"]
    assert narrow.verdict == "UNCERTAIN"
    assert wide.verdict == "CERTAIN_IN"


def test_closed_boundary_points_are_inside():
    # float32 stores 15.0 exactly.  Place the gate edges exactly on the
    # band edges as real numbers so the whole closed band is contained.
    plan = plan_from(
        ["D1", "D2"], ["M1", "M2"], [[1.0, 0.0], [0.0, 1.0]],
        gates=[{"id": "box", "parent": None,
                "closed_bounds": {"M1": [15.0, None], "M2": [None, 15.0]}}],
        absolute_tolerance=0.5,
        feasibility_tolerance=1e-9,
    )
    obs = np.array([[15.0, 15.0]], dtype=np.float32)
    decision = analyze_events(obs, plan)[0].gate_decisions["box"]
    # Band [14.5, 15.5] crosses the edges, so the event is UNCERTAIN; the
    # inside witness nevertheless sits on the closed boundary (included),
    # while the outside witness strictly violates it.
    assert decision.verdict == "UNCERTAIN"
    inside = np.asarray(decision.inside_witness)
    outside = np.asarray(decision.outside_witness)
    assert inside[0] >= 15.0 - 1e-9 and inside[1] <= 15.0 + 1e-9
    assert (outside[0] < 15.0 - 1e-9) or (outside[1] > 15.0 + 1e-9)

    # A gate containing the whole closed band must be CERTAIN_IN with no
    # outside witness.
    plan_wide = plan_from(
        ["D1", "D2"], ["M1", "M2"], [[1.0, 0.0], [0.0, 1.0]],
        gates=[{"id": "box", "parent": None,
                "closed_bounds": {"M1": [14.0, None], "M2": [None, 16.0]}}],
        absolute_tolerance=0.5,
        feasibility_tolerance=1e-9,
    )
    wide = analyze_events(obs, plan_wide)[0].gate_decisions["box"]
    assert wide.verdict == "CERTAIN_IN"
    assert wide.outside_witness is None


def test_gate_rows_cover_every_bound():
    plan = _three_event_plan([
        {"id": "g", "parent": None,
         "closed_bounds": {"M1": [5, 25], "M2": [None, 10]}},
    ])
    g_ub, g_rhs, description = gate_rows(plan.gates["g"], ["M1", "M2"])
    assert g_ub.shape == (3, 2)
    sides = {(d["marker"], d["side"], d["bound"]) for d in description}
    assert sides == {("M1", "lower", 5.0), ("M1", "upper", 25.0), ("M2", "upper", 10.0)}


def test_unknown_parent_and_cycles_are_structure_errors():
    base = {
        "detectors": ["D1"], "markers": ["M1"],
        "spillover_rows_detectors_cols_markers": [[1.0]],
        "saturation_value": 100.0, "absolute_tolerance": 0.5, "feasibility_tolerance": 1e-7,
    }
    with pytest.raises(StructureError, match="unknown parent"):
        parse_plan({**base, "gates": [
            {"id": "g", "parent": "ghost", "closed_bounds": {"M1": [0, None]}}]})
    with pytest.raises(StructureError, match="cycle"):
        parse_plan({**base, "gates": [
            {"id": "a", "parent": "b", "closed_bounds": {"M1": [0, None]}},
            {"id": "b", "parent": "a", "closed_bounds": {"M1": [0, None]}}]})


def test_empty_gate_after_inheritance_is_structure_error():
    with pytest.raises(StructureError, match="empty after parent inheritance"):
        parse_plan({
            "detectors": ["D1"], "markers": ["M1"],
            "spillover_rows_detectors_cols_markers": [[1.0]],
            "saturation_value": 100.0, "absolute_tolerance": 0.5, "feasibility_tolerance": 1e-7,
            "gates": [
                {"id": "p", "parent": None, "closed_bounds": {"M1": [0, 10]}},
                {"id": "c", "parent": "p", "closed_bounds": {"M1": [20, None]}},
            ],
        })


def test_illegal_gate_bounds_rejected():
    with pytest.raises(StructureError, match="lower .* > upper"):
        parse_plan({
            "detectors": ["D1"], "markers": ["M1"],
            "spillover_rows_detectors_cols_markers": [[1.0]],
            "saturation_value": 100.0, "absolute_tolerance": 0.5, "feasibility_tolerance": 1e-7,
            "gates": [{"id": "g", "parent": None, "closed_bounds": {"M1": [10, 5]}}],
        })
