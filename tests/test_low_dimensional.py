"""Hand-drawable low-dimensional joint-constraint examples.

These 1D/2D cases are small enough to reason about on paper.  The 2D case
in particular is constructed so that *independent* per-marker bounds would
paint a feasible vector that does not exist under the joint system; only
joint witnesses give the correct CERTAIN_OUT verdict.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    independent_feasible,
    independent_in_gate,
    independent_out_gate,
    plan_from,
)
from satflow.analysis import analyze_events
from satflow.solver import build_constraints


# ---------------------------------------------------------------------------
# 1D saturation: the detector ceiling must act as a lower bound, never as 100.
# ---------------------------------------------------------------------------

def test_1d_saturation_is_lower_bound_only():
    plan = plan_from(
        ["FL1"], ["A"], [[1.0]],
        gates=[{"id": "low", "parent": None, "closed_bounds": {"A": [None, 50]}}],
        saturation_value=100.0,
        absolute_tolerance=0.5,
    )
    obs = np.array([[100.0]], dtype=np.float32)
    results = analyze_events(obs, plan)
    result = results[0]
    assert result.status == "FEASIBLE"
    constraint = result.constraints[0]
    assert constraint.kind == "saturation_lower"
    assert constraint.lower == 99.5 and constraint.upper is None
    # Gate A <= 50: every feasible A is >= 99.5, so the event is certainly out.
    assert result.gate_decisions["low"].verdict == "CERTAIN_OUT"
    outside = result.gate_decisions["low"].outside_witness
    assert independent_feasible(outside, obs[0], plan)
    assert independent_out_gate(outside, plan.gates["low"], ["A"])
    # An inside witness must not exist.
    assert result.gate_decisions["low"].inside_witness is None


def test_1d_saturation_makes_high_gate_certain_in():
    plan = plan_from(
        ["FL1"], ["A"], [[1.0]],
        gates=[{"id": "high", "parent": None, "closed_bounds": {"A": [90, None]}}],
    )
    results = analyze_events(np.array([[100.0]], dtype=np.float32), plan)
    decision = results[0].gate_decisions["high"]
    assert decision.verdict == "CERTAIN_IN"
    assert decision.outside_witness is None


def test_1d_normal_band_straddling_gate_boundary_is_uncertain():
    # Reading 10 +/- 0.5 means A in [9.5, 10.5]; closed gate A >= 10 cuts it.
    plan = plan_from(
        ["FL1"], ["A"], [[1.0]],
        gates=[{"id": "g", "parent": None, "closed_bounds": {"A": [10, None]}}],
    )
    results = analyze_events(np.array([[10.0]], dtype=np.float32), plan)
    decision = results[0].gate_decisions["g"]
    assert decision.verdict == "UNCERTAIN"
    inside = np.asarray(decision.inside_witness)
    outside = np.asarray(decision.outside_witness)
    assert inside[0] >= 10 - 1e-7 and outside[0] < 10
    assert independent_in_gate(inside, plan.gates["g"], ["A"])
    assert independent_out_gate(outside, plan.gates["g"], ["A"])


def test_1d_closed_boundary_is_included():
    # Reading 10 with tol 1.0 gives the closed band [9, 11]; the gate A >= 11
    # still admits the boundary point 11, so the verdict is UNCERTAIN but the
    # inside witness sits exactly on the closed boundary.
    plan = plan_from(
        ["FL1"], ["A"], [[1.0]],
        gates=[{"id": "edge", "parent": None, "closed_bounds": {"A": [11, None]}}],
        absolute_tolerance=1.0,
        feasibility_tolerance=1e-9,
    )
    decision = analyze_events(np.array([[10.0]], dtype=np.float32), plan)[0].gate_decisions["edge"]
    assert decision.verdict == "UNCERTAIN"
    assert abs(decision.inside_witness[0] - 11.0) <= 1e-7
    assert decision.outside_witness[0] < 11.0


def test_1d_negative_reading_infeasible():
    plan = plan_from(
        ["FL1"], ["A"], [[1.0]],
        gates=[{"id": "g", "parent": None, "closed_bounds": {"A": [0, None]}}],
    )
    results = analyze_events(np.array([[-5.0]], dtype=np.float32), plan)
    assert results[0].status == "INFEASIBLE"
    assert "negative detector reading" in results[0].infeasibility_reason


# ---------------------------------------------------------------------------
# 2D cross-channel coupling: per-marker boxes would invent a non-existent x.
# ---------------------------------------------------------------------------

def test_2d_joint_constraint_couples_channels():
    """Identity detectors but coupled readings.

    Observation y = (30, 30) with tol 0.5 forces (x1, x2) in
    [29.5, 30.5] x [29.5, 30.5].  The gate x1 >= 25 AND x2 >= 40 is certainly
    empty for this event even though x1's independent interval reaches values
    above 25: no single vector satisfies both gate sides at once.
    """
    plan = plan_from(
        ["D1", "D2"], ["M1", "M2"], [[1.0, 0.0], [0.0, 1.0]],
        gates=[{"id": "coupled", "parent": None,
                "closed_bounds": {"M1": [25, None], "M2": [40, None]}}],
    )
    obs = np.array([[30.0, 30.0]], dtype=np.float32)
    result = analyze_events(obs, plan)[0]
    decision = result.gate_decisions["coupled"]
    assert decision.verdict == "CERTAIN_OUT"
    assert decision.inside_witness is None
    outside = np.asarray(decision.outside_witness)
    assert independent_feasible(outside, obs[0], plan)
    assert independent_out_gate(outside, plan.gates["coupled"], ["M1", "M2"])

    # The naive per-channel box (x1 in [29.5,30.5] alone, x2 in [29.5,30.5]
    # alone) would suggest a point like (30.5, 40) -- prove it is NOT feasible.
    fake = np.array([30.5, 40.0])
    assert not independent_feasible(fake, obs[0], plan)


def test_2d_off_diagonal_spillover_uncertain_with_two_witnesses():
    # y1 = x1 + 0.5 x2, y2 = x2.  Reading (10, 6) +/- 0.5.
    plan = plan_from(
        ["D1", "D2"], ["M1", "M2"], [[1.0, 0.5], [0.0, 1.0]],
        gates=[{"id": "g", "parent": None,
                "closed_bounds": {"M1": [6.5, None], "M2": [None, 10]}}],
    )
    obs = np.array([[10.0, 6.0]], dtype=np.float32)
    result = analyze_events(obs, plan)[0]
    assert result.status == "FEASIBLE"
    decision = result.gate_decisions["g"]
    assert decision.verdict == "UNCERTAIN"
    inside = np.asarray(decision.inside_witness)
    outside = np.asarray(decision.outside_witness)
    for point in (inside, outside):
        assert independent_feasible(point, obs[0], plan)
    assert independent_in_gate(inside, plan.gates["g"], ["M1", "M2"])
    # The witness violates the closed boundary by exactly the LP epsilon
    # (1e-7); re-check with a narrower epsilon so the strict gap is visible.
    assert independent_out_gate(outside, plan.gates["g"], ["M1", "M2"], eps=1e-9)
    # Both witnesses satisfy EVERY detector band simultaneously (joint, not
    # per-channel) -- check the coupled row explicitly.
    for point in (inside, outside):
        y1 = point[0] + 0.5 * point[1]
        assert 9.5 - 1e-7 <= y1 <= 10.5 + 1e-7
        assert 5.5 - 1e-7 <= point[1] <= 6.5 + 1e-7


def test_2d_infeasible_band_intersection_with_nonnegative_marker():
    # y1 = x1 + x2, y2 = x1 + x2: two detectors read the same sum but their
    # bands do not overlap -> joint intersection empty even before x >= 0.
    plan = plan_from(
        ["D1", "D2"], ["M1", "M2"], [[1.0, 1.0], [1.0, 1.0]],
        gates=[{"id": "g", "parent": None, "closed_bounds": {"M1": [0, None]}}],
        absolute_tolerance=0.5,
    )
    results = analyze_events(np.array([[10.0, 20.0]], dtype=np.float32), plan)
    assert results[0].status == "INFEASIBLE"
    assert "band" in results[0].infeasibility_reason


def test_2d_feasible_only_with_negative_marker_is_infeasible():
    # y1 = x1 + x2 in [9.5, 10.5]; y2 = -... is impossible with S >= 0, so
    # instead use y1 = x1 + x2, y2 = x2 with readings forcing x2 > y1 sum.
    # y1 in [9.5,10.5], y2 in [19.5,20.5] => x2 ~20 > x1+x2 ~10 needs x1 < 0.
    plan = plan_from(
        ["D1", "D2"], ["M1", "M2"], [[1.0, 1.0], [0.0, 1.0]],
        gates=[{"id": "g", "parent": None, "closed_bounds": {"M1": [0, None]}}],
    )
    results = analyze_events(np.array([[10.0, 20.0]], dtype=np.float32), plan)
    assert results[0].status == "INFEASIBLE"
    assert "negative components" in results[0].infeasibility_reason


def test_constraint_rows_are_joint_and_saturated_has_no_upper_row():
    plan = plan_from(
        ["D1", "D2"], ["M1", "M2"], [[1.0, 0.5], [0.0, 1.0]],
        gates=[{"id": "g", "parent": None, "closed_bounds": {"M1": [0, None]}}],
    )
    obs = np.array([100.0, 10.0], dtype=np.float32)
    a_ub, b_ub, described = build_constraints(obs, plan)
    # One saturated lower row + two normal band rows = 3 A_ub rows.
    assert a_ub.shape == (3, 2)
    assert described[0].kind == "saturation_lower" and described[0].upper is None
    # The saturated row must be -S1 x <= -99.5; nothing bounds it above.
    assert np.allclose(a_ub[0], [-1.0, -0.5])
    assert b_ub[0] == pytest.approx(-99.5)
    # Remaining rows belong to the normal D2 band only.
    assert all(c.detector == "D2" for c in described[1:])
