"""End-to-end checks on the supplied four-channel fixture."""

from __future__ import annotations

import csv
import json
import os

import numpy as np

from conftest import independent_feasible, independent_in_gate, independent_out_gate
from satflow.analysis import summarize_gates
from satflow.fcsio import load_fcs


def test_fcs_channel_matching_and_row_identity(fixture_plan):
    data, names, text = load_fcs(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "events.fcs"),
        fixture_plan.detectors,
    )
    assert names == ["FL1", "FL2", "FL3", "FL4"]
    assert data.dtype == np.float32
    assert data.shape == (65, 4)
    # Original DATA row 64 is the deliberately negative reading.
    assert data[64, 0] == np.float32(-5.0)
    assert text["$DATATYPE"] == "F"
    assert text["$MODE"] == "L"


def test_every_event_accounted_for(fixture_results):
    statuses = [r.status for r in fixture_results]
    assert len(fixture_results) == 65
    assert statuses.count("FEASIBLE") == 64
    assert statuses.count("INFEASIBLE") == 1
    assert statuses.count("SOLVER_ERROR") == 0
    # Row indices are the original zero-based DATA positions, in order.
    assert [r.event_index for r in fixture_results] == list(range(65))


def test_negative_reading_is_the_single_infeasible_event(fixture_results, fixture_data):
    bad = [r for r in fixture_results if r.status == "INFEASIBLE"]
    assert [r.event_index for r in bad] == [64]
    assert "negative detector reading" in bad[0].infeasibility_reason
    assert fixture_data[64, 0] == np.float32(-5.0)
    # The infeasible event carries no gate verdicts and never enters counts.
    assert bad[0].gate_decisions == {}
    assert bad[0].representative_point is None


def test_saturated_events_have_lower_bound_only(fixture_results, fixture_plan):
    saturated_rows = [2, 3, 4, 34]
    for index in saturated_rows:
        result = fixture_results[index]
        kinds = {(c.detector, c.kind) for c in result.constraints}
        sat = [c for c in result.constraints if c.kind == "saturation_lower"]
        assert len(sat) == 1
        assert sat[0].lower == fixture_plan.saturation_value - fixture_plan.absolute_tolerance
        assert sat[0].upper is None
        # No row in the serialised system may carry an upper edge of 100.5
        # for the saturated detector.
        for constraint in result.constraints:
            if constraint.detector == sat[0].detector:
                assert constraint.upper is None
        assert result.status == "FEASIBLE"
        assert kinds  # collection non-empty


def test_witnesses_independently_recheck(fixture_results, fixture_data, fixture_plan):
    """Every stored witness must satisfy the joint system as one vector."""
    for result in fixture_results:
        if result.status != "FEASIBLE":
            continue
        observation = fixture_data[result.event_index]
        assert independent_feasible(result.representative_point, observation, fixture_plan)
        for decision in result.gate_decisions.values():
            gate = fixture_plan.gates[decision.gate_id]
            if decision.verdict == "CERTAIN_IN":
                assert decision.outside_witness is None
                inside = np.asarray(decision.inside_witness)
                assert independent_feasible(inside, observation, fixture_plan)
                assert independent_in_gate(inside, gate, fixture_plan.markers)
            elif decision.verdict == "CERTAIN_OUT":
                assert decision.inside_witness is None
                outside = np.asarray(decision.outside_witness)
                assert independent_feasible(outside, observation, fixture_plan)
                assert independent_out_gate(outside, gate, fixture_plan.markers, eps=1e-9)
            else:
                assert decision.inside_witness is not None
                assert decision.outside_witness is not None
                inside = np.asarray(decision.inside_witness)
                outside = np.asarray(decision.outside_witness)
                assert independent_feasible(inside, observation, fixture_plan)
                assert independent_feasible(outside, observation, fixture_plan)
                assert independent_in_gate(inside, gate, fixture_plan.markers)
                assert independent_out_gate(outside, gate, fixture_plan.markers, eps=1e-9)
                # The two witnesses must be genuinely different points.
                assert not np.allclose(inside, outside)


def test_verdict_consistency_with_independent_grid(fixture_results, fixture_data, fixture_plan):
    """Verdicts must agree with an independent dense sampling of feasible boxes.

    For normal (non-saturated) events the feasible marker set is a bounded
    polyhedron; we sample a large Latin-hypercube-style point cloud, keep only
    points that independently satisfy the observation model, and compare the
    resulting gate coverage against the LP verdict.
    """
    rng = np.random.default_rng(20261114)
    for result in fixture_results:
        if result.status != "FEASIBLE" or any(result.saturated):
            continue
        observation = fixture_data[result.event_index]
        # The feasible marker box has width ~2*tol around the compensated
        # center; sample tightly so a meaningful fraction lands inside.
        center = np.linalg.solve(fixture_plan.spillover, observation.astype(float))
        cloud = center + rng.uniform(-1.0, 1.0, size=(60000, len(center)))
        cloud = cloud[np.all(cloud >= 0, axis=1)]
        feasible = np.array(
            [independent_feasible(p, observation, fixture_plan) for p in cloud]
        )
        points = cloud[feasible]
        assert len(points) > 500
        for decision in result.gate_decisions.values():
            gate = fixture_plan.gates[decision.gate_id]
            inside = np.array(
                [independent_in_gate(p, gate, fixture_plan.markers) for p in points]
            )
            sampled_any_inside = bool(inside.any())
            sampled_any_outside = bool((~inside).any())
            if decision.verdict == "CERTAIN_IN":
                assert not sampled_any_outside
            elif decision.verdict == "CERTAIN_OUT":
                assert not sampled_any_inside
            else:
                assert sampled_any_inside and sampled_any_outside


def test_gate_counts_and_denominator(fixture_results, fixture_plan):
    summary = summarize_gates(fixture_results, fixture_plan)
    assert summary["totals"]["feasible_events"] == 64
    for gate_id, gate in summary["gates"].items():
        assert (
            gate["certain_in"] + gate["certain_out"] + gate["uncertain"] == 64
        )
        assert gate["count_determined_lower_bound"] == gate["certain_in"]
        assert gate["count_possible_upper_bound"] == gate["certain_in"] + gate["uncertain"]
        assert 0.0 <= gate["proportion_lower_bound"] <= gate["proportion_upper_bound"] <= 1.0
        # Bounds are proportions over feasible events, never another denominator.
        assert gate["proportion_lower_bound"] == gate["count_determined_lower_bound"] / 64
        assert gate["proportion_upper_bound"] == gate["count_possible_upper_bound"] / 64


def test_child_gate_is_nested_in_parent(fixture_results, fixture_plan):
    summary = summarize_gates(fixture_results, fixture_plan)["gates"]
    g1, g2 = summary["G1"], summary["G2"]
    assert g2["count_determined_lower_bound"] <= g1["count_determined_lower_bound"]
    assert g2["count_possible_upper_bound"] <= g1["count_possible_upper_bound"]
    # Per-event: CERTAIN_IN the child implies CERTAIN_IN the parent;
    # CERTAIN_OUT the parent implies CERTAIN_OUT the child.
    for result in fixture_results:
        if result.status != "FEASIBLE":
            continue
        v1 = result.gate_decisions["G1"].verdict
        v2 = result.gate_decisions["G2"].verdict
        if v2 == "CERTAIN_IN":
            assert v1 == "CERTAIN_IN"
        if v1 == "CERTAIN_OUT":
            assert v2 == "CERTAIN_OUT"


def test_csv_and_json_deliverables(tmp_path, fixture_plan, fixture_data):
    from satflow.cli import main

    fcs = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "events.fcs")
    plan_json = tmp_path / "plan.json"
    plan_json.write_text(json.dumps(fixture_plan.raw))
    outdir = tmp_path / "out"
    code = main(["analyze", "--fcs", fcs, "--plan", str(plan_json), "--outdir", str(outdir)])
    assert code == 0
    for name in ("event-status.csv", "gate-counts.json", "witnesses.json", "diagnostics.json"):
        assert (outdir / name).exists()

    with open(outdir / "event-status.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 65
    assert rows[64]["status"] == "INFEASIBLE"
    assert rows[2]["saturated_FL1"] == "true"
    assert rows[2]["gate_G1_verdict"] in {"CERTAIN_IN", "CERTAIN_OUT", "UNCERTAIN"}

    diagnostics = json.loads((outdir / "diagnostics.json").read_text())
    assert diagnostics["accounting_identity_holds"] is True
    assert diagnostics["witness_checks"]["all_present_witnesses_pass"] is True
    assert diagnostics["event_accounting"]["infeasible_events"] == 1
