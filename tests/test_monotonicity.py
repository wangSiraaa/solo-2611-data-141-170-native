"""Property: enlarging a gate can never reduce its determined counts.

Runs over random 2D/4D models including saturated observations and gates
whose edges land inside observation bands, so UNCERTAIN events occur
naturally.
"""

from __future__ import annotations

import numpy as np

from conftest import make_plan_dict
from satflow.plan import parse_plan
from satflow.analysis import analyze_events, summarize_gates


def _random_setup(rng, n_dim):
    # Diagonally dominant non-negative spillover so systems stay bounded.
    base = rng.uniform(0.0, 0.2, size=(n_dim, n_dim))
    spillover = base + np.eye(n_dim) * (1.0 + rng.uniform(0.0, 0.5))
    detectors = [f"D{i}" for i in range(n_dim)]
    markers = [f"M{i}" for i in range(n_dim)]

    n_events = 24
    observations = np.empty((n_events, n_dim), dtype=np.float32)
    for e in range(n_events):
        x_true = rng.uniform(0.0, 60.0, size=n_dim)
        y = spillover @ x_true
        if e % 7 == 0:  # force some saturated readings
            y[rng.integers(n_dim)] = 100.0
        observations[e] = y.astype(np.float32)
    return detectors, markers, spillover.tolist(), observations, markers


def _narrow_bounds(rng, markers):
    bounds = {}
    for marker in markers:
        edge = rng.uniform(5.0, 70.0)
        bounds[marker] = [
            [edge, None],
            [None, edge + rng.uniform(5.0, 25.0)],
            [edge, edge + rng.uniform(5.0, 25.0)],
        ][rng.integers(3)]
    return bounds


def _enlarge(bounds, margin):
    """Return a strictly containing rectangle (each relaxed side moves out)."""
    wide = {}
    for marker, (lo, hi) in bounds.items():
        wide[marker] = [
            lo - margin if lo is not None else None,
            hi + margin if hi is not None else None,
        ]
    return wide


def test_enlargement_monotonicity_random_models():
    rng = np.random.default_rng(424242)
    for _ in range(12):
        n_dim = rng.choice([2, 4])
        detectors, markers, spillover, observations, marker_names = _random_setup(rng, n_dim)
        narrow_bounds = _narrow_bounds(rng, marker_names)
        wide_bounds = _enlarge(narrow_bounds, margin=5.0 + rng.uniform(0.0, 10.0))

        narrow_plan = parse_plan(make_plan_dict(
            detectors, markers, spillover,
            [{"id": "g", "parent": None, "closed_bounds": narrow_bounds}],
        ))
        wide_plan = parse_plan(make_plan_dict(
            detectors, markers, spillover,
            [{"id": "g", "parent": None, "closed_bounds": wide_bounds}],
        ))

        narrow_summary = summarize_gates(
            analyze_events(observations, narrow_plan), narrow_plan
        )["gates"]["g"]
        wide_summary = summarize_gates(
            analyze_events(observations, wide_plan), wide_plan
        )["gates"]["g"]

        assert wide_summary["count_determined_lower_bound"] >= narrow_summary["count_determined_lower_bound"]
        assert wide_summary["count_possible_upper_bound"] >= narrow_summary["count_possible_upper_bound"]
        assert narrow_summary["count_determined_lower_bound"] <= narrow_summary["count_possible_upper_bound"]
        assert wide_summary["count_determined_lower_bound"] <= wide_summary["count_possible_upper_bound"]


def test_parent_child_chain_monotonicity_random():
    rng = np.random.default_rng(7)
    detectors, markers, spillover, observations, marker_names = _random_setup(rng, 4)
    parent_bounds = _narrow_bounds(rng, marker_names)
    # The child only tightens existing parent sides, so its region is a
    # non-empty subset and can never contradict the parent.
    child_bounds = {}
    for marker, (lo, hi) in parent_bounds.items():
        pair = list(parent_bounds[marker])
        if lo is not None:
            pair[0] = lo + rng.uniform(0.5, 2.0)
        if hi is not None:
            pair[1] = hi - rng.uniform(0.5, 2.0)
        child_bounds[marker] = pair
    plan = parse_plan(make_plan_dict(
        detectors, markers, spillover,
        [
            {"id": "parent", "parent": None, "closed_bounds": parent_bounds},
            {"id": "child", "parent": "parent", "closed_bounds": child_bounds},
        ],
    ))
    results = analyze_events(observations, plan)
    summary = summarize_gates(results, plan)["gates"]
    assert summary["child"]["count_determined_lower_bound"] <= summary["parent"]["count_determined_lower_bound"]
    assert summary["child"]["count_possible_upper_bound"] <= summary["parent"]["count_possible_upper_bound"]

    for result in results:
        if result.status != "FEASIBLE":
            continue
        vp = result.gate_decisions["parent"].verdict
        vc = result.gate_decisions["child"].verdict
        if vc == "CERTAIN_IN":
            assert vp == "CERTAIN_IN"
        if vp == "CERTAIN_OUT":
            assert vc == "CERTAIN_OUT"
