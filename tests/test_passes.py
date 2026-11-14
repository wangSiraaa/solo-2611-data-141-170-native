"""Unit tests for the geometry, horizon, search and evidence layers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from passes.errors import InputError, PropagationError
from passes.geometry import Ephemeris
from passes.io import (
    HorizonMask,
    load_horizon,
    load_plan,
    load_station,
    load_tle,
    station_geometry,
)
from passes.search import find_node_crossings, merge_roots, unwrapped_azimuth
from passes.solver import SolverConfig, solve
from passes.verify import IndependentModel, enu_axis_checks, independent_root

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="module")
def inputs():
    tle = load_tle(FIXTURES / "synthetic.tle")
    station = load_station(FIXTURES / "station.json")
    mask = load_horizon(FIXTURES / "horizon.csv")
    plan = load_plan(FIXTURES / "plan.json")
    return tle, station, mask, plan


def _ephemeris(inputs):
    tle, station, mask, plan = inputs
    ephem = Ephemeris(tle.satrec, station, mask, plan.start)
    ephem.bind_epoch(*plan.start_jd)
    return ephem


# --------------------------------------------------------------------- horizon


def test_horizon_periodic_join_at_360():
    mask = load_horizon(FIXTURES / "horizon.csv")
    # The wrap segment 300 -> 0 must interpolate down to 8 at 360/0.
    assert mask.threshold_deg(300.0) == pytest.approx(8.0)
    assert mask.threshold_deg(0.0) == pytest.approx(8.0)
    assert mask.threshold_deg(360.0) == pytest.approx(8.0)
    assert mask.threshold_deg(-60.0) == pytest.approx(8.0)
    # Midpoint of the wrap segment.
    assert mask.threshold_deg(330.0) == pytest.approx(8.0)
    # Raised plateau interior and its linear flanks.
    assert mask.threshold_deg(271.637253460) == pytest.approx(36.087036546)
    assert mask.threshold_deg(272.137253460) == pytest.approx(36.087036546)
    mid = 0.5 * (270.687253460 + 271.637253460)
    expected = 0.5 * (8.0 + 36.087036546)
    assert mask.threshold_deg(mid) == pytest.approx(expected)
    # Strictly periodic.
    for azimuth in (0.0, 37.5, 271.0, 359.999):
        assert mask.threshold_deg(azimuth) == pytest.approx(mask.threshold_deg(azimuth + 360.0))
        assert mask.threshold_deg(azimuth) == pytest.approx(mask.threshold_deg(azimuth - 720.0))


def test_horizon_rejects_duplicate_azimuth(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "azimuth_deg,minimum_elevation_deg\n0,5\n360,5\n180,6\n", encoding="utf-8"
    )
    # 0 and 360 normalise to the same node -> rejected.
    with pytest.raises(InputError):
        load_horizon(path)


def test_azimuth_wrap_is_not_a_backwards_jump():
    times = np.array([0.0, 1.0, 2.0])
    azimuth = np.array([358.0, 359.5, 0.5])
    unwrapped = unwrapped_azimuth(times, azimuth)
    assert unwrapped[1] > unwrapped[0]
    assert unwrapped[2] > unwrapped[1]
    assert unwrapped[2] == pytest.approx(360.5)


def test_node_crossing_detects_wrap_crossing():
    times = np.array([0.0, 10.0])
    azimuth = np.array([359.0, 1.0])
    unwrapped = unwrapped_azimuth(times, azimuth)
    crossings = find_node_crossings(times, unwrapped, [0.0])
    assert len(crossings) == 1
    assert crossings[0] == pytest.approx(5.0)


# -------------------------------------------------------------------- geometry


def test_station_geometry_units_and_orthonormality():
    site, east, north, up, basis = station_geometry(35.0, 105.0, 300.0)
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-15)
    assert np.allclose(np.cross(east, north), up, atol=1e-15)
    # A 300 m ellipsoidal height must move the site outward by 0.3 km relative
    # to the h=0 surface point.
    site0, *_ = station_geometry(35.0, 105.0, 0.0)
    assert np.linalg.norm(site - site0) == pytest.approx(0.3, abs=1e-9)
    # Height must be treated as ellipsoidal (metres), not altitude in km.
    site_km, *_ = station_geometry(35.0, 105.0, 300_000.0)
    assert np.linalg.norm(site_km - site0) == pytest.approx(300.0, abs=1e-9)


def test_enu_axis_checks(inputs):
    _, station, _, _ = inputs
    checks = enu_axis_checks(station)
    assert checks["orthonormality_max_abs_error"] < 1e-12
    assert checks["east_cross_north_minus_up_norm"] < 1e-12
    assert np.linalg.norm(checks["ellipsoid_normal_minus_up"]) < 1e-12
    assert checks["up_displacement_enu"][:2] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert checks["up_displacement_enu"][2] == pytest.approx(1000.0, abs=1e-9)
    assert checks["east_displacement_enu"] == pytest.approx([1000.0, 0.0, 0.0], abs=1e-9)
    assert checks["north_displacement_enu"] == pytest.approx([0.0, 1000.0, 0.0], abs=1e-9)


def test_azimuth_convention_cardinal_vectors(inputs):
    """Construct a point due north/east of the site and check azimuth."""
    _, station, _, _ = inputs
    site = station.site_ecef_km

    def azimuth_of(point_ecef, seconds=0.0):
        # Project the ECEF offset into ENU directly; bypass SGP4 by checking
        # the same formulas the kernel uses.
        enu = station.enu_basis @ (point_ecef - site)
        return math.degrees(math.atan2(enu[0], enu[1])) % 360.0

    assert azimuth_of(site + 100.0 * station.north) == pytest.approx(0.0, abs=1e-9)
    assert azimuth_of(site + 100.0 * station.east) == pytest.approx(90.0, abs=1e-9)
    assert azimuth_of(site - 100.0 * station.north) == pytest.approx(180.0, abs=1e-9)
    assert azimuth_of(site - 100.0 * station.east) == pytest.approx(270.0, abs=1e-9)


def test_gmst_matches_readme_at_j2000(inputs):
    # At J2000, GMST is 280.46061837 deg.
    from passes.timeutils import gmst_degrees

    assert gmst_degrees(2451545.0, 0.0) == pytest.approx(280.46061837, abs=1e-9)


def test_scalar_and_vector_models_agree(inputs):
    tle, station, mask, plan = inputs
    ephem = _ephemeris(inputs)
    reference = IndependentModel(tle, station, mask, plan.start_jd, plan.start)
    for seconds in (0.0, 1234.5, 61999.1177, 86399.9):
        vector = ephem.sample(seconds)
        scalar = reference.evaluate(seconds)
        assert np.allclose(vector.teme_km, scalar.teme_km, atol=1e-9)
        assert np.allclose(vector.ecef_km, scalar.ecef_km, atol=1e-9)
        assert np.allclose(vector.enu_km, scalar.enu_km, atol=1e-8)
        assert vector.elevation_deg == pytest.approx(scalar.elevation_deg, abs=1e-9)
        assert vector.threshold_deg == pytest.approx(scalar.threshold_deg, abs=1e-9)


# ------------------------------------------------------------ failure location


def test_propagation_failure_is_located(inputs, monkeypatch):
    tle, station, mask, plan = inputs
    ephem = _ephemeris(inputs)

    class FailingArray:
        def sgp4(self, jd, fr):
            jd = np.atleast_1d(np.asarray(jd, dtype=float))
            fr = np.atleast_1d(np.asarray(fr, dtype=float))
            errors = np.zeros_like(jd, dtype=np.int32)
            errors[:] = 6  # decayed / invalid
            positions = np.zeros((1, jd.size, 3))
            return errors[None, :], positions, np.zeros((1, jd.size, 3))

    ephem.array = FailingArray()
    with pytest.raises(PropagationError) as excinfo:
        ephem.evaluate([100.0, 200.0])
    failures = excinfo.value.failures
    assert failures
    assert failures[0]["seconds_from_start_s"] == pytest.approx(100.0)
    assert failures[0]["utc"].endswith("Z")
    assert failures[0]["sgp4_error_code"] == 6


def test_non_finite_position_is_located(inputs):
    tle, station, mask, plan = inputs
    ephem = _ephemeris(inputs)

    class NanArray:
        def sgp4(self, jd, fr):
            jd = np.atleast_1d(np.asarray(jd, dtype=float))
            positions = np.full((1, jd.size, 3), np.nan)
            return np.zeros((1, jd.size), dtype=np.int32), positions, positions

    ephem.array = NanArray()
    with pytest.raises(PropagationError) as excinfo:
        ephem.evaluate([300.0])
    assert "non-finite" in str(excinfo.value)
    assert excinfo.value.failures[0]["seconds_from_start_s"] == pytest.approx(300.0)


# --------------------------------------------------------------------- search


def test_coarse_one_minute_grid_would_miss_the_narrow_occlusion(inputs):
    """The 2.7 s occlusion must not survive on a 60 s sampling grid."""
    ephem = _ephemeris(inputs)
    # The true occlusion is ~2.7 s wide around t+62000 s.
    times = np.arange(61800.0, 62300.0 + 60.0, 60.0)
    assert ephem.evaluate(times).gap_deg.shape == times.shape
    # A one-minute grid happens to land in the visible parts on both sides of
    # the gap and sees no sign change in the gap itself.
    mid = np.arange(61990.0, 62010.0, 60.0)
    assert np.all(ephem.evaluate(mid).gap_deg >= 0.0)
    # The dense search does find the two roots.
    dense = ephem.evaluate(np.linspace(61990.0, 62010.0, 2001))
    visible = dense.gap_deg >= 0.0
    changes = np.count_nonzero(visible[:-1] != visible[1:])
    assert changes == 2


def test_solver_splits_pass_and_verifies(inputs):
    tle, station, mask, plan = inputs
    config = SolverConfig(
        search_step_s=1.0,
        verify_step_s=0.2,
        audit_step_s=0.05,
        node_probe_step_s=0.02,
        observation_step_s=10.0,
    )
    result = solve(tle, station, mask, plan, config)
    assert result.verification["ok"], json.dumps(result.verification, indent=2)[:2000]
    payload = result.payload
    assert payload["summary"]["split_pass_count"] == 1
    split_pass = [p for p in payload["passes"] if p["split_by_horizon"]]
    assert len(split_pass) == 1
    assert split_pass[0]["window_count"] == 2
    # The occlusion gap is a full mask occlusion, not a sub-horizon gap.
    gap = split_pass[0]["occlusion_gaps"][0]
    assert gap["still_above_base_horizon"] is True
    assert gap["complete_occlusion"] is True
    assert 1.0 < gap["gap_duration_s"] < 5.0
    # All windows have positive length.
    assert all(w["duration_s"] > 0.0 for w in payload["windows"])
    # Boundary reasons cover both orbital and mask transitions.
    details = {
        b["transition_diagnosis"]["detail"]
        for b in payload["boundaries"]
        if b["transition_diagnosis"]
    }
    assert "occluded_by_horizon_feature" in details
    assert "horizon_feature_cleared" in details
    assert "elevation_rise_above_horizon" in details
    assert "elevation_set_below_horizon" in details


def test_boundaries_are_within_plan_tolerance_and_independently_refined(inputs):
    tle, station, mask, plan = inputs
    config = SolverConfig(
        search_step_s=1.0,
        verify_step_s=0.2,
        audit_step_s=0.05,
        node_probe_step_s=0.02,
        observation_step_s=10.0,
    )
    result = solve(tle, station, mask, plan, config)
    assert all(b["within_plan_tolerance"] for b in result.payload["boundaries"])
    for boundary in result.payload["boundaries"]:
        scalar = boundary["independent_scalar_check"]
        assert abs(scalar["root_time_delta_s"]) <= plan.boundary_tolerance_s
        assert bool(scalar["scalar_visibility_flips_at_tolerance"]) is True
        assert bool(scalar["sides_agree"]) is True
    # Scalar and vector models agree tightly everywhere sampled.
    maxima = result.verification["scalar_vector_comparison"]["max_abs_diffs"]
    assert maxima["elevation_deg"] < 1e-8
    assert maxima["azimuth_deg"] < 1e-7


def test_independent_root_solver(inputs):
    tle, station, mask, plan = inputs
    reference = IndependentModel(tle, station, mask, plan.start_jd, plan.start)
    # First obstacle crossing: visible before, occluded after.
    root = independent_root(reference, 61998.0, 62000.0, xtol=1e-7)
    assert reference.evaluate(root).gap_deg == pytest.approx(0.0, abs=1e-7)
    assert reference.evaluate(root - 0.01).gap_deg > 0.0
    assert reference.evaluate(root + 0.01).gap_deg < 0.0
    # Second obstacle crossing: occluded before, visible after.
    root2 = independent_root(reference, 62001.0, 62003.0, xtol=1e-7)
    assert reference.evaluate(root2).gap_deg == pytest.approx(0.0, abs=1e-7)
    assert reference.evaluate(root2 - 0.01).gap_deg < 0.0
    assert reference.evaluate(root2 + 0.01).gap_deg > 0.0


def test_zero_length_tangency_is_discarded(inputs):
    tle, station, mask, plan = inputs
    # A synthetic mask that just touches the pass peak creates a tangency.
    ephem = _ephemeris(inputs)
    peak_s = 2062.2
    peak_el = ephem.sample(peak_s).elevation_deg
    # Build a mask at exactly the peak elevation around all azimuths: the
    # visible set collapses to isolated tangent instants.
    nodes = np.array([0.0, 90.0, 180.0, 270.0])
    values = np.full(4, peak_el)
    extended_az = np.concatenate([nodes, [360.0]])
    extended_el = np.concatenate([values, [values[0]]])
    tangent_mask = HorizonMask(nodes, values, extended_az, extended_el)
    tangent_ephem = Ephemeris(tle.satrec, station, tangent_mask, plan.start)
    tangent_ephem.bind_epoch(*plan.start_jd)
    # At an instant slightly before/after the peak the ratio is below the mask.
    assert tangent_ephem.sample(peak_s - 1.0).gap_deg < 0.0
    assert tangent_ephem.sample(peak_s + 1.0).gap_deg < 0.0
    # Compute with a short plan around the peak: expect no positive windows.
    from passes.io import Plan

    short = Plan(
        start_utc=plan.start_utc,
        end_utc=plan.end_utc,
        start=plan.start,
        end=plan.end,
        start_jd=plan.start_jd,
        end_jd=plan.end_jd,
        duration_s=plan.duration_s,
        boundary_tolerance_s=plan.boundary_tolerance_s,
        observation_step_s=10.0,
        azimuth_convention="",
        elevation_convention="",
        raw={},
    )
    result = solve(tle, station, tangent_mask, short, SolverConfig(search_step_s=1.0, verify_step_s=0.5, audit_step_s=0.1, node_probe_step_s=0.05))
    assert all(w["duration_s"] > 0.0 for w in result.payload["windows"])


def test_merge_roots_collapses_duplicates():
    from passes.search import Root

    roots = [
        Root(100.0, 1e-9, 99.0, 101.0, ("coarse",), 5, True),
        Root(100.0 + 1e-7, -1e-10, 99.0, 101.0, ("fine_verification",), 4, True),
        Root(200.0, 1e-9, 199.0, 201.0, ("coarse",), 5, True),
    ]
    merged = merge_roots(roots, tolerance=1e-5)
    assert len(merged) == 2
    assert merged[0].seconds == pytest.approx(100.0 + 1e-7)
    assert set(merged[0].sources) == {"coarse", "fine_verification"}


def test_plan_is_validated(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "start_utc": "2026-11-14T00:00:00Z",
                "end_utc": "2026-11-13T00:00:00Z",
                "earth_rotation": "GMST",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(InputError):
        load_plan(path)


def test_plan_endpoint_truncation(inputs):
    """A plan that starts/ends inside a pass emits truncation boundaries."""
    import dataclasses
    import datetime as dt

    tle, station, mask, plan = inputs
    start = dt.datetime(2026, 11, 14, 0, 32, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 11, 14, 15, 38, tzinfo=dt.timezone.utc)
    from passes.timeutils import jd_parts_from_datetime

    truncated = dataclasses.replace(
        plan,
        start_utc="2026-11-14T00:32:00Z",
        end_utc="2026-11-14T15:38:00Z",
        start=start,
        end=end,
        start_jd=jd_parts_from_datetime(start),
        end_jd=jd_parts_from_datetime(end),
        duration_s=(end - start).total_seconds(),
    )
    config = SolverConfig(
        search_step_s=1.0,
        verify_step_s=0.2,
        audit_step_s=0.05,
        node_probe_step_s=0.02,
        observation_step_s=20.0,
    )
    result = solve(tle, station, mask, truncated, config)
    assert result.verification["ok"]
    windows = result.payload["windows"]
    assert len(windows) == 2
    assert windows[0]["entry"]["reason"] == "start_of_plan_truncation"
    assert windows[0]["start_utc"] == "2026-11-14T00:32:00Z"
    assert windows[1]["exit"]["reason"] == "end_of_plan_truncation"
    assert windows[1]["end_utc"] == "2026-11-14T15:38:00Z"
    # Truncation boundaries are not refined sign changes.
    assert windows[0]["entry"]["refined_crossing"] is False
    assert windows[1]["exit"]["refined_crossing"] is False
