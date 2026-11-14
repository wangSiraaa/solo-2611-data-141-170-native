"""End-to-end solver: search -> refine -> windows -> evidence -> audit."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import (
    DEFAULT_BOUNDARY_TOLERANCE_S,
    DEFAULT_EXTREMUM_XTOL_S,
    DEFAULT_OBSERVATION_STEP_S,
    DEFAULT_ROOT_XTOL_S,
    DEFAULT_SEARCH_STEP_S,
    DEFAULT_VERIFY_STEP_S,
)
from .geometry import Ephemeris, Sample
from .io import HorizonMask, Plan, Station, TleRecord
from .search import (
    Bracket,
    Root,
    build_windows,
    extract_reasons,
    find_node_crossings,
    find_wall_clock_extremum,
    node_probe_brackets,
    refine_brackets,
    merge_roots,
    scan,
    unwrapped_azimuth,
)
from .timeutils import iso_from_seconds
from .verify import (
    IndependentModel,
    compare_scalar_to_vector,
    enu_axis_checks,
    independent_root,
)


@dataclass
class SolverConfig:
    search_step_s: float = DEFAULT_SEARCH_STEP_S
    verify_step_s: float = DEFAULT_VERIFY_STEP_S
    audit_step_s: float = 0.005
    node_probe_step_s: float = 0.0025
    root_xtol_s: float = DEFAULT_ROOT_XTOL_S
    boundary_tolerance_s: float = DEFAULT_BOUNDARY_TOLERANCE_S
    observation_step_s: float = DEFAULT_OBSERVATION_STEP_S
    extremum_xtol_s: float = DEFAULT_EXTREMUM_XTOL_S
    independent_offsets_s: tuple[float, ...] = (0.2, 0.05, 0.001)
    scalar_time_budget: int = 400

    def validate(self) -> None:
        if self.search_step_s <= 0 or self.verify_step_s <= 0 or self.audit_step_s <= 0:
            raise ValueError("search/verify/audit steps must be positive")
        if self.node_probe_step_s <= 0:
            raise ValueError("node probe step must be positive")
        if self.boundary_tolerance_s <= 0:
            raise ValueError("boundary tolerance must be positive")
        if self.search_step_s < self.verify_step_s:
            raise ValueError("coarse search step should be >= verification step")


@dataclass
class SolveOutput:
    payload: dict[str, Any] = field(default_factory=dict)
    boundary_evidence: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    observations: list[dict[str, Any]] = field(default_factory=list)


def _sample_record(sample: Sample, start) -> dict[str, Any]:
    return {
        "utc": iso_from_seconds(start, sample.seconds),
        "seconds_from_start_s": sample.seconds,
        "jd": sample.jd,
        "gmst_deg": sample.gmst_deg,
        "teme_km": [float(v) for v in sample.teme_km],
        "ecef_km": [float(v) for v in sample.ecef_km],
        "enu_km": [float(v) for v in sample.enu_km],
        "range_km": sample.range_km,
        "azimuth_deg": sample.azimuth_deg,
        "elevation_deg": sample.elevation_deg,
        "horizon_threshold_deg": sample.threshold_deg,
        "gap_deg": sample.gap_deg,
        "visible": sample.visible,
    }


def _observation_row(sample: Sample, start, window_id: str, pass_id: int, role: str) -> dict[str, Any]:
    return {
        "utc": iso_from_seconds(start, sample.seconds),
        "jd": sample.jd,
        "seconds_from_start_s": sample.seconds,
        "pass_id": pass_id,
        "window_id": window_id,
        "row_role": role,
        "azimuth_deg": sample.azimuth_deg,
        "elevation_deg": sample.elevation_deg,
        "horizon_threshold_deg": sample.threshold_deg,
        "gap_deg": sample.gap_deg,
        "visible": sample.visible,
        "gmst_deg": sample.gmst_deg,
        "range_km": sample.range_km,
        "teme_x_km": float(sample.teme_km[0]),
        "teme_y_km": float(sample.teme_km[1]),
        "teme_z_km": float(sample.teme_km[2]),
        "ecef_x_km": float(sample.ecef_km[0]),
        "ecef_y_km": float(sample.ecef_km[1]),
        "ecef_z_km": float(sample.ecef_km[2]),
        "enu_east_km": float(sample.enu_km[0]),
        "enu_north_km": float(sample.enu_km[1]),
        "enu_up_km": float(sample.enu_km[2]),
    }


def _direction_label(entry: bool, cause: str) -> tuple[str, str]:
    if entry:
        if cause == "horizon_mask_feature":
            return "enter_visibility", "horizon_feature_cleared"
        return "enter_visibility", "elevation_rise_above_horizon"
    if cause == "horizon_mask_feature":
        return "exit_visibility", "occluded_by_horizon_feature"
    return "exit_visibility", "elevation_set_below_horizon"


def _locate_reason(ephem: Ephemeris, root: Root, start) -> dict[str, Any]:
    detail = extract_reasons(ephem, root)
    entry = bool(detail["entry"])
    label, human = _direction_label(entry, detail["cause"])
    detail["reason"] = label
    detail["detail"] = human
    return detail


def _gap_summary(ephem: Ephemeris, start_s: float, end_s: float, base_threshold_deg: float) -> dict[str, Any]:
    span = end_s - start_s
    if span <= 0.0:
        return {}
    # Refined roots carry a ~xtol residual, so trim a thin guard at each end.
    guard = min(1.0e-3, 0.01 * span)
    inner_start, inner_end = start_s + guard, end_s - guard
    if inner_end <= inner_start:
        inner_start, inner_end = start_s, end_s
    inner_span = inner_end - inner_start
    step = min(0.25, max(0.002, inner_span / 2000.0))
    n = max(2, int(math.ceil(inner_span / step)) + 1)
    times = np.linspace(inner_start, inner_end, n)
    geometry = ephem.evaluate(times)
    i_min = int(np.argmin(geometry.gap_deg))
    i_max = int(np.argmax(geometry.gap_deg))
    return {
        "gap_duration_s": span,
        "audited_interval_s": [inner_start, inner_end],
        "endpoint_guard_s": guard,
        "min_elevation_deg": float(geometry.elevation_deg[i_min]),
        "max_elevation_deg": float(geometry.elevation_deg[i_max]),
        "min_gap_deg": float(geometry.gap_deg[i_min]),
        "max_gap_deg": float(geometry.gap_deg[i_max]),
        "min_clearance_to_base_horizon_deg": float(
            np.min(geometry.elevation_deg - base_threshold_deg)
        ),
        "samples": int(n),
        "sample_step_s": float(step),
    }


def solve(
    tle: TleRecord,
    station: Station,
    mask: HorizonMask,
    plan: Plan,
    config: SolverConfig,
) -> SolveOutput:
    config.validate()
    base_threshold = float(np.min(mask.minimum_elevation_deg))
    start = plan.start
    t0, t1 = 0.0, plan.duration_s

    ephem = Ephemeris(tle.satrec, station, mask, start)
    ephem.bind_epoch(*plan.start_jd)

    # ----------------------------------------------------------- search phase
    coarse = scan(ephem, t0, t1, config.search_step_s, source="coarse")
    verify = scan(
        ephem, t0, t1, config.verify_step_s, source="fine_verification", collect_azimuth=True
    )

    azimuth_unwrapped = unwrapped_azimuth(verify.times, verify.azimuth_deg)
    crossings = find_node_crossings(
        verify.times, azimuth_unwrapped, mask.azimuth_deg
    )
    node_brackets, node_stats = node_probe_brackets(
        ephem,
        t0,
        t1,
        crossings,
        base_threshold_deg=base_threshold,
        probe_step=config.node_probe_step_s,
    )

    all_brackets: list[Bracket] = list(coarse.brackets) + list(verify.brackets) + node_brackets
    roots_raw, rejected = refine_brackets(ephem, all_brackets, xtol=config.root_xtol_s)
    # Coarse, verification and node-probe scans bracket the same transition from
    # different grid phases; their roots differ by ~xtol.  Merge within a few
    # xtol so that duplicates cannot create zero-length windows, while a real
    # short occlusion (seconds to milliseconds) stays distinct.
    roots = merge_roots(roots_raw, tolerance=max(4.0 * config.root_xtol_s, 1e-6))
    windows, kept_roots, discarded = build_windows(ephem, roots, t0, t1)

    # Free the dense azimuth series now that crossings are known.
    verify.times = None
    verify.azimuth_deg = None

    # ----------------------------------------------------------- pass grouping
    window_pass_ids: list[int] = []
    pass_count = 0
    previous_window = None
    for window in windows:
        if previous_window is None:
            pass_count += 1
        else:
            same_pass = _same_pass(ephem, previous_window, window, base_threshold)
            if not same_pass:
                pass_count += 1
        window_pass_ids.append(pass_count)
        previous_window = window

    # ----------------------------------------------------------- evidence
    boundaries: list[dict[str, Any]] = []
    boundary_summaries: dict[str, dict[str, Any]] = {}
    window_records: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    boundary_counter = 0

    for index, (window, pass_id) in enumerate(zip(windows, window_pass_ids), start=1):
        window_id = f"pass_{pass_id:02d}_window_{index:02d}"
        start_sample = ephem.sample(window.start_s)
        end_sample = ephem.sample(window.end_s)
        peak_s, peak_el = find_wall_clock_extremum(ephem, window.start_s, window.end_s, config.extremum_xtol_s)
        peak_sample = ephem.sample(peak_s)

        start_boundary = _window_boundary(
            ephem,
            start,
            window,
            role="entry",
            sample=start_sample,
            root=window.entry_root,
            reason=window.entry_reason,
            boundary_counter=boundary_counter,
            plan=plan,
        )
        boundary_counter += 1
        end_boundary = _window_boundary(
            ephem,
            start,
            window,
            role="exit",
            sample=end_sample,
            root=window.exit_root,
            reason=window.exit_reason,
            boundary_counter=boundary_counter,
            plan=plan,
        )
        boundary_counter += 1
        boundaries.extend([start_boundary["record"], end_boundary["record"]])
        boundary_summaries[start_boundary["record"]["boundary_id"]] = start_boundary["summary"]
        boundary_summaries[end_boundary["record"]["boundary_id"]] = end_boundary["summary"]

        window_record = {
            "window_id": window_id,
            "pass_id": pass_id,
            "start_utc": iso_from_seconds(start, window.start_s),
            "end_utc": iso_from_seconds(start, window.end_s),
            "start_seconds_from_start_s": window.start_s,
            "end_seconds_from_start_s": window.end_s,
            "duration_s": window.duration_s,
            "start_jd": start_sample.jd,
            "end_jd": end_sample.jd,
            "entry": start_boundary["summary"],
            "exit": end_boundary["summary"],
            "peak": {
                "utc": iso_from_seconds(start, peak_s),
                "seconds_from_start_s": peak_s,
                "elevation_deg": peak_sample.elevation_deg,
                "azimuth_deg": peak_sample.azimuth_deg,
                "horizon_threshold_deg": peak_sample.threshold_deg,
            },
        }
        window_records.append(window_record)

        # Observations: boundary rows, regular interior grid, and the peak.
        times = [window.start_s]
        count = int(math.floor(window.duration_s / config.observation_step_s))
        for k in range(1, count + 1):
            value = window.start_s + k * config.observation_step_s
            if value < window.end_s - 1e-9:
                times.append(value)
        times.extend([peak_s, window.end_s])
        times = sorted(set(round(v, 9) for v in times))
        for value in times:
            if value == round(window.start_s, 9):
                role = "window_start"
            elif value == round(window.end_s, 9):
                role = "window_end"
            elif value == round(peak_s, 9):
                role = "peak_elevation"
            else:
                role = "interior"
            sample = ephem.sample(value)
            observations.append(_observation_row(sample, start, window_id, pass_id, role))

        start_boundary["window_id"] = window_id
        end_boundary["window_id"] = window_id
        start_boundary["record"]["window_id"] = window_id
        start_boundary["record"]["pass_id"] = pass_id
        end_boundary["record"]["window_id"] = window_id
        end_boundary["record"]["pass_id"] = pass_id

    # ----------------------------------------------------------- passes tree
    passes_payload: list[dict[str, Any]] = []
    for pass_id in sorted(set(window_pass_ids)):
        member_indices = [i for i, pid in enumerate(window_pass_ids) if pid == pass_id]
        member_windows = [window_records[i] for i in member_indices]
        gaps = []
        for left, right in zip(member_windows[:-1], member_windows[1:]):
            gap = _gap_summary(
                ephem,
                left["end_seconds_from_start_s"],
                right["start_seconds_from_start_s"],
                base_threshold,
            )
            gap.update(
                {
                    "before_window_id": left["window_id"],
                    "after_window_id": right["window_id"],
                    "complete_occlusion": bool(gap.get("max_gap_deg", 0.0) < 0.0),
                    "still_above_base_horizon": bool(
                        gap.get("min_clearance_to_base_horizon_deg", -1.0) >= 0.0
                    ),
                }
            )
            gaps.append(gap)
        passes_payload.append(
            {
                "pass_id": pass_id,
                "window_ids": [w["window_id"] for w in member_windows],
                "window_count": len(member_windows),
                "split_by_horizon": len(member_windows) > 1,
                "windows": member_windows,
                "occlusion_gaps": gaps,
            }
        )

    # ----------------------------------------------------------- audit
    audit_step = config.audit_step_s
    audit = scan(ephem, t0, t1, audit_step, source="independent_fine_audit")
    audit_roots_raw, audit_rejected = refine_brackets(ephem, audit.brackets, xtol=config.root_xtol_s)
    audit_roots = merge_roots(
        audit_roots_raw, tolerance=max(4.0 * config.root_xtol_s, 1e-6)
    )

    independent = IndependentModel(tle, station, mask, plan.start_jd, start)
    verification = _build_verification(
        ephem=ephem,
        independent=independent,
        station=station,
        mask=mask,
        start=start,
        t0=t0,
        t1=t1,
        roots=kept_roots,
        windows=windows,
        window_pass_ids=window_pass_ids,
        audit=audit,
        audit_roots=audit_roots,
        audit_rejected=audit_rejected,
        config=config,
        base_threshold=base_threshold,
    )

    # Attach independent scalar comparisons to each boundary record.
    for record in boundaries:
        comparison = verification["boundary_checks"].get(record.get("root_key"))
        if comparison is not None:
            record["independent_scalar_check"] = comparison

    # Recompute window records with the extra boundary evidence.
    for record in window_records:
        record["entry"]["boundary_id"] = next(
            b["boundary_id"] for b in boundaries if b["window_id"] == record["window_id"] and b["role"] == "entry"
        )
        record["exit"]["boundary_id"] = next(
            b["boundary_id"] for b in boundaries if b["window_id"] == record["window_id"] and b["role"] == "exit"
        )

    for boundary in boundaries:
        boundary["tolerance_s"] = plan.boundary_tolerance_s
        tolerance_ok = True
        scalar = boundary.get("independent_scalar_check", {})
        if scalar.get("independent_root_s") is not None:
            tolerance_ok = (
                abs(scalar["independent_root_s"] - boundary["seconds_from_start_s"])
                <= plan.boundary_tolerance_s
            )
        boundary["within_plan_tolerance"] = bool(tolerance_ok)
        summary = boundary_summaries.get(boundary["boundary_id"])
        if summary is not None:
            summary["within_plan_tolerance"] = bool(tolerance_ok)
    for record in window_records:
        for side in ("entry", "exit"):
            record[side]["within_plan_tolerance"] = boundary_summaries[record[side]["boundary_id"]][
                "within_plan_tolerance"
            ]

    verification["all_boundaries_within_tolerance"] = all(
        b["within_plan_tolerance"] for b in boundaries
    )
    verification["observation_rows"] = len(observations)

    # ----------------------------------------------------------- assembly
    payload = {
        "command": "passes",
        "version": "1.0.0",
        "status": "ok",
        "model": {
            "propagator": "SGP4 (sgp4==2.23), TEME position in km",
            "earth_rotation": "README GMST polynomial; UT1=UTC; polar motion = 0",
            "gmst_formula_deg": "((280.46061837 + 360.98564736629*D + 0.000387933*T^2 - T^3/38710000) mod 360), D=JD-2451545.0, T=D/36525",
            "frames": "TEME -> ECEF (GMST z-rotation) -> station ENU (WGS84 local basis)",
            "station": {
                "station_id": station.station_id,
                "latitude_deg": station.latitude_deg,
                "longitude_deg_east": station.longitude_deg_east,
                "ellipsoid_height_m": station.ellipsoid_height_m,
                "datum": station.datum,
                "site_ecef_km": [float(v) for v in station.site_ecef_km],
                "east_ecef": [float(v) for v in station.east],
                "north_ecef": [float(v) for v in station.north],
                "up_ecef": [float(v) for v in station.up],
            },
            "wgs84": {
                "a_km": 6378.137,
                "f": 1.0 / 298.257223563,
                "ellipsoid_height_used_as_geodetic_height": True,
                "height_unit_conversion": "metres -> kilometres performed exactly once at input",
            },
            "horizon": {
                "azimuth_convention": "degrees clockwise from true north, [0,360)",
                "interpolation": "periodic piecewise linear; last node joins first through 360 deg",
                "nodes": [
                    {"azimuth_deg": float(a), "minimum_elevation_deg": float(e)}
                    for a, e in zip(mask.azimuth_deg, mask.minimum_elevation_deg)
                ],
                "base_threshold_deg": base_threshold,
            },
            "refraction": "none",
            "visibility_rule": "geometric elevation >= horizon(azimuth); zero-duration tangencies are not windows",
            "time": "UTC continuous over the plan; half-open [start,end); plans do not wrap to local day",
        },
        "tle": {
            "name": tle.name,
            "line1": tle.line1,
            "line2": tle.line2,
            "epoch_utc": tle.epoch_utc,
        },
        "plan": {
            "start_utc": plan.start_utc,
            "end_utc": plan.end_utc,
            "start_jd": list(plan.start_jd),
            "duration_s": plan.duration_s,
            "boundary_tolerance_seconds": plan.boundary_tolerance_s,
            "half_open": True,
        },
        "search": {
            "coarse_step_s": config.search_step_s,
            "verification_step_s": config.verify_step_s,
            "audit_step_s": config.audit_step_s,
            "node_probe_step_s": config.node_probe_step_s,
            "root_xtol_s": config.root_xtol_s,
            "coarse_samples": coarse.sample_count,
            "verification_samples": verify.sample_count,
            "audit_samples": audit.sample_count,
            "coarse_visible_samples": coarse.visible_count,
            "verification_visible_samples": verify.visible_count,
            "coarse_min_gap_deg": coarse.min_gap_deg,
            "coarse_max_gap_deg": coarse.max_gap_deg,
            "verification_min_gap_deg": verify.min_gap_deg,
            "verification_max_gap_deg": verify.max_gap_deg,
            "audit_min_gap_deg": audit.min_gap_deg,
            "audit_max_gap_deg": audit.max_gap_deg,
            "coarse_brackets": len(coarse.brackets),
            "verification_brackets": len(verify.brackets),
            "node_probe": node_stats,
            "node_probe_brackets": len(node_brackets),
            "node_crossing_times_utc": [iso_from_seconds(start, c) for c in crossings],
            "raw_roots": len(roots_raw),
            "merged_roots": len(roots),
            "rejected_brackets": rejected,
            "discarded_tangencies": discarded,
        },
        "summary": {
            "pass_count": len(passes_payload),
            "window_count": len(windows),
            "visible_seconds_total": float(sum(w.duration_s for w in windows)),
            "split_pass_count": sum(1 for p in passes_payload if p["split_by_horizon"]),
            "unmatched_audit_roots": verification["audit"]["unmatched_audit_roots"],
            "unmatched_solver_roots": verification["audit"]["unmatched_solver_roots"],
            "independent_checks_ok": verification["ok"],
        },
        "passes": passes_payload,
        "windows": window_records,
        "boundaries": boundaries,
        "verification": verification,
        "precision_limits": _precision_limits(config, plan),
    }

    evidence = {
        "command": "passes",
        "description": (
            "Boundary evidence: refined crossing, bracket, propagation coordinates at and "
            "around each boundary, transition diagnosis, and independent scalar re-evaluation."
        ),
        "plan_tolerance_s": plan.boundary_tolerance_s,
        "boundaries": boundaries,
        "rejected_brackets": rejected,
        "discarded_tangencies": discarded,
    }

    return SolveOutput(
        payload=payload,
        boundary_evidence=evidence,
        verification=verification,
        observations=observations,
    )


def _same_pass(ephem: Ephemeris, left, right, base_threshold_deg: float) -> bool:
    span = right.start_s - left.end_s
    if span <= 0.0:
        return True
    step = min(0.25, max(0.002, span / 2000.0))
    n = max(2, int(math.ceil(span / step)) + 1)
    times = np.linspace(left.end_s, right.start_s, n)
    elevation = ephem.evaluate(times).elevation_deg
    return bool(np.min(elevation - base_threshold_deg) >= -1e-9)


def _window_boundary(
    ephem: Ephemeris,
    start,
    window,
    role: str,
    sample: Sample,
    root,
    reason: str,
    boundary_counter: int,
    plan: Plan,
) -> dict[str, Any]:
    boundary_id = f"boundary_{boundary_counter:03d}"
    seconds = window.start_s if role == "entry" else window.end_s
    before = ephem.sample(max(0.0, seconds - 0.2))
    after = ephem.sample(min(plan.duration_s, seconds + 0.2))
    record: dict[str, Any] = {
        "boundary_id": boundary_id,
        "window_id": None,
        "pass_id": None,
        "root_key": None if root is None else f"{root.seconds:.9f}",
        "role": role,
        "reason": reason,
        "utc": iso_from_seconds(start, seconds),
        "seconds_from_start_s": seconds,
        "jd": sample.jd,
        "gmst_deg": sample.gmst_deg,
        "azimuth_deg": sample.azimuth_deg,
        "elevation_deg": sample.elevation_deg,
        "horizon_threshold_deg": sample.threshold_deg,
        "gap_deg": sample.gap_deg,
        "range_km": sample.range_km,
        "teme_km": [float(v) for v in sample.teme_km],
        "ecef_km": [float(v) for v in sample.ecef_km],
        "enu_km": [float(v) for v in sample.enu_km],
        "sample_before_0_2s": _sample_record(before, start),
        "sample_after_0_2s": _sample_record(after, start),
    }
    if root is not None:
        record.update(
            {
                "refined_crossing": True,
                "bracket_s": [root.bracket_lo, root.bracket_hi],
                "bracket_width_s": root.bracket_hi - root.bracket_lo,
                "root_residual_deg": root.f_value,
                "root_iterations": root.iterations,
                "root_converged": root.converged,
                "root_sources": list(root.sources),
            }
        )
        diagnosis = _locate_reason(ephem, root, start)
        record["transition_diagnosis"] = {
            "cause": diagnosis["cause"],
            "detail": diagnosis["detail"],
            "entry": diagnosis["entry"],
            "gap_rate_deg_per_s": diagnosis["gap_rate_deg_per_s"],
            "elevation_rate_deg_per_s": diagnosis["elevation_rate_deg_per_s"],
            "threshold_rate_deg_per_s": diagnosis["threshold_rate_deg_per_s"],
            "azimuth_rate_deg_per_s": diagnosis["azimuth_rate_deg_per_s"],
            "sample_before": _sample_record(diagnosis["sample_before"], start),
            "sample_at": _sample_record(diagnosis["sample_at"], start),
            "sample_after": _sample_record(diagnosis["sample_after"], start),
        }
    else:
        record.update(
            {
                "refined_crossing": False,
                "bracket_s": None,
                "bracket_width_s": None,
                "root_residual_deg": None,
                "root_iterations": None,
                "root_converged": None,
                "root_sources": [],
                "transition_diagnosis": None,
            }
        )
    summary = {
        "boundary_id": boundary_id,
        "utc": record["utc"],
        "seconds_from_start_s": seconds,
        "role": role,
        "reason": reason,
        "azimuth_deg": record["azimuth_deg"],
        "elevation_deg": record["elevation_deg"],
        "horizon_threshold_deg": record["horizon_threshold_deg"],
        "gap_deg": record["gap_deg"],
        "refined_crossing": record["refined_crossing"],
    }
    return {"record": record, "summary": summary}


def _build_verification(
    ephem: Ephemeris,
    independent: IndependentModel,
    station: Station,
    mask: HorizonMask,
    start,
    t0: float,
    t1: float,
    roots: list[Root],
    windows,
    window_pass_ids: list[int],
    audit,
    audit_roots: list[Root],
    audit_rejected: list[dict[str, Any]],
    config: SolverConfig,
    base_threshold: float,
) -> dict[str, Any]:
    tolerance = config.boundary_tolerance_s

    # --- ENU axial checks -------------------------------------------------
    enu_checks = enu_axis_checks(station)
    enu_ok = (
        enu_checks["orthonormality_max_abs_error"] < 1e-12
        and enu_checks["east_cross_north_minus_up_norm"] < 1e-12
        and np.linalg.norm(enu_checks["ellipsoid_normal_minus_up"]) < 1e-12
        and abs(enu_checks["up_displacement_enu"][0]) < 1e-9
        and abs(enu_checks["up_displacement_enu"][1]) < 1e-9
        and abs(enu_checks["up_displacement_enu"][2] - 1000.0) < 1e-9
        and abs(enu_checks["east_displacement_enu"][1]) < 1e-9
        and abs(enu_checks["east_displacement_enu"][2]) < 1e-9
        and abs(enu_checks["north_displacement_enu"][0]) < 1e-9
        and abs(enu_checks["north_displacement_enu"][2]) < 1e-9
    )

    # --- boundary independent scalar checks ------------------------------
    boundary_checks: dict[str, Any] = {}
    scalar_times: list[float] = []
    boundary_ok = True
    for boundary_index, root in enumerate(sorted(roots, key=lambda r: r.seconds)):
        root_key = f"{root.seconds:.9f}"
        bracket_lo = root.bracket_lo
        bracket_hi = root.bracket_hi
        try:
            independent_s = independent_root(
                independent, bracket_lo, bracket_hi, xtol=1e-6
            )
        except Exception as exc:  # pragma: no cover - defensive
            independent_s = None
            independent_error = str(exc)
        else:
            independent_error = None
        offsets = list(config.independent_offsets_s)
        sides = []
        for offset in offsets:
            for sign in (-1.0, 1.0):
                value = root.seconds + sign * offset
                if t0 <= value <= t1:
                    scalar = independent.evaluate(value)
                    vector = ephem.sample(value)
                    sides.append(
                        {
                            "offset_s": sign * offset,
                            "utc": iso_from_seconds(start, value),
                            "scalar": scalar.as_dict(),
                            "vector": _sample_record(vector, start),
                            "scalar_visible": scalar.visible,
                            "vector_visible": vector.visible,
                            "visibility_agree": scalar.visible == vector.visible,
                        }
                    )
                    scalar_times.append(value)
        scalar_at = independent.evaluate(root.seconds)
        vector_at = ephem.sample(root.seconds)
        if independent_s is not None:
            delta_s = independent_s - root.seconds
            within = abs(delta_s) <= tolerance
        else:
            delta_s = None
            within = False
        # Visibility must flip across the boundary (checked at plan tolerance).
        left = independent.evaluate(max(t0, root.seconds - tolerance))
        right = independent.evaluate(min(t1, root.seconds + tolerance))
        flips = left.visible != right.visible
        agrees = (
            scalar_at.visible == vector_at.visible
            and abs(scalar_at.azimuth_deg - vector_at.azimuth_deg) < 1e-9
            and abs(scalar_at.elevation_deg - vector_at.elevation_deg) < 1e-9
        )
        all_sides_agree = all(side["visibility_agree"] for side in sides)
        ok = bool(within and flips and agrees and all_sides_agree)
        boundary_ok = boundary_ok and ok
        boundary_checks[str(root_key)] = {
            "root_seconds_from_start_s": root.seconds,
            "independent_root_s": independent_s,
            "root_time_delta_s": delta_s,
            "within_plan_tolerance": within,
            "scalar_at_boundary": scalar_at.as_dict(),
            "vector_at_boundary": _sample_record(vector_at, start),
            "scalar_visibility_flips_at_tolerance": flips,
            "scalar_visibility_at_minus_tolerance": left.visible,
            "scalar_visibility_at_plus_tolerance": right.visible,
            "scalar_and_vector_agree_at_boundary": agrees,
            "sides_agree": all_sides_agree,
            "sides": sides,
            "independent_error": independent_error,
            "ok": ok,
        }
        scalar_times.append(root.seconds)
        scalar_times.append(independent_s if independent_s is not None else root.seconds)

    # --- global scalar/vector comparison ---------------------------------
    uniform_times = np.linspace(t0, t1, config.scalar_time_budget)
    comparison_times = sorted(
        set(
            [round(float(v), 9) for v in uniform_times]
            + [round(float(v), 9) for v in scalar_times if t0 <= v <= t1]
        )
    )
    thresholds = {
        "teme_km": 1e-9,
        "ecef_km": 1e-9,
        "enu_km": 1e-9,
        "range_km": 1e-9,
        "azimuth_deg": 1e-7,
        "elevation_deg": 1e-9,
        "threshold_deg": 1e-9,
    }
    scalar_comparison = compare_scalar_to_vector(
        independent, ephem, comparison_times, thresholds
    )

    # --- audit root matching ---------------------------------------------
    unmatched_audit = 0
    unmatched_solver = 0
    matches = []
    for audit_root in audit_roots:
        nearest = min((abs(audit_root.seconds - r.seconds), r) for r in roots) if roots else (math.inf, None)
        matched = nearest[0] <= tolerance
        if not matched:
            unmatched_audit += 1
        matches.append(
            {
                "audit_seconds": audit_root.seconds,
                "solver_seconds": None if nearest[1] is None else nearest[1].seconds,
                "delta_s": None if nearest[1] is None else nearest[1].seconds - audit_root.seconds,
                "matched_within_tolerance": bool(matched),
            }
        )
    for solver_root in roots:
        nearest = min((abs(audit_root.seconds - solver_root.seconds) for audit_root in audit_roots), default=math.inf)
        if nearest > tolerance:
            unmatched_solver += 1

    # --- occlusion gap fine audit ----------------------------------------
    gap_audits = []
    fine_gap_ok = True
    for pair_index in range(len(windows) - 1):
        left, right = windows[pair_index], windows[pair_index + 1]
        if window_pass_ids[pair_index] != window_pass_ids[pair_index + 1]:
            # Different passes: the long gap is not at risk of hiding a sub-window.
            continue
        gap_start, gap_end = left.end_s, right.start_s
        if gap_end <= gap_start:
            continue
        span = gap_end - gap_start
        n = max(2, int(math.ceil(span / config.audit_step_s)) + 1)
        if n > 200_000:
            step = span / 199_999.0
            n = 200_000
        else:
            step = config.audit_step_s
        # Refined roots carry a ~xtol time residual, so exclude a thin guard at
        # both ends; the question is whether the *interior* of the gap is fully
        # occluded, not whether the endpoint equals zero to the last bit.
        guard = max(1e-3, 20.0 * config.root_xtol_s)
        inner_start, inner_end = gap_start + guard, gap_end - guard
        if inner_end <= inner_start:
            inner_start, inner_end = gap_start, gap_end
        times = np.linspace(inner_start, inner_end, n)
        geometry = ephem.evaluate(times)
        visible = geometry.gap_deg >= 0.0
        sign_changes = int(np.count_nonzero(visible[:-1] != visible[1:]))
        min_gap = float(np.min(geometry.gap_deg))
        max_gap = float(np.max(geometry.gap_deg))
        # A confirmation is only meaningful when at least two interior samples
        # fall strictly inside the gap; otherwise report it as unresolved.
        sufficiently_sampled = int(n) >= 3 and step < span / 2.0
        confirmed = bool(sufficiently_sampled and sign_changes == 0 and max_gap < 0.0)
        fine_gap_ok = fine_gap_ok and confirmed
        gap_audits.append(
            {
                "before_window_id": f"pass_{window_pass_ids[pair_index]:02d}",
                "before_end_utc": iso_from_seconds(start, gap_start),
                "after_start_utc": iso_from_seconds(start, gap_end),
                "audited_interval_utc": [
                    iso_from_seconds(start, inner_start),
                    iso_from_seconds(start, inner_end),
                ],
                "endpoint_guard_s": guard,
                "duration_s": span,
                "sample_step_s": float(step),
                "samples": int(n),
                "sufficient_resolution": sufficiently_sampled,
                "sign_changes": sign_changes,
                "min_gap_deg": min_gap,
                "max_gap_deg": max_gap,
                "confirmed_full_occlusion": confirmed,
            }
        )

    # --- boundary-neighbourhood fine checks ------------------------------
    # Confirm each reported crossing is the only sign change in a +/- window.
    neighbourhood_ok = True
    neighbourhoods = []
    for root in roots:
        lo = max(t0, root.seconds - max(1.0, 10 * config.audit_step_s))
        hi = min(t1, root.seconds + max(1.0, 10 * config.audit_step_s))
        n = max(3, int(math.ceil((hi - lo) / config.audit_step_s)) + 1)
        times = np.linspace(lo, hi, n)
        geometry = ephem.evaluate(times)
        visible = geometry.gap_deg >= 0.0
        changes = np.flatnonzero(visible[:-1] != visible[1:])
        count = int(changes.size)
        ok = count == 1
        neighbourhood_ok = neighbourhood_ok and ok
        neighbourhoods.append(
            {
                "root_seconds_from_start_s": root.seconds,
                "window_s": [lo, hi],
                "sample_step_s": config.audit_step_s,
                "sign_changes": count,
                "ok": ok,
            }
        )

    ok = bool(
        enu_ok
        and boundary_ok
        and scalar_comparison["ok"]
        and unmatched_audit == 0
        and unmatched_solver == 0
        and fine_gap_ok
        and neighbourhood_ok
    )
    return {
        "ok": ok,
        "enu_axis_checks": enu_checks,
        "enu_axis_checks_ok": bool(enu_ok),
        "boundary_checks": boundary_checks,
        "boundary_checks_ok": bool(boundary_ok),
        "scalar_vector_comparison": scalar_comparison,
        "audit": {
            "step_s": config.audit_step_s,
            "samples": audit.sample_count,
            "brackets": len(audit.brackets),
            "roots": len(audit_roots),
            "rejected_brackets": audit_rejected,
            "unmatched_audit_roots": unmatched_audit,
            "unmatched_solver_roots": unmatched_solver,
            "matches": matches,
        },
        "gap_fine_audit": gap_audits,
        "gap_fine_audit_ok": bool(fine_gap_ok),
        "boundary_neighbourhood_audit": neighbourhoods,
        "boundary_neighbourhood_audit_ok": bool(neighbourhood_ok),
    }


def _precision_limits(config: SolverConfig, plan: Plan) -> list[str]:
    return [
        (
            "Orbital propagation: SGP4 on the supplied synthetic TLE. The TLE itself is "
            "fictional and no external ephemeris, EOP or leap-second table is used."
        ),
        (
            "Earth orientation: UT1 is set equal to UTC and polar motion is zero, per the "
            "task model. Real UT1-UTC can reach ~0.9 s (about 0.004 deg of Earth rotation), "
            "which would shift azimuth/elevation at the sub-0.01 deg level and boundary "
            "times by much less than one second; this is outside the task model."
        ),
        (
            "No atmospheric refraction is applied; reported elevations are geometric. "
            "Near the horizon real refraction would raise apparent elevation by ~0.5 deg."
        ),
        (
            "The horizon mask is a periodic piecewise-linear interpolation of the supplied "
            "nodes. It is treated as exact; any true obstacle shape between nodes is not "
            "modelled."
        ),
        (
            f"Search: coarse {config.search_step_s:g} s is only used to propose brackets. "
            f"Transitions are confirmed by a {config.verify_step_s:g} s rescan, dense "
            f"{config.node_probe_step_s:g} s probes on mask-node segments, and an "
            f"independent {config.audit_step_s:g} s full-day audit."
        ),
        (
            f"Boundary refinement: Brent's method with xtol {config.root_xtol_s:g} s on the "
            f"continuous margin, far below the plan {plan.boundary_tolerance_s:g} s tolerance. "
            "Each boundary is independently re-solved with the scalar model; the reported "
            "agreement is recorded per boundary."
        ),
        (
            f"Detection floor: a complete occlusion must last longer than roughly two times "
            f"the sampling step to be guaranteed. On raised mask-node segments that is about "
            f"{2 * config.node_probe_step_s:g} s; elsewhere about {2 * config.audit_step_s:g} s. "
            "The fixture's narrowest occlusion is orders of magnitude longer than this and "
            "is independently audited."
        ),
        (
            "Times are reported in UTC as continuous seconds from the plan start. datetime "
            "strings are rounded to 1 microsecond; the authoritative value is "
            "seconds_from_start_s."
        ),
        (
            "Zero-duration tangencies (margin touches zero but no positive-length visible "
            "segment remains) are explicitly discarded and listed in the search diagnostics."
        ),
    ]
