"""Search for visibility transitions and refine them to sub-second precision.

Design:

* A **coarse** grid is the search pass.  It only proposes brackets that may
  contain a transition; it is never used to connect samples into windows.
* A **verification** grid (much finer) is scanned over the same interval.  Any
  transition it finds is also accepted, so a short occlusion between coarse
  samples cannot be silently skipped.
* **Mask-node probes** densely sample the stretches where the satellite crosses
  azimuth node directions whose horizon threshold is raised.  This catches
  narrow obstacle crossings even when the pass itself is fast.
* Every proposed bracket is refined with Brent's method on the continuous
  margin ``f(t) = elevation(t) - threshold(azimuth(t))``.  The threshold is
  continuous across azimuth nodes, so ``f`` is continuous even though its
  derivative is not.
* Duplicate roots are merged and positive-length visible segments are emitted
  separately.  A zero-duration tangency therefore produces no window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
from scipy.optimize import brentq

from .constants import (
    DEFAULT_EXTREMUM_XTOL_S,
    DEFAULT_ROOT_XTOL_S,
    ROOT_MERGE_TOL_S,
    TANGENCY_EPS_S,
)
from .geometry import Ephemeris

_CHUNK = 200_000
# Hard cap on points emitted by one mask-node probe segment.
_NODE_PROBE_MAX_POINTS = 40_000


@dataclass
class Bracket:
    """A time bracket known to contain one visibility transition."""

    lo: float
    hi: float
    source: str
    f_lo: float
    f_hi: float


@dataclass
class ScanResult:
    step_s: float
    source: str
    brackets: list[Bracket] = field(default_factory=list)
    sample_count: int = 0
    visible_count: int = 0
    min_gap_deg: float = math.inf
    max_gap_deg: float = -math.inf
    times: Optional[np.ndarray] = None
    azimuth_deg: Optional[np.ndarray] = None


def _grid_count(t0: float, t1: float, step: float) -> tuple[int, bool]:
    span = t1 - t0
    if span <= 0.0:
        raise ValueError("empty time range")
    if step <= 0.0:
        raise ValueError("step must be positive")
    n_full = int(math.floor(span / step))
    # Include the exact final instant when it does not already lie on the grid.
    tail = span - n_full * step > 1e-9
    return n_full + 1 + (1 if tail else 0), tail


def _times_for_indices(t0: float, t1: float, step: float, start: int, stop: int, count: int, tail: bool) -> np.ndarray:
    idx = np.arange(start, stop, dtype=float)
    times = t0 + idx * step
    if tail and stop == count:  # final chunk includes the exact endpoint
        times[-1] = t1
    return times


def scan(
    ephemeris: Ephemeris,
    t0: float,
    t1: float,
    step: float,
    source: str,
    collect_azimuth: bool = False,
) -> ScanResult:
    """Scan ``[t0, t1]`` at ``step`` and return sign-change brackets.

    Chunks overlap by one sample so transitions between chunks are not lost.
    The full grid is never materialised for fine scans.
    """
    count, tail = _grid_count(t0, t1, step)
    result = ScanResult(step_s=step, source=source)
    if collect_azimuth:
        result.times = np.empty(count, dtype=float)
        result.azimuth_deg = np.empty(count, dtype=float)
    for start in range(0, count, _CHUNK):
        stop = min(start + _CHUNK, count)
        # One extra sample closes the last pair of this chunk.
        eval_stop = min(stop + 1, count)
        times = _times_for_indices(t0, t1, step, start, eval_stop, count, tail)
        geometry = ephemeris.evaluate(times)
        gaps = geometry.gap_deg
        result.sample_count += len(times)
        result.visible_count += int(np.count_nonzero(gaps >= 0.0))
        result.min_gap_deg = min(result.min_gap_deg, float(np.min(gaps)))
        result.max_gap_deg = max(result.max_gap_deg, float(np.max(gaps)))
        if collect_azimuth:
            # Only the chunk's own samples belong in the dense series.
            own = stop - start
            result.times[start:stop] = times[:own]
            result.azimuth_deg[start:stop] = geometry.azimuth_deg[:own]
        visible = gaps >= 0.0
        changes = np.flatnonzero(visible[:-1] != visible[1:])
        for offset in changes:
            lo = float(times[offset])
            hi = float(times[offset + 1])
            result.brackets.append(
                Bracket(
                    lo=lo,
                    hi=hi,
                    source=source,
                    f_lo=float(gaps[offset]),
                    f_hi=float(gaps[offset + 1]),
                )
            )
    return result


def unwrapped_azimuth(times: np.ndarray, azimuth_deg: np.ndarray) -> np.ndarray:
    """Unwrap azimuth in degrees so that 359 -> 1 is a +2 degree step."""
    if times.size < 2:
        return np.asarray(azimuth_deg, dtype=float).copy()
    return np.degrees(np.unwrap(np.radians(np.asarray(azimuth_deg, dtype=float))))


def find_node_crossings(
    times: np.ndarray, azimuth_unwrapped_deg: np.ndarray, node_azimuths_deg: Iterable[float]
) -> list[float]:
    """Times at which the unwrapped azimuth crosses any node direction.

    Crossing ``v + 360*k`` is detected by a change of
    ``floor((az - v) / 360)``; this is robust against azimuth passing a node
    multiple times and against the 0/360 wrap.
    """
    az = np.asarray(azimuth_unwrapped_deg, dtype=float)
    times_arr = np.asarray(times, dtype=float)
    crossings: list[float] = []
    for node in np.asarray(list(node_azimuths_deg), dtype=float):
        q = np.floor((az - node) / 360.0)
        dq = np.diff(q)
        for i in np.flatnonzero(dq != 0):
            delta = az[i + 1] - az[i]
            if delta == 0.0:
                continue
            lo_q, hi_q = int(q[i]), int(q[i + 1])
            if hi_q > lo_q:
                levels = range(lo_q + 1, hi_q + 1)
            else:
                levels = range(lo_q, hi_q, -1)
            t0, t1 = times_arr[i], times_arr[i + 1]
            for level in levels:
                target = node + 360.0 * level
                frac = (target - az[i]) / delta
                if -1e-12 <= frac <= 1.0 + 1e-12:
                    crossings.append(float(t0 + frac * (t1 - t0)))
    crossings.sort()
    # Collapse duplicates from numerically coincident nodes.
    merged: list[float] = []
    for value in crossings:
        if not merged or abs(value - merged[-1]) > 1e-6:
            merged.append(value)
    return merged


def node_probe_brackets(
    ephemeris: Ephemeris,
    t0: float,
    t1: float,
    crossings: list[float],
    base_threshold_deg: float,
    probe_step: float,
    source: str = "mask_node_probe",
) -> tuple[list[Bracket], dict[str, Any]]:
    """Densely probe trajectory pieces adjacent to raised mask nodes.

    Azimuth nodes partition the sweep into pieces where the threshold is a
    single linear function of azimuth.  Pieces whose endpoint nodes are raised
    above the base horizon can hide a short occlusion between ordinary
    samples, so they are sampled densely here.
    """
    stats = {"segments": 0, "raised_segments": 0, "samples": 0, "crossings": len(crossings)}
    brackets: list[Bracket] = []
    points = sorted(set([t0, t1] + [c for c in crossings if t0 <= c <= t1]))
    for a, b in zip(points[:-1], points[1:]):
        if b <= a:
            continue
        stats["segments"] += 1
        # Threshold at the segment ends; at a crossing this is the node's own
        # threshold.  If neither end is raised, the ordinary scans suffice.
        sample_a = ephemeris.sample(a)
        sample_b = ephemeris.sample(b)
        raised = max(sample_a.threshold_deg, sample_b.threshold_deg) > base_threshold_deg + 1e-7
        if not raised:
            continue
        stats["raised_segments"] += 1
        span = b - a
        step = min(probe_step, max(span / 64.0, 1e-6))
        n = int(math.ceil(span / step)) + 1
        if n > _NODE_PROBE_MAX_POINTS:
            n = _NODE_PROBE_MAX_POINTS
        probe_times = np.linspace(a, b, n)
        geometry = ephemeris.evaluate(probe_times)
        gaps = geometry.gap_deg
        stats["samples"] += len(probe_times)
        visible = gaps >= 0.0
        for offset in np.flatnonzero(visible[:-1] != visible[1:]):
            brackets.append(
                Bracket(
                    lo=float(probe_times[offset]),
                    hi=float(probe_times[offset + 1]),
                    source=source,
                    f_lo=float(gaps[offset]),
                    f_hi=float(gaps[offset + 1]),
                )
            )
    return brackets, stats


@dataclass
class Root:
    seconds: float
    f_value: float
    bracket_lo: float
    bracket_hi: float
    sources: tuple[str, ...]
    iterations: int
    converged: bool
    reason: str = ""


def refine_bracket(
    ephemeris: Ephemeris,
    bracket: Bracket,
    xtol: float = DEFAULT_ROOT_XTOL_S,
) -> Optional[Root]:
    """Brent root of the continuous visibility margin inside a bracket."""
    f_lo = ephemeris.gap(bracket.lo)
    f_hi = ephemeris.gap(bracket.hi)
    if f_lo == 0.0:
        return Root(bracket.lo, 0.0, bracket.lo, bracket.hi, (bracket.source,), 0, True)
    if f_hi == 0.0:
        return Root(bracket.hi, 0.0, bracket.lo, bracket.hi, (bracket.source,), 0, True)
    if f_lo * f_hi > 0.0:
        return None
    try:
        root, info = brentq(
            ephemeris.gap,
            bracket.lo,
            bracket.hi,
            xtol=xtol,
            rtol=4.0 * np.finfo(float).eps,
            maxiter=200,
            full_output=True,
            disp=False,
        )
    except (ValueError, RuntimeError):
        return None
    residual = ephemeris.gap(float(root))
    return Root(
        seconds=float(root),
        f_value=residual,
        bracket_lo=bracket.lo,
        bracket_hi=bracket.hi,
        sources=(bracket.source,),
        iterations=int(info.iterations),
        converged=bool(info.converged),
    )


def refine_brackets(
    ephemeris: Ephemeris, brackets: Iterable[Bracket], xtol: float = DEFAULT_ROOT_XTOL_S
) -> tuple[list[Root], list[dict[str, Any]]]:
    roots: list[Root] = []
    rejected: list[dict[str, Any]] = []
    for bracket in brackets:
        root = refine_bracket(ephemeris, bracket, xtol=xtol)
        if root is None:
            rejected.append(
                {
                    "bracket_s": [bracket.lo, bracket.hi],
                    "source": bracket.source,
                    "f_lo": bracket.f_lo,
                    "f_hi": bracket.f_hi,
                }
            )
        else:
            roots.append(root)
    return roots, rejected


def merge_roots(roots: Iterable[Root], tolerance: float = ROOT_MERGE_TOL_S) -> list[Root]:
    """Merge numerically duplicate roots, keeping the smallest residual."""
    ordered = sorted(roots, key=lambda r: r.seconds)
    groups: list[list[Root]] = []
    for root in ordered:
        if groups and abs(root.seconds - groups[-1][-1].seconds) <= tolerance:
            groups[-1].append(root)
        else:
            groups.append([root])
    merged: list[Root] = []
    for group in groups:
        best = min(group, key=lambda r: abs(r.f_value))
        sources: list[str] = []
        for root in group:
            for source in root.sources:
                if source not in sources:
                    sources.append(source)
        merged.append(
            Root(
                seconds=best.seconds,
                f_value=best.f_value,
                bracket_lo=min(r.bracket_lo for r in group),
                bracket_hi=max(r.bracket_hi for r in group),
                sources=tuple(sources),
                iterations=max(r.iterations for r in group),
                converged=all(r.converged for r in group),
            )
        )
    return merged


@dataclass
class Window:
    start_s: float
    end_s: float
    entry_reason: str
    exit_reason: str
    entry_root: Optional[Root] = None
    exit_root: Optional[Root] = None
    # Filled by the reporter.
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def build_windows(
    ephemeris: Ephemeris,
    roots: Iterable[Root],
    t0: float,
    t1: float,
    edge_epsilon: float = 1e-6,
) -> tuple[list[Window], list[Root], list[dict[str, Any]]]:
    """Build positive-length visible segments between refined roots.

    Roots within ``edge_epsilon`` of the plan edges are treated as edge
    truncations.  Each segment is classified independently at its midpoint.
    Segments shorter than ``TANGENCY_EPS_S`` are discarded as tangencies.
    """
    kept: list[Root] = []
    for root in roots:
        if root.seconds <= t0 + edge_epsilon or root.seconds >= t1 - edge_epsilon:
            continue
        kept.append(root)
    kept.sort(key=lambda r: r.seconds)

    cut_points: list[tuple[float, Optional[Root]]] = [(t0, None)]
    cut_points.extend((root.seconds, root) for root in kept)
    cut_points.append((t1, None))

    windows: list[Window] = []
    discarded: list[dict[str, Any]] = []
    for (a, root_a), (b, root_b) in zip(cut_points[:-1], cut_points[1:]):
        if b <= a:
            continue
        mid = 0.5 * (a + b)
        visible = ephemeris.gap(mid) >= 0.0
        if not visible:
            continue
        duration = b - a
        if duration <= TANGENCY_EPS_S:
            discarded.append(
                {
                    "start_s": a,
                    "end_s": b,
                    "duration_s": duration,
                    "reason": "zero_or_negative_duration_tangency",
                }
            )
            continue
        entry_reason = "start_of_plan_truncation" if root_a is None else "enter_visibility"
        exit_reason = "end_of_plan_truncation" if root_b is None else "exit_visibility"
        windows.append(
            Window(
                start_s=a,
                end_s=b,
                entry_reason=entry_reason,
                exit_reason=exit_reason,
                entry_root=root_a,
                exit_root=root_b,
            )
        )
    return windows, kept, discarded


def extract_reasons(ephemeris: Ephemeris, root: Root, delta: float = 0.1) -> dict[str, Any]:
    """Characterise a transition as orbital-elevation or horizon-mask driven.

    Uses central differences on elevation, threshold and azimuth.  This is
    evidence only; it does not influence the root.
    """
    before = ephemeris.sample(root.seconds - delta)
    at = ephemeris.sample(root.seconds)
    after = ephemeris.sample(root.seconds + delta)
    span = 2.0 * delta
    el_rate = (after.elevation_deg - before.elevation_deg) / span
    thr_rate = (after.threshold_deg - before.threshold_deg) / span
    az_rate = (after.azimuth_deg - before.azimuth_deg) / span
    # Azimuth rate must respect the 0/360 seam.
    if az_rate > 180.0 / span:
        az_rate -= 360.0 / span
    elif az_rate < -180.0 / span:
        az_rate += 360.0 / span
    gap_rate = el_rate - thr_rate
    entry = after.visible and not before.visible
    cause = "horizon_mask_feature" if abs(thr_rate) > abs(el_rate) else "orbital_elevation"
    if entry:
        label = "enter_visibility"
        if cause == "orbital_elevation":
            detail = "elevation_rise_above_horizon"
        else:
            detail = "azimuth_rate_through_horizon_feature"
    else:
        label = "exit_visibility"
        if cause == "orbital_elevation":
            detail = "elevation_set_below_horizon"
        else:
            detail = "occluded_by_horizon_feature"
    return {
        "reason": label,
        "cause": cause,
        "detail": detail,
        "entry": bool(entry),
        "gap_rate_deg_per_s": float(gap_rate),
        "elevation_rate_deg_per_s": float(el_rate),
        "threshold_rate_deg_per_s": float(thr_rate),
        "azimuth_rate_deg_per_s": float(az_rate),
        "sample_before": before,
        "sample_at": at,
        "sample_after": after,
    }


def find_wall_clock_extremum(
    ephemeris: Ephemeris, start_s: float, end_s: float, xtol: float = DEFAULT_EXTREMUM_XTOL_S
) -> tuple[float, float]:
    """Peak elevation (seconds, degrees) inside a window."""
    from scipy.optimize import minimize_scalar

    if end_s <= start_s:
        sample = ephemeris.sample(start_s)
        return start_s, sample.elevation_deg
    result = minimize_scalar(
        lambda s: -ephemeris.elevation(float(s)),
        bounds=(float(start_s), float(end_s)),
        method="bounded",
        options={"xatol": xtol},
    )
    return float(result.x), float(-result.fun)
