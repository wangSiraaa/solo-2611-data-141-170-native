"""Bounded two-parameter likelihood fit, joint support region, and profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import optimize

from .likelihood import (
    PatternSummary,
    log_likelihood_from_basis,
)
from .model import ReliabilityModel

_EPS = 1e-12


@dataclass
class FitResult:
    parameters: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    theta_mle: np.ndarray
    loglik_mle: float
    support_drop: float
    support_threshold: float
    grid_size: int
    grid_lo: np.ndarray
    grid_hi: np.ndarray
    grid_first: np.ndarray
    grid_second: np.ndarray
    grid_ll: np.ndarray
    grid_mask: np.ndarray
    starts_used: int
    optimizer_message: str
    at_boundary: dict[str, bool]
    boundary_hits: list[dict]
    marginal_intervals: dict[str, tuple[float, float]]
    profile_intervals: dict[str, tuple[float, float]]
    profile_curves: dict[str, dict[str, np.ndarray]]
    identifiability: dict
    local_maxima: list[dict] = field(default_factory=list)

    @property
    def axis_spacing(self) -> tuple[float, float]:
        d0 = (self.grid_hi[0] - self.grid_lo[0]) / (self.grid_size - 1)
        d1 = (self.grid_hi[1] - self.grid_lo[1]) / (self.grid_size - 1)
        return float(d0), float(d1)


def _basis_and_derivatives(p0: float, p1: float):
    basis = np.array(
        [(1 - p0) * (1 - p1), (1 - p0) * p1, p0 * (1 - p1), p0 * p1]
    )
    d0 = np.array([-(1 - p1), -p1, (1 - p1), p1])
    d1 = np.array([-(1 - p0), (1 - p0), -p0, p0])
    return basis, d0, d1


def _neg_loglik_and_grad(
    theta: np.ndarray, weights: np.ndarray, counts: np.ndarray
) -> tuple[float, np.ndarray]:
    p0, p1 = float(theta[0]), float(theta[1])
    basis, d0, d1 = _basis_and_derivatives(p0, p1)
    probs = weights @ basis
    active = counts > 0
    p = probs[active]
    if np.any(p <= 0.0):
        return float("inf"), np.zeros(2)
    c = counts[active]
    loglik = float(c @ np.log(p))
    dp0 = weights[active] @ d0
    dp1 = weights[active] @ d1
    grad = np.array([float((c * dp0 / p).sum()), float((c * dp1 / p).sum())])
    return -loglik, -grad


def _loglik_at(theta: Sequence[float], weights: np.ndarray, counts: np.ndarray) -> float:
    p0, p1 = float(theta[0]), float(theta[1])
    basis, _, _ = _basis_and_derivatives(p0, p1)
    probs = weights @ basis
    active = counts > 0
    p = probs[active]
    if np.any(p <= 0.0):
        return float("-inf")
    return float(counts[active] @ np.log(p))


def _make_axis(lo: float, hi: float, n: int) -> np.ndarray:
    return np.linspace(lo, hi, n)


def _grid_ll_from_summary(
    summary: PatternSummary, first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    p0 = first.reshape(-1)
    p1 = second.reshape(-1)
    basis = np.stack(
        [
            (1 - p0) * (1 - p1),
            (1 - p0) * p1,
            p0 * (1 - p1),
            p0 * p1,
        ],
        axis=0,
    )
    return np.asarray(log_likelihood_from_basis(summary, basis), dtype=float)


def optimise_mle(
    model: ReliabilityModel,
    summary: PatternSummary,
    grid_first: np.ndarray,
    grid_second: np.ndarray,
    grid_ll: np.ndarray,
    n_starts: int = 12,
) -> tuple[np.ndarray, float, int, str, list[dict]]:
    """Multi-start bounded optimisation, seeded by the best grid cells."""

    lo = np.array([b[0] for b in model.plan.bounds.values()], dtype=float)
    hi = np.array([b[1] for b in model.plan.bounds.values()], dtype=float)
    weights = summary.weights
    counts = summary.counts

    flat_ll = grid_ll.reshape(-1)
    finite = np.where(np.isfinite(flat_ll))[0]
    if finite.size == 0:
        raise ValueError("likelihood is zero (or invalid) everywhere on the grid")
    order = finite[np.argsort(flat_ll[finite])[::-1]]
    seed_points: list[np.ndarray] = []
    first_flat = grid_first.reshape(-1)
    second_flat = grid_second.reshape(-1)
    seen: set[tuple[float, float]] = set()
    for idx in order:
        key = (round(float(first_flat[idx]), 9), round(float(second_flat[idx]), 9))
        if key in seen:
            continue
        seen.add(key)
        seed_points.append(np.array([first_flat[idx], second_flat[idx]]))
        if len(seed_points) >= n_starts:
            break

    best_theta = seed_points[0].copy()
    best_value = -float(flat_ll[order[0]])
    messages: list[str] = []
    local_maxima: list[dict] = []
    for start in seed_points:
        result = optimize.minimize(
            _neg_loglik_and_grad,
            x0=start,
            args=(weights, counts),
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(lo, hi)),
            options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000},
        )
        theta = np.clip(result.x, lo, hi)
        value = -_loglik_at(theta, weights, counts)
        messages.append(str(result.message))
        local_maxima.append(
            {
                "start": start.tolist(),
                "theta": theta.tolist(),
                "loglik": -float(value),
                "converged": bool(result.success),
            }
        )
        if np.isfinite(value) and value < best_value:
            best_value = value
            best_theta = theta
    message = (
        f"{len(seed_points)} starts; all converged"
        if all("CONVERGENCE" in m.upper() or "SUCCESS" in m.upper() for m in messages)
        else f"{len(seed_points)} starts; messages: {sorted(set(messages))}"
    )
    return best_theta, -float(best_value), len(seed_points), message, local_maxima


def _profile_curves(
    ll: np.ndarray, axis0: np.ndarray, axis1: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    profile0 = np.nanmax(ll, axis=1)
    profile1 = np.nanmax(ll, axis=0)
    return {
        "axis0": {"grid_value": axis0, "profile_loglik": profile0},
        "axis1": {"grid_value": axis1, "profile_loglik": profile1},
    }


def _interval_from_profile(
    grid_value: np.ndarray, profile: np.ndarray, threshold: float
) -> tuple[float, float]:
    valid = profile >= threshold
    if not np.any(valid):
        return float("nan"), float("nan")
    idx = np.where(valid)[0]
    return float(grid_value[idx[0]]), float(grid_value[idx[-1]])


def _refine_profile_endpoint(
    summary: PatternSummary,
    axis0: np.ndarray,
    axis1: np.ndarray,
    grid_ll: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    threshold: float,
    axis: int,
    side: str,
    grid_estimate: float,
) -> float:
    """Push a profile endpoint to the exact support boundary with SLSQP.

    Maximise (side='max') or minimise (side='min') one parameter subject to
    ``loglik >= threshold`` and the parameter box.  Falls back to the grid
    estimate when the solver fails or leaves the support.
    """

    def objective(z: np.ndarray) -> float:
        return -z[axis] if side == "max" else z[axis]

    def gradient(z: np.ndarray) -> np.ndarray:
        g = np.zeros(2)
        g[axis] = -1.0 if side == "max" else 1.0
        return g

    def constraint(z: np.ndarray) -> float:
        return _loglik_at(z, summary.weights, summary.counts) - threshold

    # Seed using the grid column/row where the endpoint is attained, with the
    if axis == 0:
        profile = np.nanmax(grid_ll, axis=1)
        valid = profile >= threshold
        idx = np.where(valid)[0][0 if side == "min" else -1]
        j = int(np.nanargmax(grid_ll[idx, :]))
        keystart = np.array([axis0[idx], axis1[j]])
    else:
        profile = np.nanmax(grid_ll, axis=0)
        valid = profile >= threshold
        j = np.where(valid)[0][0 if side == "min" else -1]
        i = int(np.nanargmax(grid_ll[:, j]))
        keystart = np.array([axis0[i], axis1[j]])
    keystart = np.clip(keystart, lo, hi)
    if constraint(keystart) < -1e-9:
        return float(grid_estimate)
    result = optimize.minimize(
        objective,
        x0=keystart,
        jac=gradient,
        method="SLSQP",
        bounds=list(zip(lo, hi)),
        constraints=[{"type": "ineq", "fun": constraint}],
        options={"ftol": 1e-13, "maxiter": 1000},
    )
    point = np.clip(result.x, lo, hi)
    if constraint(point) < -1e-7:
        return float(grid_estimate)
    # Both the refined point and the grid estimate are feasible; keep the
    # wider interval.
    value = float(point[axis])
    if side == "min":
        return min(value, float(grid_estimate))
    return max(value, float(grid_estimate))


def _boundary_hits(
    mask: np.ndarray,
    axis0: np.ndarray,
    axis1: np.ndarray,
    theta_mle: np.ndarray,
    drop: float,
) -> tuple[dict[str, bool], list[dict]]:
    """Identify support contact with each edge of the parameter box."""

    edges = {
        "axis0_low": mask[0, :],
        "axis0_high": mask[-1, :],
        "axis1_low": mask[:, 0],
        "axis1_high": mask[:, -1],
    }
    hits: list[dict] = []
    for name, edge_mask in edges.items():
        count = int(np.sum(edge_mask))
        hits.append(
            {
                "edge": name,
                "grid_points_in_support": count,
                "fraction": float(count / edge_mask.size),
                "covered": bool(count > 0),
            }
        )
    # MLE itself against bounds; contact to numerical tolerance.
    tol = 1e-9
    lo = np.array([axis0[0], axis1[0]])
    hi = np.array([axis0[-1], axis1[-1]])
    at_boundary = {
        "axis0_low": bool(theta_mle[0] - lo[0] <= tol),
        "axis0_high": bool(hi[0] - theta_mle[0] <= tol),
        "axis1_low": bool(theta_mle[1] - lo[1] <= tol),
        "axis1_high": bool(hi[1] - theta_mle[1] <= tol),
    }
    return at_boundary, hits


def profile_identifiability(
    model: ReliabilityModel,
    summary: PatternSummary,
    theta: np.ndarray,
    loglik_mle: float,
) -> dict:
    """Fisher information and near-flat directions at the MLE."""

    p0, p1 = float(theta[0]), float(theta[1])
    basis, d0, d1 = _basis_and_derivatives(p0, p1)
    active = summary.counts > 0
    probs = summary.weights[active] @ basis
    if np.any(probs <= 0):
        return {"valid": False, "reason": "observed pattern has zero probability at MLE"}
    counts = summary.counts[active]
    n_active = summary.weights[active]
    dp0 = n_active @ d0
    dp1 = n_active @ d1
    score0 = dp0 / probs
    score1 = dp1 / probs
    fisher = np.array(
        [
            [np.sum(counts * score0 * score0), np.sum(counts * score0 * score1)],
            [np.sum(counts * score0 * score1), np.sum(counts * score1 * score1)],
        ]
    )
    eigvals, eigvecs = np.linalg.eigh(fisher)
    positive = eigvals > max(eigvals[-1], 1.0) * 1e-10
    rank = int(np.sum(positive))
    # Direction of weakest curvature in parameter space.
    weak = eigvecs[:, 0]
    # Empirically test the weak direction: how much does theta move and does
    # the log-likelihood stay almost constant along it?
    step = 0.02
    lo = np.array([b[0] for b in model.plan.bounds.values()])
    hi = np.array([b[1] for b in model.plan.bounds.values()])
    probes = []
    for sign in (-1.0, 1.0):
        point = np.clip(theta + sign * step * weak, lo, hi)
        delta = float(_loglik_at(point, summary.weights, summary.counts)) - loglik_mle
        probes.append(
            {
                "direction": "negative" if sign < 0 else "positive",
                "theta": point.tolist(),
                "theta_change": float(np.linalg.norm(point - theta)),
                "loglik_change": delta,
            }
        )
    identified_separately = bool(rank == 2 and eigvals[0] > 1e-6 * max(eigvals[-1], 1.0))
    return {
        "valid": True,
        "fisher_information": fisher.tolist(),
        "eigenvalues": eigvals.tolist(),
        "condition_number": float(eigvals[-1] / eigvals[0]) if eigvals[0] > 0 else float("inf"),
        "rank": rank,
        "weak_direction": weak.tolist(),
        "weak_direction_probes": probes,
        "identified_separately": identified_separately,
        "near_equivalent_family_note": (
            None
            if identified_separately
            else "the likelihood is near-flat along at least one parameter "
            "direction; distinct (p_surge, p_heat) pairs can fit almost equally "
            "well, so the two causes are not separated by the data"
        ),
    }


def fit(
    model: ReliabilityModel,
    summary: PatternSummary,
    grid_size: int = 401,
) -> FitResult:
    lo = np.array([b[0] for b in model.plan.bounds.values()], dtype=float)
    hi = np.array([b[1] for b in model.plan.bounds.values()], dtype=float)
    axis0 = _make_axis(lo[0], hi[0], grid_size)
    axis1 = _make_axis(lo[1], hi[1], grid_size)
    first, second = np.meshgrid(axis0, axis1, indexing="ij")
    grid_ll = _grid_ll_from_summary(summary, first, second).reshape(first.shape)

    theta_mle, ll_mle, starts, message, local_maxima = optimise_mle(
        model, summary, first, second, grid_ll
    )
    drop = float(model.plan.support_loglik_drop)
    threshold = ll_mle - drop
    mask = np.isfinite(grid_ll) & (grid_ll >= threshold)

    at_boundary, boundary_hits = _boundary_hits(mask, axis0, axis1, theta_mle, drop)
    profiles = _profile_curves(grid_ll, axis0, axis1)
    profile_intervals = {
        name: _interval_from_profile(
            profiles[key]["grid_value"], profiles[key]["profile_loglik"], threshold
        )
        for name, key in zip(model.parameters, ("axis0", "axis1"))
    }
    # Refine each endpoint to the exact support boundary, keeping the wider
    # of the grid estimate and the refined SLSQP boundary point.
    for axis_index, name in enumerate(model.parameters):
        lo_grid, hi_grid = profile_intervals[name]
        if np.isfinite(lo_grid):
            lo_grid = _refine_profile_endpoint(
                summary, axis0, axis1, grid_ll, lo, hi, threshold,
                axis_index, "min", lo_grid,
            )
        if np.isfinite(hi_grid):
            hi_grid = _refine_profile_endpoint(
                summary, axis0, axis1, grid_ll, lo, hi, threshold,
                axis_index, "max", hi_grid,
            )
        profile_intervals[name] = (float(lo_grid), float(hi_grid))
    # The axis-aligned extent of the joint support is exactly the union of
    # the profile intervals, because a profile maximum over the other axis
    # is the marginal projection.  Keep both for reporting; neither is used
    # to build a risk witness.
    marginal = {
        key: profile_intervals[name]
        for name, key in zip(model.parameters, ("axis0", "axis1"))
    }
    identifiability = profile_identifiability(model, summary, theta_mle, ll_mle)

    result = FitResult(
        parameters=tuple(model.parameters),
        bounds=tuple(
            (float(lo[i]), float(hi[i])) for i in range(len(model.parameters))
        ),
        theta_mle=theta_mle,
        loglik_mle=ll_mle,
        support_drop=drop,
        support_threshold=threshold,
        grid_size=grid_size,
        grid_lo=lo,
        grid_hi=hi,
        grid_first=first,
        grid_second=second,
        grid_ll=grid_ll,
        grid_mask=mask,
        starts_used=starts,
        optimizer_message=message,
        at_boundary=at_boundary,
        boundary_hits=boundary_hits,
        marginal_intervals={
            model.parameters[0]: marginal["axis0"],
            model.parameters[1]: marginal["axis1"],
        },
        profile_intervals=profile_intervals,
        profile_curves=profiles,
        identifiability=identifiability,
        local_maxima=local_maxima,
    )
    return result


def support_contains(
    fit_result: FitResult,
    summary: PatternSummary,
    theta: Sequence[float],
    tol: float = 1e-7,
) -> bool:
    theta = np.asarray(theta, dtype=float)
    lo = fit_result.grid_lo
    hi = fit_result.grid_hi
    if np.any(theta < lo - tol) or np.any(theta > hi + tol):
        return False
    value = _loglik_at(theta, summary.weights, summary.counts)
    return bool(value >= fit_result.support_threshold - tol)
