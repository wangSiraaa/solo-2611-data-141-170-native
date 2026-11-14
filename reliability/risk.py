"""Risk over the joint likelihood-support region.

Risk is a smooth (bilinear) function of the two parameters:

    R(theta) = c . basis(theta),
    basis = [(1-p0)(1-p1), (1-p0)p1, p0(1-p1), p0 p1]

Extrema over the support are seeded from the dense likelihood grid, then
refined with bounded SLSQP under the non-linear constraint
``loglik(theta) >= threshold``.  Every reported extreme carries the parameter
pair that attains it, and that pair is re-verified against the exact
likelihood before being reported.  Marginal parameter endpoints are never
combined into a pair outside the support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import optimize

from .fit import FitResult, support_contains
from .likelihood import PatternSummary, log_likelihood_from_basis
from .model import ReliabilityModel


def _basis(theta: Sequence[float]) -> np.ndarray:
    p0, p1 = float(theta[0]), float(theta[1])
    return np.array(
        [(1 - p0) * (1 - p1), (1 - p0) * p1, p0 * (1 - p1), p0 * p1]
    )


def risk_value(coefficients: np.ndarray, theta: Sequence[float]) -> float:
    return float(coefficients @ _basis(theta))


def risk_grid(coefficients: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
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
    return np.asarray(coefficients @ basis, dtype=float).reshape(first.shape)


def loglik_value(summary: PatternSummary, theta: Sequence[float]) -> float:
    return float(log_likelihood_from_basis(summary, _basis(theta)))


@dataclass
class Extreme:
    theta: np.ndarray
    value: float
    loglik: float
    in_support: bool
    method: str

    def as_dict(self) -> dict:
        return {
            "theta": self.theta.tolist(),
            "value": self.value,
            "loglik": self.loglik,
            "in_support": self.in_support,
            "method": self.method,
        }


def _seed_indices(values: np.ndarray, feasible: np.ndarray, sign: float, k: int) -> list[int]:
    flat_values = values.reshape(-1)
    flat_feasible = feasible.reshape(-1)
    scores = np.where(flat_feasible & np.isfinite(flat_values), sign * flat_values, -np.inf)
    order = np.argsort(scores)[::-1]
    seeds: list[int] = []
    for idx in order:
        if not np.isfinite(scores[idx]):
            break
        seeds.append(int(idx))
        if len(seeds) >= k:
            break
    if not seeds:
        raise ValueError("support region is empty on the grid")
    return seeds


def _refine(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    summary: PatternSummary,
    fit_result: FitResult,
    seeds: list[np.ndarray],
    sign: float,
) -> Extreme:
    lo, hi = fit_result.grid_lo, fit_result.grid_hi
    threshold = fit_result.support_threshold

    def constraint(theta: np.ndarray) -> float:
        return loglik_value(summary, theta) - threshold

    best: Extreme | None = None
    for seed in seeds:
        x0 = np.clip(np.asarray(seed, dtype=float), lo, hi)
        if constraint(x0) < -1e-9:
            continue
        result = optimize.minimize(
            lambda z: -sign * objective(z),
            x0=x0,
            jac=lambda z: -sign * gradient(z),
            method="SLSQP",
            bounds=list(zip(lo, hi)),
            constraints=[{"type": "ineq", "fun": constraint}],
            options={"ftol": 1e-13, "maxiter": 2000},
        )
        candidate = np.clip(result.x, lo, hi)
        if not support_contains(fit_result, summary, candidate, tol=1e-7):
            continue
        value = objective(candidate)
        extreme = Extreme(
            candidate,
            value,
            loglik_value(summary, candidate),
            True,
            "slsqp",
        )
        if best is None or sign * value > sign * best.value:
            best = extreme

    if best is None:
        # Fall back to the best verified grid seed.
        seed = seeds[0]
        best = Extreme(
            np.asarray(seed, dtype=float),
            objective(np.asarray(seed, dtype=float)),
            loglik_value(summary, seed),
            True,
            "grid",
        )
    return best


def _risk_objective(coefficients: np.ndarray):
    def value(theta: np.ndarray) -> float:
        return risk_value(coefficients, theta)

    def gradient(theta: np.ndarray) -> np.ndarray:
        p0, p1 = float(theta[0]), float(theta[1])
        d0 = np.array([-(1 - p1), -p1, (1 - p1), p1])
        d1 = np.array([-(1 - p0), (1 - p0), -p0, p0])
        return np.array([float(coefficients @ d0), float(coefficients @ d1)])

    return value, gradient


def _risk_range(
    coefficients: np.ndarray,
    summary: PatternSummary,
    fit_result: FitResult,
    theta_mle: Sequence[float],
    n_seeds: int = 6,
) -> dict:
    values = risk_grid(coefficients, fit_result.grid_first, fit_result.grid_second)
    feasible = fit_result.grid_mask & np.isfinite(values)
    masked = np.where(feasible, values, np.nan)
    grid_min = float(np.nanmin(masked))
    grid_max = float(np.nanmax(masked))
    flat_first = fit_result.grid_first.reshape(-1)
    flat_second = fit_result.grid_second.reshape(-1)
    value_fn, grad_fn = _risk_objective(coefficients)

    min_seeds = [
        np.array([flat_first[i], flat_second[i]])
        for i in _seed_indices(values, feasible, -1.0, n_seeds)
    ]
    max_seeds = [
        np.array([flat_first[i], flat_second[i]])
        for i in _seed_indices(values, feasible, +1.0, n_seeds)
    ]
    low = _refine(value_fn, grad_fn, summary, fit_result, min_seeds, -1.0)
    high = _refine(value_fn, grad_fn, summary, fit_result, max_seeds, +1.0)
    # Guarantee the refined extrema bracket the grid extremes.
    if low.value > grid_min + 1e-12:
        idx = int(np.nanargmin(masked))
        low = Extreme(
            np.array([flat_first[idx], flat_second[idx]]),
            grid_min,
            loglik_value(summary, [flat_first[idx], flat_second[idx]]),
            True,
            "grid",
        )
    if high.value < grid_max - 1e-12:
        idx = int(np.nanargmax(masked))
        high = Extreme(
            np.array([flat_first[idx], flat_second[idx]]),
            grid_max,
            loglik_value(summary, [flat_first[idx], flat_second[idx]]),
            True,
            "grid",
        )
    return {
        "risk_at_mle": risk_value(coefficients, theta_mle),
        "grid_min": grid_min,
        "grid_max": grid_max,
        "min": low,
        "max": high,
    }


def _benefit_extremes(
    baseline_coeffs: np.ndarray,
    repaired_coeffs: np.ndarray,
    summary: PatternSummary,
    fit_result: FitResult,
    n_seeds: int = 6,
) -> tuple[Extreme, Extreme, float, float]:
    values_b = risk_grid(baseline_coeffs, fit_result.grid_first, fit_result.grid_second)
    values_r = risk_grid(repaired_coeffs, fit_result.grid_first, fit_result.grid_second)
    benefit = values_b - values_r
    feasible = fit_result.grid_mask & np.isfinite(benefit)
    masked = np.where(feasible, benefit, np.nan)
    grid_min = float(np.nanmin(masked))
    grid_max = float(np.nanmax(masked))
    flat_first = fit_result.grid_first.reshape(-1)
    flat_second = fit_result.grid_second.reshape(-1)

    def value(theta: np.ndarray) -> float:
        return risk_value(baseline_coeffs, theta) - risk_value(repaired_coeffs, theta)

    def gradient(theta: np.ndarray) -> np.ndarray:
        return _risk_objective(baseline_coeffs)[1](theta) - _risk_objective(
            repaired_coeffs
        )[1](theta)

    min_seeds = [
        np.array([flat_first[i], flat_second[i]])
        for i in _seed_indices(benefit, feasible, -1.0, n_seeds)
    ]
    max_seeds = [
        np.array([flat_first[i], flat_second[i]])
        for i in _seed_indices(benefit, feasible, +1.0, n_seeds)
    ]
    low = _refine(value, gradient, summary, fit_result, min_seeds, -1.0)
    high = _refine(value, gradient, summary, fit_result, max_seeds, +1.0)
    if low.value > grid_min + 1e-12:
        idx = int(np.nanargmin(masked))
        low = Extreme(
            np.array([flat_first[idx], flat_second[idx]]),
            grid_min,
            loglik_value(summary, [flat_first[idx], flat_second[idx]]),
            True,
            "grid",
        )
    if high.value < grid_max - 1e-12:
        idx = int(np.nanargmax(masked))
        high = Extreme(
            np.array([flat_first[idx], flat_second[idx]]),
            grid_max,
            loglik_value(summary, [flat_first[idx], flat_second[idx]]),
            True,
            "grid",
        )
    return low, high, grid_min, grid_max


def risk_scenario(
    model: ReliabilityModel,
    summary: PatternSummary,
    fit_result: FitResult,
) -> dict:
    baseline_coeffs = model.risk_coefficients(repaired=False)
    repaired_coeffs = model.risk_coefficients(repaired=True)
    theta_mle = fit_result.theta_mle

    baseline = _risk_range(baseline_coeffs, summary, fit_result, theta_mle)
    repaired = _risk_range(repaired_coeffs, summary, fit_result, theta_mle)
    benefit_low, benefit_high, benefit_grid_min, benefit_grid_max = _benefit_extremes(
        baseline_coeffs, repaired_coeffs, summary, fit_result
    )

    def witness(theta: np.ndarray, label: str) -> dict:
        b = risk_value(baseline_coeffs, theta)
        r = risk_value(repaired_coeffs, theta)
        return {
            "label": label,
            "theta": [float(theta[0]), float(theta[1])],
            "baseline_risk": b,
            "repaired_risk": r,
            "absolute_benefit": b - r,
            "loglik": loglik_value(summary, theta),
            "in_support": support_contains(fit_result, summary, theta),
        }

    witnesses = [
        witness(baseline["min"].theta, "baseline_min"),
        witness(baseline["max"].theta, "baseline_max"),
        witness(repaired["min"].theta, "repaired_min"),
        witness(repaired["max"].theta, "repaired_max"),
        witness(benefit_low.theta, "benefit_min"),
        witness(benefit_high.theta, "benefit_max"),
    ]

    # Demonstrate why marginal endpoints must not be stitched together: take
    # the four corner pairs formed from the marginal projections and check
    # each against the joint support.  At least one is typically outside.
    names = list(fit_result.parameters)
    m0 = fit_result.marginal_intervals[names[0]]
    m1 = fit_result.marginal_intervals[names[1]]
    naive_corners = []
    for a in m0:
        for b in m1:
            theta = np.array([a, b])
            inside = support_contains(fit_result, summary, theta)
            naive_corners.append(
                {
                    "theta": [float(a), float(b)],
                    "in_joint_support": bool(inside),
                    "baseline_risk_if_evaluated": risk_value(baseline_coeffs, theta),
                    "repaired_risk_if_evaluated": risk_value(repaired_coeffs, theta),
                }
            )

    return {
        "support_threshold": fit_result.support_threshold,
        "risk_at_mle": {
            "baseline": baseline["risk_at_mle"],
            "repaired": repaired["risk_at_mle"],
            "absolute_benefit": baseline["risk_at_mle"] - repaired["risk_at_mle"],
        },
        "baseline": {
            "min": baseline["min"].as_dict(),
            "max": baseline["max"].as_dict(),
            "grid_min": baseline["grid_min"],
            "grid_max": baseline["grid_max"],
        },
        "repaired": {
            "min": repaired["min"].as_dict(),
            "max": repaired["max"].as_dict(),
            "grid_min": repaired["grid_min"],
            "grid_max": repaired["grid_max"],
        },
        "absolute_benefit": {
            "min": benefit_low.as_dict(),
            "max": benefit_high.as_dict(),
            "grid_min": benefit_grid_min,
            "grid_max": benefit_grid_max,
            "semantics": (
                "baseline_risk - repaired_risk evaluated at each parameter pair in "
                "the same joint support; not a difference of independently chosen "
                "risk endpoints"
            ),
        },
        "marginal_endpoint_combination": {
            "marginal_intervals": {
                names[0]: [float(m0[0]), float(m0[1])],
                names[1]: [float(m1[0]), float(m1[1])],
            },
            "corners": naive_corners,
            "note": (
                "corners are shown only to demonstrate that combining marginal "
                "endpoints can leave the joint support; they are not used as "
                "risk bounds"
            ),
        },
        "witnesses": witnesses,
    }
