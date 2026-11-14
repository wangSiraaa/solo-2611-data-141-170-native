"""Exact joint-observation likelihood under partial, masked snapshots.

Each cycle contributes the probability that *all* observed node states hold
simultaneously, marginalising over every primitive configuration that is not
observed.  Because the same primitive events (surge / heat) feed several
gates, cycles cannot be split into per-component samples.

For a configuration ``x`` of the primitive events and the two unknown bits
``(b_surge, b_heat)``::

    P(x, b) = fixed_weight(x) * basis[b](p_surge, p_heat)

The weight of an observation pattern ``s`` is therefore

    P(s | theta) = sum_g n[s, g] * basis_g(theta),

with ``n[s, g]`` a *constant* over theta (a sum of fixed-event weights).
Grouped patterns make the log-likelihood a cheap function of ``n``, so dense
parameter grids can be evaluated without re-enumerating the circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .model import ReliabilityModel

UNOBSERVED = -1


@dataclass(frozen=True)
class PatternSummary:
    """Distinct observation patterns and their grouped configuration weights."""

    names: tuple[str, ...]
    patterns: tuple[tuple[int, ...], ...]
    counts: np.ndarray            # (S,) number of cycles per pattern
    weights: np.ndarray           # (S, G) sum of fixed weights, grouped by unknown bits
    cycle_pattern: np.ndarray     # (n_cycles,) index into patterns
    feasible: np.ndarray          # (S,) whether some primitive config satisfies the pattern

    @property
    def n_patterns(self) -> int:
        return len(self.patterns)

    def describe(self, index: int) -> str:
        parts = []
        for name, value in zip(self.names, self.patterns[index]):
            parts.append(f"{name}={'NA' if value == UNOBSERVED else value}")
        return ",".join(parts)


def summarise_patterns(
    model: ReliabilityModel,
    observations: Sequence[Mapping[str, int | None]],
) -> PatternSummary:
    """Map cycles to distinct patterns and precompute grouped weights.

    A pattern records only the values that were actually observed.  ``NA`` is
    carried as a wildcard; it is never imputed as healthy.
    """

    names = tuple(model.observables)
    index_of: dict[tuple[int, ...], int] = {}
    pattern_list: list[tuple[int, ...]] = []
    counts: list[int] = []
    cycle_pattern = np.empty(len(observations), dtype=np.int64)
    for cycle, obs in enumerate(observations):
        key = tuple(
            UNOBSERVED if obs.get(name) is None else int(obs[name]) for name in names
        )
        idx = index_of.get(key)
        if idx is None:
            idx = len(pattern_list)
            index_of[key] = idx
            pattern_list.append(key)
            counts.append(0)
        counts[idx] += 1
        cycle_pattern[cycle] = idx

    # Constraint masks per distinct pattern, evaluated on primitive configs.
    values = model.evaluate_nodes()
    groups = model.group_codes
    fixed = model.fixed_weights
    n_groups = 2 ** len(model.parameters)
    weights = np.zeros((len(pattern_list), n_groups), dtype=float)
    feasible = np.ones(len(pattern_list), dtype=bool)
    for idx, key in enumerate(pattern_list):
        mask = np.ones(values[model.root].shape[0], dtype=bool)
        for name, value in zip(names, key):
            if value == UNOBSERVED:
                continue
            mask &= values[name] == bool(value)
        if np.any(mask):
            bincount = np.bincount(
                groups[mask].astype(np.int64), weights=fixed[mask], minlength=n_groups
            )
            weights[idx, :] = bincount
        else:
            feasible[idx] = False
        # An infeasible pattern has all-zero weights and is a genuine
        # graph/snapshot contradiction, never a reason to drop the row.

    return PatternSummary(
        names=names,
        patterns=tuple(pattern_list),
        counts=np.asarray(counts, dtype=float),
        weights=weights,
        cycle_pattern=cycle_pattern,
        feasible=feasible,
    )


@dataclass
class PatternReport:
    pattern: str
    count: int
    probability: float
    log_probability: float


def pattern_probabilities(
    summary: PatternSummary, basis: np.ndarray
) -> np.ndarray:
    """Probability of each pattern for a single theta (basis shape (G,))."""

    return summary.weights @ basis


def log_likelihood_from_basis(summary: PatternSummary, basis: np.ndarray) -> float:
    """Log-likelihood for a single basis vector or a (G, N) basis matrix.

    Zero-probability observed patterns yield ``-inf``; they are not clipped to
    a small positive number.
    """

    probs = summary.weights @ basis  # (S,) or (S, N)
    active = summary.counts > 0
    if not np.any(active):
        return 0.0
    p = probs[active]
    if p.ndim == 1:
        if np.any(p <= 0.0):
            return float("-inf")
        return float(summary.counts[active] @ np.log(p))
    counts = summary.counts[active][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(p > 0.0, np.log(p), -np.inf)
        result = np.sum(counts * logs, axis=0)
    return result


def log_likelihood(model: ReliabilityModel, summary: PatternSummary, theta: Sequence[float]) -> float:
    return log_likelihood_from_basis(summary, model.basis(theta))


def log_likelihood_gradient(summary: PatternSummary, theta: Sequence[float]) -> np.ndarray:
    """Analytic gradient of the log-likelihood with respect to theta."""

    p0, p1 = float(theta[0]), float(theta[1])
    basis = np.array(
        [(1 - p0) * (1 - p1), (1 - p0) * p1, p0 * (1 - p1), p0 * p1]
    )
    dc0 = np.array([-(1 - p1), -p1, (1 - p1), p1])
    dc1 = np.array([-(1 - p0), (1 - p0), -p0, p0])
    probs = summary.weights @ basis
    active = summary.counts > 0
    p = probs[active]
    if np.any(p <= 0.0):
        return np.zeros(2)
    c = summary.counts[active]
    w = summary.weights[active]
    dp0 = w @ dc0
    dp1 = w @ dc1
    return np.array([float((c * dp0 / p).sum()), float((c * dp1 / p).sum())])


def log_likelihood_grid(
    model: ReliabilityModel,
    summary: PatternSummary,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Vectorised log-likelihood on a flattened parameter grid."""

    first_flat = np.asarray(first, dtype=float).reshape(-1)
    second_flat = np.asarray(second, dtype=float).reshape(-1)
    basis = model.basis_flat(first_flat, second_flat)
    return np.asarray(log_likelihood_from_basis(summary, basis), dtype=float)


def pattern_table(
    model: ReliabilityModel,
    summary: PatternSummary,
    theta: Sequence[float],
) -> list[PatternReport]:
    probs = pattern_probabilities(summary, model.basis(theta))
    rows: list[PatternReport] = []
    for idx in range(summary.n_patterns):
        probability = float(probs[idx])
        rows.append(
            PatternReport(
                pattern=summary.describe(idx),
                count=int(summary.counts[idx]),
                probability=probability,
                log_probability=float(np.log(probability)) if probability > 0 else float("-inf"),
            )
        )
    rows.sort(key=lambda r: (-r.count, r.pattern))
    return rows


def inconsistent_patterns(summary: PatternSummary) -> list[int]:
    """Patterns that no primitive configuration can satisfy."""

    return [int(idx) for idx in np.where(~summary.feasible)[0]]


def contradiction_report(
    model: ReliabilityModel,
    summary: PatternSummary,
) -> dict:
    bad = inconsistent_patterns(summary)
    details = [
        {
            "pattern": summary.describe(idx),
            "cycles": int(summary.counts[idx]),
        }
        for idx in bad
    ]
    # Also flag any cycle whose observed states violate a gate locally.
    local = local_contradictions(model, summary)
    return {
        "inconsistent_patterns": details,
        "local_gate_contradictions": local,
        "consistent": not details and not local,
    }


def local_contradictions(model: ReliabilityModel, summary: PatternSummary) -> list[dict]:
    """Cheap per-node sanity check independent of the full enumeration.

    For every fully observed gate with an AND/OR/k-of-n law, a set of input
    states that cannot produce the observed output is reported.  Inputs which
    are themselves unobserved are skipped.
    """

    findings: list[dict] = []
    for idx, pattern in enumerate(summary.patterns):
        state = {
            name: (None if value == UNOBSERVED else bool(value))
            for name, value in zip(summary.names, pattern)
        }
        for node in model.nodes.values():
            out = state.get(node.id)
            inputs = [state.get(name) for name in node.inputs]
            if out is None or any(v is None for v in inputs):
                continue
            required = model.nodes[node.id]
            if required.op == "or":
                implied = any(inputs)
            elif required.op == "and":
                implied = all(inputs)
            else:
                implied = sum(bool(v) for v in inputs) >= int(required.k or 0)
            if implied != out:
                findings.append(
                    {
                        "pattern": summary.describe(idx),
                        "node": node.id,
                        "observed_output": int(out),
                        "inputs": {n: int(v) for n, v in zip(node.inputs, inputs)},
                        "cycles": int(summary.counts[idx]),
                    }
                )
    return findings


def design_rank(model: ReliabilityModel, summary: PatternSummary) -> dict:
    """Local identifiability diagnostic of the two parameters.

    Uses the Fisher-information-like matrix at a symmetric reference point,
    built from pattern probabilities and their theta-derivatives.  A rank
    deficiency is a warning of an exactly unidentifiable family, not a proof
    for the whole box.
    """

    theta = np.array([0.05, 0.05], dtype=float)
    basis = model.basis(theta)
    probs = summary.weights @ basis
    active = summary.counts > 0
    p = probs[active]
    if np.any(p <= 0):
        return {"valid": False, "reason": "zero-probability pattern at reference theta"}
    # d basis / d theta for the two parameters.
    p0, p1 = theta
    derivs = np.stack(
        [
            np.array([-(1 - p1), -p1, (1 - p1), p1]),   # d/d p_surge
            np.array([-(1 - p0), (1 - p0), -p0, p0]),   # d/d p_heat
        ]
    )
    # per-pattern derivative = n . deriv
    n_active = summary.weights[active]
    dp = n_active @ derivs.T  # (S, 2)
    score = dp / p[:, None]
    fisher = (summary.counts[active][:, None] * score).T @ score
    eigvals = np.linalg.eigvalsh(fisher)
    cond = float(eigvals[-1] / eigvals[0]) if eigvals[0] > 0 else float("inf")
    return {
        "valid": True,
        "fisher_information": fisher.tolist(),
        "eigenvalues": eigvals.tolist(),
        "condition_number": cond,
        "rank": int(np.linalg.matrix_rank(fisher, tol=1e-9)),
        "reference_theta": [float(p0), float(p1)],
    }
