"""Joint feasible-set LP machinery.

For one event with detector reading ``y`` the feasible marker vectors are

    F = { x >= 0 : l <= S x <= u }

where normal detector rows get a two-sided band
``[y - tol, y + tol]`` and saturated rows (``y >= saturation_value``) get
only a lower bound ``[saturation_value - tol, +inf)``.  The band is built
*before* any inversion, so a saturated channel is never assigned a fake
upper edge.

Membership questions about a closed gate region ``G`` are answered by
feasibility LPs on the *joint* constraint system:

* ``F`` empty                          -> event INFEASIBLE
* ``F subset of G`` (no outside point) -> CERTAIN_IN
* ``F disjoint from G`` (no inside)    -> CERTAIN_OUT
* otherwise                            -> UNCERTAIN, with one inside and
  one outside witness, each a single vector satisfying every row of the
  joint system simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from .errors import SolverError
from .plan import Gate, Plan

# LP statuses returned by scipy's HiGHS wrapper.
_LP_OPTIMAL = 0
_LP_INFEASIBLE = 2


@dataclass
class EventConstraint:
    """Serialisable description of one row of the event's joint system."""

    detector: str
    detector_index: int
    observed: float
    kind: str  # "normal_band" | "saturation_lower"
    lower: float | None
    upper: float | None
    row: list[float]  # coefficients over markers

    def to_json(self) -> dict:
        return {
            "detector": self.detector,
            "detector_index": self.detector_index,
            "observed": self.observed,
            "kind": self.kind,
            "lower": self.lower,
            "upper": self.upper,
            "row_detector_by_markers": self.row,
        }


@dataclass
class LPOutcome:
    feasible: bool
    point: np.ndarray | None
    status: int
    message: str


@dataclass
class GateDecision:
    gate_id: str
    verdict: str  # CERTAIN_IN | CERTAIN_OUT | UNCERTAIN
    reason: str
    inside_witness: list[float] | None = None
    outside_witness: list[float] | None = None
    # Independent re-check of each witness against the full joint system.
    inside_witness_check: dict | None = None
    outside_witness_check: dict | None = None
    detail: dict = field(default_factory=dict)


def build_constraints(
    observation: np.ndarray, plan: Plan
) -> tuple[np.ndarray, np.ndarray, list[EventConstraint]]:
    """Build the A_ub x <= b_ub form of the joint system.

    Lower bounds ``S_i x >= l_i`` become ``-S_i x <= -l_i``; upper bounds
    ``S_i x <= u_i`` are appended directly.  Non-negativity of every marker
    is enforced through LP variable bounds, not through these rows.
    """
    matrix = plan.spillover
    tol = plan.absolute_tolerance
    sat = plan.saturation_value

    a_rows: list[np.ndarray] = []
    b_rows: list[float] = []
    described: list[EventConstraint] = []

    for i, detector in enumerate(plan.detectors):
        reading = float(observation[i])
        row = matrix[i, :].astype(float)
        if reading >= sat:
            lower = sat - tol
            a_rows.append(-row)
            b_rows.append(-lower)
            described.append(
                EventConstraint(
                    detector=detector,
                    detector_index=i,
                    observed=reading,
                    kind="saturation_lower",
                    lower=lower,
                    upper=None,
                    row=row.tolist(),
                )
            )
        else:
            lower = reading - tol
            upper = reading + tol
            a_rows.append(-row)
            b_rows.append(-lower)
            a_rows.append(row)
            b_rows.append(upper)
            described.append(
                EventConstraint(
                    detector=detector,
                    detector_index=i,
                    observed=reading,
                    kind="normal_band",
                    lower=lower,
                    upper=upper,
                    row=row.tolist(),
                )
            )

    return np.asarray(a_rows, dtype=float), np.asarray(b_rows, dtype=float), described


def _solve(a_ub: np.ndarray, b_ub: np.ndarray, n_markers: int) -> LPOutcome:
    """Feasibility LP with a zero objective (HiGHS returns any feasible x)."""
    result = linprog(
        c=np.zeros(n_markers),
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0.0, None)] * n_markers,
        method="highs",
    )
    if result.status == _LP_OPTIMAL:
        return LPOutcome(True, np.asarray(result.x, dtype=float), result.status, str(result.message))
    if result.status == _LP_INFEASIBLE:
        return LPOutcome(False, None, result.status, str(result.message))
    raise SolverError(f"LP backend status {result.status}: {result.message}")


def feasibility_lp(a_ub: np.ndarray, b_ub: np.ndarray, n_markers: int) -> LPOutcome:
    return _solve(a_ub, b_ub, n_markers)


def gate_rows(gate: Gate, markers: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Closed gate ``G`` as A_g x <= b_g rows, plus a serialisable description."""
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    description: list[dict] = []
    for marker, (lower, upper) in gate.bounds.items():
        j = markers.index(marker)
        unit = np.zeros(len(markers))
        unit[j] = 1.0
        if lower is not None:
            rows.append(-unit)
            rhs.append(-float(lower))
            description.append({"marker": marker, "side": "lower", "bound": float(lower)})
        if upper is not None:
            rows.append(unit)
            rhs.append(float(upper))
            description.append({"marker": marker, "side": "upper", "bound": float(upper)})
    if not rows:
        # A gate with no effective bounds is the whole space.
        return np.zeros((0, len(markers))), np.zeros(0), description
    return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float), description


def classify_gate(
    gate: Gate,
    plan: Plan,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    eps: float,
) -> GateDecision:
    """Decide gate membership against the whole joint feasible set."""
    n = len(plan.markers)
    g_ub, g_rhs, g_description = gate_rows(gate, plan.markers)

    # Inside witness: F intersect G.
    inside = _solve(np.vstack([a_ub, g_ub]), np.concatenate([b_ub, g_rhs]), n)
    # Outside witness: F with at least one gate row violated (v >= eps).
    outside = _outside_witness(a_ub, b_ub, g_ub, g_rhs, n, eps)

    inside_ok = inside.feasible
    outside_ok = outside.feasible

    if inside_ok and not outside_ok:
        verdict, reason = (
            "CERTAIN_IN",
            "inside witness exists; no joint-feasible vector can violate any closed gate bound "
            "(outside LP infeasible)",
        )
    elif not inside_ok and outside_ok:
        verdict, reason = (
            "CERTAIN_OUT",
            "no joint-feasible vector satisfies every closed gate bound (inside LP infeasible); "
            "outside witness exists",
        )
    elif inside_ok and outside_ok:
        verdict, reason = (
            "UNCERTAIN",
            "the joint feasible set crosses the gate boundary: both an inside and an outside "
            "joint-feasible witness exist",
        )
    else:
        raise SolverError(
            f"gate {gate.gate_id}: both inside and outside LPs are infeasible, "
            "which contradicts the event feasibility check"
        )

    decision = GateDecision(
        gate_id=gate.gate_id,
        verdict=verdict,
        reason=reason,
        inside_witness=inside.point.tolist() if inside.point is not None else None,
        outside_witness=outside.point.tolist() if outside.point is not None else None,
        detail={
            "effective_bounds": {
                marker: list(bounds) for marker, bounds in gate.bounds.items()
            },
            "gate_rows": g_description,
            "inside_lp_status": inside.status,
            "outside_lp_status": outside.status,
            "parent_chain": list(gate.chain),
        },
    )
    decision.inside_witness_check = check_witness(
        decision.inside_witness, a_ub, b_ub, g_ub, g_rhs, eps
    )
    decision.outside_witness_check = check_witness(
        decision.outside_witness, a_ub, b_ub, g_ub, g_rhs, eps
    )
    return decision


def _outside_witness(
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    g_ub: np.ndarray,
    g_rhs: np.ndarray,
    n: int,
    eps: float,
) -> LPOutcome:
    """Find x in F with some gate row violated by at least ``eps``.

    The disjunction (row k violated) is handled by one LP per gate row; a
    single vector from any successful LP is the outside witness.  This keeps
    the witness a genuine point of the original joint system instead of an
    auxiliary-variable artefact.
    """
    if g_ub.shape[0] == 0:
        # Gate is the whole space; there is no outside.
        return LPOutcome(False, None, _LP_INFEASIBLE, "gate has no bounds: outside is empty")

    failures: list[str] = []
    for k in range(g_ub.shape[0]):
        # Gate row k is  g_k . x <= b_k.  Violating it by eps means
        # g_k . x >= b_k + eps, i.e.  -g_k . x <= -b_k - eps.  Negating the
        # row (not tightening the original rhs) is essential: keeping the
        # original row would ask the point to satisfy the gate more strictly.
        rows = np.vstack([a_ub, -g_ub[k:k + 1, :]])
        rhs = np.concatenate([b_ub, np.array([-g_rhs[k] - eps])])
        outcome = _solve(rows, rhs, n)
        if outcome.feasible:
            return outcome
        failures.append(f"row {k} status {outcome.status}")
    return LPOutcome(
        False,
        None,
        _LP_INFEASIBLE,
        "every gate-side violation LP infeasible (" + "; ".join(failures) + ")",
    )


def check_witness(
    point: list[float] | None,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    g_ub: np.ndarray,
    g_rhs: np.ndarray,
    eps: float,
) -> dict:
    """Independent NumPy re-check of a witness (no LP solver involved).

    A witness is accepted only if one vector satisfies *all* rows of the
    joint observation system at once, is non-negative, and has the claimed
    inside/outside relationship to every gate row.  Residuals are reported so
    another program can substitute its own tolerance.
    """
    if point is None:
        return {"checked": False, "passes": False, "note": "no witness point"}
    x = np.asarray(point, dtype=float)
    nonneg_slack = x
    obs_slack = b_ub - a_ub @ x if a_ub.size else np.zeros(0)
    gate_slack = g_rhs - g_ub @ x if g_ub.size else np.zeros(0)

    nonneg_ok = bool((nonneg_slack >= -eps).all())
    obs_ok = bool((obs_slack >= -eps).all()) if obs_slack.size else True
    # Closed gate: inside means every row satisfied, with eps absorbing
    # solver round-off; outside means a row is strictly beyond the closed
    # boundary.  Because the outside LP enforces g_k x >= b_k + eps, a
    # genuine outside witness has slack <= -eps; points with
    # -eps < slack <= 0 merely sit on the boundary and count as inside.
    inside_gate_ok = bool((gate_slack >= -eps).all()) if gate_slack.size else True
    outside_gate_ok = bool((gate_slack <= -eps).any()) if gate_slack.size else False
    boundary_only = (
        bool((gate_slack < 0).any()) and not outside_gate_ok
    ) if gate_slack.size else False

    return {
        "checked": True,
        "tolerance": eps,
        "passes_nonnegativity": nonneg_ok,
        "passes_joint_observation_constraints": obs_ok,
        "inside_gate_all_rows_satisfied": inside_gate_ok,
        "outside_gate_some_row_violated": outside_gate_ok,
        "boundary_only_not_strictly_outside": boundary_only,
        "passes_as_inside_witness": nonneg_ok and obs_ok and inside_gate_ok,
        "passes_as_outside_witness": nonneg_ok and obs_ok and outside_gate_ok,
        "min_nonnegativity_slack": float(np.min(nonneg_slack)),
        "min_observation_slack": float(np.min(obs_slack)) if obs_slack.size else None,
        "gate_row_slacks": [float(v) for v in gate_slack],
    }


def diagnose_infeasibility(
    a_ub: np.ndarray, b_ub: np.ndarray, described: list[EventConstraint], plan: Plan
) -> str:
    """Produce a human-readable hint for why the joint system has no x >= 0."""
    negative_band = [c.detector for c in described if c.kind == "normal_band" and c.upper is not None and c.upper < 0]
    if negative_band:
        return (
            "physical infeasibility: negative detector reading on "
            f"{negative_band} whose upper band edge is below zero while y = S x with "
            "S >= 0 and x >= 0 forbids negative detector values"
        )
    # Check whether dropping the non-negativity requirement helps: if the
    # unconstrained band intersection is empty it is a band contradiction.
    free = linprog(
        c=np.zeros(len(plan.markers)),
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * len(plan.markers),
        method="highs",
    )
    if free.status == _LP_INFEASIBLE:
        return "physical infeasibility: the detector observation bands have no joint intersection under y = S x"
    return (
        "physical infeasibility: band intersection exists only for marker vectors with "
        "negative components; x >= 0 makes the joint system empty"
    )
