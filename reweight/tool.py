"""Bounded entropy re-weighting for nonresponse adjustment.

Public API
----------
load_inputs            read respondents / margins / replicate weights / plan
build_constraint_matrix exact 0/1 margin indicator rows
check_feasibility      structural, exact-rational, and LP feasibility pre-check
fit                    bounded calibration (entropy distance) for one weight vector
estimate               weighted domain means from a fitted weight vector
replicate_estimates    re-solve every supplied design replicate
replicate_variance     plan-coefficient design-replicate variance
verify                 independent recomputation of constraints/bounds/optimality
render_report          static text report

Design notes
------------
* The objective is the plan's entropy distance
  ``sum_i [w_i log(w_i/d_i) - w_i + d_i]`` with linear margins and box bounds
  ``[relative_lower*d_i, relative_upper*d_i]``; ``d_i == 0`` forces ``w_i == 0``.
* Constraints may be redundant (overlapping population margins).  They are
  *not* dropped silently: each redundant total is expressed as an exact
  rational combination of an independent basis and checked for consistency.
* Everything is deterministic and input-order invariant: respondents and
  constraints are canonically sorted on load.
* No sampling, imputation, or nonrespondent score synthesis is performed.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Numerical policy (published in every report)
# ---------------------------------------------------------------------------
# Constraint tolerances are *relative to the supplied population totals*:
#   constraint tol_i = REL_TOL * max(|b_i|, 1)
# Bound tolerances are *relative to each respondent's upper bound*:
#   bound tol_i = BOUND_REL_TOL * max(relative_upper*d_i, 1)
# OPT_TOL gates dual-gradient and interior-KKT stationarity.
REL_TOL = 1.0e-10
BOUND_REL_TOL = 1.0e-10
OPT_TOL = 1.0e-9
MAX_NEWTON_ITERS = 200
LP_FEAS_TOL = 1.0e-9        # HiGHS primal feasibility tolerance for pre-check


def constraint_tolerance(rhs: float) -> float:
    return REL_TOL * max(abs(rhs), 1.0)


def _to_fraction(text: str) -> Fraction:
    return Fraction(str(text).strip())


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Respondent:
    respondent_id: str
    groups: dict[str, str]
    score: float
    base_weight: float


@dataclass
class Margin:
    constraint_id: str
    field: str
    value: str
    total: Fraction
    raw_total: str

    @property
    def total_float(self) -> float:
        return float(self.total)


@dataclass
class Plan:
    relative_lower: float
    relative_upper: float
    domains: list[dict]
    variance_coefficient: float
    replicate_ids: list[str]
    nonresponse_assumption: str


@dataclass
class Inputs:
    respondents: list[Respondent]
    margins: list[Margin]
    plan: Plan
    replicates: dict[str, dict[str, float]]
    replicate_ids: list[str]
    replicate_source: str


# ---------------------------------------------------------------------------
# Loading (canonical ordering -> row/constraint order invariance)
# ---------------------------------------------------------------------------
def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _parse_variance_coefficient(text: str, n_rep: int) -> float:
    m = re.match(r"\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)", text)
    if m:
        return float(Fraction(int(m.group(1)), int(m.group(2))))
    m = re.match(r"\s*(\d+)\s*/\s*(\d+)", text)
    if m:
        return float(Fraction(int(m.group(1)), int(m.group(2))))
    return 1.0 / n_rep if n_rep else float("nan")


def load_inputs(respondents_path: str, margins_path: str, plan_path: str,
                replicate_path: str) -> Inputs:
    resp_rows = _read_csv(respondents_path)
    if not resp_rows:
        raise ValueError("respondents file is empty")
    header = list(resp_rows[0].keys())
    for needed in ("respondent_id", "score", "base_weight"):
        if needed not in header:
            raise ValueError(f"respondents file missing column {needed!r}")
    group_fields = [c for c in header
                    if c not in ("respondent_id", "score", "base_weight")]
    respondents: list[Respondent] = []
    seen: set[str] = set()
    for raw in resp_rows:
        rid = raw["respondent_id"].strip()
        if rid in seen:
            raise ValueError(f"duplicate respondent_id {rid!r}")
        seen.add(rid)
        bw = _to_fraction(raw["base_weight"])
        if bw < 0:
            raise ValueError(f"negative base weight for {rid!r}")
        # Scores may be integer, decimal, or rational ("3/2") per the spec.
        score = _to_fraction(raw["score"])
        respondents.append(Respondent(
            respondent_id=rid,
            groups={g: raw[g].strip() for g in group_fields},
            score=float(score),
            base_weight=float(bw),
        ))

    margins: list[Margin] = []
    seen_cid: set[str] = set()
    for raw in _read_csv(margins_path):
        cid = raw["constraint_id"].strip()
        if cid in seen_cid:
            raise ValueError(f"duplicate constraint_id {cid!r}")
        seen_cid.add(cid)
        total = _to_fraction(raw["population_total"])
        if total < 0:
            raise ValueError(f"negative margin total for {cid!r}")
        margins.append(Margin(cid, raw["field"].strip(), raw["value"].strip(),
                              total, raw["population_total"].strip()))

    plan_raw = json.load(open(plan_path, encoding="utf-8"))
    for key in ("relative_lower", "relative_upper", "domains",
                "replicate_variance"):
        if key not in plan_raw:
            raise ValueError(f"plan is missing required key {key!r}")
    lower = float(plan_raw["relative_lower"])
    upper = float(plan_raw["relative_upper"])
    if not (np.isfinite(lower) and np.isfinite(upper)):
        raise ValueError("plan relative bounds must be finite")
    if lower < 0 or upper <= 0 or lower > upper:
        raise ValueError("plan requires 0 <= relative_lower <= relative_upper")
    repl_raw = _read_csv(replicate_path)
    if not repl_raw:
        raise ValueError("replicate weights file is empty")
    rep_header = list(repl_raw[0].keys())
    if "respondent_id" not in rep_header:
        raise ValueError("replicate weights file missing respondent_id")
    replicate_ids = sorted(c for c in rep_header if c != "respondent_id")
    replicates: dict[str, dict[str, float]] = {}
    for raw in repl_raw:
        rid = raw["respondent_id"].strip()
        replicates[rid] = {r: float(raw[r]) for r in replicate_ids}
    resp_ids = {r.respondent_id for r in respondents}
    if set(replicates) != resp_ids:
        raise ValueError(
            "replicate weights must cover exactly the respondent IDs; "
            f"missing={sorted(resp_ids - set(replicates))}, "
            f"extra={sorted(set(replicates) - resp_ids)}")
    for rep in replicate_ids:
        for r in respondents:
            v = replicates[r.respondent_id][rep]
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"replicate {rep!r} has invalid base weight "
                                 f"{v} for {r.respondent_id!r}")

    plan = Plan(
        relative_lower=lower,
        relative_upper=upper,
        domains=list(plan_raw["domains"]),
        variance_coefficient=_parse_variance_coefficient(
            plan_raw["replicate_variance"], len(replicate_ids)),
        replicate_ids=replicate_ids,
        nonresponse_assumption=plan_raw.get("nonresponse_assumption", ""),
    )

    # Canonical order: guarantees row/constraint order cannot change results.
    respondents.sort(key=lambda r: r.respondent_id)
    margins.sort(key=lambda m: m.constraint_id)
    return Inputs(respondents, margins, plan, replicates, replicate_ids,
                  replicate_path)


# ---------------------------------------------------------------------------
# Constraint matrix and exact algebra
# ---------------------------------------------------------------------------
def build_constraint_matrix(respondents: Sequence[Respondent],
                            margins: Sequence[Margin]) -> np.ndarray:
    rows = []
    known = respondents[0].groups if respondents else {}
    for m in margins:
        if m.field == "*":
            rows.append(np.ones(len(respondents), dtype=float))
            continue
        if m.field not in known:
            raise ValueError(f"margin {m.constraint_id!r} references unknown "
                             f"field {m.field!r}")
        rows.append(np.array([1.0 if r.groups[m.field] == m.value else 0.0
                              for r in respondents], dtype=float))
    return np.vstack(rows) if rows else np.zeros((0, len(respondents)))


def empty_rows(A: np.ndarray) -> list[int]:
    return [i for i in range(A.shape[0]) if not np.any(A[i] != 0)]


def _exact_rank(rows: Sequence[Sequence[Fraction]]) -> int:
    mat = [list(r) for r in rows]
    if not mat:
        return 0
    nrow, ncol = len(mat), len(mat[0])
    rank = 0
    for col in range(ncol):
        pivot = next((r for r in range(rank, nrow) if mat[r][col] != 0), None)
        if pivot is None:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        pv = mat[rank][col]
        mat[rank] = [x / pv for x in mat[rank]]
        for r in range(nrow):
            if r != rank and mat[r][col] != 0:
                fac = mat[r][col]
                mat[r] = [x - fac * y for x, y in zip(mat[r], mat[rank])]
        rank += 1
        if rank == nrow:
            break
    return rank


def independent_rows(A: np.ndarray) -> list[int]:
    """Order-invariant exact rational independent-row selection."""
    exact = [[Fraction(int(round(x))) for x in row] for row in A]
    chosen: list[int] = []
    for i, row in enumerate(exact):
        if _exact_rank([exact[j] for j in chosen] + [row]) == len(chosen) + 1:
            chosen.append(i)
    return chosen


def _exact_combination(basis: Sequence[Sequence[Fraction]],
                       target: Sequence[Fraction]) -> list[Fraction]:
    """Solve basis^T coef = target exactly (basis rows are independent)."""
    k = len(basis)
    n = len(target)
    # Augmented matrix: columns = basis rows + target, one equation per entry.
    aug = [[basis[j][i] for j in range(k)] + [target[i]] for i in range(n)]
    rank = 0
    for col in range(k):
        pivot = next((r for r in range(rank, n) if aug[r][col] != 0), None)
        if pivot is None:
            continue
        aug[rank], aug[pivot] = aug[pivot], aug[rank]
        pv = aug[rank][col]
        aug[rank] = [x / pv for x in aug[rank]]
        for r in range(n):
            if r != rank and aug[r][col] != 0:
                fac = aug[r][col]
                aug[r] = [x - fac * y for x, y in zip(aug[r], aug[rank])]
        rank += 1
    if rank != k:
        raise ValueError("basis rows are not independent")
    # RREF rows 0..k-1 now have leading 1 in col j; RHS is coef[j].
    return [aug[j][k] for j in range(k)]


@dataclass
class DependencyCertificate:
    dependent_id: str
    independent_ids: list[str]
    coefficients: list[Fraction]
    implied_total: Fraction
    stated_total: Fraction

    @property
    def consistent(self) -> bool:
        return self.implied_total == self.stated_total

    @property
    def gap(self) -> Fraction:
        return self.stated_total - self.implied_total


def dependency_certificates(inputs: Inputs, A: np.ndarray,
                             chosen: Sequence[int]) -> list[DependencyCertificate]:
    """Express every redundant margin exactly in the independent basis."""
    exact = [[Fraction(int(round(x))) for x in row] for row in A]
    basis = [exact[j] for j in chosen]
    certs: list[DependencyCertificate] = []
    for i, row in enumerate(exact):
        if i in chosen:
            continue
        coef = _exact_combination(basis, row)
        implied = sum(c * inputs.margins[j].total for c, j in zip(coef, chosen))
        certs.append(DependencyCertificate(
            dependent_id=inputs.margins[i].constraint_id,
            independent_ids=[inputs.margins[j].constraint_id for j in chosen],
            coefficients=coef,
            implied_total=implied,
            stated_total=inputs.margins[i].total,
        ))
    return certs


# ---------------------------------------------------------------------------
# Feasibility pre-check
# ---------------------------------------------------------------------------
@dataclass
class Feasibility:
    feasible: bool
    status: str
    messages: list[str]
    evidence: list[dict]
    independent: list[int]
    certificates: list[DependencyCertificate]
    conflict_set: list[str] = field(default_factory=list)


def _lp_feasible(Ai: np.ndarray, bi: np.ndarray, lo: np.ndarray,
                 hi: np.ndarray) -> bool:
    from scipy.optimize import linprog
    if len(bi) == 0:
        return True
    bounds = list(zip(lo.tolist(), hi.tolist()))
    res = linprog(np.zeros(Ai.shape[1]), A_eq=Ai, b_eq=bi, bounds=bounds,
                  method="highs",
                  options={"primal_feasibility_tolerance": LP_FEAS_TOL,
                           "dual_feasibility_tolerance": LP_FEAS_TOL,
                           "presolve": True})
    return bool(res.status == 0)


def _minimal_conflict_set(Ai: np.ndarray, bi: np.ndarray, lo: np.ndarray,
                          hi: np.ndarray) -> list[int]:
    """Deletion filter: smallest row subset still infeasible on the box."""
    keep = list(range(len(bi)))
    if _lp_feasible(Ai, bi, lo, hi):
        return []
    i = 0
    while i < len(keep):
        trial = keep[:i] + keep[i + 1:]
        if trial and not _lp_feasible(Ai[trial], bi[trial], lo, hi):
            keep = trial
        else:
            i += 1
    return keep


def farkas_certificate(Ai: np.ndarray, bi: np.ndarray, lo: np.ndarray,
                       hi: np.ndarray) -> dict | None:
    """Explicit, hand-checkable infeasibility certificate.

    Find multipliers y and per-variable bounds t_j so that y·b lies outside
    [sum_j bounds(yᵀA)_j].  Concretely solve the LP

        max  y·b − Σ_j t_j
        s.t. t_j >= w_j·lo_j,  t_j >= w_j·hi_j,  w = Aᵀy,  −1 <= y <= 1

    A positive optimum proves that for *every* box-feasible w the combination
    y·(A w) cannot equal y·b.  Returns None when no such certificate is found
    (in which case the LP must have been feasible).
    """
    from scipy.optimize import linprog
    m, n = Ai.shape
    if m == 0:
        return None
    # variables: y (m), t (n)
    c = np.zeros(m + n)
    c[:m] = -bi            # minimise -(y·b) + sum t
    c[m:] = 1.0
    A_ub, b_ub = [], []
    # t_j - w_j lo_j >= 0  ->  -t_j + lo_j * (A^T y)_j <= 0
    for j in range(n):
        wj = Ai[:, j]
        row = np.zeros(m + n)
        row[:m] = lo[j] * wj
        row[m + j] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
        row = np.zeros(m + n)
        row[:m] = hi[j] * wj
        row[m + j] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
    bounds = [(-1.0, 1.0)] * m + [(None, None)] * n
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds,
                  method="highs")
    if res.status != 0:
        return None
    value = -res.fun
    if value <= 1e-9 * max(1.0, float(np.max(np.abs(bi)))):
        return None
    y = res.x[:m]
    # Snap near-integer / simple-rational multipliers for the human reader.
    y_show = [Fraction(float(v)).limit_denominator(1000) for v in y]
    w = Ai.T @ y
    lower_possible = float(np.sum(np.where(w > 0, w * lo, w * hi)))
    upper_possible = float(np.sum(np.where(w > 0, w * hi, w * lo)))
    return {
        "type": "farkas_combination",
        "multipliers": [str(v) for v in y_show],
        "combination_target": float(y @ bi),
        "box_min": lower_possible,
        "box_max": upper_possible,
        "violation": value,
    }


def check_feasibility(inputs: Inputs, A: np.ndarray, d: np.ndarray,
                      lower: float, upper: float) -> Feasibility:
    msgs: list[str] = []
    evidence: list[dict] = []
    lo = lower * d
    hi = upper * d
    chosen = independent_rows(A)
    certs = dependency_certificates(inputs, A, chosen)
    feasible = True

    # 1. structural empty cells
    for i in empty_rows(A):
        b = inputs.margins[i].total
        if b != 0:
            feasible = False
            msgs.append(f"structural empty cell: margin "
                        f"{inputs.margins[i].constraint_id!r} selects no "
                        f"respondents but demands total {b}")
            evidence.append({"type": "structural_empty_cell",
                             "constraint_id": inputs.margins[i].constraint_id,
                             "target": str(b)})

    # 2. exact redundancy consistency
    for cert in certs:
        if not cert.consistent:
            feasible = False
            msgs.append(
                f"redundant margin {cert.dependent_id!r} inconsistent: stated "
                f"{cert.stated_total} != exact implied {cert.implied_total} "
                f"(gap {cert.gap})")
            evidence.append({"type": "dependency_gap",
                             "constraint_id": cert.dependent_id,
                             "stated": str(cert.stated_total),
                             "implied": str(cert.implied_total)})

    # 3. per-row box reachability (cheap, human-checkable evidence)
    row_conflicts: list[str] = []
    for i in chosen:
        b = inputs.margins[i].total_float
        lb, ub = float(A[i] @ lo), float(A[i] @ hi)
        if lb > b + constraint_tolerance(b):
            feasible = False
            row_conflicts.append(inputs.margins[i].constraint_id)
            msgs.append(f"margin {inputs.margins[i].constraint_id!r} "
                        f"unreachable: box minimum {lb:.6f} > target {b:.6f}")
            evidence.append({"type": "box_lower_violation",
                             "constraint_id": inputs.margins[i].constraint_id,
                             "minimum": lb, "target": b})
        if ub < b - constraint_tolerance(b):
            feasible = False
            row_conflicts.append(inputs.margins[i].constraint_id)
            msgs.append(f"margin {inputs.margins[i].constraint_id!r} "
                        f"unreachable: box maximum {ub:.6f} < target {b:.6f}")
            evidence.append({"type": "box_upper_violation",
                             "constraint_id": inputs.margins[i].constraint_id,
                             "maximum": ub, "target": b})

    # 4. joint (overlapping) feasibility via LP, with a minimal conflict set
    conflict_ids: list[str] = list(dict.fromkeys(row_conflicts))
    b_all = np.array([m.total_float for m in inputs.margins])
    if feasible and not _lp_feasible(A, b_all, lo, hi):
        feasible = False
        subset = _minimal_conflict_set(A, b_all, lo, hi)
        conflict_ids = [inputs.margins[i].constraint_id for i in subset]
        msgs.append("overlapping margins are jointly unreachable; minimal "
                    f"infeasible subset: {conflict_ids}")
        for i in subset:
            b = inputs.margins[i].total_float
            evidence.append({
                "type": "box_range",
                "constraint_id": inputs.margins[i].constraint_id,
                "target": b,
                "minimum": float(A[i] @ lo),
                "maximum": float(A[i] @ hi)})
        # Multiplier certificate over the *minimal* conflicting subset.
        subset_A = A[subset]
        subset_b = np.array([inputs.margins[i].total_float for i in subset])
        cert = farkas_certificate(subset_A, subset_b, lo, hi)
        if cert is not None:
            cert["constraint_ids"] = [inputs.margins[i].constraint_id
                                      for i in subset]
            cert["statement"] = (
                "no w in the box can satisfy the combination "
                + " + ".join(f"({c})*[{cid}]" for c, cid in
                             zip(cert["multipliers"], cert["constraint_ids"]))
                + f": target {cert['combination_target']:.6f} outside "
                f"achievable [{cert['box_min']:.6f}, {cert['box_max']:.6f}]")
            evidence.append(cert)
            msgs.append(cert["statement"])

    status = "feasible" if feasible else "infeasible"
    if feasible:
        msgs.append("pre-check passed: structural cells non-empty, redundant "
                    "totals exact-consistent, box and LP jointly feasible")
    return Feasibility(feasible, status, msgs, evidence, chosen, certs,
                       conflict_ids)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    weights: np.ndarray
    objective: float
    status: str            # feasible | infeasible | failed
    iterations: int
    max_residual: float
    max_margin_tol: float
    max_kkt: float
    kkt_lower_min: float
    kkt_upper_max: float
    optimality_ok: bool
    dual: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    targets: np.ndarray
    residuals: np.ndarray
    residual_tols: np.ndarray
    at_lower: np.ndarray
    at_upper: np.ndarray
    independent: list[int]
    diagnostics: list[str]
    solver_converged: bool
    tolerance: dict


def _newton_dual(Ai, bi, d, lo, hi, lam):
    """Damped Newton on the smooth (C^1) dual.  Returns lam and status.

    The dual Hessian is ``A diag(w) A^T`` clipped to interior weights; at box
    kinks the second derivative drops to zero, so this fast path can stall
    close to a bound.  Callers fall back to ``_lbfgs_dual`` when it does.
    """
    m = len(bi)
    scale_b = max(float(np.max(np.abs(bi))) if m else 1.0, 1.0)
    grad_stop = 1e-13 * scale_b
    it = 0
    gnorm = float("inf")
    for it in range(1, MAX_NEWTON_ITERS + 1):
        _, grad, w, curv = _dual_eval(lam, Ai, bi, d, lo, hi)
        gnorm = float(np.max(np.abs(grad))) if m else 0.0
        if gnorm <= grad_stop:
            return lam, True, it, gnorm, []
        hess = Ai @ (curv[:, None] * Ai.T)
        ridge = 1e-13 * max(1.0, float(np.max(np.abs(hess)))) if m else 0.0
        hess = hess + ridge * np.eye(m)
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(hess, grad, rcond=None)[0]
        directional = float(grad @ delta)
        if not np.isfinite(directional) or directional <= 0:
            delta = grad
            directional = float(grad @ grad)
        t = 1.0
        f0 = _dual_eval(lam, Ai, bi, d, lo, hi)[0]
        accepted = False
        while t > 1e-14:
            f1 = _dual_eval(lam + t * delta, Ai, bi, d, lo, hi)[0]
            if np.isfinite(f1) and f1 >= f0 + 1e-4 * t * directional:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            return lam, False, it, gnorm, ["Newton line search stalled"]
        lam = lam + t * delta
    return lam, False, it, gnorm, ["maximum Newton iterations reached"]


def _lbfgs_dual(Ai, bi, d, lo, hi, lam0):
    """Fallback quasi-Newton solve for the dual near box kinks."""
    from scipy.optimize import minimize
    m = len(bi)

    def neg(lam):
        return -_dual_eval(lam, Ai, bi, d, lo, hi)[0]

    def neg_grad(lam):
        return -_dual_eval(lam, Ai, bi, d, lo, hi)[1]

    res = minimize(neg, lam0, jac=neg_grad, method="L-BFGS-B",
                   options={"ftol": 1e-16, "gtol": 1e-14, "maxiter": 5000,
                            "maxfun": 20000})
    _, grad, _, _ = _dual_eval(res.x, Ai, bi, d, lo, hi)
    gnorm = float(np.max(np.abs(grad))) if m else 0.0
    return res.x, bool(res.success), int(res.nit), gnorm, \
        ([] if res.success else [f"L-BFGS-B: {res.message}"])


def _dual_eval(lam, Ai, bi, d, lo, hi):
    c = Ai.T @ lam
    z = d * np.exp(np.clip(c, -700.0, 700.0))
    w = np.clip(z, lo, hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_d = np.where(d > 0, d, 1.0)
        safe_w = np.where(w > 0, w, 1.0)
        f = np.where(d > 0, w * np.log(safe_w / safe_d) - w + d, 0.0)
    grad = bi - Ai @ w
    active = (w > lo + 1e-15 * np.maximum(hi, 1.0)) & \
             (w < hi - 1e-15 * np.maximum(hi, 1.0)) & (d > 0)
    curv = np.where(active, w, 0.0)
    return float(bi @ lam + np.sum(f - c * w)), grad, w, curv


def _entropy_objective(w, d):
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sum(np.where(
            d > 0, w * np.log(np.where(w > 0, w, 1) /
                              np.where(d > 0, d, 1)) - w + d, 0.0)))


def fit(inputs: Inputs, A: np.ndarray, d: np.ndarray, *,
        lower: float | None = None,
        upper: float | None = None) -> FitResult:
    lower = inputs.plan.relative_lower if lower is None else lower
    upper = inputs.plan.relative_upper if upper is None else upper
    d = np.asarray(d, dtype=float)
    targets = np.array([m.total_float for m in inputs.margins])
    tol_arr = np.array([constraint_tolerance(b) for b in targets])
    lo, hi = lower * d, upper * d
    diagnostics: list[str] = []
    tolerance = {"rel_tol": REL_TOL, "bound_rel_tol": BOUND_REL_TOL,
                 "opt_tol": OPT_TOL, "lp_feas_tol": LP_FEAS_TOL,
                 "constraint_tol_policy":
                     "REL_TOL*max(|target|,1) per margin",
                 "bound_tol_policy":
                     "BOUND_REL_TOL*max(relative_upper*base,1) per respondent"}

    def _empty(status: str, msgs: list[str], chosen: list[int]) -> FitResult:
        return FitResult(np.zeros_like(d), float("nan"), status, 0,
                         float("nan"), float("nan"), float("nan"),
                         float("nan"), float("nan"), False,
                         np.zeros(len(A)), lo, hi, targets,
                         np.full(len(targets), np.nan), tol_arr,
                         np.zeros(d.shape, bool), np.zeros(d.shape, bool),
                         chosen, msgs, False, tolerance)

    if d.size == 0:
        return _empty("failed", ["no respondents"], [])
    if not (np.all(np.isfinite(d)) and np.all(d >= 0)):
        return _empty("failed", ["base weights must be finite and >= 0"], [])

    feas = check_feasibility(inputs, A, d, lower, upper)
    if not feas.feasible:
        return _empty("infeasible", feas.messages, feas.independent)

    chosen = feas.independent
    Ai, bi = A[chosen], targets[chosen]
    m = len(chosen)

    lam = np.zeros(m)
    lam, converged, it, gnorm, msgs = _newton_dual(Ai, bi, d, lo, hi, lam)
    diagnostics.extend(msgs)
    # Fast path stalled near a bound kink?  Retry with a quasi-Newton dual.
    if not converged:
        lam2, conv2, it2, gn2, msgs2 = _lbfgs_dual(Ai, bi, d, lo, hi, lam)
        it += it2
        if conv2 and gn2 <= gnorm:
            lam, converged, gnorm = lam2, conv2, gn2
            diagnostics.append("Newton stalled; recovered with L-BFGS-B dual")
        else:
            diagnostics.extend(msgs2)
            if gn2 < gnorm:
                lam, gnorm = lam2, gn2

    _, _, w, _ = _dual_eval(lam, Ai, bi, d, lo, hi)
    residuals = A @ w - targets
    max_residual = float(np.max(np.abs(residuals))) if len(targets) else 0.0
    max_margin_tol = float(np.max(tol_arr)) if len(tol_arr) else 0.0

    full_lam = np.zeros(len(A))
    full_lam[chosen] = lam
    c = A.T @ full_lam
    with np.errstate(divide="ignore", invalid="ignore"):
        kkt = np.where(d > 0, np.log(np.where(w > 0, w, 1) /
                                     np.where(d > 0, d, 1)) - c, 0.0)
    bound_tol = BOUND_REL_TOL * np.maximum(hi, 1.0)
    at_lower = (d > 0) & (np.abs(w - lo) <= bound_tol)
    at_upper = (d > 0) & (np.abs(w - hi) <= bound_tol)
    interior = (d > 0) & ~at_lower & ~at_upper
    max_kkt = float(np.max(np.abs(kkt[interior]))) if np.any(interior) else 0.0
    kkt_lower_min = float(np.min(kkt[at_lower])) if np.any(at_lower) else 0.0
    kkt_upper_max = float(np.max(kkt[at_upper])) if np.any(at_upper) else 0.0
    optimality_ok = (max_kkt <= OPT_TOL and kkt_lower_min >= -OPT_TOL
                     and kkt_upper_max <= OPT_TOL)

    residual_ok = bool(np.all(np.abs(residuals) <= tol_arr + 1e-14)) \
        if len(residuals) else True
    bound_violation = 0.0
    if d.size:
        bound_violation = float(max(np.max(lo - w), np.max(w - hi), 0.0))
    bounds_ok = bound_violation <= float(np.max(bound_tol))
    objective = _entropy_objective(w, d)

    if not converged:
        diagnostics.append(f"optimiser convergence flag not set "
                           f"(final max|dual gradient|={gnorm:.3e})")
    if not residual_ok:
        diagnostics.append(f"margin residual {max_residual:.3e} exceeds "
                           f"per-constraint tolerance {max_margin_tol:.3e}")
    if not optimality_ok:
        diagnostics.append(
            f"optimality check failed: interior|KKT|={max_kkt:.3e}, "
            f"lower-min={kkt_lower_min:.3e}, upper-max={kkt_upper_max:.3e}")
    if not bounds_ok:
        diagnostics.append(f"bound violation {bound_violation:.3e}")

    # For this smooth convex program, primal residual + KKT sign/magnitude +
    # bound checks are a complete optimality certificate.  Acceptance never
    # loosens those checks; the optimiser flag is recorded separately.
    ok = residual_ok and optimality_ok and bounds_ok
    status = "feasible" if ok else "failed"
    if ok and not converged:
        diagnostics.append("optimiser flag not converged, but residual/KKT/"
                           "bound checks certify optimality at published "
                           "tolerances")
    solver_converged = converged
    return FitResult(
        weights=w, objective=objective, status=status, iterations=it,
        max_residual=max_residual, max_margin_tol=max_margin_tol,
        max_kkt=max_kkt, kkt_lower_min=kkt_lower_min,
        kkt_upper_max=kkt_upper_max, optimality_ok=optimality_ok,
        dual=full_lam, lower=lo, upper=hi, targets=targets,
        residuals=residuals, residual_tols=tol_arr,
        at_lower=at_lower, at_upper=at_upper, independent=chosen,
        diagnostics=diagnostics, solver_converged=solver_converged,
        tolerance=tolerance)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def domain_mask(respondents: Sequence[Respondent], dom: dict) -> np.ndarray:
    if dom["field"] == "*":
        return np.ones(len(respondents), dtype=bool)
    if respondents and dom["field"] not in respondents[0].groups:
        raise ValueError(f"domain {dom['id']!r} references unknown field "
                         f"{dom['field']!r}")
    return np.array([r.groups.get(dom["field"]) == dom["value"]
                     for r in respondents], dtype=bool)


def estimate(inputs: Inputs, weights: np.ndarray) -> dict[str, float]:
    scores = np.array([r.score for r in inputs.respondents], dtype=float)
    out: dict[str, float] = {}
    for dom in inputs.plan.domains:
        mask = domain_mask(inputs.respondents, dom)
        wsum = float(weights[mask].sum()) if np.any(mask) else 0.0
        out[dom["id"]] = (float(weights[mask] @ scores[mask]) / wsum
                          if wsum > 0 else float("nan"))
    return out


# ---------------------------------------------------------------------------
# Design replicates
# ---------------------------------------------------------------------------
@dataclass
class ReplicateResult:
    replicate_id: str
    status: str
    weights: np.ndarray | None
    estimates: dict[str, float]
    diagnostics: list[str]
    fit: FitResult | None


def replicate_estimates(inputs: Inputs, A: np.ndarray) -> list[ReplicateResult]:
    out: list[ReplicateResult] = []
    base_ids = [r.respondent_id for r in inputs.respondents]
    for rep in inputs.replicate_ids:
        d_rep = np.array([inputs.replicates[i][rep] for i in base_ids],
                         dtype=float)
        zero = int(np.sum(d_rep <= 0))
        fr = fit(inputs, A, d_rep)
        diag = list(fr.diagnostics)
        if zero:
            diag.insert(0, f"{zero} respondents have zero replicate base weight "
                           f"(forced to zero adjusted weight)")
        if fr.status == "feasible":
            out.append(ReplicateResult(rep, fr.status, fr.weights,
                                       estimate(inputs, fr.weights), diag, fr))
        else:
            out.append(ReplicateResult(rep, fr.status, None, {}, diag, fr))
    return out


@dataclass
class VarianceResult:
    estimates: dict[str, float]
    replicate_estimates: dict[str, dict[str, float]]
    replicate_status: dict[str, str]
    variances: dict[str, float]
    standard_errors: dict[str, float]
    complete: dict[str, bool]
    n_used: dict[str, int]
    n_total: int
    coefficient: float
    notes: list[str]


def replicate_variance(inputs: Inputs, full_estimates: dict[str, float],
                       reps: Sequence[ReplicateResult]) -> VarianceResult:
    coef = inputs.plan.variance_coefficient
    n_total = len(reps)
    rep_estimates: dict[str, dict[str, float]] = {}
    rep_status: dict[str, str] = {}
    for rr in reps:
        rep_status[rr.replicate_id] = rr.status
        if rr.status == "feasible":
            rep_estimates[rr.replicate_id] = rr.estimates
    variances, ses, complete, n_used, notes = {}, {}, {}, {}, []
    for dom in inputs.plan.domains:
        did = dom["id"]
        vals = [rep_estimates[r][did] for r in rep_estimates
                if did in rep_estimates[r] and np.isfinite(rep_estimates[r][did])]
        used = len(vals)
        n_used[did] = used
        full = full_estimates.get(did, float("nan"))
        if used == 0 or not np.isfinite(full):
            variances[did] = float("nan")
            ses[did] = float("nan")
            complete[did] = False
            notes.append(f"domain {did!r}: no usable replicates; SE incomplete")
            continue
        var = coef * sum((v - full) ** 2 for v in vals)
        variances[did] = var
        ses[did] = float(np.sqrt(max(var, 0.0)))
        complete[did] = used == n_total
        if used < n_total:
            notes.append(
                f"domain {did!r}: {n_total - used}/{n_total} replicates failed; "
                f"SE uses {used} terms with the plan's fixed coefficient and is "
                f"marked INCOMPLETE")
    return VarianceResult(full_estimates, rep_estimates, rep_status, variances,
                          ses, complete, n_used, n_total, coef, notes)


# ---------------------------------------------------------------------------
# Independent verification
# ---------------------------------------------------------------------------
@dataclass
class VerifyReport:
    ok: bool
    constraint_checks: list[dict]
    bound_checks: list[dict]
    zero_checks: list[dict]
    total_check: dict
    objective_check: dict
    optimality_check: dict
    recalc_variance: dict
    replicate_checks: list[dict]
    problems: list[str]


def verify(inputs: Inputs, A: np.ndarray, d: np.ndarray, result: FitResult,
           reps: Sequence[ReplicateResult] | None = None,
           var: VarianceResult | None = None) -> VerifyReport:
    """Recompute everything from the returned weights; never calls the solver,
    ``estimate`` or ``replicate_variance`` for the values it checks."""
    problems: list[str] = []
    w = np.asarray(result.weights, dtype=float)
    targets = np.array([m.total_float for m in inputs.margins])
    residuals = A @ w - targets
    tols = np.array([constraint_tolerance(b) for b in targets])

    constraint_checks = []
    for i, m in enumerate(inputs.margins):
        ok = abs(float(residuals[i])) <= float(tols[i])
        constraint_checks.append({
            "constraint_id": m.constraint_id, "field": m.field,
            "value": m.value, "lhs": float(A[i] @ w),
            "rhs": m.total_float, "residual": float(residuals[i]),
            "tol": float(tols[i]), "ok": bool(ok),
            "independent": i in result.independent})
        if not ok:
            problems.append(f"constraint {m.constraint_id!r} residual "
                            f"{residuals[i]:.3e} exceeds tol {tols[i]:.3e}")

    bound_checks = []
    # Recompute expected bounds from the plan and base weights, so the check
    # does not trust the bound arrays produced by the solver.
    expected_lower = inputs.plan.relative_lower * np.asarray(d, dtype=float)
    expected_upper = inputs.plan.relative_upper * np.asarray(d, dtype=float)
    bound_tols = BOUND_REL_TOL * np.maximum(expected_upper, 1.0)
    for i, resp in enumerate(inputs.respondents):
        lo, hi = float(expected_lower[i]), float(expected_upper[i])
        bt = float(bound_tols[i])
        stored_ok = (abs(float(result.lower[i]) - lo) <= bt and
                     abs(float(result.upper[i]) - hi) <= bt)
        ok = (w[i] >= lo - bt) and (w[i] <= hi + bt) and stored_ok
        bound_checks.append({
            "respondent_id": resp.respondent_id, "weight": float(w[i]),
            "lower": lo, "upper": hi, "ok": bool(ok),
            "bounds_match_plan": bool(stored_ok),
            "at_lower": bool(abs(w[i] - lo) <= bt),
            "at_upper": bool(abs(w[i] - hi) <= bt)})
        if not stored_ok:
            problems.append(f"respondent {resp.respondent_id!r} bound array "
                            f"does not match plan relative bounds")
        if not ((w[i] >= lo - bt) and (w[i] <= hi + bt)):
            problems.append(f"respondent {resp.respondent_id!r} weight "
                            f"{w[i]:.6f} outside [{lo:.6f}, {hi:.6f}]")

    zero_checks = []
    for i, resp in enumerate(inputs.respondents):
        if d[i] == 0:
            ok = w[i] == 0.0
            zero_checks.append({"respondent_id": resp.respondent_id,
                                "weight": float(w[i]), "ok": bool(ok)})
            if not ok:
                problems.append(f"zero base weight {resp.respondent_id!r} has "
                                f"w={w[i]:.3e}")

    total_rows = [i for i, m in enumerate(inputs.margins) if m.field == "*"]
    total_check = {"present": bool(total_rows), "sum_weights": float(w.sum())}
    if total_rows:
        stated = inputs.margins[total_rows[0]].total_float
        total_check.update({"stated": stated,
                            "gap": float(w.sum()) - stated,
                            "ok": abs(w.sum() - stated) <=
                                  constraint_tolerance(stated)})
        if not total_check["ok"]:
            problems.append("sum of weights differs from '*' total")

    objective_check = {
        "reported": float(result.objective),
        "recomputed": _entropy_objective(w, d),
    }
    objective_check["gap"] = abs(objective_check["reported"] -
                                 objective_check["recomputed"])
    objective_check["ok"] = objective_check["gap"] <= \
        REL_TOL * max(abs(objective_check["recomputed"]), 1.0)
    if not objective_check["ok"]:
        problems.append("reported objective does not match recomputation")

    # Optimality signs recomputed from the returned weights and dual.
    c = A.T @ result.dual
    with np.errstate(divide="ignore", invalid="ignore"):
        kkt = np.where(d > 0,
                       np.log(np.where(w > 0, w, 1) / np.where(d > 0, d, 1)) - c,
                       0.0)
    interior = (d > 0) & ~result.at_lower & ~result.at_upper
    low = result.at_lower & (d > 0)
    up = result.at_upper & (d > 0)
    opt = {
        "interior_max_abs": float(np.max(np.abs(kkt[interior]))) if np.any(interior) else 0.0,
        "lower_min": float(np.min(kkt[low])) if np.any(low) else 0.0,
        "upper_max": float(np.max(kkt[up])) if np.any(up) else 0.0,
        "opt_tol": OPT_TOL,
    }
    opt["ok"] = (opt["interior_max_abs"] <= OPT_TOL and
                 opt["lower_min"] >= -OPT_TOL and opt["upper_max"] <= OPT_TOL)
    if not opt["ok"]:
        problems.append("KKT optimality sign/magnitude check failed")

    # Variance: plug the already-computed replicate estimates straight into
    # the plan formula, independently of replicate_variance().
    recalc_variance: dict = {"coefficient": inputs.plan.variance_coefficient}
    if reps is not None and var is not None:
        all_ok = True
        for dom in inputs.plan.domains:
            did = dom["id"]
            full = var.estimates[did]
            direct = [rr.estimates[did] for rr in reps
                      if rr.status == "feasible" and did in rr.estimates
                      and np.isfinite(rr.estimates[did])]
            if not direct or not np.isfinite(full):
                recalc_variance[did] = {"ok": False,
                                        "reason": "no usable replicates"}
                all_ok = False
                continue
            manual = inputs.plan.variance_coefficient * \
                sum((x - full) ** 2 for x in direct)
            match = abs(manual - var.variances[did]) <= \
                REL_TOL * max(abs(manual), 1.0) + 1e-14
            recalc_variance[did] = {"direct": manual,
                                    "reported": var.variances[did],
                                    "n": len(direct), "ok": bool(match)}
            if not match:
                all_ok = False
                problems.append(f"variance for {did!r} mismatches direct formula")
        recalc_variance["ok"] = all_ok

    # Independently recheck every *successful* replicate's returned weights
    # against all margins and bounds (and confirm failed ones really are
    # infeasible, not dropped for convenience).
    replicate_checks = []
    if reps is not None:
        dtols = np.array([constraint_tolerance(b) for b in targets])
        for rr in reps:
            if rr.status == "feasible" and rr.fit is not None:
                wr = np.asarray(rr.fit.weights, dtype=float)
                rr_res = A @ wr - targets
                rlo = inputs.plan.relative_lower * np.asarray(
                    [inputs.replicates[r.respondent_id][rr.replicate_id]
                     for r in inputs.respondents])
                rhi = inputs.plan.relative_upper * np.asarray(
                    [inputs.replicates[r.respondent_id][rr.replicate_id]
                     for r in inputs.respondents])
                margins_ok = bool(np.all(np.abs(rr_res) <= dtols + 1e-14))
                bounds_ok = bool(np.all(wr >= rlo - 1e-9) and
                                 np.all(wr <= rhi + 1e-9))
                replicate_checks.append({
                    "replicate_id": rr.replicate_id, "status": rr.status,
                    "margins_ok": margins_ok, "bounds_ok": bounds_ok,
                    "max_margin_residual": float(np.max(np.abs(rr_res))),
                    "ok": margins_ok and bounds_ok})
                if not (margins_ok and bounds_ok):
                    problems.append(f"replicate {rr.replicate_id!r} returned "
                                    f"weights violate margins or bounds")
            else:
                d_rep = np.array([inputs.replicates[r.respondent_id][rr.replicate_id]
                                  for r in inputs.respondents])
                rlo = inputs.plan.relative_lower * d_rep
                rhi = inputs.plan.relative_upper * d_rep
                independent_check = not _lp_feasible(A, targets, rlo, rhi)
                replicate_checks.append({
                    "replicate_id": rr.replicate_id, "status": rr.status,
                    "confirmed_infeasible_by_lp": independent_check,
                    "ok": (rr.status == "infeasible" and independent_check) or
                          (rr.status == "failed")})
                if rr.status == "infeasible" and not independent_check:
                    problems.append(f"replicate {rr.replicate_id!r} was "
                                    f"reported infeasible but LP finds it "
                                    f"feasible")

    return VerifyReport(ok=not problems, constraint_checks=constraint_checks,
                        bound_checks=bound_checks, zero_checks=zero_checks,
                        total_check=total_check, objective_check=objective_check,
                        optimality_check=opt, recalc_variance=recalc_variance,
                        replicate_checks=replicate_checks, problems=problems)
