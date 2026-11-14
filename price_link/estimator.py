"""Estimation, robust weighting, uncertainty and stability for the price index.

Method (one interpretable approach, as required)
--------------------------------------------
Additive model on log prices::

    log p(s,m,i) = a_i + b_m + c_s + e

* ``a_i``  persistent product/SKU price level over the window,
* ``b_m``  common month effect shared by every SKU,
* ``c_s``  persistent supplier quote level,
* ``e``    disturbance, possibly correlated within a supplier trajectory.

Only observed supplier/item/month overlap identifies ``b_m``.  Product families
are never treated as equivalence keys.

Weights
-------
Every supplier trajectory receives equal total weight: a quote from supplier
``s`` gets base weight ``1 / n_s`` where ``n_s`` is the number of quotes in
that supplier's observed trajectory.  The weights are then scaled to mean 1.
This prevents a supplier that happened to quote more SKUs or more months from
dominating the common month movement.  Weighting is applied to every estimate,
including replicates, so the estimand is a supplier-balanced common movement.

Error assumption / robust handling
----------------------------------
Errors are heteroskedastic across suppliers only through the known design
(counts), so ``Var(e)`` is approximated as ``sigma^2 / w_base``.  Outlying
quotes are not deleted: they are Huber-downweighted (tuning 1.345 on
MAD-standardized residuals) in an IRLS loop, and the final Huber weights
multiply the base weights.  This is a description of the assumed error
structure, not an exact finite-sample coverage claim.

Uncertainty
-----------
Whole supplier trajectories are resampled with replacement (not individual
quotes), preserving within-supplier dependence.  In every replicate the
exact rational identification test is re-run.  A replicate in which a contrast
is not identifiable is counted as a failure and is never imputed, zero-filled
or silently dropped from the report.  A percentile interval is only declared
*reportable* if the identified replicate fraction reaches
``min_valid_fraction``; otherwise the point index (if identified in the full
sample) is reported with status ``insufficient_replicate_support`` and no
interval.  Leave-one-supplier-out estimates and the list of critical bridge
suppliers are always reported alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Sequence

import numpy as np

from .data import Inputs, Quote
from .identification import (
    Columns,
    Diagnosis,
    ExactSystem,
    build_columns,
    diagnose,
    unique_cells,
)

HUBER_C = 1.345
MAD_TO_SIGMA = 1.4826
DEFAULT_MIN_VALID_FRACTION = 0.80
DEFAULT_MIN_VALID_REPLICATES = 20
DEFAULT_BOOTSTRAP = 500


# ---------------------------------------------------------------------------
# Weighted least squares machinery
# ---------------------------------------------------------------------------


class PivotDesign:
    """Design matrix restricted to an exactly independent column set."""

    def __init__(self, cols: Columns, pivots: Sequence[int], rows: Sequence[tuple[str, str, str]]):
        self.cols = cols
        self.pivots = tuple(pivots)
        self.row_keys = list(rows)
        self.position = {j: k for k, j in enumerate(self.pivots)}
        self.x = np.zeros((len(rows), len(self.pivots)), dtype=float)
        for r, (supplier, month, item) in enumerate(rows):
            for label, value in (
                (("item", item), 1.0),
                (("month", month), 1.0 if month != cols.base_month else 0.0),
                (("supplier", supplier), 1.0 if supplier != cols.ref_supplier else 0.0),
            ):
                j = cols.index.get(label)
                if j is not None and value and j in self.position:
                    self.x[r, self.position[j]] += value

    def contrast_in_pivots(self, contrast: Sequence[int]) -> np.ndarray:
        out = np.zeros(len(self.pivots), dtype=float)
        for j, val in enumerate(contrast):
            if val and j in self.position:
                out[self.position[j]] += float(val)
        return out

    def predict(self, theta: np.ndarray) -> np.ndarray:
        return self.x @ theta


@dataclass
class FitResult:
    theta: np.ndarray
    fitted: np.ndarray
    raw_residuals: np.ndarray
    weights: np.ndarray          # final Huber * base
    base_weights: np.ndarray
    scale: float
    n_iterations: int
    converged: bool
    design: PivotDesign


def _base_weights(quotes: Sequence[Quote], rows: Sequence[tuple[str, str, str]]) -> np.ndarray:
    counts: dict[str, int] = {}
    for q in quotes:
        counts[q.supplier_id] = counts.get(q.supplier_id, 0) + 1
    raw = np.array([1.0 / counts[s] for s, _, _ in rows], dtype=float)
    return raw * (len(raw) / raw.sum())


def _huber_irls(
    design: PivotDesign,
    y: np.ndarray,
    base_w: np.ndarray,
    robust: bool,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> FitResult:
    w = base_w.copy()
    theta = np.linalg.lstsq(
        design.x * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None
    )[0]
    scale = 0.0
    iterations = 0
    converged = not robust
    for iterations in range(1, max_iter + 1):
        resid = y - design.x @ theta
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        scale = MAD_TO_SIGMA * mad
        if scale <= 1e-12:
            converged = True
            break
        if not robust:
            converged = True
            break
        u = resid / scale
        huber = np.where(np.abs(u) <= HUBER_C, 1.0, HUBER_C / np.abs(u))
        new_w = base_w * huber
        delta = np.max(np.abs(new_w - w))
        # Re-fit with the updated weights.
        theta_new = np.linalg.lstsq(
            design.x * np.sqrt(new_w)[:, None], y * np.sqrt(new_w), rcond=None
        )[0]
        w = new_w
        theta = theta_new
        if delta <= tol:
            converged = True
            break
    return FitResult(
        theta=theta,
        fitted=design.x @ theta,
        raw_residuals=y - design.x @ theta,
        weights=w,
        base_weights=base_w,
        scale=scale,
        n_iterations=iterations,
        converged=converged,
        design=design,
    )


def _fit_rows(
    cells: Sequence[tuple[str, str, str]],
    y: np.ndarray,
    cols: Columns,
    pivots: Sequence[int],
    base_w: np.ndarray,
    robust: bool,
) -> FitResult:
    design = PivotDesign(cols, pivots, cells)
    return _huber_irls(design, y, base_w, robust=robust)


# ---------------------------------------------------------------------------
# Contrast bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class ValueEstimate:
    status: str
    log_value: float | None = None
    value_ratio: float | None = None
    ci_log: tuple[float, float] | None = None
    ci_ratio: tuple[float, float] | None = None
    valid_replicates: int = 0
    total_replicates: int = 0
    valid_fraction: float | None = None
    failures: dict[str, int] = field(default_factory=dict)
    interval_reportable: bool = False
    reason: str | None = None
    bootstrap_values: list[float] = field(default_factory=list)


def _summarize_replicates(
    estimates: list[float],
    failures: dict[str, int],
    total: int,
    min_valid_fraction: float,
    min_valid_replicates: int,
    level: float,
) -> ValueEstimate:
    n_valid = len(estimates)
    frac = n_valid / total if total else 0.0
    est = ValueEstimate(
        status="identified",
        valid_replicates=n_valid,
        total_replicates=total,
        valid_fraction=frac,
        failures=dict(failures),
        bootstrap_values=list(estimates),
    )
    if n_valid == 0:
        est.status = "insufficient_support"
        est.reason = "no resampled trajectory set retained identification of this contrast"
        return est
    lo, hi = np.percentile(np.array(estimates, dtype=float), [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    est.ci_log = (float(lo), float(hi))
    est.ci_ratio = (float(np.exp(lo)), float(np.exp(hi)))
    if frac < min_valid_fraction or n_valid < min_valid_replicates:
        est.status = "insufficient_replicate_support"
        parts = [
            f"only {n_valid}/{total} resampled trajectory sets ({frac:.1%}) retained "
            "identification",
            f"required >= {min_valid_fraction:.0%} and >= {min_valid_replicates} replicates",
        ]
        if failures:
            parts.append(
                "unidentified replicate reasons: "
                + "; ".join(f"{k}: {v}" for k, v in sorted(failures.items()))
            )
        est.reason = "; ".join(parts)
        # Keep the conditional percentile range as a diagnostic only; mark
        # interval_reportable=False so no consumer treats it as a valid interval.
        est.interval_reportable = False
    else:
        est.interval_reportable = True
    return est


# ---------------------------------------------------------------------------
# Main result object
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    inputs: Inputs
    diagnosis: Diagnosis
    months: list[str]
    fit: FitResult
    month_estimates: dict[str, ValueEstimate]
    item_level_estimates: dict[str, ValueEstimate]
    item_spreads: list[dict[str, Any]]
    supplier_levels: list[dict[str, Any]]
    stability: dict[str, Any]
    uncertainty: dict[str, Any]
    outliers: list[dict[str, Any]]
    weights_report: dict[str, Any]
    effective_support: dict[str, Any]
    quote_trace: list[dict[str, Any]]
    trace_note: str
    warnings: list[str]
    limitations: list[str]
    seed: int

    # -- structured serialization ------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        def ve(v: ValueEstimate) -> dict[str, Any]:
            d = {
                "status": v.status,
                "log_value": v.log_value,
                "ratio": v.value_ratio,
                "ci_log": list(v.ci_log) if v.ci_log else None,
                "ci_ratio": list(v.ci_ratio) if v.ci_ratio else None,
                "interval_reportable": v.interval_reportable,
                "valid_replicates": v.valid_replicates,
                "total_replicates": v.total_replicates,
                "valid_fraction": v.valid_fraction,
                "failure_reasons": v.failures,
                "reason": v.reason,
            }
            if v.log_value is not None and v.value_ratio is not None:
                d["index_base100"] = 100.0 * v.value_ratio
            return d

        diag = self.diagnosis.to_dict()
        return {
            "metadata": {
                "method": (
                    "supplier-balanced, Huber-robust additive log-price model "
                    "with exact rational identification of month contrasts"
                ),
                "base_month": self.inputs.plan.base_month,
                "months": self.months,
                "n_quotes": len(self.inputs.quotes),
                "n_suppliers": len({q.supplier_id for q in self.inputs.quotes}),
                "n_items": len({q.item_id for q in self.inputs.quotes}),
                "seed": self.seed,
            },
            "model": {
                "equation": "log price(s,m,i) = item_level(i) + month_effect(m) + supplier_level(s) + error",
                "normalization": diag["normalization"],
                "weights": (
                    "each supplier trajectory has equal total weight; a quote gets "
                    "1/n_s for its supplier, rescaled to mean 1"
                ),
                "error_assumption": (
                    "var(error) = sigma^2 / base_weight; within-supplier correlation "
                    "is not modelled parametrically but is preserved by trajectory resampling"
                ),
                "outlier_handling": (
                    "Huber IRLS, tuning 1.345 on MAD-standardized residuals; quotes are "
                    "downweighted, never deleted, so all observed support stays in the design"
                ),
                "product_semantics": (
                    "families are used only to label old/new replacement pairs; "
                    "they are not equivalence keys for estimation"
                ),
            },
            "identification": diag,
            "estimates": {
                "month_index": [
                    {
                        "month": m,
                        "identified_in_full_sample": self.diagnosis.month_identification[m],
                        **ve(self.month_estimates[m]),
                    }
                    for m in self.months
                ],
                "item_levels": {
                    item: ve(est) for item, est in self.item_level_estimates.items()
                },
                "item_spreads_within_family": self.item_spreads,
                "supplier_levels": self.supplier_levels,
            },
            "stability": self.stability,
            "uncertainty": self.uncertainty,
            "outliers": self.outliers,
            "weights": self.weights_report,
            "effective_support": self.effective_support,
            "quote_trace": {
                "note": self.trace_note,
                "rows": self.quote_trace,
            },
            "warnings": self.warnings,
            "limitations": self.limitations,
        }


# ---------------------------------------------------------------------------
# Fitting driver
# ---------------------------------------------------------------------------


def _row_keys(quotes: Sequence[Quote]) -> list[tuple[str, str, str]]:
    return [(q.supplier_id, q.month, q.item_id) for q in quotes]


def _y(quotes: Sequence[Quote]) -> np.ndarray:
    return np.array([np.log(q.price) for q in quotes], dtype=float)


def _contrast_log(theta: np.ndarray, design: PivotDesign, contrast: Sequence[int]) -> float:
    cp = design.contrast_in_pivots(contrast)
    return float(cp @ theta)


def _value_from_log(est: ValueEstimate, log_value: float) -> ValueEstimate:
    est.log_value = log_value
    est.value_ratio = float(np.exp(log_value))
    return est


def _build_quote_trace(
    quotes: Sequence[Quote],
    cols: Columns,
    theta_full: np.ndarray,
    full_fit: FitResult,
    diag: Diagnosis,
) -> list[dict[str, Any]]:
    """Trace every quote to SKU level + month effect + supplier level + residual.

    Uses one exact-pivot solution.  When the model is not fully identified the
    trace is explicitly labelled as normalization-dependent; identified
    contrasts (reported elsewhere) do not depend on it.
    """
    rows = []
    for i, q in enumerate(quotes):
        item_j = cols.index[("item", q.item_id)]
        m_j = cols.index.get(("month", q.month))
        s_j = cols.index.get(("supplier", q.supplier_id))
        a = float(theta_full[item_j])
        b = 0.0 if m_j is None else float(theta_full[m_j])
        c = 0.0 if s_j is None else float(theta_full[s_j])
        log_price = float(np.log(q.price))
        rows.append(
            {
                "quote_id": q.quote_id,
                "supplier_id": q.supplier_id,
                "month": q.month,
                "item_id": q.item_id,
                "price": q.price,
                "log_price": log_price,
                "item_level": a,
                "month_effect": b,
                "supplier_level": c,
                "predicted_log_price": a + b + c,
                "residual": log_price - (a + b + c),
                "huber_weight": float(full_fit.weights[i] / full_fit.base_weights[i])
                if full_fit.base_weights[i] > 0
                else 0.0,
                "item_level_identified": diag.exact.unit_identified(("item", q.item_id)),
                "supplier_level_identified": (
                    diag.exact.unit_identified(("supplier", q.supplier_id))
                    if s_j is not None
                    else True
                ),
            }
        )
    return rows


def _bootstrap_months(
    quotes: Sequence[Quote],
    cols: Columns,
    month_contrasts: dict[str, list[int]],
    month_identified: dict[str, bool],
    suppliers: Sequence[str],
    n_boot: int,
    seed: int,
    min_valid_fraction: float,
    min_valid_replicates: int,
    level: float,
) -> tuple[dict[str, ValueEstimate], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n_s = len(suppliers)
    month_vals: dict[str, list[float]] = {m: [] for m in month_contrasts}
    month_fail: dict[str, dict[str, int]] = {m: {} for m in month_contrasts}
    ident_cache: dict[frozenset[str], tuple[list[int], list[list[Fraction]]]] = {}
    replicate_components: dict[str, int] = {}

    item_contrasts: dict[str, list[int]] = {}
    supports_full = frozenset(suppliers)
    full_cells = unique_cells(quotes)
    full_exact = ExactSystem(full_cells, cols)
    ident_cache[supports_full] = (full_exact.pivots, full_exact.nullspace)

    for _ in range(n_boot):
        sample = rng.choice(np.array(suppliers, dtype=object), size=n_s, replace=True)
        present = frozenset(str(s) for s in sample)
        if present not in ident_cache:
            cells = unique_cells(quotes, present=present)
            sys = ExactSystem(cells, cols)
            ident_cache[present] = (sys.pivots, sys.nullspace)

        pivots, nullspace = ident_cache[present]

        def identifiable(contrast: Sequence[int]) -> bool:
            return ExactSystem.identified_static(contrast, nullspace)
        # Build replicate rows by walking the sampled trajectory list, so a
        # supplier sampled twice contributes its whole trajectory twice.
        rep_quotes: list[Quote] = []
        for s in sample:
            rep_quotes.extend(q for q in quotes if q.supplier_id == s)
        rows = _row_keys(rep_quotes)
        y = _y(rep_quotes)
        base_w = _base_weights(quotes, rows)
        fit_rep = _fit_rows(rows, y, cols, pivots, base_w, robust=True)
        design = fit_rep.design

        failed_any = False
        for m, contrast in month_contrasts.items():
            if identifiable(contrast):
                month_vals[m].append(_contrast_log(fit_rep.theta, design, contrast))
            else:
                failed_any = True
                key = "contrast_not_identified_in_replicate"
                month_fail[m][key] = month_fail[m].get(key, 0) + 1
        if failed_any:
            ncomp = len(_replicate_components(quotes, present))
            key = f"replicate_components={ncomp}"
            replicate_components[key] = replicate_components.get(key, 0) + 1

    estimates = {}
    for m, contrast in month_contrasts.items():
        est = _summarize_replicates(
            month_vals[m],
            month_fail[m],
            n_boot,
            min_valid_fraction,
            min_valid_replicates,
            level,
        )
        if not month_identified[m]:
            est.status = "insufficient_support"
            est.ci_log = None
            est.ci_ratio = None
            est.interval_reportable = False
            est.log_value = None
            est.value_ratio = None
            est.reason = (
                est.reason
                or "month effect is not identified in the full sample; no replicate "
                "can repair a missing overlap and no external bridge is used"
            )
        estimates[m] = est

    diagnostics = {
        "n_boot": n_boot,
        "seed": seed,
        "n_suppliers": n_s,
        "n_distinct_present_sets": len(ident_cache),
        "replicate_component_counts": replicate_components,
    }
    return estimates, diagnostics


def _replicate_components(quotes: Sequence[Quote], present: frozenset[str]) -> list:
    from .identification import item_month_components

    return item_month_components(quotes, present=present)


def fit(
    quotes_path: str | None = None,
    products_path: str | None = None,
    plan_path: str | None = None,
    *,
    inputs: Inputs | None = None,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 20250101,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    min_valid_replicates: int = DEFAULT_MIN_VALID_REPLICATES,
    ci_level: float = 0.95,
    robust: bool = True,
) -> AnalysisResult:
    """Fit the comparable-month index.

    Either pass the three input paths or pre-loaded :class:`Inputs`.
    """
    if inputs is None:
        if not (quotes_path and products_path and plan_path):
            raise ValueError("provide inputs= or all three input paths")
        from .data import load_inputs

        inputs = load_inputs(quotes_path, products_path, plan_path)

    quotes = inputs.quotes
    plan = inputs.plan
    base_month = plan.base_month
    cols = build_columns(quotes, base_month)
    months = sorted({q.month for q in quotes})
    suppliers = sorted({q.supplier_id for q in quotes})
    diag = diagnose(quotes, inputs.products, base_month)

    # Fit on the exact pivot columns for the full sample.
    cells = unique_cells(quotes)
    rows = _row_keys(quotes)
    y = _y(quotes)
    base_w = _base_weights(quotes, rows)
    full_fit = _fit_rows(rows, y, cols, diag.exact.pivots, base_w, robust=robust)
    design = full_fit.design

    month_contrasts = {m: diag.exact.month_contrast(m) for m in months}
    month_identified = dict(diag.month_identification)

    # Point estimates for identified month contrasts.
    month_est: dict[str, ValueEstimate] = {}
    for m in months:
        c = month_contrasts[m]
        if month_identified[m]:
            log_v = _contrast_log(full_fit.theta, design, c)
            month_est[m] = _value_from_log(ValueEstimate(status="identified"), log_v)
        else:
            month_est[m] = ValueEstimate(
                status="insufficient_support",
                reason=(
                    "month effect not identified from observed overlap; "
                    "the exact design null space contains a direction with a non-zero "
                    f"coefficient on {m}"
                ),
            )
    # Base month is 100 by normalization.
    month_est[base_month].log_value = 0.0
    month_est[base_month].value_ratio = 1.0
    month_est[base_month].status = "identified"
    month_est[base_month].reason = "base-month normalization; index fixed at 100"

    # Item centered levels and family pair spreads.
    item_level_est: dict[str, ValueEstimate] = {}
    n_items = len(cols.items)
    for item in cols.items:
        c = [Fraction(0)] * cols.ncol
        c[cols.index[("item", item)]] = Fraction(1)
        for other in cols.items:
            c[cols.index[("item", other)]] -= Fraction(1, n_items)
        if diag.exact.identified(c):
            log_v = _contrast_log(full_fit.theta, design, c)
            item_level_est[item] = _value_from_log(ValueEstimate(status="identified"), log_v)
        else:
            item_level_est[item] = ValueEstimate(
                status="insufficient_support",
                reason="centered item level is not identified",
            )

    item_spreads: list[dict[str, Any]] = []
    if inputs.products:
        by_family: dict[str, list[str]] = {}
        for item, prod in inputs.products.items():
            if item in cols.items:
                by_family.setdefault(prod.family, []).append(item)
        for family in sorted(by_family):
            items = sorted(by_family[family])
            for a_idx in range(len(items)):
                for b_idx in range(a_idx + 1, len(items)):
                    # Put the "-old" SKU first when the naming makes it clear.
                    first, second = items[a_idx], items[b_idx]
                    if "-old" in second and "-new" in first:
                        first, second = second, first
                    item_a, item_b = first, second
                    c = diag.exact.item_contrast(item_b, item_a)
                    identified = diag.exact.identified(c)
                    rec: dict[str, Any] = {
                        "family": family,
                        "item_a": item_a,
                        "item_b": item_b,
                        "contrast": f"log_level({item_b}) - log_level({item_a})",
                        "ratio_definition": f"{item_b} price / {item_a} price at equal month/supplier",
                        "status": "identified" if identified else "insufficient_support",
                    }
                    if identified:
                        log_v = _contrast_log(full_fit.theta, design, c)
                        rec["log_value"] = log_v
                        rec["price_ratio_b_over_a"] = float(np.exp(log_v))
                        rec["reason"] = (
                            "observed overlap identifies the persistent SKU level difference; "
                            "this is not an equality claim from the family label"
                        )
                    else:
                        rec["log_value"] = None
                        rec["price_ratio_b_over_a"] = None
                        rec["reason"] = (
                            "SKU level difference not identified: no observed overlap "
                            "separates it from month movement"
                        )
                    item_spreads.append(rec)

    # Supplier persistent levels (centered across suppliers).
    supplier_levels: list[dict[str, Any]] = []
    n_sup = len(suppliers)
    for supplier in suppliers:
        # level(s) with reference supplier at 0, centered over all suppliers.
        c = [0] * cols.ncol
        if supplier != cols.ref_supplier:
            c[cols.index[("supplier", supplier)]] += 1.0
        for other in cols.suppliers_nonref:
            c[cols.index[("supplier", other)]] -= 1.0 / n_sup
        identified = diag.exact.identified(c)
        rec: dict[str, Any] = {
            "supplier_id": supplier,
            "status": "identified" if identified else "insufficient_support",
            "n_quotes": sum(1 for q in quotes if q.supplier_id == supplier),
            "months": sorted({q.month for q in quotes if q.supplier_id == supplier}),
            "items": sorted({q.item_id for q in quotes if q.supplier_id == supplier}),
        }
        if identified:
            log_v = _contrast_log(full_fit.theta, design, c)
            rec["centered_log_level"] = log_v
            rec["level_ratio_to_mean"] = float(np.exp(log_v))
        else:
            rec["centered_log_level"] = None
            rec["level_ratio_to_mean"] = None
            rec["reason"] = "supplier level contrast not identified"
        supplier_levels.append(rec)

    # Per-quote trace: an explicit decomposition into persistent SKU level,
    # month effect and supplier trajectory level.  When the full sample has a
    # non-trivial null space, a single representative decomposition is chosen
    # by the exact pivot rule; the identified contrasts remain the estimand.
    theta_full = np.zeros(cols.ncol, dtype=float)
    for pos, col in enumerate(diag.exact.pivots):
        theta_full[col] = full_fit.theta[pos]
    identifiability_note = (
        "all parameters identified" if not diag.exact.nullspace else
        "individual raw levels depend on the chosen normalization; use identified contrasts"
    )
    quote_trace = _build_quote_trace(quotes, cols, theta_full, full_fit, diag)
    boot_est, boot_diag = _bootstrap_months(
        quotes,
        cols,
        month_contrasts,
        month_identified,
        suppliers,
        n_boot,
        seed,
        min_valid_fraction,
        min_valid_replicates,
        ci_level,
    )

    # Merge point estimates with bootstrap intervals.
    for m in months:
        point = month_est[m]
        boot = boot_est[m]
        point.valid_replicates = boot.valid_replicates
        point.total_replicates = boot.total_replicates
        point.valid_fraction = boot.valid_fraction
        point.failures = boot.failures
        point.bootstrap_values = boot.bootstrap_values
        if not month_identified[m]:
            continue
        if m == base_month:
            # Base normalization; interval is exactly 100 by construction, but
            # report replicate share for transparency.
            point.ci_log = (0.0, 0.0)
            point.ci_ratio = (1.0, 1.0)
            point.interval_reportable = True
            point.status = "identified"
            continue
        point.status = boot.status
        point.reason = boot.reason
        point.ci_log = boot.ci_log if boot.interval_reportable else None
        point.ci_ratio = boot.ci_ratio if boot.interval_reportable else None
        point.interval_reportable = boot.interval_reportable

    # Bootstrap intervals for item pair spreads (same resampling scheme).
    item_boot = _bootstrap_item_spreads(
        quotes,
        cols,
        item_spreads,
        suppliers,
        n_boot,
        seed + 17,
        min_valid_fraction,
        min_valid_replicates,
        ci_level,
    )
    for rec, (lo, hi, n_valid, failures, reportable) in zip(item_spreads, item_boot):
        if rec["status"] == "identified" and reportable:
            rec["ci_log"] = [lo, hi]
            rec["ci_price_ratio"] = [float(np.exp(lo)), float(np.exp(hi))]
            rec["valid_replicates"] = n_valid
            rec["valid_fraction"] = n_valid / n_boot
        elif rec["status"] == "identified":
            rec["ci_log"] = None
            rec["ci_price_ratio"] = None
            rec["valid_replicates"] = n_valid
            rec["valid_fraction"] = n_valid / n_boot
            rec["status"] = "insufficient_replicate_support"
            rec["reason"] = (
                rec["reason"]
                + f"; only {n_valid}/{n_boot} resampled trajectory sets retained "
                "identification, so no interval is reported"
            )

    # Leave-one-supplier-out stability for month indices.
    stability = _leave_one_out(quotes, cols, months, month_contrasts, diag)
    critical = {m: v for m, v in diag.critical_suppliers.items() if v}
    stability["critical_bridge_suppliers"] = critical

    # Outlier / weight reporting.
    z = full_fit.raw_residuals / full_fit.scale if full_fit.scale > 0 else np.zeros_like(y)
    outliers = []
    for i, q in enumerate(quotes):
        hw = full_fit.weights[i] / full_fit.base_weights[i] if full_fit.base_weights[i] > 0 else 0
        if hw < 0.85 or abs(z[i]) > 2.5:
            outliers.append(
                {
                    "quote_id": q.quote_id,
                    "supplier_id": q.supplier_id,
                    "month": q.month,
                    "item_id": q.item_id,
                    "price": q.price,
                    "log_residual": float(full_fit.raw_residuals[i]),
                    "standardized_residual": float(z[i]),
                    "huber_weight": float(hw),
                    "disposition": "downweighted, retained",
                }
            )
    outliers.sort(key=lambda r: -abs(r["standardized_residual"]))

    weights_report = _weights_report(quotes, full_fit, suppliers)
    effective_support = {
        "n_quotes": len(quotes),
        "n_cells": len(cells),
        "n_replicates": n_boot,
        "min_valid_fraction_required": min_valid_fraction,
        "min_valid_replicates_required": min_valid_replicates,
        "months": {
            m: {
                "identified_full_sample": month_identified[m],
                "valid_replicate_fraction": month_est[m].valid_fraction,
                "interval_reportable": month_est[m].interval_reportable,
            }
            for m in months
        },
        "bootstrap": boot_diag,
        "critical_bridge_suppliers": critical,
    }

    uncertainty = {
        "method": (
            "whole-supplier-trajectory bootstrap with replacement; identification "
            "re-tested exactly in every replicate; percentile intervals"
        ),
        "resampling_unit": "supplier trajectory (all quotes of one supplier)",
        "n_boot": n_boot,
        "seed": seed,
        "ci_level": ci_level,
        "min_valid_fraction": min_valid_fraction,
        "min_valid_replicates": min_valid_replicates,
        "coverage_note": (
            "Percentile intervals from a finite supplier bootstrap are descriptive; "
            "no exact finite-sample coverage guarantee is claimed. Replicates that "
            "lose identification are failures, not zero estimates."
        ),
    }

    warnings = list(inputs.warnings) + list(diag.warnings)
    for m in months:
        if month_est[m].status == "insufficient_replicate_support":
            warnings.append(
                f"month {m}: point index is identified in the full sample but "
                f"only {month_est[m].valid_fraction:.1%} of resampled trajectory sets "
                "retain identification; interval withheld (evidence insufficient)"
            )
        if month_est[m].status == "insufficient_support":
            warnings.append(
                f"month {m}: index not reported; no observed overlap supports it and "
                "no external bridge is used"
            )

    limitations = [
        "The model is additive on log prices with persistent item and supplier levels; "
        "if relative SKU prices drift within the six-month window the month effect is "
        "biased toward the average drift.",
        "Families only record a replacement relationship. Any old/new level difference "
        "reported here is a measured persistent level difference from overlap, not a "
        "quality or equivalence claim.",
        "Identification is exact and conditional on the observed quote pattern; "
        "absence of overlap is reported as insufficient support rather than bridged "
        "with external product prices.",
        "Supplier-level disturbance dependence is handled only by whole-trajectory "
        "resampling; no parametric covariance model is fitted.",
        "Percentile bootstrap intervals are descriptive and may be optimistic when "
        "the bridge rests on a single supplier, which is shown explicitly by the "
        "delete-one stability table.",
        "No forecasting, hedonic regression, rolling analysis or automatic product matching is performed.",
    ]

    result = AnalysisResult(
        inputs=inputs,
        diagnosis=diag,
        months=months,
        fit=full_fit,
        month_estimates=month_est,
        item_level_estimates=item_level_est,
        item_spreads=item_spreads,
        supplier_levels=supplier_levels,
        stability=stability,
        uncertainty=uncertainty,
        outliers=outliers,
        weights_report=weights_report,
        effective_support=effective_support,
        quote_trace=quote_trace,
        trace_note=identifiability_note,
        warnings=warnings,
        limitations=limitations,
        seed=seed,
    )
    result.metadata = {  # convenience for report code
        "n_boot": n_boot,
        "ci_level": ci_level,
    }
    return result


def _bootstrap_item_spreads(
    quotes: Sequence[Quote],
    cols: Columns,
    spreads: Sequence[dict[str, Any]],
    suppliers: Sequence[str],
    n_boot: int,
    seed: int,
    min_valid_fraction: float,
    min_valid_replicates: int,
    level: float,
) -> list[tuple[float, float, int, dict[str, int], bool]]:
    rng = np.random.default_rng(seed)
    n_s = len(suppliers)
    vals: list[list[float]] = [[] for _ in spreads]
    fails: list[dict[str, int]] = [{} for _ in spreads]
    cache: dict[frozenset[str], tuple[list[int], list[list]]] = {}
    full_present = frozenset(suppliers)
    full_exact = ExactSystem(unique_cells(quotes), cols)
    cache[full_present] = (full_exact.pivots, full_exact.nullspace)
    for _ in range(n_boot):
        sample = rng.choice(np.array(suppliers, dtype=object), size=n_s, replace=True)
        present = frozenset(str(s) for s in sample)
        if present not in cache:
            sys = ExactSystem(unique_cells(quotes, present=present), cols)
            cache[present] = (sys.pivots, sys.nullspace)
        pivots, nullspace = cache[present]
        rep_quotes: list[Quote] = []
        for s in sample:
            rep_quotes.extend(q for q in quotes if q.supplier_id == s)
        rows = _row_keys(rep_quotes)
        y = _y(rep_quotes)
        base_w = _base_weights(quotes, rows)
        fit_rep = _fit_rows(rows, y, cols, pivots, base_w, robust=True)
        for k, rec in enumerate(spreads):
            c = [0] * cols.ncol
            c[cols.index[("item", rec["item_b"])]] += 1
            c[cols.index[("item", rec["item_a"])]] -= 1
            if ExactSystem.identified_static(c, nullspace):
                vals[k].append(_contrast_log(fit_rep.theta, fit_rep.design, c))
            else:
                fails[k]["item_contrast_not_identified_in_replicate"] = (
                    fails[k].get("item_contrast_not_identified_in_replicate", 0) + 1
                )
    out = []
    for k in range(len(spreads)):
        n_valid = len(vals[k])
        frac = n_valid / n_boot if n_boot else 0.0
        reportable = frac >= min_valid_fraction and n_valid >= min_valid_replicates
        if n_valid > 0:
            lo, hi = np.percentile(np.array(vals[k]), [100 * (1 - level) / 2, 100 * (1 + level) / 2])
        else:
            lo, hi = float("nan"), float("nan")
        out.append((float(lo), float(hi), n_valid, fails[k], reportable))
    return out


def _leave_one_out(
    quotes: Sequence[Quote],
    cols: Columns,
    months: Sequence[str],
    month_contrasts: dict[str, list[int]],
    diag: Diagnosis,
) -> dict[str, Any]:
    suppliers = sorted({q.supplier_id for q in quotes})
    loads: list[dict[str, Any]] = []
    for s in suppliers:
        present = frozenset(x for x in suppliers if x != s)
        cells = unique_cells(quotes, present=present)
        sys = ExactSystem(cells, cols)
        sub_quotes = [q for q in quotes if q.supplier_id != s]
        if not sub_quotes:
            continue
        rows = _row_keys(sub_quotes)
        y = _y(sub_quotes)
        base_w = _base_weights(sub_quotes, rows)
        fit_sub = _fit_rows(rows, y, cols, sys.pivots, base_w, robust=True)
        est_months: dict[str, Any] = {}
        for m in months:
            c = month_contrasts[m]
            if sys.identified(c):
                est_months[m] = {
                    "status": "identified",
                    "index_base100": 100.0 * float(np.exp(_contrast_log(fit_sub.theta, fit_sub.design, c))),
                }
            else:
                est_months[m] = {
                    "status": "insufficient_support",
                    "index_base100": None,
                }
        loads.append({"removed_supplier": s, "months": est_months})
    return {"leave_one_supplier_out": loads}


def _weights_report(quotes: Sequence[Quote], fit_result: FitResult, suppliers: Sequence[str]) -> dict[str, Any]:
    base = fit_result.base_weights
    final = fit_result.weights
    total_base = float(base.sum())
    total_final = float(final.sum())
    per_supplier = []
    for s in suppliers:
        idx = [i for i, q in enumerate(quotes) if q.supplier_id == s]
        per_supplier.append(
            {
                "supplier_id": s,
                "n_quotes": len(idx),
                "base_weight_share": float(base[idx].sum() / total_base),
                "final_weight_share": float(final[idx].sum() / total_final),
                "mean_huber_weight": float((final[idx] / base[idx]).mean()),
            }
        )
    return {
        "scheme": "supplier-balanced: quote weight = 1/n_supplier_quotes, scaled to mean 1",
        "total_base_weight": total_base,
        "total_final_weight": total_final,
        "effective_n_quotes": float(total_final ** 2 / float((final ** 2).sum())) if len(final) else 0.0,
        "per_supplier": per_supplier,
        "n_downweighted": int((final / base < 0.85).sum()),
        "residual_scale_log": fit_result.scale,
        "irls_iterations": fit_result.n_iterations,
        "irls_converged": fit_result.converged,
    }
