"""Independent micro-data verification of the estimator.

Three properties are checked against small synthetic designs whose answer is
known analytically:

1. Fixed SKUs with a given common multiplicative price increase.  The recovered
   base-100 index must equal the injected multipliers (leaving aside the
   persistent SKU/supplier levels).  This is a correctness check of the
   estimator, independent of the task fixtures.
2. Whole-scalar price rescaling (multiply every price by a constant) and row
   reordering must leave every reported month index unchanged.
3. A design whose only cross-block link is carried by one supplier: deleting
   that whole supplier must produce explicit non-identifiability, not a
   slightly different index.  The check asserts the status flips to
   ``insufficient_support`` and the interval disappears.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import Inputs, Plan, Product, Quote
from .estimator import fit


def _synth_inputs(
    n_months: int,
    month_multipliers: dict[int, float],
    item_levels: dict[str, float],
    supplier_levels: dict[str, float],
    schedule: dict[str, list[str]],
    noise_sd: float = 0.0,
    seed: int = 0,
    families: dict[str, str] | None = None,
    base_month: str = "m0",
) -> Inputs:
    """Build synthetic quotes in memory.

    ``schedule[supplier]`` is the list of months that supplier quotes; every
    scheduled month quotes every item the supplier covers (all items here).
    """
    rng = np.random.default_rng(seed)
    quotes: list[Quote] = []
    qid = 0
    for supplier, months in schedule.items():
        items = list(item_levels)  # all suppliers quote all fixed SKUs
        for m_idx in months:
            month = f"m{m_idx}"
            mult = month_multipliers.get(m_idx, 1.0)
            for item in items:
                logp = np.log(item_levels[item]) + np.log(mult) + np.log(supplier_levels[supplier])
                if noise_sd > 0:
                    logp += rng.normal(0, noise_sd)
                qid += 1
                quotes.append(
                    Quote(
                        quote_id=f"s{qid:04d}",
                        supplier_id=supplier,
                        month=month,
                        item_id=item,
                        price=float(np.exp(logp)),
                    )
                )
    products = {
        item: Product(
            item_id=item,
            family=(families or {}).get(item, item + "-fam"),
            relationship="replacement-in-family; quality/price equality is NOT asserted",
        )
        for item in item_levels
    }
    plan = Plan(
        base_month=base_month,
        price_unit="one fixed physical unit in one currency",
        model_assumption="synthetic micro check",
    )
    return Inputs(quotes=quotes, products=products, plan=plan)


def check_common_multiplier(n_boot: int = 120, seed: int = 7) -> dict[str, Any]:
    """Fixed SKUs, known month multipliers, no noise: index must reproduce them."""
    month_multipliers = {0: 1.0, 1: 1.05, 2: 1.10, 3: 1.21}
    item_levels = {"sku-1": 10.0, "sku-2": 20.0}
    supplier_levels = {"sup-a": 1.0, "sup-b": 1.20, "sup-c": 0.90}
    schedule = {s: [0, 1, 2, 3] for s in supplier_levels}
    inputs = _synth_inputs(
        n_months=4,
        month_multipliers=month_multipliers,
        item_levels=item_levels,
        supplier_levels=supplier_levels,
        schedule=schedule,
        noise_sd=0.0,
        seed=seed,
    )
    result = fit(inputs=inputs, n_boot=n_boot, seed=seed, robust=True)
    observed = {}
    expected = {}
    max_abs_err = 0.0
    for m_idx in range(4):
        m = f"m{m_idx}"
        est = result.month_estimates[m]
        expected[m] = 100.0 * month_multipliers[m_idx]
        if est.status != "identified" or est.value_ratio is None:
            observed[m] = None
            max_abs_err = float("inf")
            continue
        observed[m] = 100.0 * est.value_ratio
        max_abs_err = max(max_abs_err, abs(observed[m] - expected[m]))
    passed = max_abs_err < 1e-6
    return {
        "check": "common_multiplicative_increase_on_fixed_skus",
        "passed": passed,
        "expected_index": expected,
        "observed_index": observed,
        "max_abs_error_index_points": max_abs_err,
        "tolerance": 1e-6,
        "note": "SKUs are fixed across all months and every supplier quotes every SKU/month; "
                "month contrasts are exactly identified and unconfounded.",
    }


def check_scale_and_order(n_boot: int = 60, seed: int = 11) -> dict[str, Any]:
    """Multiplying all prices by a constant and reordering rows must not move the index."""
    item_levels = {"sku-1": 10.0, "sku-2": 18.0, "sku-3": 45.0}
    supplier_levels = {"sup-a": 1.0, "sup-b": 1.10}
    month_multipliers = {0: 1.0, 1: 1.05, 2: 1.10, 3: 1.21}
    schedule = {s: [0, 1, 2, 3] for s in supplier_levels}
    inputs = _synth_inputs(
        n_months=4,
        month_multipliers=month_multipliers,
        item_levels=item_levels,
        supplier_levels=supplier_levels,
        schedule=schedule,
        noise_sd=0.0,
        seed=seed,
    )
    base = fit(inputs=inputs, n_boot=n_boot, seed=seed)

    scaled = Inputs(
        quotes=[Quote(q.quote_id, q.supplier_id, q.month, q.item_id, q.price * 3.14159) for q in inputs.quotes],
        products=dict(inputs.products),
        plan=inputs.plan,
        warnings=list(inputs.warnings),
    )
    scaled_result = fit(inputs=scaled, n_boot=n_boot, seed=seed)

    reordered = Inputs(
        quotes=list(reversed(inputs.quotes)),
        products=dict(inputs.products),
        plan=inputs.plan,
        warnings=list(inputs.warnings),
    )
    reordered_result = fit(inputs=reordered, n_boot=n_boot, seed=seed)

    diffs = {}
    for name, other in (("scaled", scaled_result), ("reordered", reordered_result)):
        d = {}
        for m in sorted({q.month for q in inputs.quotes}):
            a = base.month_estimates[m].value_ratio
            b = other.month_estimates[m].value_ratio
            d[m] = abs(100.0 * (a - b))
        diffs[name] = d
    tol = 1e-6
    passed = all(v < tol for d in diffs.values() for v in d.values())
    return {
        "check": "scale_and_row_order_invariance",
        "passed": passed,
        "max_abs_index_change_points": {
            k: max(v.values()) if v else 0.0 for k, v in diffs.items()
        },
        "per_month_diffs_points": diffs,
        "tolerance": tol,
        "note": "A global price constant is absorbed by item levels; row order is not used "
                "for parameter ordering, which is fixed from sorted identifiers.",
    }


def check_rotation_separates_gap(n_boot: int = 120, seed: int = 13) -> dict[str, Any]:
    """Known SKU rotation with a known old/new level gap and known multipliers.

    Old SKU is quoted in months 0-1, new SKU in months 2-3, and both are quoted
    in the boundary month 1 so that overlap exists.  The additive estimator
    must recover BOTH the injected common month multipliers and the injected
    persistent old/new level gap.  A naive month-average index would instead
    absorb the SKU gap into the "price movement"; this check fails in that case.
    """
    month_multipliers = {0: 1.0, 1: 1.04, 2: 1.10, 3: 1.16}
    old_level, new_level = 10.0, 15.0  # known 1.5x persistent gap, NOT equality
    suppliers = {"s1": 1.00, "s2": 1.08, "s3": 0.95, "s4": 1.02}
    quotes: list[Quote] = []
    qid = 0
    for supplier, sup_level in suppliers.items():
        plan = {
            0: ["old"],
            1: ["old", "new"],   # overlap here identifies the gap
            2: ["new"],
            3: ["new"],
        }
        for month, items in plan.items():
            for item in items:
                qid += 1
                base_level = old_level if item == "old" else new_level
                price = base_level * month_multipliers[month] * sup_level
                quotes.append(Quote(f"s{qid:04d}", supplier, f"m{month}", item, price))
    products = {
        "old": Product("old", "fam", "replacement-in-family; quality/price equality is NOT asserted"),
        "new": Product("new", "fam", "replacement-in-family; quality/price equality is NOT asserted"),
    }
    plan_obj = Plan(base_month="m0", price_unit="fixed physical unit")
    inputs = Inputs(quotes=quotes, products=products, plan=plan_obj)
    result = fit(inputs=inputs, n_boot=n_boot, seed=seed)

    observed_index = {}
    max_month_err = 0.0
    for month, mult in month_multipliers.items():
        est = result.month_estimates[f"m{month}"]
        if est.value_ratio is None:
            observed_index[f"m{month}"] = None
            max_month_err = float("inf")
            continue
        observed_index[f"m{month}"] = 100.0 * est.value_ratio
        max_month_err = max(max_month_err, abs(observed_index[f"m{month}"] - 100.0 * mult))

    spread = next(
        (r for r in result.item_spreads if r["family"] == "fam"), None
    )
    observed_gap = None
    gap_err = float("inf")
    if spread is not None and spread["price_ratio_b_over_a"] is not None:
        raw = spread["price_ratio_b_over_a"]
        # The pair order is fixed by sorted SKU ids, so compare the unordered gap.
        observed_gap = max(raw, 1.0 / raw) if raw > 0 else raw
        gap_err = abs(observed_gap - new_level / old_level)

    naive = {
        f"m{month}": 100.0 * float(
            np.mean([
                (old_level if month <= 1 else new_level) * month_multipliers[month] * s
                for s in suppliers.values()
            ])
        )
        for month in month_multipliers
    }
    # Normalize naive index to base 100 to show it is distorted by the rotation.
    base_naive = naive["m0"]
    naive_index = {k: 100.0 * v / base_naive for k, v in naive.items()}
    naive_err = max(
        abs(naive_index[f"m{m}"] - 100.0 * mult) for m, mult in month_multipliers.items()
    )

    passed = (
        max_month_err < 1e-6
        and gap_err < 1e-6
        and naive_err > 10.0  # the naive approach is clearly biased
    )
    return {
        "check": "known_rotation_and_gap_are_separated_by_overlap",
        "passed": passed,
        "expected_month_index": {f"m{m}": 100.0 * v for m, v in month_multipliers.items()},
        "observed_month_index": observed_index,
        "max_abs_month_error_points": max_month_err,
        "expected_new_over_old_gap": new_level / old_level,
        "observed_new_over_old_gap": observed_gap,
        "abs_gap_error": gap_err,
        "naive_ungrouped_index": naive_index,
        "naive_max_abs_error_points": naive_err,
        "note": (
            "Overlap in the boundary month must separate the 1.5x SKU level gap from "
            "the common multipliers. The ungrouped month-average index is shown to be "
            "badly biased, so passing requires the grouped estimate rather than chance."
        ),
    }


def check_unique_bridge_deletion(n_boot: int = 200, seed: int = 5) -> dict[str, Any]:
    """Only one supplier links the old-SKU block to the new-SKU block.

    Deleting that whole supplier must make the post-rotation months
    non-identifiable.  The check fails if an index is returned at all after
    deletion (that would mean the code manufactured a bridge or silently
    imputed one).
    """
    # Months 0-1 quote old SKUs; months 2-3 quote new SKUs.  'bridge' supplier
    # quotes BOTH cohorts in months 1 and 2, providing the only cross-cohort
    # overlap.  Other suppliers quote each cohort independently.
    item_levels = {"old": 10.0, "new": 13.0}
    supplier_levels = {"bridge": 1.0, "backer-1": 1.05, "backer-2": 0.92}
    month_multipliers = {0: 1.0, 1: 1.05, 2: 1.10, 3: 1.21}
    # Build the structured schedule manually.
    quotes: list[Quote] = []
    qid = 0

    def add(supplier, month, item):
        nonlocal qid
        qid += 1
        price = item_levels[item] * month_multipliers[month] * supplier_levels[supplier]
        quotes.append(Quote(f"s{qid:04d}", supplier, f"m{month}", item, price))

    for supplier in ("backer-1", "backer-2"):
        for month in (0, 1):
            add(supplier, month, "old")
        for month in (2, 3):
            add(supplier, month, "new")
    for month in (0, 1, 2, 3):
        add("bridge", month, "old" if month < 2 else "new")
    # The bridge links the cohorts by quoting both in month 1 and month 2.
    add("bridge", 1, "new")
    add("bridge", 2, "old")

    products = {
        "old": Product("old", "family-x", "replacement-in-family; quality/price equality is NOT asserted"),
        "new": Product("new", "family-x", "replacement-in-family; quality/price equality is NOT asserted"),
    }
    plan = Plan(base_month="m0", price_unit="fixed unit")
    inputs = Inputs(quotes=quotes, products=products, plan=plan)

    full = fit(inputs=inputs, n_boot=n_boot, seed=seed)
    bridge_ids = [q.quote_id for q in quotes if q.supplier_id == "bridge"]
    deleted = Inputs(
        quotes=[q for q in quotes if q.supplier_id != "bridge"],
        products=products,
        plan=plan,
    )
    deleted_result = fit(inputs=deleted, n_boot=min(n_boot, 100), seed=seed)

    full_crit = {m: v for m, v in full.diagnosis.critical_suppliers.items() if v}
    post_months = ["m2", "m3"]
    full_status = {m: full.month_estimates[m].status for m in post_months}
    deleted_status = {m: deleted_result.month_estimates[m].status for m in post_months}
    deleted_components = len(deleted_result.diagnosis.components)
    passed = (
        full_status["m2"] != "insufficient_support"
        and "bridge" in full_crit.get("m2", [])
        and all(deleted_status[m] == "insufficient_support" for m in post_months)
        and deleted_components >= 2
    )
    return {
        "check": "unique_bridge_supplier_deletion_exposes_nonidentifiability",
        "passed": passed,
        "full_sample_status_post_rotation": full_status,
        "full_sample_critical_bridge_months": full_crit,
        "deleted_supplier": "bridge",
        "deleted_status_post_rotation": deleted_status,
        "deleted_graph_components": deleted_components,
        "note": "Deleting the sole cross-cohort supplier must flip the post-rotation "
                "months to insufficient_support; a merely shifted index would fail this check.",
    }


def check_family_labels_are_not_keys(n_boot: int = 80, seed: int = 3) -> dict[str, Any]:
    """Renaming family labels must not change any month index."""
    item_levels = {"s1": 10.0, "s2": 12.0}
    supplier_levels = {"a": 1.0, "b": 1.1}
    month_multipliers = {0: 1.0, 1: 1.05, 2: 1.10}
    schedule = {s: [0, 1, 2] for s in supplier_levels}
    inputs = _synth_inputs(
        3, month_multipliers, item_levels, supplier_levels, schedule, noise_sd=0.0, seed=seed,
        families={"s1": "F", "s2": "F"},
    )
    alt = Inputs(
        quotes=list(inputs.quotes),
        products={k: Product(k, "totally-different-" + v.family, v.relationship) for k, v in inputs.products.items()},
        plan=inputs.plan,
    )
    r1 = fit(inputs=inputs, n_boot=n_boot, seed=seed)
    r2 = fit(inputs=alt, n_boot=n_boot, seed=seed)
    diffs = {
        m: abs(100.0 * (r1.month_estimates[m].value_ratio - r2.month_estimates[m].value_ratio))
        for m in sorted({q.month for q in inputs.quotes})
    }
    passed = max(diffs.values()) < 1e-9
    return {
        "check": "family_label_rename_preserves_index",
        "passed": passed,
        "max_abs_index_change_points": max(diffs.values()),
        "note": "Family labels are not used as equivalence keys, so renaming them "
                "cannot move the month index.",
    }


def check_single_bridge_fixture_behavior(out_path: str | Path | None = None) -> dict[str, Any]:
    """Document the single_bridge and disconnected cases at the case level."""
    from .data import load_inputs

    root = Path(__file__).resolve().parent.parent
    outcomes = {}
    for name, quotes_path in (
        ("single_bridge", root / "cases" / "single_bridge.csv"),
        ("disconnected", root / "cases" / "disconnected.csv"),
    ):
        inputs = load_inputs(quotes_path, root / "fixtures" / "products.csv", root / "fixtures" / "plan.json")
        result = fit(inputs=inputs, n_boot=150, seed=101)
        crit = {m: v for m, v in result.diagnosis.critical_suppliers.items() if v}
        outcomes[name] = {
            "month_status": {m: result.month_estimates[m].status for m in result.months},
            "month_identified_full": dict(result.diagnosis.month_identification),
            "valid_fraction": {
                m: result.month_estimates[m].valid_fraction for m in result.months
            },
            "critical_bridge_suppliers": crit,
            "components": [
                {"items": c.items, "months": c.months} for c in result.diagnosis.components
            ],
        }
        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    return {"check": "case_level_single_bridge_and_disconnected", "cases": outcomes}


def run_micro_checks(out_path: str | Path) -> dict[str, Any]:
    checks = {
        "common_multiplier": check_common_multiplier(),
        "rotation_and_gap": check_rotation_separates_gap(),
        "scale_and_order": check_scale_and_order(),
        "unique_bridge_deletion": check_unique_bridge_deletion(),
        "family_labels": check_family_labels_are_not_keys(),
        "case_level": check_single_bridge_fixture_behavior(),
    }
    all_passed = all(v["passed"] for k, v in checks.items() if k != "case_level")
    payload = {"all_checks_passed": all_passed, "checks": checks}
    Path(out_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
