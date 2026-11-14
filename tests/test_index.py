"""Tests for the comparable-month price index.

Run with:  .runtime/venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from price_link.data import Inputs, Plan, Product, Quote, load_inputs  # noqa: E402
from price_link.estimator import fit  # noqa: E402
from price_link.identification import build_columns, diagnose, ExactSystem, unique_cells  # noqa: E402

QUOTES = ROOT / "fixtures" / "quotes.csv"
PRODUCTS = ROOT / "fixtures" / "products.csv"
PLAN = ROOT / "fixtures" / "plan.json"
SINGLE = ROOT / "cases" / "single_bridge.csv"
DISCONNECTED = ROOT / "cases" / "disconnected.csv"


@pytest.fixture(scope="module")
def base_inputs():
    return load_inputs(QUOTES, PRODUCTS, PLAN)


@pytest.fixture(scope="module")
def base_result(base_inputs):
    return fit(inputs=base_inputs, n_boot=80, seed=123)


def test_base_everything_identified(base_result):
    for m in base_result.months:
        assert base_result.diagnosis.month_identification[m]
        assert base_result.month_estimates[m].status == "identified"
        assert base_result.month_estimates[m].interval_reportable


def test_base_index_shape(base_result):
    assert base_result.month_estimates["2025-01"].value_ratio == pytest.approx(1.0)
    ratios = [base_result.month_estimates[m].value_ratio for m in base_result.months]
    assert ratios == sorted(ratios), "fixture has a monotone month path"
    # Index is measured on log prices; a constant shift would move it.
    assert 1.10 < ratios[-1] < 1.20


def test_family_labels_not_used_as_keys(base_inputs):
    """Renaming family labels must not change any estimate."""
    r1 = fit(inputs=base_inputs, n_boot=40, seed=9)
    renamed = Inputs(
        quotes=list(base_inputs.quotes),
        products={
            k: Product(k, "renamed-" + v.family, v.relationship)
            for k, v in base_inputs.products.items()
        },
        plan=base_inputs.plan,
    )
    r2 = fit(inputs=renamed, n_boot=40, seed=9)
    for m in r1.months:
        assert r1.month_estimates[m].log_value == pytest.approx(
            r2.month_estimates[m].log_value, abs=1e-12
        )


def test_scale_and_row_order_invariance(base_inputs):
    r1 = fit(inputs=base_inputs, n_boot=40, seed=3)
    scaled = Inputs(
        quotes=[Quote(q.quote_id, q.supplier_id, q.month, q.item_id, q.price * 12.5) for q in base_inputs.quotes],
        products=dict(base_inputs.products),
        plan=base_inputs.plan,
    )
    r2 = fit(inputs=scaled, n_boot=40, seed=3)
    reordered = Inputs(
        quotes=list(reversed(base_inputs.quotes)),
        products=dict(base_inputs.products),
        plan=base_inputs.plan,
    )
    r3 = fit(inputs=reordered, n_boot=40, seed=3)
    for m in r1.months:
        assert r1.month_estimates[m].log_value == pytest.approx(
            r2.month_estimates[m].log_value, abs=1e-10
        )
        assert r1.month_estimates[m].log_value == pytest.approx(
            r3.month_estimates[m].log_value, abs=1e-10
        )


def test_exact_nullspace_base_is_only_normalization(base_inputs):
    cols = build_columns(base_inputs.quotes, "2025-01")
    exact = ExactSystem(unique_cells(base_inputs.quotes), cols)
    assert len(exact.nullspace) == 0
    assert len(exact.pivots) == cols.ncol


def test_single_bridge_critical_supplier():
    inputs = load_inputs(SINGLE, PRODUCTS, PLAN)
    result = fit(inputs=inputs, n_boot=120, seed=77)
    diag = result.diagnosis
    # supplier-1 is the only supplier quoting old and new SKUs in months 3/4.
    assert diag.month_identification["2025-04"]  # identified in the full sample
    assert diag.critical_suppliers["2025-04"] == ["supplier-1"]
    assert diag.critical_suppliers["2025-06"] == ["supplier-1"]
    # Sole-supplier evidence cells are reported.
    cells = {(c["item_id"], c["month"]) for c in diag.sole_supplier_cells}
    assert ("A-new", "2025-03") in cells
    assert ("A-old", "2025-04") in cells
    # Delete-one sweep exposes the loss of identification.
    del1 = next(d for d in diag.deletion if d["removed_supplier"] == "supplier-1")
    assert del1["unidentified_months"] == ["2025-04", "2025-05", "2025-06"]
    # Replicate support is below the threshold, so the interval is withheld.
    est = result.month_estimates["2025-04"]
    assert est.status == "insufficient_replicate_support"
    assert est.interval_reportable is False
    assert 0.0 < est.valid_fraction < 1.0


def test_single_bridge_deletion_not_a_small_shift():
    """Removing the unique bridge must change status, not just the index."""
    inputs = load_inputs(SINGLE, PRODUCTS, PLAN)
    full = fit(inputs=inputs, n_boot=60, seed=5)
    reduced = Inputs(
        quotes=[q for q in inputs.quotes if q.supplier_id != "supplier-1"],
        products=dict(inputs.products),
        plan=inputs.plan,
    )
    cut = fit(inputs=reduced, n_boot=40, seed=5)
    for m in ("2025-04", "2025-05", "2025-06"):
        assert full.month_estimates[m].status != "insufficient_support"
        assert cut.month_estimates[m].status == "insufficient_support"
        assert cut.month_estimates[m].value_ratio is None
        assert cut.month_estimates[m].ci_ratio is None
    # Early months stay identified.
    for m in ("2025-01", "2025-02", "2025-03"):
        assert cut.month_estimates[m].status == "identified"


def test_disconnected_blocks_are_not_bridged():
    inputs = load_inputs(DISCONNECTED, PRODUCTS, PLAN)
    result = fit(inputs=inputs, n_boot=60, seed=5)
    diag = result.diagnosis
    assert len(diag.components) == 2
    for m in ("2025-04", "2025-05", "2025-06"):
        assert diag.month_identification[m] is False
        assert result.month_estimates[m].status == "insufficient_support"
        assert result.month_estimates[m].value_ratio is None
        assert result.month_estimates[m].valid_replicates == 0
        conf = diag.confounding[m]
        assert conf["null_direction"]["month_effects"]
        assert conf["n_witness_quotes"] > 0
    for m in ("2025-01", "2025-02", "2025-03"):
        assert result.month_estimates[m].status == "identified"
    # The family split must be reported, not silently bridged.
    assert diag.family_cross_component
    # Item level spreads across the break are not identified.
    for rec in result.item_spreads:
        assert rec["status"] == "insufficient_support"


def test_no_family_cross_component_in_base():
    inputs = load_inputs(QUOTES, PRODUCTS, PLAN)
    diag = diagnose(inputs.quotes, inputs.products, "2025-01")
    assert diag.family_cross_component == []
    assert len(diag.components) == 1


def test_supplier_balanced_weights(base_result):
    w = base_result.weights_report
    shares = [rec["base_weight_share"] for rec in w["per_supplier"]]
    # Equal total base weight per trajectory.
    assert max(shares) - min(shares) == pytest.approx(0.0, abs=1e-12)


def test_micro_common_multiplier():
    from price_link.verify import check_common_multiplier

    out = check_common_multiplier(n_boot=40)
    assert out["passed"], out


def test_micro_unique_bridge_deletion():
    from price_link.verify import check_unique_bridge_deletion

    out = check_unique_bridge_deletion(n_boot=80)
    assert out["passed"], out


def test_micro_rotation_and_gap():
    from price_link.verify import check_rotation_separates_gap

    out = check_rotation_separates_gap(n_boot=40)
    assert out["passed"], out
    assert out["naive_max_abs_error_points"] > 10.0  # naive index is biased


def test_micro_scale_and_order():
    from price_link.verify import check_scale_and_order

    out = check_scale_and_order(n_boot=30)
    assert out["passed"], out


def test_result_is_json_serializable(base_result):
    import json

    payload = json.dumps(base_result.to_dict())
    assert "month_index" in payload
    d = base_result.to_dict()
    for row in d["estimates"]["month_index"]:
        assert "index_base100" in row or row["status"] == "insufficient_support"


def test_invalid_price_rejected(tmp_path):
    from price_link.data import InputError, load_quotes

    bad = tmp_path / "bad.csv"
    bad.write_text("quote_id,supplier_id,month,item_id,price\nq1,s1,m0,i1,-1\n", encoding="utf-8")
    with pytest.raises(InputError):
        load_quotes(bad)
