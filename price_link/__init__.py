"""Comparable-month price index for rotating SKUs with repeated supplier quotes.

Public API
----------
load_inputs(quotes_path, products_path, plan_path) -> Inputs
fit(quotes, products, plan, ...) -> AnalysisResult
explain(result) -> str

The estimator is a supplier-balanced, Huber-robust additive model on log prices

    log price(s, m, i) = item_level(i) + month_effect(m) + supplier_level(s) + error

with the base-month effect fixed at zero and one supplier level fixed at zero.
Product families are never used as equivalence keys; only observed
supplier/item/month overlap is allowed to identify the month effects.
"""

from .data import Inputs, load_inputs, load_products, load_plan, load_quotes
from .estimator import AnalysisResult, fit
from .explain import explain

__all__ = [
    "Inputs",
    "load_inputs",
    "load_quotes",
    "load_products",
    "load_plan",
    "fit",
    "AnalysisResult",
    "explain",
]
