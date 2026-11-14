"""Input loading and validation.

All inputs are read from CSV/JSON.  No file name is hard-coded; the caller
supplies paths.  Validation is strict about the semantics the task fixes:
one currency, one fixed physical unit per line, distinct SKUs per quote,
and a declared family that only records a replacement relationship.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

QUOTE_FIELDS = ("quote_id", "supplier_id", "month", "item_id", "price")


@dataclass(frozen=True)
class Quote:
    quote_id: str
    supplier_id: str
    month: str
    item_id: str
    price: float


@dataclass(frozen=True)
class Product:
    item_id: str
    family: str
    relationship: str


@dataclass(frozen=True)
class Plan:
    base_month: str
    price_unit: str
    analysis_target: str = ""
    model_assumption: str = ""
    replacement_semantics: str = ""
    uncertainty: str = ""
    output: str = ""
    scope: str = ""


@dataclass
class Inputs:
    quotes: list[Quote]
    products: dict[str, Product]
    plan: Plan
    warnings: list[str] = field(default_factory=list)


class InputError(ValueError):
    """Raised when the input violates a premise the method relies on."""


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_quotes(path: str | Path) -> list[Quote]:
    rows = _read_csv(path)
    if not rows:
        raise InputError(f"{path}: no quote rows")
    missing = set(QUOTE_FIELDS) - set(rows[0].keys())
    if missing:
        raise InputError(f"{path}: missing quote fields {sorted(missing)}")

    seen_ids: set[str] = set()
    seen_cells: dict[tuple[str, str, str], int] = {}
    quotes: list[Quote] = []
    for line_no, row in enumerate(rows, start=2):
        try:
            price = float(row["price"])
        except (TypeError, ValueError):
            raise InputError(f"{path}:{line_no}: non-numeric price {row['price']!r}")
        if price <= 0:
            raise InputError(f"{path}:{line_no}: price must be positive, got {price}")
        qid = row["quote_id"].strip()
        if not qid:
            raise InputError(f"{path}:{line_no}: empty quote_id")
        if qid in seen_ids:
            raise InputError(f"{path}:{line_no}: duplicate quote_id {qid!r}")
        seen_ids.add(qid)
        key = (row["supplier_id"].strip(), row["month"].strip(), row["item_id"].strip())
        if key in seen_cells:
            raise InputError(
                f"{path}:{line_no}: duplicate supplier/month/item cell {key} "
                f"(also at row {seen_cells[key]})"
            )
        seen_cells[key] = line_no
        quotes.append(
            Quote(
                quote_id=qid,
                supplier_id=key[0],
                month=key[1],
                item_id=key[2],
                price=price,
            )
        )
    return quotes


def load_products(path: str | Path) -> dict[str, Product]:
    rows = _read_csv(path)
    missing = {"item_id", "family", "relationship"} - set(rows[0].keys() if rows else ())
    if missing:
        raise InputError(f"{path}: missing product fields {sorted(missing)}")
    products: dict[str, Product] = {}
    for row in rows:
        item = row["item_id"].strip()
        if item in products:
            raise InputError(f"{path}: duplicate item_id {item!r}")
        products[item] = Product(
            item_id=item,
            family=row["family"].strip(),
            relationship=row["relationship"].strip(),
        )
    return products


def load_plan(path: str | Path) -> Plan:
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    if "base_month" not in obj:
        raise InputError(f"{path}: missing base_month")
    return Plan(
        base_month=str(obj["base_month"]).strip(),
        price_unit=str(obj.get("price_unit", "")).strip(),
        analysis_target=str(obj.get("analysis_target", "")).strip(),
        model_assumption=str(obj.get("model_assumption", "")).strip(),
        replacement_semantics=str(obj.get("replacement_semantics", "")).strip(),
        uncertainty=str(obj.get("uncertainty", "")).strip(),
        output=str(obj.get("output", "")).strip(),
        scope=str(obj.get("scope", "")).strip(),
    )


def load_inputs(quotes_path: str | Path, products_path: str | Path, plan_path: str | Path) -> Inputs:
    quotes = load_quotes(quotes_path)
    products = load_products(products_path)
    plan = load_plan(plan_path)
    warnings: list[str] = []

    months = sorted({q.month for q in quotes})
    if plan.base_month not in months:
        raise InputError(
            f"base_month {plan.base_month!r} not present in quotes months {months}"
        )
    if not plan.price_unit:
        warnings.append("plan.price_unit is empty; assuming one currency and one fixed physical unit")
    unknown = sorted({q.item_id for q in quotes} - set(products))
    if unknown:
        raise InputError(f"quotes reference items absent from products.csv: {unknown}")
    if products:
        fam_months: dict[str, set[str]] = {}
        for q in quotes:
            fam_months.setdefault(products[q.item_id].family, set()).add(q.month)
        for fam, ms in fam_months.items():
            seen_outside = [m for m in months if m not in ms]
            if seen_outside:
                warnings.append(
                    f"family {fam!r} has no observed SKU in month(s) {sorted(seen_outside)}; "
                    "no overlap-based bridge exists for those months"
                )
    return Inputs(quotes=quotes, products=products, plan=plan, warnings=warnings)


def validate_months(quotes: Iterable[Quote], base_month: str) -> list[str]:
    months = sorted({q.month for q in quotes})
    if base_month not in months:
        raise InputError(f"base_month {base_month!r} absent; months are {months}")
    return months
