"""Exact identification diagnostics for the additive log-price model.

The model is

    log price(s, m, i) = item_level(i) + month_effect(m) + supplier_level(s) + error

with ``month_effect(base_month) = 0`` and ``supplier_level(ref_supplier) = 0``.
Those are the only conventional normalizations.  Every other linear
combination is either exactly identifiable from the observed quote pattern or
it is not; deciding that is a rational linear-algebra question, not a rank
threshold and not something a numerical least-squares routine may decide.

This module therefore works with exact rational arithmetic.  It returns

* the null space of the (centered) design matrix,
* which unit parameters and contrasts are identifiable,
* a maximal independent set of design columns (the RREF pivot columns) that a
  caller may use to fit without ever dropping a column numerically,
* item/month/supplier graph components for human-readable evidence,
* concrete confounding directions when a month is not identifiable,
* sole-supplier cells, critical suppliers and bridge cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Sequence

from .data import Quote

COLUMN_KINDS = ("item", "month", "supplier")


@dataclass(frozen=True)
class Columns:
    """Fixed, deterministic parameter ordering of the centered design."""

    base_month: str
    ref_supplier: str
    items: tuple[str, ...]
    months_nonbase: tuple[str, ...]
    suppliers_nonref: tuple[str, ...]
    labels: tuple[tuple[str, str], ...]
    index: dict[tuple[str, str], int]

    @property
    def ncol(self) -> int:
        return len(self.labels)


def build_columns(quotes: Sequence[Quote], base_month: str) -> Columns:
    items = tuple(sorted({q.item_id for q in quotes}))
    months = tuple(sorted({q.month for q in quotes}))
    suppliers = tuple(sorted({q.supplier_id for q in quotes}))
    if base_month not in months:
        raise ValueError(f"base month {base_month!r} missing from months {months}")
    ref = suppliers[0]
    months_nonbase = tuple(m for m in months if m != base_month)
    suppliers_nonref = tuple(s for s in suppliers if s != ref)
    labels: list[tuple[str, str]] = []
    labels.extend(("item", i) for i in items)
    labels.extend(("month", m) for m in months_nonbase)
    labels.extend(("supplier", s) for s in suppliers_nonref)
    index = {lab: k for k, lab in enumerate(labels)}
    return Columns(
        base_month=base_month,
        ref_supplier=ref,
        items=items,
        months_nonbase=months_nonbase,
        suppliers_nonref=suppliers_nonref,
        labels=tuple(labels),
        index=index,
    )


def unique_cells(quotes: Iterable[Quote], present: frozenset[str] | None = None) -> list[tuple[str, str, str]]:
    """Sorted unique (supplier, month, item) cells, optionally restricted."""
    cells = {
        (q.supplier_id, q.month, q.item_id)
        for q in quotes
        if present is None or q.supplier_id in present
    }
    return sorted(cells)


def design_rows(cells: Sequence[tuple[str, str, str]], cols: Columns) -> list[list[int]]:
    """Integer 0/1 design rows in the centered parametrization."""
    ncol = cols.ncol
    rows: list[list[int]] = []
    for supplier, month, item in cells:
        row = [0] * ncol
        row[cols.index[("item", item)]] = 1
        if month != cols.base_month:
            row[cols.index[("month", month)]] = 1
        if supplier != cols.ref_supplier:
            row[cols.index[("supplier", supplier)]] = 1
        rows.append(row)
    return rows


def exact_rref(rows: Sequence[Sequence[int]], ncol: int) -> tuple[list[int], list[list[Fraction]]]:
    """Reduced row echelon form over Q.

    Returns ``(pivot_columns, null_space_basis)``.  The pivot columns are a
    maximal independent set of design columns: fitting on exactly those columns
    spans the same column space and never relies on a numerical rank decision.
    """
    mat = [[Fraction(x) for x in row] for row in rows]
    nrow = len(mat)
    pivots: list[int] = []
    r = 0
    for c in range(ncol):
        pivot_row = None
        for i in range(r, nrow):
            if mat[i][c] != 0:
                pivot_row = i
                break
        if pivot_row is None:
            continue
        mat[r], mat[pivot_row] = mat[pivot_row], mat[r]
        scale = mat[r][c]
        mat[r] = [x / scale for x in mat[r]]
        for i in range(nrow):
            if i != r and mat[i][c] != 0:
                factor = mat[i][c]
                mat[i] = [mat[i][j] - factor * mat[r][j] for j in range(ncol)]
        pivots.append(c)
        r += 1
        if r == nrow:
            break

    pivot_set = set(pivots)
    free = [c for c in range(ncol) if c not in pivot_set]
    basis: list[list[Fraction]] = []
    for fc in free:
        vec = [Fraction(0)] * ncol
        vec[fc] = Fraction(1)
        for pi, pc in enumerate(pivots):
            vec[pc] = -mat[pi][fc]
        basis.append(vec)
    return pivots, basis


class ExactSystem:
    """Exact identification state for one set of observed cells."""

    def __init__(self, cells: Sequence[tuple[str, str, str]], cols: Columns):
        self.cells = list(cells)
        self.cols = cols
        self.pivots, self.nullspace = exact_rref(design_rows(cells, cols), cols.ncol)

    def identified(self, contrast: Sequence[int | Fraction]) -> bool:
        """True iff the linear combination is invariant over the null space."""
        return self.identified_static(contrast, self.nullspace)

    @staticmethod
    def identified_static(
        contrast: Sequence[int | Fraction], nullspace: Sequence[Sequence[Fraction]]
    ) -> bool:
        """Allocation-light exact test reusable with a cached null space."""
        for vec in nullspace:
            total = Fraction(0)
            for c, v in zip(contrast, vec):
                if c:
                    total += Fraction(c) * v
            if total != 0:
                return False
        return True

    def unit_identified(self, label: tuple[str, str]) -> bool:
        c = [0] * self.cols.ncol
        c[self.cols.index[label]] = 1
        return self.identified(c)

    def month_contrast(self, month: str) -> list[int]:
        """Contrast for month effect relative to the base month (all-zero for base)."""
        c = [0] * self.cols.ncol
        if month != self.cols.base_month:
            c[self.cols.index[("month", month)]] = 1
        return c

    def item_contrast(self, new: str, old: str) -> list[int]:
        c = [0] * self.cols.ncol
        c[self.cols.index[("item", new)]] += 1
        c[self.cols.index[("item", old)]] -= 1
        return c

    def supplier_contrast(self, supplier: str) -> list[int]:
        c = [0] * self.cols.ncol
        if supplier != self.cols.ref_supplier:
            c[self.cols.index[("supplier", supplier)]] = 1
        return c

    def column_position(self, label: tuple[str, str]) -> int | None:
        j = self.cols.index.get(label)
        if j is None:
            return None
        try:
            return self.pivots.index(j)
        except ValueError:
            return None

    def column_space_dimension(self) -> int:
        return len(self.pivots)


# ---------------------------------------------------------------------------
# Human-readable graph evidence
# ---------------------------------------------------------------------------


@dataclass
class Component:
    component_id: int
    items: list[str]
    months: list[str]
    suppliers: list[str]
    cells: list[tuple[str, str, str]]
    quote_ids: list[str]


def _cell_quote_map(quotes: Sequence[Quote]) -> dict[tuple[str, str, str], list[Quote]]:
    out: dict[tuple[str, str, str], list[Quote]] = {}
    for q in quotes:
        out.setdefault((q.supplier_id, q.month, q.item_id), []).append(q)
    return out


def item_month_components(
    quotes: Sequence[Quote], present: frozenset[str] | None = None
) -> list[Component]:
    """Connected components of the bipartite item-month graph.

    This graph deliberately ignores product families: an edge exists only for
    an actually observed quote.  It is descriptive evidence; final
    identification is always decided by :class:`ExactSystem`.
    """
    adj: dict[tuple[str, str], set[tuple[str, str]]] = {}
    edges: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for q in quotes:
        if present is not None and q.supplier_id not in present:
            continue
        a = ("item", q.item_id)
        b = ("month", q.month)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
        edges.append((a, b))
    cell_map = _cell_quote_map([q for q in quotes if present is None or q.supplier_id in present])

    seen: set[tuple[str, str]] = set()
    comps: list[Component] = []
    for node in sorted(adj):
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        items: set[str] = set()
        months: set[str] = set()
        while stack:
            u = stack.pop()
            if u[0] == "item":
                items.add(u[1])
            else:
                months.add(u[1])
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        cells = sorted(
            (s, m, i)
            for (s, m, i) in cell_map
            if i in items and m in months
        )
        suppliers = sorted({s for (s, _, _) in cells})
        quote_ids = sorted(q.quote_id for cell in cells for q in cell_map[cell])
        comps.append(
            Component(
                component_id=len(comps),
                items=sorted(items),
                months=sorted(months),
                suppliers=suppliers,
                cells=cells,
                quote_ids=quote_ids,
            )
        )
    return comps


def _component_of(comps: Sequence[Component]) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for comp in comps:
        for i in comp.items:
            out[("item", i)] = comp.component_id
        for m in comp.months:
            out[("month", m)] = comp.component_id
    return out


def _bridge_quotes(
    quotes: Sequence[Quote], deleted_supplier: str, present: frozenset[str]
) -> list[dict]:
    """Quotes of the deleted supplier that join different components of the graph
    built from the remaining suppliers.

    These are the observations that, if the supplier's whole trajectory is
    removed, carry the only remaining evidence separating month movement from
    product level differences.
    """
    comps = item_month_components(quotes, present=present)
    comp_of = _component_of(comps)
    out = []
    for q in quotes:
        if q.supplier_id != deleted_supplier:
            continue
        a = comp_of.get(("item", q.item_id))
        b = comp_of.get(("month", q.month))
        if a is None or b is None or a != b:
            out.append(
                {
                    "quote_id": q.quote_id,
                    "supplier_id": q.supplier_id,
                    "month": q.month,
                    "item_id": q.item_id,
                    "price": q.price,
                    "reason": (
                        f"connects item component {a} to month component {b} "
                        "in the remaining-supplier graph"
                    ),
                }
            )
    return sorted(out, key=lambda r: r["quote_id"])


def _quote_records(quotes: Sequence[Quote], cell_filter) -> list[dict]:
    out = []
    for q in quotes:
        if cell_filter(q):
            out.append(
                {
                    "quote_id": q.quote_id,
                    "supplier_id": q.supplier_id,
                    "month": q.month,
                    "item_id": q.item_id,
                    "price": q.price,
                }
            )
    return sorted(out, key=lambda r: r["quote_id"])


@dataclass
class Diagnosis:
    columns: Columns
    exact: ExactSystem
    components: list[Component]
    month_identification: dict[str, bool]
    month_component: dict[str, int]
    critical_suppliers: dict[str, list[str]]
    deletion: list[dict]
    sole_supplier_cells: list[dict]
    bridge_cells: dict[str, list[dict]]
    confounding: dict[str, dict]
    family_cross_component: list[dict]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "normalization": {
                "base_month": self.columns.base_month,
                "reference_supplier": self.columns.ref_supplier,
                "base_month_effect_fixed_to": 0.0,
                "reference_supplier_level_fixed_to": 0.0,
            },
            "design": {
                "n_parameters": self.columns.ncol,
                "n_independent_columns": self.exact.column_space_dimension(),
                "n_null_directions": len(self.exact.nullspace),
                "columns": [
                    {"kind": k, "name": n} for k, n in self.columns.labels
                ],
                "pivot_columns": [
                    {"kind": self.columns.labels[j][0], "name": self.columns.labels[j][1]}
                    for j in self.exact.pivots
                ],
                "note": (
                    "Pivot columns are a maximal independent set of the design; "
                    "they are used for fitting so no numerical routine decides rank."
                ),
            },
            "components": [
                {
                    "component_id": c.component_id,
                    "items": c.items,
                    "months": c.months,
                    "suppliers": c.suppliers,
                    "n_cells": len(c.cells),
                    "n_quotes": len(c.quote_ids),
                    "quote_ids": c.quote_ids,
                }
                for c in self.components
            ],
            "months": [
                {
                    "month": m,
                    "identified": self.month_identification[m],
                    "component_id": self.month_component.get(m),
                    "critical_suppliers": self.critical_suppliers.get(m, []),
                    "confounding_direction": self.confounding.get(m),
                }
                for m in sorted(self.month_component)
            ],
            "sole_supplier_cells": self.sole_supplier_cells,
            "bridge_cells": self.bridge_cells,
            "deletion": self.deletion,
            "family_cross_component": self.family_cross_component,
            "warnings": self.warnings,
        }


def diagnose(
    quotes: Sequence[Quote],
    products: dict,
    base_month: str,
) -> Diagnosis:
    cols = build_columns(quotes, base_month)
    cells = unique_cells(quotes)
    exact = ExactSystem(cells, cols)
    comps = item_month_components(quotes)
    comp_of = _component_of(comps)

    suppliers = sorted({q.supplier_id for q in quotes})
    months = sorted({q.month for q in quotes})
    sole_map: dict[tuple[str, str, str], list[Quote]] = {}
    for q in quotes:
        sole_map.setdefault((q.supplier_id, q.month, q.item_id), []).append(q)
    cell_suppliers: dict[tuple[str, str], set[str]] = {}
    for q in quotes:
        cell_suppliers.setdefault((q.month, q.item_id), set()).add(q.supplier_id)
    sole_supplier_cells = []
    for (s, m, i), qs in sorted(sole_map.items()):
        if len(cell_suppliers[(m, i)]) == 1:
            sole_supplier_cells.append(
                {
                    "item_id": i,
                    "month": m,
                    "supplier_id": s,
                    "quotes": [
                        {"quote_id": q.quote_id, "price": q.price} for q in sorted(qs, key=lambda z: z.quote_id)
                    ],
                }
            )

    month_identification = {
        m: exact.identified(exact.month_contrast(m)) for m in months
    }
    month_component = {m: comp_of.get(("month", m), -1) for m in months}

    # Confounding directions: a null-space vector with a non-zero month entry.
    confounding: dict[str, dict] = {}
    for m in months:
        if month_identification[m]:
            continue
        j = cols.index.get(("month", m))
        if j is None:
            continue
        chosen = None
        for vec in exact.nullspace:
            if vec[j] != 0:
                chosen = vec
                break
        if chosen is None:
            continue
        items = {
            name: str(chosen[cols.index[("item", name)]])
            for name in cols.items
            if chosen[cols.index[("item", name)]] != 0
        }
        mons = {
            name: str(chosen[cols.index[("month", name)]])
            for name in cols.months_nonbase
            if chosen[cols.index[("month", name)]] != 0
        }
        sups = {
            name: str(chosen[cols.index[("supplier", name)]])
            for name in cols.suppliers_nonref
            if chosen[cols.index[("supplier", name)]] != 0
        }
        # Quotes witnessing the invariance: those in the affected month block.
        affected_months = set(mons)
        witness = _quote_records(
            quotes, lambda q: q.month in affected_months
        )
        confounding[m] = {
            "null_direction": {
                "item_levels": items,
                "month_effects": mons,
                "supplier_levels": sups,
            },
            "explanation": (
                "Adding this direction to the parameters leaves every observed "
                "quote unchanged, so the month effect cannot be separated from "
                "the listed item/supplier levels."
            ),
            "n_witness_quotes": len(witness),
            "witness_quote_ids": [r["quote_id"] for r in witness],
        }

    # Delete-one-supplier identification and bridge evidence.
    deletion = []
    critical_suppliers: dict[str, list[str]] = {m: [] for m in months}
    bridge_cells: dict[str, list[dict]] = {}
    for s in suppliers:
        present = frozenset(x for x in suppliers if x != s)
        sub_cells = unique_cells(quotes, present=present)
        sub_exact = ExactSystem(sub_cells, cols)
        sub_comps = item_month_components(quotes, present=present)
        sub_comp_of = _component_of(sub_comps)
        sub_ident = {m: sub_exact.identified(sub_exact.month_contrast(m)) for m in months}
        newly_unidentified = [
            m for m in months if month_identification[m] and not sub_ident[m]
        ]
        for m in newly_unidentified:
            critical_suppliers[m].append(s)
        deletion.append(
            {
                "removed_supplier": s,
                "n_quotes_remaining": sum(1 for q in quotes if q.supplier_id != s),
                "identified_months": [m for m in months if sub_ident[m]],
                "unidentified_months": [m for m in months if not sub_ident[m]],
                "newly_unidentified_months": newly_unidentified,
                "components": [
                    {
                        "component_id": c.component_id,
                        "items": c.items,
                        "months": c.months,
                        "suppliers": c.suppliers,
                    }
                    for c in sub_comps
                ],
                "month_component": {m: sub_comp_of.get(("month", m)) for m in months},
            }
        )
        if newly_unidentified:
            bridge_cells[s] = _bridge_quotes(quotes, s, present)

    # Families that span components but do not connect them by an observed SKU.
    family_cross = []
    if products:
        by_family: dict[str, list[str]] = {}
        for item, prod in products.items():
            by_family.setdefault(prod.family, []).append(item)
        for family, items in sorted(by_family.items()):
            comps_seen = sorted({comp_of.get(("item", i)) for i in items if ("item", i) in comp_of})
            if len(comps_seen) > 1:
                family_cross.append(
                    {
                        "family": family,
                        "items": sorted(items),
                        "components": comps_seen,
                        "note": (
                            "Family membership spans components, but no single SKU "
                            "is observed in both; the family name is not used as an "
                            "equivalence key."
                        ),
                    }
                )

    warns = []
    for m in months:
        if not month_identification[m]:
            warns.append(
                f"month {m}: month effect is not identifiable relative to "
                f"{base_month}; see confounding_direction"
            )
        elif critical_suppliers[m]:
            warns.append(
                f"month {m}: identified only because supplier(s) "
                f"{critical_suppliers[m]} bridge otherwise disconnected SKU groups"
            )

    return Diagnosis(
        columns=cols,
        exact=exact,
        components=comps,
        month_identification=month_identification,
        month_component=month_component,
        critical_suppliers=critical_suppliers,
        deletion=deletion,
        sole_supplier_cells=sole_supplier_cells,
        bridge_cells=bridge_cells,
        confounding=confounding,
        family_cross_component=family_cross,
        warnings=warns,
    )
