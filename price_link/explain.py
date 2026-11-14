"""Text explanation of a fitted result, plus JSON/Markdown/HTML report writing."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .estimator import AnalysisResult


def _fmt_index(est) -> str:
    if est.status == "insufficient_support":
        return "NOT IDENTIFIED"
    if est.log_value is None:
        return "n/a"
    return f"{100.0 * est.value_ratio:8.2f}"


def _fmt_interval(est) -> str:
    if est.ci_ratio is None:
        return "-"
    lo, hi = 100.0 * est.ci_ratio[0], 100.0 * est.ci_ratio[1]
    tag = "" if est.interval_reportable else "  (conditional, withheld)"
    return f"[{lo:.2f}, {hi:.2f}]{tag}"


def explain(result: AnalysisResult) -> str:
    lines: list[str] = []
    add = lines.append
    d = result.diagnosis
    add("=" * 78)
    add("COMPARABLE-MONTH PRICE INDEX  (supplier-balanced additive log-price model)")
    add("=" * 78)
    add("")
    add(f"Quotes: {len(result.inputs.quotes)}   months: {len(result.months)}   "
        f"suppliers: {len({q.supplier_id for q in result.inputs.quotes})}   "
        f"SKUs: {len({q.item_id for q in result.inputs.quotes})}")
    add(f"Base month (index = 100): {result.inputs.plan.base_month}")
    add(f"Normalization: month_effect({result.inputs.plan.base_month}) = 0, "
        f"supplier_level({d.columns.ref_supplier}) = 0")
    add("")

    add("-" * 78)
    add("1. IDENTIFICATION DIAGNOSTIC (exact rational null space)")
    add("-" * 78)
    add(f"Design parameters: {d.columns.ncol}   "
        f"independent columns: {d.exact.column_space_dimension()}   "
        f"null directions: {len(d.exact.nullspace)}")
    add("Pivot columns (exactly independent fit basis): "
        + ", ".join(f"{k}:{n}" for k, n in (d.columns.labels[j] for j in d.exact.pivots)))
    add("")
    add("Observed item-month graph components (edges are actual quotes, never families):")
    for comp in d.components:
        add(f"  component {comp.component_id}:")
        add(f"    items  : {comp.items}")
        add(f"    months : {comp.months}")
        add(f"    suppliers present: {comp.suppliers}")
        add(f"    cells/quotes: {len(comp.cells)}/{len(comp.quote_ids)}")
    add("")
    add("Month index identification:")
    for m in result.months:
        ident = d.month_identification[m]
        status = "identified" if ident else "NOT IDENTIFIED"
        add(f"  {m}: {status}")
        if not ident:
            conf = d.confounding.get(m, {})
            nd = conf.get("null_direction", {})
            add(f"    confounding null direction: item_levels={nd.get('item_levels')}, "
                f"month_effects={nd.get('month_effects')}")
            add(f"    witness quotes ({conf.get('n_witness_quotes', 0)}): "
                f"{conf.get('witness_quote_ids')}")
    add("")
    add("Sole-supplier cells (cells quoted by exactly one supplier):")
    if not d.sole_supplier_cells:
        add("  none")
    for rec in d.sole_supplier_cells:
        qids = [q["quote_id"] for q in rec["quotes"]]
        prices = [q["price"] for q in rec["quotes"]]
        add(f"  {rec['item_id']} in {rec['month']}: {rec['supplier_id']} "
            f"({qids}, price {prices})")
    add("")
    add("Critical bridge suppliers (removing a whole supplier makes a month unidentified):")
    crit = {m: v for m, v in d.critical_suppliers.items() if v}
    if not crit:
        add("  none")
    for m, sups in sorted(crit.items()):
        add(f"  {m}: {sups}")
        for s in sups:
            bridge = d.bridge_cells.get(s, [])
            if bridge:
                add(f"    bridge quotes that carry the link when {s} is removed:")
                for q in bridge[:12]:
                    add(f"      {q['quote_id']} {q['supplier_id']} {q['month']} "
                        f"{q['item_id']} price={q['price']}")
                if len(bridge) > 12:
                    add(f"      ... {len(bridge) - 12} more")
    add("")
    if d.family_cross_component:
        add("Families spanning disconnected components (NOT usable as an equivalence key):")
        for rec in d.family_cross_component:
            add(f"  {rec['family']}: items {rec['items']} span components {rec['components']}")
        add("")
    add("Delete-one-supplier identification sweep:")
    for rec in d.deletion:
        add(f"  remove {rec['removed_supplier']}: "
            f"identified months {rec['identified_months']}; "
            f"unidentified {rec['unidentified_months']}; "
            f"graph components now {len(rec['components'])}")
    add("")

    add("-" * 78)
    add("2. MONTH INDICES (base = 100)")
    add("-" * 78)
    add(f"  {'month':10s} {'index':>8s}  {'status':22s} {'95% interval (same base)':>28s}")
    for m in result.months:
        est = result.month_estimates[m]
        add(f"  {m:10s} {_fmt_index(est):>8s}  {est.status:22s} {_fmt_interval(est):>28s}")
    add("")
    add("Replicate support per month (whole-supplier resampling):")
    for m in result.months:
        est = result.month_estimates[m]
        if m == result.inputs.plan.base_month:
            add(f"  {m}: base normalization, {est.valid_replicates}/{est.total_replicates} "
                "replicates identified")
            continue
        add(f"  {m}: {est.valid_replicates}/{est.total_replicates} "
            f"({(est.valid_fraction or 0):.1%}) replicates retained identification; "
            f"interval_reportable={est.interval_reportable}")
        if est.failures:
            add(f"      failure reasons: {est.failures}")
        if est.status == "insufficient_replicate_support":
            add(f"      reason: {est.reason}")
    add("")

    add("-" * 78)
    add("3. PERSISTENT SKU LEVELS AND FAMILY PAIR SPREADS")
    add("-" * 78)
    add("Centered item levels (log scale; identified from overlap only):")
    for item in sorted(result.item_level_estimates):
        est = result.item_level_estimates[item]
        if est.log_value is None:
            add(f"  {item}: {est.status}")
        else:
            add(f"  {item}: {est.log_value:+.4f}  (exp = {est.value_ratio:.4f})")
    add("")
    add("Measured price level ratios for old/new SKUs within a family:")
    for rec in result.item_spreads:
        if rec["log_value"] is None:
            add(f"  {rec['family']} {rec['item_b']} / {rec['item_a']}: "
                f"{rec['status']}  ({rec.get('reason')})")
        else:
            ci = rec.get("ci_price_ratio")
            frac = rec.get("valid_fraction")
            if ci:
                ci_txt = f"  CI [{ci[0]:.4f}, {ci[1]:.4f}]"
                if frac is not None:
                    ci_txt += f" (replicate support {frac:.1%})"
            elif rec["status"] == "insufficient_replicate_support":
                ci_txt = "  interval withheld (insufficient replicate support)"
            else:
                ci_txt = ""
            add(f"  {rec['family']} {rec['item_b']} / {rec['item_a']}: "
                f"{rec['price_ratio_b_over_a']:.4f}{ci_txt}  (log {rec['log_value']:+.4f})")
    add("  NOTE: these ratios are measured level differences, not quality-equality claims.")
    add("")

    add("-" * 78)
    add("4. SUPPLIER TRAJECTORY LEVELS (centered; not a price index)")
    add("-" * 78)
    for rec in result.supplier_levels:
        if rec.get("centered_log_level") is None:
            add(f"  {rec['supplier_id']}: {rec['status']} ({rec.get('reason')})")
        else:
            add(f"  {rec['supplier_id']}: {rec['centered_log_level']:+.4f} "
                f"(ratio {rec['level_ratio_to_mean']:.4f}), "
                f"{rec['n_quotes']} quotes, months {rec['months']}, items {rec['items']}")
    add("")

    add("-" * 78)
    add("5. STABILITY: LEAVE-ONE-SUPPLIER-OUT")
    add("-" * 78)
    for rec in result.stability["leave_one_supplier_out"]:
        parts = []
        for m in result.months:
            v = rec["months"][m]
            parts.append(f"{m}: " + (f"{v['index_base100']:.2f}" if v["index_base100"] is not None else "UNIDENTIFIED"))
        add(f"  remove {rec['removed_supplier']}: " + "; ".join(parts))
    add("")

    add("-" * 78)
    add("6. WEIGHTS, OUTLIERS, ERROR ASSUMPTION")
    add("-" * 78)
    w = result.weights_report
    add(f"  {w['scheme']}")
    add(f"  effective number of quotes: {w['effective_n_quotes']:.1f}; "
        f"downweighted: {w['n_downweighted']}")
    add(f"  residual scale (log): {w['residual_scale_log']:.6f}; "
        f"IRLS iterations {w['irls_iterations']}, converged={w['irls_converged']}")
    for rec in w["per_supplier"]:
        add(f"    {rec['supplier_id']}: {rec['n_quotes']} quotes, "
            f"weight share {rec['final_weight_share']:.1%}, "
            f"mean Huber weight {rec['mean_huber_weight']:.3f}")
    if result.outliers:
        add("  Outlying quotes (Huber-downweighted, retained):")
        for rec in result.outliers:
            add(f"    {rec['quote_id']} {rec['supplier_id']} {rec['month']} {rec['item_id']} "
                f"z={rec['standardized_residual']:+.2f} huber_w={rec['huber_weight']:.3f}")
    else:
        add("  no quote falls below Huber weight 0.85 or |z| > 2.5")
    add("")

    add("-" * 78)
    add("7. TRACEABILITY")
    add("-" * 78)
    add("Every quote is decomposed in the result JSON / quote_trace.csv as")
    add("  log_price = item_level + month_effect + supplier_level + residual,")
    add("with per-quote identification flags and Huber weights.")
    add(f"Trace note: {result.trace_note}")
    add("")
    add("-" * 78)
    add("8. WARNINGS AND LIMITATIONS")
    add("-" * 78)
    for wmsg in result.warnings:
        add(f"  WARNING: {wmsg}")
    for lim in result.limitations:
        add(f"  LIMIT: {lim}")
    add("")
    return "\n".join(lines)


def result_summary(result: AnalysisResult) -> dict[str, Any]:
    """Compact machine-readable summary across cases."""
    return {
        "base_month": result.inputs.plan.base_month,
        "months": result.months,
        "month_index": {
            m: {
                "status": result.month_estimates[m].status,
                "index_base100": (
                    100.0 * result.month_estimates[m].value_ratio
                    if result.month_estimates[m].value_ratio is not None
                    else None
                ),
                "ci_ratio": result.month_estimates[m].ci_ratio,
                "interval_reportable": result.month_estimates[m].interval_reportable,
                "valid_fraction": result.month_estimates[m].valid_fraction,
            }
            for m in result.months
        },
        "components": [
            {"items": c.items, "months": c.months, "suppliers": c.suppliers}
            for c in result.diagnosis.components
        ],
        "critical_bridge_suppliers": {
            m: v for m, v in result.diagnosis.critical_suppliers.items() if v
        },
        "item_spreads": [
            {
                "family": r["family"],
                "item_a": r["item_a"],
                "item_b": r["item_b"],
                "status": r["status"],
                "price_ratio_b_over_a": r["price_ratio_b_over_a"],
                "ci_price_ratio": r.get("ci_price_ratio"),
            }
            for r in result.item_spreads
        ],
        "warnings": result.warnings,
    }


def write_reports(result: AnalysisResult, out_dir: str | Path, case_name: str) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_json = out / f"{case_name}.result.json"
    result_md = out / f"{case_name}.report.md"
    result_html = out / f"{case_name}.report.html"
    result_txt = out / f"{case_name}.explain.txt"
    result_trace = out / f"{case_name}.quote_trace.csv"

    result_json.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    _write_trace_csv(result, result_trace)
    text = explain(result)
    result_txt.write_text(text, encoding="utf-8")
    result_md.write_text(_to_markdown(result, case_name), encoding="utf-8")
    result_html.write_text(_to_html(result, case_name), encoding="utf-8")
    return {
        "json": str(result_json),
        "markdown": str(result_md),
        "html": str(result_html),
        "text": str(result_txt),
        "quote_trace_csv": str(result_trace),
    }


def _write_trace_csv(result: AnalysisResult, path: Path) -> None:
    import csv

    fields = [
        "quote_id", "supplier_id", "month", "item_id", "price", "log_price",
        "item_level", "month_effect", "supplier_level", "predicted_log_price",
        "residual", "huber_weight", "item_level_identified", "supplier_level_identified",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in result.quote_trace:
            writer.writerow({k: row[k] for k in fields})


def _to_markdown(result: AnalysisResult, case_name: str) -> str:
    d = result.diagnosis
    base = result.inputs.plan.base_month
    L: list[str] = []
    L.append(f"# Comparable-month price index — `{case_name}`\n")
    L.append(f"Base month **{base}** (index = 100). Additive log-price model with "
             f"persistent SKU levels and supplier levels; families are **not** equivalence keys.\n")
    L.append("## Month index\n")
    L.append("| month | index (base 100) | status | 95% interval | replicates identified | interval reportable |")
    L.append("|---|---:|---|---|---:|---|")
    for m in result.months:
        est = result.month_estimates[m]
        idx = "—" if est.value_ratio is None else f"{100.0 * est.value_ratio:.2f}"
        if est.ci_ratio and est.interval_reportable:
            ci = f"[{100*est.ci_ratio[0]:.2f}, {100*est.ci_ratio[1]:.2f}]"
        else:
            ci = "withheld"
        L.append(f"| {m} | {idx} | {est.status} | {ci} | "
                 f"{est.valid_replicates}/{est.total_replicates} "
                 f"({(est.valid_fraction or 0):.1%}) | {est.interval_reportable} |")
    L.append("")
    L.append("## Identification evidence\n")
    L.append(f"- Design: {d.columns.ncol} parameters, "
             f"{d.exact.column_space_dimension()} exactly independent columns, "
             f"{len(d.exact.nullspace)} null direction(s).")
    L.append(f"- Normalization: month effect at base = 0; supplier level "
             f"`{d.columns.ref_supplier}` = 0. These are the only normalizations.")
    L.append("- Item–month graph components (edges are real quotes only):")
    for c in d.components:
        L.append(f"  - component {c.component_id}: items {c.items}; months {c.months}; suppliers {c.suppliers}")
    L.append("")
    for m in result.months:
        if not d.month_identification[m]:
            conf = d.confounding.get(m, {})
            L.append(f"- **{m} is not identifiable.** Confounding null direction: "
                     f"item levels {conf.get('null_direction', {}).get('item_levels')}, "
                     f"month effects {conf.get('null_direction', {}).get('month_effects')}. "
                     f"Witness quotes: {conf.get('witness_quote_ids')}.")
    crit = {m: v for m, v in d.critical_suppliers.items() if v}
    for m, sups in sorted(crit.items()):
        L.append(f"- **{m}** is identified only through critical bridge supplier(s) {sups}.")
        for s in sups:
            br = d.bridge_cells.get(s, [])
            if br:
                L.append(f"  - bridge quotes for `{s}`: "
                         + ", ".join(f"{q['quote_id']} ({q['month']} {q['item_id']} {q['price']})" for q in br))
    L.append("")
    L.append("## SKU level spreads within families (measured, not equality claims)\n")
    L.append("| family | pair | ratio B/A | 95% interval | status |")
    L.append("|---|---|---:|---|---|")
    for rec in result.item_spreads:
        ratio = "—" if rec["price_ratio_b_over_a"] is None else f"{rec['price_ratio_b_over_a']:.4f}"
        ci = rec.get("ci_price_ratio")
        cit = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "—"
        L.append(f"| {rec['family']} | {rec['item_b']} / {rec['item_a']} | {ratio} | {cit} | {rec['status']} |")
    L.append("")
    L.append("## Leave-one-supplier-out stability\n")
    L.append("| removed | " + " | ".join(result.months) + " |")
    L.append("|---|" + "---:|" * len(result.months))
    for rec in result.stability["leave_one_supplier_out"]:
        cells = []
        for m in result.months:
            v = rec["months"][m]
            cells.append("UNIDENT." if v["index_base100"] is None else f"{v['index_base100']:.2f}")
        L.append(f"| {rec['removed_supplier']} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Warnings\n")
    for w in result.warnings:
        L.append(f"- {w}")
    L.append("")
    L.append("## Limitations\n")
    for lim in result.limitations:
        L.append(f"- {lim}")
    L.append("")
    return "\n".join(L)


def _to_html(result: AnalysisResult, case_name: str) -> str:
    md = _to_markdown(result, case_name)
    # Minimal, dependency-free converter: emit the markdown in a <pre> block
    # plus a small table of month indices for readability.
    esc = html.escape(md)
    d = result.diagnosis
    rows = []
    for m in result.months:
        est = result.month_estimates[m]
        idx = "—" if est.value_ratio is None else f"{100.0 * est.value_ratio:.2f}"
        if est.ci_ratio and est.interval_reportable:
            ci = f"[{100*est.ci_ratio[0]:.2f}, {100*est.ci_ratio[1]:.2f}]"
        elif est.status == "insufficient_replicate_support":
            ci = "withheld (insufficient replicates)"
        else:
            ci = "—"
        cls = "ok" if est.interval_reportable else "warn"
        rows.append(
            f"<tr class='{cls}'><td>{html.escape(m)}</td><td>{idx}</td>"
            f"<td>{html.escape(est.status)}</td><td>{html.escape(ci)}</td></tr>"
        )
    ident = "".join(
        f"<li>{html.escape(m)}: {'identified' if d.month_identification[m] else 'NOT IDENTIFIED'}"
        f"{'' if d.month_identification[m] else ' — ' + html.escape(str(d.confounding.get(m, {}).get('witness_quote_ids')))}</li>"
        for m in result.months
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Comparable price index — {html.escape(case_name)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;line-height:1.5}}
table{{border-collapse:collapse;margin:1rem 0}}td,th{{border:1px solid #ccc;padding:.35rem .6rem;text-align:left}}
tr.warn{{background:#fff3cd}} tr.ok{{background:#e8f5e9}}
pre{{background:#f6f8fa;padding:1rem;overflow:auto;font-size:.85rem}}
</style></head><body>
<h1>Comparable-month price index — {html.escape(case_name)}</h1>
<p>Base month <b>{html.escape(result.inputs.plan.base_month)}</b> = 100. Additive log-price model;
product families are replacement labels only, not equivalence keys.</p>
<table><thead><tr><th>month</th><th>index</th><th>status</th><th>95% interval</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Identification</h2><ul>{ident}</ul>
<pre>{esc}</pre>
</body></html>
"""
