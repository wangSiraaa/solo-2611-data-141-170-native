"""build 命令主流程：读 plan / 注释 / BAM → 片段归组 → 兼容性评估 → 输出。"""
from __future__ import annotations

import json
import os

from .annotation import load_exons
from .compat import evaluate_fragment
from .fragments import (
    EQUIVALENCE,
    EXCL_ANOMALY,
    EXCL_DUPLICATE,
    EXCL_UNSUPPORTED_CIGAR,
    NO_COMPAT,
    ORPHAN,
    PAIRED,
    UNIQUE,
    collect_fragments,
)
from .report import (
    build_counts,
    status_tally,
    write_counts_csvs,
    write_fragments_csv,
)

STATUS_LABELS = {
    UNIQUE: "唯一兼容",
    EQUIVALENCE: "等价类(多候选)",
    NO_COMPAT: "无兼容",
    ORPHAN: "孤儿",
    EXCL_DUPLICATE: "排除-duplicate",
    EXCL_UNSUPPORTED_CIGAR: "排除-不支持的CIGAR",
    EXCL_ANOMALY: "排除-结构异常",
}


def load_plan(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        plan = json.load(fh)
    # 只支持 plan 声明的约定；其它约定显式拒绝而不是静默误判
    if plan.get("annotation_coordinates") != "zero-based half-open":
        raise NotImplementedError(f"不支持的注释坐标约定: {plan.get('annotation_coordinates')}")
    if plan.get("library") != "read1_matches_transcript_strand":
        raise NotImplementedError(f"不支持的文库链向约定: {plan.get('library')}")
    if plan.get("fragment_identity") != "query_name":
        raise NotImplementedError(f"不支持的片段标识: {plan.get('fragment_identity')}")
    for key in ("reference_name", "accepted_cigar"):
        if key not in plan:
            raise ValueError(f"plan 缺少必要字段: {key}")
    return plan


def run_build(bam_path: str, exons_path: str, plan_path: str, out_dir: str) -> dict:
    plan = load_plan(plan_path)
    transcripts = load_exons(exons_path, plan["reference_name"])
    fragments, record_diag, header_info = collect_fragments(bam_path, plan)

    # 兼容性评估（仅对通过过滤的成对片段）
    for frag in fragments:
        if frag.status == PAIRED:
            evaluate_fragment(frag, transcripts)

    # 数量闭合：每个 query_name 必落入且只落入一个类别
    tally = status_tally(fragments)
    n_groups = len(fragments)
    fragment_closure_ok = sum(tally.values()) == n_groups
    # 比对记录闭合：总记录 = 纳入分组的主比对 + 各类记录级排除
    kept = record_diag["kept_primary"]
    record_closure_ok = record_diag["total"] == kept + sum(
        record_diag[k]
        for k in ("secondary", "supplementary", "unmapped", "not_paired", "missing_cigar", "other_reference")
    ) and kept == sum(len(f.records) for f in fragments)
    if not fragment_closure_ok or not record_closure_ok:
        raise RuntimeError(f"闭合检查失败: tally={tally}, record_diag={record_diag}")

    unique_rows, eq_rows = build_counts(fragments, transcripts)

    diagnostics = {
        "inputs": {"bam": bam_path, "exons": exons_path, "plan": plan_path},
        "plan_applied": {
            "library": plan["library"],
            "exclude_flagged_duplicate": plan.get("exclude_flagged_duplicate"),
            "exclude_secondary_supplementary": plan.get("exclude_secondary_supplementary"),
            "accepted_cigar": plan["accepted_cigar"],
            "mate_gap": plan.get("mate_gap"),
            "compatibility": plan.get("compatibility"),
        },
        "bam_header": header_info,
        "alignments": record_diag,
        "fragments": {"total_query_names": n_groups, **tally},
        "closure": {
            "fragment_equation": " + ".join(f"{k}={v}" for k, v in tally.items())
            + f" = {sum(tally.values())} == total_query_names={n_groups}",
            "fragment_ok": fragment_closure_ok,
            "alignment_equation": (
                f"total={record_diag['total']} == kept_primary={kept} + "
                + " + ".join(
                    f"{k}={record_diag[k]}"
                    for k in ("secondary", "supplementary", "unmapped", "not_paired", "missing_cigar", "other_reference")
                )
            ),
            "alignment_ok": record_closure_ok,
        },
        "details": {
            "unique": {f.qname: sorted(f.compatible) for f in fragments if f.status == UNIQUE},
            "equivalence": {f.qname: sorted(f.compatible) for f in fragments if f.status == EQUIVALENCE},
            "no_compat": {f.qname: f.reason for f in fragments if f.status == NO_COMPAT},
            "orphan": {f.qname: f.reason for f in fragments if f.status == ORPHAN},
            "excluded_duplicate": [f.qname for f in fragments if f.status == EXCL_DUPLICATE],
            "excluded_unsupported_cigar": {
                f.qname: f.reason for f in fragments if f.status == EXCL_UNSUPPORTED_CIGAR
            },
            "excluded_anomaly": {f.qname: f.reason for f in fragments if f.status == EXCL_ANOMALY},
            "pair_notes": {f.qname: f.reason for f in fragments if f.reason and f.status in (UNIQUE, EQUIVALENCE)},
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    write_fragments_csv(fragments, os.path.join(out_dir, "fragments.csv"))
    write_counts_csvs(unique_rows, eq_rows, out_dir)
    with open(os.path.join(out_dir, "diagnostics.json"), "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, ensure_ascii=False, indent=2)
    _write_summary(diagnostics, unique_rows, eq_rows, os.path.join(out_dir, "summary.txt"))
    return diagnostics


def _write_summary(diag: dict, unique_rows: list[dict], eq_rows: list[dict], path: str) -> None:
    lines: list[str] = []
    a = lines.append
    a("rnacount build 摘要")
    a("=" * 60)
    inp = diag["inputs"]
    a(f"输入: bam={inp['bam']}  exons={inp['exons']}  plan={inp['plan']}")
    hdr = diag["bam_header"]
    a(f"参考: {','.join(hdr['references'])}  排序: {hdr['sort_order']}")
    a("")
    al = diag["alignments"]
    a(f"比对记录: 总计 {al['total']}，纳入分组主比对 {al['kept_primary']}")
    a(
        "  记录级排除: secondary={secondary} supplementary={supplementary} "
        "unmapped={unmapped} not_paired={not_paired} missing_cigar={missing_cigar} "
        "other_reference={other_reference}".format(**al)
    )
    a("")
    fr = diag["fragments"]
    a(f"片段(按 query_name): {fr['total_query_names']}")
    for status, label in STATUS_LABELS.items():
        a(f"  {label:<18} {fr[status]}")
    a("")
    a("闭合检查:")
    a(f"  片段: {diag['closure']['fragment_equation']}  -> {'OK' if diag['closure']['fragment_ok'] else 'FAIL'}")
    a(f"  比对: {diag['closure']['alignment_equation']}  -> {'OK' if diag['closure']['alignment_ok'] else 'FAIL'}")
    a("")
    a("唯一支持计数:")
    for row in unique_rows:
        a(
            f"  {row['transcript_id']} (gene {row['gene_id']}, {row['strand']}): "
            f"{row['unique_support']}"
            + (f"  [{row['fragment_ids']}]" if row["fragment_ids"] else "")
        )
    a("")
    a("等价类(多候选):")
    if eq_rows:
        for row in eq_rows:
            a(
                f"  {row['class_id']} {{{row['transcripts']}}}: {row['n_fragments']}"
                f"  [{row['fragment_ids']}]"
            )
    else:
        a("  (无)")
    a("")
    det = diag["details"]
    if det["no_compat"]:
        a("无兼容原因:")
        for name, reason in det["no_compat"].items():
            a(f"  {name}: {reason}")
        a("")
    for key, label in (
        ("orphan", "孤儿"),
        ("excluded_duplicate", "排除-duplicate"),
        ("excluded_unsupported_cigar", "排除-不支持的CIGAR"),
        ("excluded_anomaly", "排除-结构异常"),
    ):
        items = det[key]
        if items:
            a(f"{label}:")
            if isinstance(items, dict):
                for name, reason in items.items():
                    a(f"  {name}: {reason}")
            else:
                for name in items:
                    a(f"  {name}")
            a("")
    if det["pair_notes"]:
        a("成对备注(非排除性):")
        for name, msg in det["pair_notes"].items():
            a(f"  {name}: {msg}")
        a("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
