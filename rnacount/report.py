"""输出报告：fragments.csv、唯一支持 / 等价类计数、诊断与摘要。

- fragments.csv：每个 query_name 一行，含状态、兼容集合、定位信息与原因，
  是被接受 / 排除 / 孤儿 / 无兼容片段数量闭合的逐片段账目；
- counts_unique.csv：每个转录本的唯一支持片段数（含 0），附来源片段名；
- counts_equivalence.csv：多候选等价类计数，附来源片段名；
- diagnostics.json：记录级与片段级计数、闭合等式、逐片段诊断明细；
- summary.txt：人读摘要。
"""
from __future__ import annotations

import csv
import json
import os

from .annotation import Transcript
from .fragments import (
    EQUIVALENCE,
    EXCL_ANOMALY,
    EXCL_DUPLICATE,
    EXCL_UNSUPPORTED_CIGAR,
    NO_COMPAT,
    ORPHAN,
    UNIQUE,
    Fragment,
)

FRAGMENT_FIELDS = [
    "fragment_id",
    "status",
    "compatible_transcripts",
    "n_compatible",
    "declared_strand",
    "read1_cigar",
    "read1_pos0",
    "read2_cigar",
    "read2_pos0",
    "union_covered_bp",
    "mate_overlap_bp",
    "n_records",
    "reason",
]


def write_fragments_csv(fragments: list[Fragment], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FRAGMENT_FIELDS)
        w.writeheader()
        for frag in fragments:
            r1, r2 = frag.read1, frag.read2
            w.writerow(
                {
                    "fragment_id": frag.qname,
                    "status": frag.status,
                    "compatible_transcripts": ";".join(sorted(frag.compatible)),
                    "n_compatible": len(frag.compatible) if frag.compatible else 0,
                    "declared_strand": frag.declared_strand,
                    "read1_cigar": r1.cigarstring if r1 else "",
                    "read1_pos0": r1.pos0 if r1 else "",
                    "read2_cigar": r2.cigarstring if r2 else "",
                    "read2_pos0": r2.pos0 if r2 else "",
                    "union_covered_bp": frag.union_covered_bp,
                    "mate_overlap_bp": frag.mate_overlap_bp,
                    "n_records": len(frag.records),
                    "reason": frag.reason,
                }
            )


def build_counts(
    fragments: list[Fragment], transcripts: list[Transcript]
) -> tuple[list[dict], list[dict]]:
    """唯一支持计数与等价类计数（含来源片段名，作为来源定位）。"""
    unique_rows = []
    for tx in transcripts:
        donors = sorted(
            f.qname for f in fragments if f.status == UNIQUE and f.compatible == {tx.transcript_id}
        )
        unique_rows.append(
            {
                "transcript_id": tx.transcript_id,
                "gene_id": tx.gene_id,
                "strand": tx.strand,
                "unique_support": len(donors),
                "fragment_ids": ";".join(donors),
            }
        )

    classes: dict[frozenset[str], list[str]] = {}
    for frag in fragments:
        if frag.status == EQUIVALENCE:
            classes.setdefault(frozenset(frag.compatible), []).append(frag.qname)
    eq_rows = []
    for i, (members, donors) in enumerate(
        sorted(classes.items(), key=lambda kv: (sorted(kv[0]))), start=1
    ):
        eq_rows.append(
            {
                "class_id": f"EC{i}",
                "transcripts": ";".join(sorted(members)),
                "n_transcripts": len(members),
                "n_fragments": len(donors),
                "fragment_ids": ";".join(sorted(donors)),
            }
        )
    return unique_rows, eq_rows


def write_counts_csvs(unique_rows: list[dict], eq_rows: list[dict], out_dir: str) -> None:
    with open(os.path.join(out_dir, "counts_unique.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["transcript_id", "gene_id", "strand", "unique_support", "fragment_ids"]
        )
        w.writeheader()
        w.writerows(unique_rows)
    with open(
        os.path.join(out_dir, "counts_equivalence.csv"), "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(
            fh, fieldnames=["class_id", "transcripts", "n_transcripts", "n_fragments", "fragment_ids"]
        )
        w.writeheader()
        w.writerows(eq_rows)


def status_tally(fragments: list[Fragment]) -> dict:
    tally = {
        UNIQUE: 0,
        EQUIVALENCE: 0,
        NO_COMPAT: 0,
        ORPHAN: 0,
        EXCL_DUPLICATE: 0,
        EXCL_UNSUPPORTED_CIGAR: 0,
        EXCL_ANOMALY: 0,
    }
    for frag in fragments:
        tally[frag.status] = tally.get(frag.status, 0) + 1
    return tally
