"""端到端验证：fixtures 上的 build 结果与手工推导、独立重算交叉核对。

手工推导（不读程序输出，直接从 BAM 与注释推出）：

- unique_A1: R1 20M60N20M@120 → 块 120-140/200-220，N=140-200 精确等于
  A1 与 A3 的内含子 1；R2 5S20M@315 → 315-335 只落在 A1/A2 的 300-340。
  两 mate 的交集 = {A1}（软剪切不产生参考足迹）。
- shared_A1_A2: R1 110-130 在 A1/A2/A3 共享外显子 1，R2 320-340 在 A1/A2
  共享外显子 3 而 A3 的第三外显子是 360-400 → 等价类 {A1,A2}。
- unique_A2: R1 10M110N10M@130 → N=140-250 只等于 A2 内含子 → {A2}。
- reverse_B1: read1 反向 → 声明链向 '-'；R2 20M60N15M@520 的 N=540-600
  只等于 B1 内含子（B2 为 540-650）→ {B1}。
- indel_C: 10M2I10M2D8M@810，I 不占参考、D 830-832 仍在 C1 外显子内 → {C1}。
- overlap_mates_C: 两 mate 815-845 / 825-855 重叠 20bp，都在 C1 内 → {C1}，
  片段只计一次。
- bad_splice: N=140-201 不等于任何内含子 → 无兼容（错位剪接）。
- duplicate_C: 双 mate 带 0x400 → 排除。
- unsupported_H: R1 含 5H → 排除（不支持的 CIGAR 操作）。
- orphan_A: 只有 read1 → 孤儿。

闭合：5 唯一 + 1 等价类 + 1 无兼容 + 1 孤儿 + 2 排除 = 10 个 query_name；
19 条比对 = 9×2 + 1。
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from rnacount.cli import main

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"

# 手工推导的期望结果（status, 兼容集合）
EXPECTED = {
    "unique_A1": ("unique", {"A1"}),
    "shared_A1_A2": ("equivalence", {"A1", "A2"}),
    "unique_A2": ("unique", {"A2"}),
    "reverse_B1": ("unique", {"B1"}),
    "indel_C": ("unique", {"C1"}),
    "overlap_mates_C": ("unique", {"C1"}),
    "bad_splice": ("no_compat", set()),
    "duplicate_C": ("excluded_duplicate", None),
    "unsupported_H": ("excluded_unsupported_cigar", None),
    "orphan_A": ("orphan", None),
}


def independent_compat_sets():
    """独立重算：不复用 rnacount 的注释/兼容代码，直接解析 TSV 与 BAM。"""
    exons: dict[str, dict] = {}
    with open(FIXTURES / "exons.tsv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            d = exons.setdefault(row["transcript_id"], {"strand": row["strand"], "exons": []})
            d["exons"].append((int(row["exon_index"]), int(row["start0"]), int(row["end0"])))
    txs = {}
    for tid, d in exons.items():
        ordered = [(s, e) for i, s, e in sorted(d["exons"])]
        introns = set()
        for a, b in zip(ordered, ordered[1:]):
            lo, hi = (a, b) if a[0] < b[0] else (b, a)
            introns.add((lo[1], hi[0]))
        txs[tid] = (d["strand"], ordered, introns)

    consumes_ref = {0, 2, 3, 7, 8}
    groups: dict[str, list] = {}
    with pysam.AlignmentFile(str(FIXTURES / "reads.bam"), "rb") as bam:
        for r in bam.fetch(until_eof=True):
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            blocks, dels, ns, ops = [], [], [], []
            ref = r.reference_start
            for op, ln in r.cigartuples:
                ops.append(op)
                if op in (0, 7, 8):
                    blocks.append((ref, ref + ln))
                elif op == 2:
                    dels.append((ref, ref + ln))
                elif op == 3:
                    ns.append((ref, ref + ln))
                if op in consumes_ref:
                    ref += ln
            groups.setdefault(r.query_name, []).append(
                dict(is_read1=r.is_read1, rev=r.is_reverse, blocks=blocks,
                     dels=dels, ns=ns, ops=ops, flag=r.flag)
            )

    expected = {}
    for qn, members in groups.items():
        if len(members) != 2:
            expected[qn] = ("orphan", None)
            continue
        if any(m["flag"] & 0x400 for m in members):
            expected[qn] = ("excluded_duplicate", None)
            continue
        if any(op not in (0, 1, 2, 3, 4, 7, 8) for m in members for op in m["ops"]):
            expected[qn] = ("excluded_unsupported_cigar", None)
            continue
        r1 = next(m for m in members if m["is_read1"])
        r2 = next(m for m in members if not m["is_read1"])
        strand = "-" if r1["rev"] else "+"
        compat = set()
        for tid, (tstrand, exon_list, introns) in txs.items():
            if tstrand != strand:
                continue
            ok = True
            for m in (r1, r2):
                for s, e in m["blocks"] + m["dels"]:
                    if not any(es <= s and e <= ee for es, ee in exon_list):
                        ok = False
                for iv in m["ns"]:
                    if iv not in introns:
                        ok = False
            if ok:
                compat.add(tid)
        status = "unique" if len(compat) == 1 else ("equivalence" if len(compat) >= 2 else "no_compat")
        expected[qn] = (status, compat)
    return expected


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("build")
    rc = main(["build", "--bam", str(FIXTURES / "reads.bam"), "--exons",
               str(FIXTURES / "exons.tsv"), "--plan", str(FIXTURES / "plan.json"),
               "--out", str(out)])
    assert rc == 0
    rows = {}
    with open(out / "fragments.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[row["fragment_id"]] = row
    with open(out / "diagnostics.json", encoding="utf-8") as fh:
        diag = json.load(fh)
    return out, rows, diag


def test_hand_derived_expectations(built):
    """手工推导的兼容集合与程序输出一致。"""
    _, rows, _ = built
    assert set(rows) == set(EXPECTED)
    for qn, (status, compat) in EXPECTED.items():
        row = rows[qn]
        assert row["status"] == status, f"{qn}: {row['status']} != {status}"
        if compat is not None:
            got = set(row["compatible_transcripts"].split(";")) if row["compatible_transcripts"] else set()
            assert got == compat, f"{qn}: {got} != {compat}"


def test_independent_recomputation_matches(built):
    """独立重算的兼容集合与程序输出一致（引擎交叉核对）。"""
    _, rows, _ = built
    expected = independent_compat_sets()
    assert set(rows) == set(expected)
    for qn, (status, compat) in expected.items():
        assert rows[qn]["status"] == status, qn
        if compat is not None:
            got = set(rows[qn]["compatible_transcripts"].split(";")) if rows[qn]["compatible_transcripts"] else set()
            assert got == compat, qn


def test_count_closure(built):
    """被接受、排除、孤儿、无兼容的数量闭合；比对记录数也闭合。"""
    _, rows, diag = built
    assert diag["closure"]["fragment_ok"] and diag["closure"]["alignment_ok"]
    fr = diag["fragments"]
    assert fr["total_query_names"] == 10
    assert fr["unique"] == 5 and fr["equivalence"] == 1 and fr["no_compat"] == 1
    assert fr["orphan"] == 1
    assert fr["excluded_duplicate"] == 1 and fr["excluded_unsupported_cigar"] == 1
    assert sum(fr[k] for k in fr if k != "total_query_names") == 10
    al = diag["alignments"]
    assert al["total"] == 19 and al["kept_primary"] == 19
    assert sum(len(r) for r in [rows]) == 10  # 每个 query_name 一行


def test_unique_support_and_equivalence_counts(built):
    """唯一支持按片段计一次；共享外显子片段进入等价类而非任意分配。"""
    out, _, _ = built
    with open(out / "counts_unique.csv", newline="", encoding="utf-8") as fh:
        uniq = {r["transcript_id"]: r for r in csv.DictReader(fh)}
    assert {t: int(r["unique_support"]) for t, r in uniq.items()} == {
        "A1": 1, "A2": 1, "A3": 0, "B1": 1, "B2": 0, "C1": 2,
    }
    assert uniq["C1"]["fragment_ids"] == "indel_C;overlap_mates_C"
    with open(out / "counts_equivalence.csv", newline="", encoding="utf-8") as fh:
        eq = list(csv.DictReader(fh))
    assert len(eq) == 1
    assert eq[0]["transcripts"] == "A1;A2" and eq[0]["n_fragments"] == "1"
    assert eq[0]["fragment_ids"] == "shared_A1_A2"


def test_overlapping_mates_counted_once(built):
    """重叠 mate：并集 40bp、重叠 20bp，C1 唯一支持只计该片段一次。"""
    _, rows, _ = built
    row = rows["overlap_mates_C"]
    assert row["status"] == "unique" and row["compatible_transcripts"] == "C1"
    assert int(row["union_covered_bp"]) == 40
    assert int(row["mate_overlap_bp"]) == 20


def test_no_compat_reason_pinpoints_bad_intron(built):
    """错位剪接的原因必须指到具体的 N 区间。"""
    _, rows, _ = built
    reason = rows["bad_splice"]["reason"]
    assert "140-201" in reason and "内含子" in reason


def test_orphan_and_exclusions_diagnosed_separately(built):
    _, rows, diag = built
    assert rows["orphan_A"]["status"] == "orphan"
    assert "缺少 read2" in rows["orphan_A"]["reason"]
    assert rows["unsupported_H"]["reason"].startswith("CIGAR 含不支持的操作: H")
    det = diag["details"]
    assert set(det["orphan"]) == {"orphan_A"}
    assert det["excluded_duplicate"] == ["duplicate_C"]
    assert set(det["excluded_unsupported_cigar"]) == {"unsupported_H"}


def test_declared_strand_from_read1(built):
    """文库链向以 read1 为准：reverse_B1 声明为负链并唯一支持 B1。"""
    _, rows, _ = built
    assert rows["reverse_B1"]["declared_strand"] == "-"
    assert rows["reverse_B1"]["compatible_transcripts"] == "B1"
    assert rows["unique_A1"]["declared_strand"] == "+"


def test_cli_subprocess(tmp_path):
    """python -m rnacount build 子进程端到端可运行。"""
    out = tmp_path / "cli_out"
    proc = subprocess.run(
        [sys.executable, "-m", "rnacount", "build", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "闭合检查: OK" in proc.stdout
    for name in ("fragments.csv", "counts_unique.csv", "counts_equivalence.csv",
                 "diagnostics.json", "summary.txt"):
        assert (out / name).exists(), name
