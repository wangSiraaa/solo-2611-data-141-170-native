"""片段归组与过滤：BAM 按坐标排序，同名 mate 可能相距很远，

因此全部读入内存后按 query_name 归组（小输入允许内存处理）。
在此检查成对身份、主比对，并应用 plan 的过滤规则：

- 记录级诊断：secondary / supplementary / unmapped / 非双端标志 /
  缺失 CIGAR / 非计划参考序列（均不进入分组）；
- 片段级：成对身份（恰为 read1+read2 各一条主比对）、duplicate 标志、
  不支持的 CIGAR 操作；只有一条记录的组诊断为孤儿；
- 成对交叉核对：mate 声明的对方坐标应与实际记录一致（不一致仅备注）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pysam

from .cigarops import Footprint, decode_cigar

# 片段状态
PAIRED = "paired"  # 通过全部过滤，待兼容性评估（内部状态，不写出）
UNIQUE = "unique"  # 恰好兼容一个转录本
EQUIVALENCE = "equivalence"  # 兼容多个转录本
NO_COMPAT = "no_compat"  # 评估后无任何兼容转录本
ORPHAN = "orphan"  # 组内只有一条主比对
EXCL_DUPLICATE = "excluded_duplicate"
EXCL_UNSUPPORTED_CIGAR = "excluded_unsupported_cigar"
EXCL_ANOMALY = "excluded_anomaly"  # 成对结构异常（记录数/读段标志不对）


@dataclass
class AlignedRead:
    """精简化的主比对记录。"""

    qname: str
    is_read1: bool
    is_read2: bool
    is_reverse: bool
    ref_name: str
    pos0: int
    pnext: int  # 记录中声明的 mate 起点（用于成对交叉核对）
    cigarstring: str
    flag: int
    footprint: Footprint


@dataclass
class Fragment:
    qname: str
    records: list[AlignedRead]
    status: str = PAIRED
    reason: str = ""
    # 评估后填写
    declared_strand: str = ""
    compatible: set[str] = field(default_factory=set)
    union_covered_bp: int = 0
    mate_overlap_bp: int = 0

    @property
    def read1(self) -> AlignedRead | None:
        return next((r for r in self.records if r.is_read1), None)

    @property
    def read2(self) -> AlignedRead | None:
        return next((r for r in self.records if r.is_read2), None)


def collect_fragments(bam_path: str, plan: dict) -> tuple[list[Fragment], dict, dict]:
    """读取 BAM，按 query_name 归组并应用过滤规则。

    返回 (片段列表, 记录级诊断计数, BAM 头信息)。
    """
    accepted = frozenset(plan["accepted_cigar"])
    reference_name = plan["reference_name"]

    record_diag = {
        "total": 0,
        "secondary": 0,
        "supplementary": 0,
        "unmapped": 0,
        "not_paired": 0,
        "missing_cigar": 0,
        "other_reference": 0,
        "kept_primary": 0,
    }
    groups: dict[str, list[AlignedRead]] = {}

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        header_info = {
            "sort_order": bam.header.get("HD", {}).get("SO", "unknown"),
            "references": list(bam.references),
        }
        if reference_name not in bam.references:
            raise ValueError(f"plan 参考 {reference_name} 不在 BAM 参考中: {list(bam.references)}")
        for rec in bam.fetch(until_eof=True):
            record_diag["total"] += 1
            if rec.is_secondary:
                record_diag["secondary"] += 1
                continue
            if rec.is_supplementary:
                record_diag["supplementary"] += 1
                continue
            if rec.is_unmapped:
                record_diag["unmapped"] += 1
                continue
            if not rec.is_paired:
                record_diag["not_paired"] += 1
                continue
            if rec.reference_name != reference_name:
                record_diag["other_reference"] += 1
                continue
            if rec.cigartuples is None:
                record_diag["missing_cigar"] += 1
                continue
            record_diag["kept_primary"] += 1
            groups.setdefault(rec.query_name, []).append(
                AlignedRead(
                    qname=rec.query_name,
                    is_read1=rec.is_read1,
                    is_read2=rec.is_read2,
                    is_reverse=rec.is_reverse,
                    ref_name=rec.reference_name,
                    pos0=rec.reference_start,
                    pnext=rec.next_reference_start,
                    cigarstring=rec.cigarstring,
                    flag=rec.flag,
                    footprint=decode_cigar(rec.cigartuples, rec.reference_start, accepted),
                )
            )

    fragments = [_classify(qname, members, plan) for qname, members in sorted(groups.items())]
    return fragments, record_diag, header_info


def _classify(qname: str, members: list[AlignedRead], plan: dict) -> Fragment:
    """确定片段的过滤后状态；通过过滤的保持 PAIRED 待兼容性评估。"""
    frag = Fragment(qname=qname, records=members)
    read1s = [r for r in members if r.is_read1]
    read2s = [r for r in members if r.is_read2]

    # 1) 成对身份：恰为 read1 + read2 各一条主比对
    if len(members) == 1:
        have = "read1" if read1s else ("read2" if read2s else "未标记 read1/read2")
        missing = "read2" if read1s else "read1"
        frag.status = ORPHAN
        frag.reason = f"组内仅有 {have} 一条主比对，缺少 {missing}"
        return frag
    if not (len(members) == 2 and len(read1s) == 1 and len(read2s) == 1):
        frag.status = EXCL_ANOMALY
        frag.reason = (
            f"成对结构异常: {len(members)} 条主比对 "
            f"(read1={len(read1s)}, read2={len(read2s)})"
        )
        return frag

    # 2) plan 过滤：duplicate 标志（任一 mate 命中即排除整个片段）
    if plan.get("exclude_flagged_duplicate") and any(r.flag & 0x400 for r in members):
        frag.status = EXCL_DUPLICATE
        frag.reason = "片段带有 duplicate 标志 (0x400)"
        return frag

    # 3) 不支持的 CIGAR 操作
    bad_ops = sorted({op for r in members for op in r.footprint.unsupported_ops})
    if bad_ops:
        frag.status = EXCL_UNSUPPORTED_CIGAR
        frag.reason = f"CIGAR 含不支持的操作: {','.join(bad_ops)}"
        return frag

    # 4) 成对交叉核对（非排除性，仅备注）
    r1, r2 = read1s[0], read2s[0]
    problems = []
    if r1.pnext != r2.pos0 or r2.pnext != r1.pos0:
        problems.append(
            f"mate 交叉坐标不一致: read1.pnext={r1.pnext} vs read2.pos={r2.pos0}; "
            f"read2.pnext={r2.pnext} vs read1.pos={r1.pos0}"
        )
    if r1.ref_name != r2.ref_name:
        problems.append(f"两个 mate 位于不同参考序列: {r1.ref_name} / {r2.ref_name}")
    if r1.is_reverse == r2.is_reverse:
        problems.append("read1 与 read2 比对链相同，与本文库预期的反向成对不符")
    if problems:
        frag.reason = "；".join(problems)  # 备注，不影响后续评估
    return frag
