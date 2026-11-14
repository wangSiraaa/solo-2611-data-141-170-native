"""片段-转录本兼容性判定。

规则（来自 plan）：
- 文库链向以 read1 约定为准：read1 的比对链与转录本同向；
- 一个转录本兼容当且仅当：两个 mate 的全部覆盖块与缺失位置都落在
  该转录本外显子内，且每个 N 精确等于该转录本连续外显子间的内含子；
- 两个 mate 必须同时满足同一个转录本（取交集，不是并集）；
- mate 之间未测到的插入区不构成剪接证据：不对两 mate 之间的外显子
  连续性做任何推断，每个 mate 独立对照同一转录本评估。

计数单位为片段：每个有效片段只贡献一次，两个 mate 覆盖重叠不重复计数。
"""
from __future__ import annotations

from .annotation import Transcript
from .fragments import EQUIVALENCE, NO_COMPAT, PAIRED, UNIQUE, Fragment


def fit_alignment(fp, tx: Transcript) -> list[str]:
    """单条比对足迹对转录本的兼容性；返回失败原因列表，空列表表示兼容。"""
    reasons: list[str] = []
    for s, e in fp.blocks:
        if not tx.covers(s, e):
            reasons.append(f"区段 {s}-{e} 不在任何外显子内")
    for s, e in fp.deletions:
        if not tx.covers(s, e):
            reasons.append(f"缺失 {s}-{e} 不在任何外显子内")
    for s, e in fp.introns:
        if (s, e) not in tx.introns:
            reasons.append(f"N {s}-{e} 不等于任何已注释内含子")
    return reasons


def _union_len(intervals: list[tuple[int, int]]) -> int:
    """区间并集长度（用于合并两个 mate 的覆盖，重叠只算一次）。"""
    total = 0
    cur_s = cur_e = None
    for s, e in sorted(intervals):
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        total += cur_e - cur_s
    return total


def evaluate_fragment(frag: Fragment, transcripts: list[Transcript]) -> None:
    """评估一个通过过滤的片段，填写兼容集合与状态。"""
    r1, r2 = frag.read1, frag.read2
    assert r1 is not None and r2 is not None and frag.status == PAIRED

    # 文库约定：read1 的比对链与转录本同向
    frag.declared_strand = "-" if r1.is_reverse else "+"

    # 两个 mate 的覆盖并集与重叠（计数仍按片段计一次，此处为诊断与定位）
    blocks1 = r1.footprint.blocks
    blocks2 = r2.footprint.blocks
    union = _union_len(blocks1 + blocks2)
    raw = sum(e - s for s, e in blocks1) + sum(e - s for s, e in blocks2)
    frag.union_covered_bp = union
    frag.mate_overlap_bp = raw - union

    compatible: set[str] = set()
    failures: dict[str, dict[str, list[str]]] = {}
    for tx in transcripts:
        if tx.strand != frag.declared_strand:
            continue
        fail1 = fit_alignment(r1.footprint, tx)
        fail2 = fit_alignment(r2.footprint, tx)
        if not fail1 and not fail2:
            compatible.add(tx.transcript_id)
        else:
            failures[tx.transcript_id] = {"read1": fail1, "read2": fail2}

    frag.compatible = compatible
    if len(compatible) == 1:
        frag.status = UNIQUE
    elif len(compatible) >= 2:
        frag.status = EQUIVALENCE
    else:
        frag.status = NO_COMPAT
        frag.reason = (
            _summarize_failures(failures)
            if failures
            else f"声明链向 {frag.declared_strand} 上没有任何候选转录本"
        )


def _summarize_failures(failures: dict[str, dict[str, list[str]]], limit: int = 600) -> str:
    """把各候选转录本的失败原因压缩成一行诊断。"""
    parts = []
    for tid in sorted(failures):
        segs = []
        for mate in ("read1", "read2"):
            rs = failures[tid][mate]
            if rs:
                segs.append(f"{mate}: " + ", ".join(rs))
        if segs:
            parts.append(f"{tid}[{' ; '.join(segs)}]")
    text = " | ".join(parts)
    return text if len(text) <= limit else text[: limit - 3] + "..."
