"""注释载入：零基半开坐标的外显子表，exon_index 按转录本方向（5'→3'）。

负链转录本的 exon_index 从高坐标到低坐标排列，因此内含子必须由
转录本方向上相邻的外显子对计算，而不是按基因组起点排序后相邻。
"""
from __future__ import annotations

import csv
from bisect import bisect_right
from dataclasses import dataclass, field


@dataclass
class Transcript:
    transcript_id: str
    gene_id: str
    chrom: str
    strand: str  # '+' / '-'
    exons_tx: list[tuple[int, int]]  # 按 exon_index（转录本 5'→3' 方向）
    exons_genomic: list[tuple[int, int]] = field(default_factory=list)  # 按基因组坐标升序
    introns: frozenset[tuple[int, int]] = frozenset()  # 转录本方向相邻外显子之间的内含子

    def __post_init__(self):
        self.exons_genomic = sorted(self.exons_tx)
        self._starts = [s for s, _ in self.exons_genomic]

    def covers(self, s: int, e: int) -> bool:
        """[s, e) 是否完整落在某个外显子内。

        外显子互不重叠，故"落在外显子并集内"等价于"落在某一个外显子内"：
        跨越外显子边界的区段必然覆盖内含子位置。
        """
        i = bisect_right(self._starts, s) - 1
        return i >= 0 and e <= self.exons_genomic[i][1]


def _compute_introns(exons_tx: list[tuple[int, int]], tid: str) -> frozenset[tuple[int, int]]:
    """转录本方向上相邻外显子之间的基因组内含子集合。"""
    introns = set()
    for a, b in zip(exons_tx, exons_tx[1:]):
        left, right = (a, b) if a[0] < b[0] else (b, a)
        if left[1] >= right[0]:
            raise ValueError(f"转录本 {tid} 的外显子重叠或相邻: {a} vs {b}")
        introns.add((left[1], right[0]))
    return frozenset(introns)


def load_exons(path: str, reference_name: str) -> list[Transcript]:
    """读取 exons.tsv，返回按 transcript_id 排序的转录本列表。"""
    rows: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            tid = row["transcript_id"]
            rec = rows.setdefault(
                tid,
                {
                    "gene_id": row["gene_id"],
                    "strand": row["strand"],
                    "exons": [],
                },
            )
            if rec["gene_id"] != row["gene_id"] or rec["strand"] != row["strand"]:
                raise ValueError(f"转录本 {tid} 的 gene_id/strand 不一致")
            rec["exons"].append((int(row["exon_index"]), int(row["start0"]), int(row["end0"])))

    transcripts = []
    for tid in sorted(rows):
        rec = rows[tid]
        if rec["strand"] not in ("+", "-"):
            raise ValueError(f"转录本 {tid} 的链向非法: {rec['strand']}")
        exons = sorted(rec["exons"])  # 按 exon_index
        indices = [ix for ix, _, _ in exons]
        if indices != list(range(1, len(exons) + 1)):
            raise ValueError(f"转录本 {tid} 的 exon_index 不是从 1 开始的连续编号: {indices}")
        exons_tx = [(s, e) for _, s, e in exons]
        for s, e in exons_tx:
            if not 0 <= s < e:
                raise ValueError(f"转录本 {tid} 的外显子坐标非法: [{s}, {e})")
        tx = Transcript(
            transcript_id=tid,
            gene_id=rec["gene_id"],
            chrom=reference_name,
            strand=rec["strand"],
            exons_tx=exons_tx,
            introns=_compute_introns(exons_tx, tid),
        )
        transcripts.append(tx)
    return transcripts
