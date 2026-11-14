"""CIGAR 解码：从 CIGAR 恢复参考序列上的实际覆盖块、缺失与内含子跳跃。

按操作对 query / 参考序列的消耗分别处理：

- M/=/X：同时消耗 query 与参考 —— 产生覆盖块 block；
- I：只消耗 query —— 参考上无足迹；
- D：只消耗参考 —— 缺失仍占用参考位置，须落在外显子内；
- N：只消耗参考 —— 内含子跳跃，须精确等于已注释内含子；
- S：只消耗 query —— 软剪切，参考上无足迹；
- 其余（H/P/B 等）：不在 plan.accepted_cigar 内，记录为不支持的操作，
  由上层诊断并排除对应片段。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# pysam cigartuples 的操作码顺序与 SAM 规范一致：MIDNSHP=XB
OP_NAMES = "MIDNSHP=XB"

# 消耗参考序列位置的操作（M D N = X）
REF_CONSUMING = frozenset({0, 2, 3, 7, 8})


@dataclass
class Footprint:
    """一条比对在参考序列上的实际足迹（零基半开坐标）。"""

    ref_start: int
    ref_end: int  # 最后一个消耗参考位置之后的坐标
    blocks: list[tuple[int, int]] = field(default_factory=list)  # M/=/X 覆盖块
    deletions: list[tuple[int, int]] = field(default_factory=list)  # D 占用的参考位置
    introns: list[tuple[int, int]] = field(default_factory=list)  # N 跳过区域
    unsupported_ops: list[str] = field(default_factory=list)  # 不在 accepted_cigar 中的操作


def decode_cigar(cigartuples, pos0: int, accepted: frozenset[str]) -> Footprint:
    """把 (op, length) 序列解码为 Footprint。

    pos0 为比对在参考上的零基起点。即使出现不支持的操作也照常解码，
    不支持的操作名记入 unsupported_ops（H/P/B 不消耗参考位置，不影响坐标推进）。
    """
    fp = Footprint(ref_start=pos0, ref_end=pos0)
    ref = pos0
    for op, length in cigartuples or []:
        name = OP_NAMES[op] if 0 <= op < len(OP_NAMES) else f"op{op}"
        if name not in accepted:
            fp.unsupported_ops.append(name)
        if op in (0, 7, 8):  # M / = / X：覆盖块
            fp.blocks.append((ref, ref + length))
        elif op == 1:  # I：仅消耗 query
            pass
        elif op == 2:  # D：缺失，仍占用参考位置
            fp.deletions.append((ref, ref + length))
        elif op == 3:  # N：内含子跳跃
            fp.introns.append((ref, ref + length))
        elif op == 4:  # S：软剪切，仅消耗 query
            pass
        # H/P/B 等：不消耗参考位置，仅诊断
        if op in REF_CONSUMING:
            ref += length
    fp.ref_end = ref
    return fp
