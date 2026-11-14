"""已对齐 FASTA 的读取与校验。

只接受 ACGTN 五个符号；N 表示该位点未知，不是第五种碱基，
其在每个潜隐碱基下的发射似然均为 1。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import CompareError

ALPHABET = "ACGTN"
_BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
# 未知位点在四个碱基下全部保留（发射似然 1）。
UNKNOWN = 4


@dataclass(frozen=True)
class Alignment:
    """按叶名索引的多序列比对结果。

    sequences[name] 是长度等于位点数的整数数组：0..3 为 ACGT，4 为 N。
    """

    names: tuple[str, ...]
    sequences: dict[str, "np.ndarray"]
    n_sites: int


def read_fasta(path: str | Path) -> Alignment:
    """读取已对齐 FASTA，并完成等长、唯一叶名、字母表校验。"""
    path = Path(path)
    if not path.is_file():
        raise CompareError(f"FASTA 文件不存在: {path}")

    order: list[str] = []
    raw: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:
                raw[current] = "".join(buffer)
            header = line[1:].strip()
            if not header:
                raise CompareError(f"{path}:{lineno} 存在空的序列名")
            name = header.split()[0]
            if name in raw or name in order:
                raise CompareError(f"{path}:{lineno} 叶名重复: {name}")
            order.append(name)
            current = name
            buffer = []
        else:
            if current is None:
                raise CompareError(f"{path}:{lineno} 在首个 '>' 之前出现序列字符")
            buffer.append(line.upper())
    if current is not None:
        raw[current] = "".join(buffer)

    if not order:
        raise CompareError(f"{path} 中没有任何序列")

    lengths = {len(seq) for seq in raw.values()}
    if len(lengths) != 1:
        detail = ", ".join(f"{n}={len(raw[n])}" for n in order)
        raise CompareError(f"{path} 序列长度不一致，无法视为已对齐: {detail}")
    n_sites = lengths.pop()
    if n_sites == 0:
        raise CompareError(f"{path} 序列长度为 0")

    sequences: dict[str, "np.ndarray"] = {}
    for name in order:
        seq = raw[name]
        bad = sorted({ch for ch in seq if ch not in ALPHABET})
        if bad:
            raise CompareError(
                f"{path} 序列 {name} 含 ACGTN 之外的字符: {''.join(bad)}"
            )
        sequences[name] = np.fromiter(
            (_BASE_INDEX.get(ch, UNKNOWN) for ch in seq), dtype=np.int8, count=n_sites
        )

    return Alignment(names=tuple(order), sequences=sequences, n_sites=n_sites)
