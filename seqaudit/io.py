"""输入文件的读取与解析。

解析失败（文件缺失、表头错误、字段类型错误、plan 自相矛盾等）抛出 InputError，
这属于"输入错误"，与抽样资格失败（eligibility）分开处理。
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

#: 本工具实现的唯一抽查顺序算法（plan 中必须逐字声明，否则无法复现顺序）。
EXPECTED_DRAW_ALGORITHM = (
    "Python 3.11 random.Random(seed).shuffle applied once "
    "to Unicode-sorted unique item IDs"
)

#: 题目约定：最多 120 件。
MAX_LOT_SIZE = 120


class InputError(Exception):
    """输入文件无法解析或字段非法（区别于抽样资格失败）。"""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Plan:
    lot_size: int
    d0: int            # reject_if_defects_at_least：零假设边界，H0: D >= d0
    alpha: Fraction    # 误接受风险上限（精确有理数）
    alpha_raw: str     # plan 中的原始写法，如 "0.05"
    seed: int
    draw_algorithm: str
    batches: tuple[int, ...]  # 累计查看点（严格递增）


def load_plan(path: str | Path) -> Plan:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"无法读取 plan: {exc}") from exc
    try:
        lot_size = int(raw["lot_size"])
        d0 = int(raw["reject_if_defects_at_least"])
        alpha_raw = str(raw["max_false_accept_probability"])
        alpha = Fraction(alpha_raw)  # 精确解析 "0.05"->1/20、"1/3" 等
        seed = int(raw["draw_seed"])
        algorithm = str(raw["draw_algorithm"])
        batches = tuple(int(b) for b in raw["batches"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InputError(f"plan 字段缺失或无法解析: {exc}") from exc
    if not 1 <= lot_size <= MAX_LOT_SIZE:
        raise InputError(f"lot_size={lot_size} 超出本工具范围 1..{MAX_LOT_SIZE}")
    if not 1 <= d0 <= lot_size:
        raise InputError(f"reject_if_defects_at_least={d0} 必须落在 1..{lot_size}")
    if not 0 < alpha < 1:
        raise InputError(
            f"max_false_accept_probability={alpha_raw!r} 必须落在 (0,1) 内"
        )
    if algorithm != EXPECTED_DRAW_ALGORITHM:
        raise InputError(
            "plan 的 draw_algorithm 不是本工具实现的算法，无法复现抽查顺序。\n"
            f"  期望: {EXPECTED_DRAW_ALGORITHM!r}\n  实际: {algorithm!r}"
        )
    if (
        not batches
        or batches[0] < 1
        or batches[-1] > lot_size
        or any(b2 <= b1 for b1, b2 in zip(batches, batches[1:]))
    ):
        raise InputError(
            f"batches 必须是 1..{lot_size} 内严格递增的累计查看点，实际: {list(batches)}"
        )
    return Plan(lot_size, d0, alpha, alpha_raw, seed, algorithm, batches)


def load_lot(path: str | Path) -> list[str]:
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = [r for r in csv.reader(fh) if r]
    except OSError as exc:
        raise InputError(f"无法读取 lot: {exc}") from exc
    if not rows or [c.strip() for c in rows[0]] != ["item_id"]:
        raise InputError("lot.csv 表头必须是单列 item_id")
    ids: list[str] = []
    for i, r in enumerate(rows[1:], start=2):
        if len(r) != 1 or not r[0].strip():
            raise InputError(f"lot.csv 第 {i} 行必须恰好一个非空 item_id")
        ids.append(r[0].strip())
    if not ids:
        raise InputError("lot.csv 没有任何件号")
    return ids


@dataclass(frozen=True)
class ObsRow:
    line: int  # 文件内行号（表头为第 1 行）
    draw_no_raw: str
    item_id: str
    defect_raw: str


def load_observations(path: str | Path) -> list[ObsRow]:
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = [r for r in csv.reader(fh) if r]
    except OSError as exc:
        raise InputError(f"无法读取 observations: {exc}") from exc
    if not rows or [c.strip() for c in rows[0]] != ["draw_no", "item_id", "defect"]:
        raise InputError("observations 表头必须是 draw_no,item_id,defect")
    out: list[ObsRow] = []
    for i, r in enumerate(rows[1:], start=2):
        if len(r) != 3:
            raise InputError(f"observations 第 {i} 行必须恰好 3 列")
        out.append(ObsRow(i, r[0].strip(), r[1].strip(), r[2].strip()))
    return out
