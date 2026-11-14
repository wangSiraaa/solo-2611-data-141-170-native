"""按 plan 所述算法复现抽查顺序。"""
from __future__ import annotations

import random


def derive_order(item_ids: list[str], seed: int) -> list[str]:
    """plan 冻结的算法：Unicode 排序后的唯一件号列表，random.Random(seed).shuffle 一次。

    返回完整抽查顺序；合法观察必须是该顺序的连续前缀。
    """
    order = sorted(set(item_ids))
    random.Random(seed).shuffle(order)
    return order
