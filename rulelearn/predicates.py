"""谓词求值与案例位掩码。

谓词语言固定为给定字段上的 eq / ge / le，不做任何扩展。
"""

from __future__ import annotations

from .data import Case, Predicate

_OPS = {
    "eq": lambda a, b: a == b,
    "ge": lambda a, b: a >= b,
    "le": lambda a, b: a <= b,
}


def evaluate(pred: Predicate, fields: dict) -> bool:
    if pred.field not in fields:
        raise KeyError(f"字段 {pred.field} 不存在（可用字段：{sorted(fields)}）")
    return _OPS[pred.op](fields[pred.field], pred.value)


def build_case_masks(predicates: list[Predicate], cases: list[Case]) -> list[int]:
    """返回每个谓词命中的案例位掩码（案例顺序即传入顺序）。"""
    masks = []
    for pred in predicates:
        bits = 0
        for j, case in enumerate(cases):
            if evaluate(pred, case.fields):
                bits |= 1 << j
        masks.append(bits)
    return masks
