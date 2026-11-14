"""有序规则的纯重放。

只接受保存下来的规则 JSON（谓词原文 + 目标队列 + 默认队列），
逐行独立判断首次命中，不依赖求解器、不训练、不访问标签。
"""

from __future__ import annotations

from .data import Predicate
from .predicates import evaluate


def rule_from_dict(item: dict) -> tuple[Predicate, str]:
    p = item["predicate"]
    return (
        Predicate(id=p["id"], field=p["field"], op=p["op"], value=p["value"]),
        item["queue"],
    )


def replay_one(rules: list[dict], default_queue: str, fields: dict) -> dict:
    """对单行字段重放首次命中。

    返回 {assigned_queue, hit_index, predicate_id}；hit_index 为 None 表示走默认。
    """
    for k, item in enumerate(rules):
        pred, queue = rule_from_dict(item)
        if evaluate(pred, fields):
            return {"assigned_queue": queue, "hit_index": k, "predicate_id": pred.id}
    return {"assigned_queue": default_queue, "hit_index": None, "predicate_id": None}


def replay_rows(rules: list[dict], default_queue: str, rows: list[dict]) -> list[dict]:
    """对多行（每行含 id、fields）重放。"""
    out = []
    for row in rows:
        result = replay_one(rules, default_queue, row["fields"])
        out.append({"id": row["id"], **result})
    return out
