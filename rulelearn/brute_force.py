"""独立穷举器（仅用于核验，不参与生产求解）。

与子集 DP 完全独立：直接枚举「谓词不重复的有序序列 × 每规则队列 × 默认队列」，
逐行重放首次命中并按原始定义计算目标值。适用范围 ≤4 谓词（自制基准）。
"""

from __future__ import annotations

import itertools

from .data import Case, CostMatrix, Plan, Predicate
from .predicates import evaluate


def _route(rules: list[tuple], default_queue: str, fields: dict) -> str:
    for pred, queue in rules:
        if evaluate(pred, fields):
            return queue
    return default_queue


def objective_value(
    rules: list[tuple],
    default_queue: str,
    cases: list[Case],
    costs: CostMatrix,
    plan: Plan,
) -> float:
    total = plan.rule_penalty * len(rules)
    for case in cases:
        q = _route(rules, default_queue, case.fields)
        total += case.weight * costs.cost[case.target][q]
    return total


def enumerate_plans(
    predicates: list[Predicate],
    cases: list[Case],
    costs: CostMatrix,
    plan: Plan,
):
    """生成 (value, rules[(pred,queue)], default) 的全部方案，按稳定次序。"""
    k = min(plan.max_rules, len(predicates))
    ordered = sorted(predicates, key=lambda p: p.id)
    for r in range(k + 1):
        for chosen in itertools.permutations(ordered, r):
            queue_choices = sorted(costs.queues)
            for qs in itertools.product(queue_choices, repeat=r):
                rules = list(zip(chosen, qs))
                for d in sorted(plan.allow_default):
                    yield objective_value(rules, d, cases, costs, plan), rules, d


def brute_force_optimum(
    predicates: list[Predicate],
    cases: list[Case],
    costs: CostMatrix,
    plan: Plan,
) -> dict:
    """返回最小值与全部同值最优方案（结构化去重）。"""
    best = float("inf")
    winners: list[dict] = []
    for value, rules, d in enumerate_plans(predicates, cases, costs, plan):
        sig = (tuple((p.id, q) for p, q in rules), d)
        if value < best - 1e-12:
            best = value
            winners = [{"value": value, "rules": rules, "default": d, "sig": sig}]
        elif abs(value - best) <= 1e-12 and sig not in {w["sig"] for w in winners}:
            winners.append({"value": value, "rules": rules, "default": d, "sig": sig})
    return {"value": best, "winners": winners}


def probe_queues_across_optima(
    winners: list[dict], probe_rows: list[dict]
) -> list[dict]:
    """穷举视角下，每个探针在全部最优方案上的队列集合与两份证据。"""
    reports = []
    for row in probe_rows:
        per_queue: dict = {}
        for w in winners:
            q = _route(w["rules"], w["default"], row["fields"])
            per_queue.setdefault(
                q,
                {
                    "rules": [
                        {"predicate": p.to_dict(), "queue": qq} for p, qq in w["rules"]
                    ],
                    "default_queue": w["default"],
                },
            )
        reports.append(
            {
                "probe_id": row["id"],
                "reachable_queues": sorted(per_queue),
                "determined": len(per_queue) == 1,
                "witnesses": [
                    {"queue": q, "plan": per_queue[q]} for q in sorted(per_queue)
                ]
                if len(per_queue) > 1
                else [],
            }
        )
    return reports
