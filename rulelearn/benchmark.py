"""自制微型穷举基准：3 个谓词（覆盖 eq/ge/le）、8 个案例、2 个队列。

完全在代码内构造，不读取业务 fixtures，不写死任何样例答案；
用于独立核验子集 DP 的目标值、最优终端集合与探针歧义结论。

构造意图（由 run_benchmark_checks 断言，而非假设）：
- p_one / q_one 在训练案例上匹配集完全相同，但在探针上分开 → 探针歧义；
- aged 与 p_one 在 z1/z5 上重叠 → 同一规则集合交换顺序会改变目标值；
- 0 条规则（最短）严格劣于带规则方案 → 最短未必代价最小。
"""

from __future__ import annotations

import itertools

from .brute_force import brute_force_optimum, objective_value, probe_queues_across_optima
from .data import Case, CostMatrix, Plan, Predicate


def build_benchmark() -> dict:
    predicates = [
        Predicate("p_one", "f1", "eq", 1),       # eq
        Predicate("q_one", "f2", "eq", 1),       # eq：训练集上与 p_one 同匹配集
        Predicate("aged", "age", "ge", 1),       # ge：与 p_one 部分重叠
        Predicate("newborn", "age", "le", 0),    # le：保证三种算子都出现
    ]
    # id, f1, f2, age, target, weight
    raw = [
        ("z1", 1, 1, 1, "specialist", 3),  # p_one/q_one/aged 全命中：顺序敏感点
        ("z2", 1, 1, 0, "specialist", 2),  # p_one/q_one
        ("z3", 0, 0, 1, "general", 2),     # 仅 aged
        ("z4", 0, 0, 0, "general", 1),     # newborn
        ("z5", 1, 1, 1, "specialist", 1),
        ("z6", 0, 0, 0, "general", 2),     # newborn
        ("z7", 0, 0, 1, "general", 1),     # 仅 aged
        ("z8", 1, 1, 0, "general", 2),     # 命中 p_one/q_one 但真实 general（错送权衡）
    ]
    cases = [
        Case(cid, {"f1": f1, "f2": f2, "age": age}, target, weight)
        for cid, f1, f2, age, target, weight in raw
    ]
    costs = CostMatrix(
        {
            "general": {"general": 0, "specialist": 2},
            "specialist": {"general": 5, "specialist": 0},
        },
        ("general", "specialist"),
    )
    plan = Plan(max_rules=4, rule_penalty=2, allow_default=("general", "specialist"))
    probes = [
        # 训练中未出现的组合：p_one 与 q_one 在此分开
        {"id": "pr-x", "fields": {"f1": 1, "f2": 0, "age": 0}},  # 仅 p_one（+newborn）
        {"id": "pr-y", "fields": {"f1": 0, "f2": 1, "age": 0}},  # 仅 q_one（+newborn）
        {"id": "pr-z", "fields": {"f1": 0, "f2": 0, "age": 0}},
    ]
    return {
        "predicates": predicates,
        "cases": cases,
        "costs": costs,
        "plan": plan,
        "probes": probes,
    }


def _best_value_by_length(bench: dict) -> dict:
    preds, cases, costs, plan = (
        bench["predicates"],
        bench["cases"],
        bench["costs"],
        bench["plan"],
    )
    queues = sorted(costs.queues)
    by_len = {}
    for r in range(plan.max_rules + 1):
        best = float("inf")
        for chosen in itertools.permutations(preds, r):
            for qs in itertools.product(queues, repeat=r):
                rules = list(zip(chosen, qs))
                for d in plan.allow_default:
                    best = min(best, objective_value(rules, d, cases, costs, plan))
        by_len[r] = best
    return by_len


def run_benchmark_checks() -> dict:
    bench = build_benchmark()
    preds, cases, costs, plan, probes = (
        bench["predicates"],
        bench["cases"],
        bench["costs"],
        bench["plan"],
        bench["probes"],
    )
    assert len(preds) <= 4 and len(cases) <= 8

    bf = brute_force_optimum(preds, cases, costs, plan)
    by_len = _best_value_by_length(bench)

    # 性质 1：同一谓词集合 {aged, p_one}，交换顺序 -> 目标值不同
    p_one, q_one, aged = preds[0], preds[1], preds[2]
    order_a = [(aged, "general"), (p_one, "specialist")]
    order_b = [(p_one, "specialist"), (aged, "general")]
    va = objective_value(order_a, "general", cases, costs, plan)
    vb = objective_value(order_b, "general", cases, costs, plan)
    order_matters = va != vb

    # 性质 2：0 条规则（最短）严格劣于全局最优；且 4 条规则也不是最优
    shortest_not_best = by_len[0] > bf["value"] + 1e-12

    # 性质 3：歧义探针的两份证据方案分流不同，且所有方案目标值都是 V*
    probe_report = probe_queues_across_optima(bf["winners"], probes)
    ambiguous = [r for r in probe_report if not r["determined"]]
    ambiguity_real = (
        bool(ambiguous)
        and all(
            len({w["queue"] for w in r["witnesses"]}) == len(r["reachable_queues"]) > 1
            and all(
                objective_value(
                    [
                        (next(p for p in preds if p.id == x["predicate"]["id"]), x["queue"])
                        for x in w["plan"]["rules"]
                    ],
                    w["plan"]["default_queue"],
                    cases,
                    costs,
                    plan,
                )
                == bf["value"]
                for w in r["witnesses"]
            )
            for r in ambiguous
        )
    )

    return {
        "benchmark": {
            "n_predicates": len(preds),
            "n_cases": len(cases),
            "n_queues": len(costs.queues),
            "rule_penalty": plan.rule_penalty,
            "max_rules": plan.max_rules,
            "ops_covered": sorted({p.op for p in preds}),
        },
        "brute_force_optimum": bf["value"],
        "n_optimal_plans": len(bf["winners"]),
        "best_value_by_rule_count": by_len,
        "order_dependence": {
            "rule_set": [p.id for p, _ in order_a],
            "value_when_aged_first": va,
            "value_when_p_one_first": vb,
            "order_matters": order_matters,
        },
        "shortest_not_best": {
            "zero_rule_value": by_len[0],
            "global_optimum": bf["value"],
            "holds": shortest_not_best,
        },
        "probes": probe_report,
        "ambiguity_real": ambiguity_real,
        "all_properties_hold": order_matters and shortest_not_best and ambiguity_real,
    }
