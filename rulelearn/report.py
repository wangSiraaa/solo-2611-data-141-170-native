"""把求解结果组装为可检查的规则 JSON 与结构化例外清单。

例外清单只如实报告错送案例，绝不为贴合标签修改任何原始数据。
"""

from __future__ import annotations

from .data import Case, CostMatrix, Plan, Predicate
from .replay import replay_one
from .solver import ExactSolver, bits


def build_rulebook(plan: Plan, canonical: dict, v_star: float, n_pred: int) -> dict:
    rules = [
        {"position": k, "predicate": item["predicate"], "queue": item["queue"]}
        for k, item in enumerate(canonical["rules"])
    ]
    return {
        "schema_version": 1,
        "default_queue": canonical["default_queue"],
        "rules": rules,
        "objective": {
            "value": v_star,
            "formula": "sum(case weight * misroute cost) + rule_penalty * number of rules",
            "rule_penalty": plan.rule_penalty,
            "rule_count": len(rules),
        },
        "meta": {
            "n_predicates_available": n_pred,
            "max_rules": plan.max_rules,
            "allowed_default_queues": list(plan.allow_default),
        },
    }


def build_exceptions(
    rulebook: dict,
    predicates: list[Predicate],
    cases: list[Case],
    costs: CostMatrix,
    plan: Plan,
    v_star: float,
) -> dict:
    rules = rulebook["rules"]
    default_queue = rulebook["default_queue"]
    pred_by_id = {p.id: p for p in predicates}

    per_rule = [
        {
            "position": item["position"],
            "predicate": item["predicate"],
            "queue": item["queue"],
            "first_hit_cases": [],
            "shadowed_cases": [],
            "rule_error_cost": 0,
        }
        for item in rules
    ]
    default_section = {"queue": default_queue, "cases": [], "error_cases": [], "error_cost": 0}
    all_errors = []
    misroute_total = 0.0

    for case in cases:
        result = replay_one(rules, default_queue, case.fields)
        assigned = result["assigned_queue"]
        unit_cost = costs.cost[case.target][assigned]
        line_cost = case.weight * unit_cost
        entry = {
            "case_id": case.id,
            "target": case.target,
            "assigned_queue": assigned,
            "weight": case.weight,
            "unit_cost": unit_cost,
            "error_cost": line_cost,
        }
        misroute_total += line_cost

        if result["hit_index"] is None:
            default_section["cases"].append(
                {"case_id": case.id, "target": case.target, "weight": case.weight}
            )
            if unit_cost > 0:
                default_section["error_cases"].append(entry)
                default_section["error_cost"] += line_cost
        else:
            k = result["hit_index"]
            if unit_cost > 0:
                per_rule[k]["first_hit_cases"].append(entry)
                per_rule[k]["rule_error_cost"] += line_cost
            else:
                per_rule[k]["first_hit_cases"].append(
                    {"case_id": case.id, "target": case.target, "weight": case.weight}
                )

        if unit_cost > 0:
            all_errors.append({**entry, "hit_index": result["hit_index"],
                               "predicate_id": result["predicate_id"]})

    # 遮蔽分析：谓词命中但首次命中发生在更早规则上的案例
    for k, item in enumerate(rules):
        pred = pred_by_id[item["predicate"]["id"]]
        for case in cases:
            if not _pred_match(pred, case):
                continue
            earlier = replay_one(rules[:k], default_queue, case.fields)
            if earlier["hit_index"] is not None:
                per_rule[k]["shadowed_cases"].append(
                    {
                        "case_id": case.id,
                        "target": case.target,
                        "captured_by_position": earlier["hit_index"],
                        "captured_by_predicate": earlier["predicate_id"],
                        "assigned_queue": earlier["assigned_queue"],
                    }
                )

    maintenance = plan.rule_penalty * len(rules)
    return {
        "summary": {
            "objective_value": v_star,
            "misroute_cost": misroute_total,
            "maintenance_cost": maintenance,
            "n_rules": len(rules),
            "default_queue": default_queue,
            "n_default_routed": len(default_section["cases"]),
            "n_error_cases": len(all_errors),
        },
        "rules": per_rule,
        "default": default_section,
        "error_cases": all_errors,
    }


def _pred_match(pred: Predicate, case: Case) -> bool:
    from .predicates import evaluate

    return evaluate(pred, case.fields)


def fit_report(
    solver: ExactSolver,
    plan: Plan,
    canonical: dict,
    terminals: list[dict],
    v_star: float,
    probe_report: list[dict],
) -> dict:
    """供 fit 入口落盘的综合结构化报告。"""
    terminal_records = [
        {
            "predicate_ids": sorted(
                solver.preds[i].id for i in bits(t["set"])
            ),
            "default_queue": t["default"],
            "value": t["value"],
        }
        for t in sorted(terminals, key=solver._terminal_key)
    ]
    return {
        "objective_value": v_star,
        "n_optimal_terminals": len(terminals),
        "optimal_terminals": terminal_records,
        "canonical_plan": {
            "rules": canonical["rules"],
            "default_queue": canonical["default_queue"],
        },
        "tie_break_rule": (
            "先按规则数升序，再按谓词 id 排序列表字典序，再按默认队列名字典序；"
            "序列内部按 (谓词id, 队列名) 字典序贪心且每步经后缀 DP 可行性校验。"
            "仅依赖谓词 id 与队列名，不依赖输入行顺序。"
        ),
        "probes": probe_report,
    }
