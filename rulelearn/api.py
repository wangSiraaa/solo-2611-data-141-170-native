"""fit / apply / verify 三个公开入口。

- fit   ：加载标记材料，子集 DP 精确求解，落盘规则 JSON、例外清单与方法报告
- apply ：只从保存的规则 JSON 重新分流（不重训、不读标签）
- verify ：独立重放每行首次命中与成本，并用自制穷举基准交叉核验
"""

from __future__ import annotations

import json
from pathlib import Path

from .benchmark import run_benchmark_checks
from .brute_force import brute_force_optimum, objective_value, probe_queues_across_optima
from .data import (
    load_cases,
    load_costs,
    load_plan,
    load_predicates,
    load_unlabeled_rows,
)
from .predicates import build_case_masks
from .report import build_exceptions, build_rulebook, fit_report
from .replay import replay_rows
from .solver import ExactSolver, bits


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _build_solver(predicates, cases, costs, plan):
    masks = build_case_masks(predicates, cases)
    solver = ExactSolver(predicates, cases, costs, plan, masks)
    solver.solve_forward()
    return solver


def fit(input_dir: str, output_dir: str) -> dict:
    """从标记案例学习规则。返回内存中的全部结构化结果。"""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    plan = load_plan(in_dir / "plan.json")
    predicates = load_predicates(in_dir / "predicates.json")
    costs = load_costs(in_dir / "costs.csv")
    cases = load_cases(in_dir / "cases.csv", costs)
    probes = load_unlabeled_rows(in_dir / "probes.csv")

    if len(predicates) > 12 or len(cases) > 50 or len(costs.queues) > 3:
        raise ValueError("输入超出承诺的精确求解范围（≤12 谓词 / ≤50 案例 / ≤3 队列）")

    solver = _build_solver(predicates, cases, costs, plan)
    terminals = solver.terminals()
    v_star = solver.optimum_value(terminals)
    term = solver.canonical_terminal(terminals)
    canonical = solver.canonical_plan(term)
    probe_report = solver.probe_ambiguity(probes, terminals, v_star)

    rulebook = build_rulebook(plan, canonical, v_star, len(predicates))
    exceptions = build_exceptions(rulebook, predicates, cases, costs, plan, v_star)
    report = fit_report(solver, plan, canonical, terminals, v_star, probe_report)

    _write_json(out_dir / "rules.json", rulebook)
    _write_json(out_dir / "exceptions.json", exceptions)
    _write_json(out_dir / "report.json", report)

    return {
        "rulebook": rulebook,
        "exceptions": exceptions,
        "report": report,
        "_objects": {
            "plan": plan,
            "predicates": predicates,
            "costs": costs,
            "cases": cases,
            "probes": probes,
            "solver": solver,
        },
    }


def apply(rulebook_path: str, input_csv: str) -> list[dict]:
    """从保存的可检查规则 JSON 重新分流任意含相同字段的 CSV。

    输入可以是未标记探针，也可以是带 target 的案例；本函数不使用标签。
    """
    rulebook = json.loads(Path(rulebook_path).read_text(encoding="utf-8"))
    rows = load_unlabeled_rows(input_csv)
    results = replay_rows(rulebook["rules"], rulebook["default_queue"], rows)
    return [
        {
            "case_id": r["id"],
            "assigned_queue": r["assigned_queue"],
            "hit_index": r["hit_index"],
            "predicate_id": r["predicate_id"],
        }
        for r in results
    ]


def verify(input_dir: str, output_dir: str) -> dict:
    """独立核验：重放目标值、行序不变性、DP 与穷举基准、探针歧义、两条演示性质。"""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    rulebook = json.loads((out_dir / "rules.json").read_text(encoding="utf-8"))

    plan = load_plan(in_dir / "plan.json")
    predicates = load_predicates(in_dir / "predicates.json")
    costs = load_costs(in_dir / "costs.csv")
    cases = load_cases(in_dir / "cases.csv", costs)
    probes = load_unlabeled_rows(in_dir / "probes.csv")

    checks: list[dict] = []

    def record(name: str, ok: bool, detail: dict) -> bool:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        return bool(ok)

    # 1) 独立重放每行首次命中与成本
    replay = replay_rows(rulebook["rules"], rulebook["default_queue"],
                         [{"id": c.id, "fields": c.fields} for c in cases])
    misroute = 0.0
    replayed_errors = []
    for case, r in zip(cases, replay):
        q = r["assigned_queue"]
        line = case.weight * costs.cost[case.target][q]
        misroute += line
        if line > 0:
            replayed_errors.append(
                {"case_id": case.id, "target": case.target,
                 "assigned_queue": q, "error_cost": line}
            )
    maintenance = plan.rule_penalty * len(rulebook["rules"])
    replayed_objective = misroute + maintenance
    record(
        "replayed_objective_matches_saved",
        abs(replayed_objective - rulebook["objective"]["value"]) <= 1e-9,
        {"replayed": replayed_objective, "saved": rulebook["objective"]["value"],
         "misroute": misroute, "maintenance": maintenance,
         "n_error_cases": len(replayed_errors)},
    )

    # 2) 行序不变性：打乱案例行与谓词输入顺序，目标值与规范方案必须一致
    shuffled_cases = cases[::-1]
    shuffled_preds = predicates[::-1]
    solver2 = _build_solver(shuffled_preds, shuffled_cases, costs, plan)
    terms2 = solver2.terminals()
    v2 = solver2.optimum_value(terms2)
    plan2 = solver2.canonical_plan(solver2.canonical_terminal(terms2))
    canonical_sig = (
        [(r["predicate"]["id"], r["queue"]) for r in rulebook["rules"]],
        rulebook["default_queue"],
    )
    shuffled_sig = (
        [(r["predicate"]["id"], r["queue"]) for r in plan2["rules"]],
        plan2["default_queue"],
    )
    record(
        "row_order_invariance",
        abs(v2 - rulebook["objective"]["value"]) <= 1e-9 and canonical_sig == shuffled_sig,
        {"optimum_original": rulebook["objective"]["value"],
         "optimum_shuffled": v2,
         "canonical_signature_equal": canonical_sig == shuffled_sig},
    )

    # 3) 主数据规模下，DP 最优终端的每条方案用穷举式目标函数复核（不枚举全空间）
    solver = _build_solver(predicates, cases, costs, plan)
    terminals = solver.terminals()
    terminal_checks = []
    for t in terminals:
        chosen = [predicates[i] for i in bits(t["set"])]
        canonical_seq = solver.canonical_plan(t)
        rule_pairs = [
            (next(p for p in predicates if p.id == item["predicate"]["id"]), item["queue"])
            for item in canonical_seq["rules"]
        ]
        direct = objective_value(rule_pairs, t["default"], cases, costs, plan)
        terminal_checks.append(
            {"predicate_ids": [p.id for p in chosen],
             "default_queue": t["default"],
             "dp_value": t["value"], "direct_replay_value": direct,
             "matches": abs(direct - t["value"]) <= 1e-9}
        )
    record(
        "every_optimal_terminal_replays_to_its_value",
        all(x["matches"] for x in terminal_checks),
        {"n_terminals": len(terminal_checks), "terminals": terminal_checks},
    )

    # 4) 自制穷举基准：DP 与穷举的目标值、最优方案数、探针歧义完全一致
    from .benchmark import build_benchmark

    bench = build_benchmark()
    b_preds, b_cases, b_costs, b_plan, b_probes = (
        bench["predicates"], bench["cases"], bench["costs"], bench["plan"], bench["probes"]
    )
    bf = brute_force_optimum(b_preds, b_cases, b_costs, b_plan)
    b_solver = _build_solver(b_preds, b_cases, b_costs, b_plan)
    b_terms = b_solver.terminals()
    b_v = b_solver.optimum_value(b_terms)
    bf_probes = probe_queues_across_optima(bf["winners"], b_probes)
    dp_probes = b_solver.probe_ambiguity(b_probes, b_terms, b_v)
    probe_match = all(
        sorted(a["reachable_queues"]) == sorted(b["reachable_queues"])
        and a["determined"] == b["determined"]
        for a, b in zip(bf_probes, dp_probes)
    )
    # 穷举与 DP 收集到的最优终端签名一致
    bf_sigs = {w["sig"] for w in bf["winners"]}
    dp_sigs = set()
    for t in b_terms:
        seq = b_solver.canonical_plan(t)
        dp_sigs.add(
            (tuple((x["predicate"]["id"], x["queue"]) for x in seq["rules"]),
             t["default"])
        )
    record(
        "benchmark_dp_equals_brute_force",
        abs(b_v - bf["value"]) <= 1e-9 and bf_sigs == dp_sigs and probe_match,
        {"dp_value": b_v, "brute_force_value": bf["value"],
         "n_optima_dp": len(dp_sigs), "n_optima_brute_force": len(bf_sigs),
         "probe_ambiguity_equal": probe_match},
    )

    # 5) 基准三性质：顺序敏感、最短非最优、探针歧义真实
    bench_props = run_benchmark_checks()
    record(
        "benchmark_required_properties",
        bench_props["all_properties_hold"],
        {
            "order_dependence": bench_props["order_dependence"],
            "shortest_not_best": bench_props["shortest_not_best"],
            "ambiguous_probes": [r["probe_id"] for r in bench_props["probes"]
                                 if not r["determined"]],
            "best_value_by_rule_count": bench_props["best_value_by_rule_count"],
        },
    )

    # 6) 主数据探针：DP 报告的每个歧义证据方案，重放目标值必须等于 V*，
    #    且两份证据对该探针确实分流不同
    saved_report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    witness_checks = []
    for item in saved_report["probes"]:
        ok_queues = set()
        all_optimal = True
        for w in item["witnesses"]:
            pairs = [
                (next(p for p in predicates if p.id == r["predicate"]["id"]), r["queue"])
                for r in w["plan"]["rules"]
            ]
            val = objective_value(pairs, w["plan"]["default_queue"], cases, costs, plan)
            if abs(val - saved_report["objective_value"]) > 1e-9:
                all_optimal = False
            # 用 apply 的同一套重放逻辑确认该探针被分到证据声称的队列
            routed = replay_rows(
                w["plan"]["rules"], w["plan"]["default_queue"],
                [{"id": item["probe_id"],
                  "fields": next(x["fields"] for x in probes if x["id"] == item["probe_id"])}],
            )[0]["assigned_queue"]
            ok_queues.add(routed)
            if routed != w["queue"]:
                all_optimal = False
        witness_ok = all_optimal and (
            item["determined"] or len(ok_queues) == len(item["reachable_queues"]) > 1
        )
        witness_checks.append({"probe_id": item["probe_id"], "passed": witness_ok,
                               "queues_proved": sorted(ok_queues)})
    record(
        "probe_witnesses_are_optimal_and_distinct",
        all(x["passed"] for x in witness_checks),
        {"probes": witness_checks},
    )

    all_passed = all(c["passed"] for c in checks)
    payload = {"all_passed": all_passed, "checks": checks,
               "benchmark_properties": bench_props}
    _write_json(out_dir / "verification.json", payload)
    if not all_passed:
        raise AssertionError("verify 发现核验失败项，详见 verification.json")
    return payload
