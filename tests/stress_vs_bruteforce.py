"""压力核验：多组随机小实例上，子集 DP 与全穷举逐项一致。

覆盖不同谓词数、案例数、队列数、算子、惩罚与默认队列集合；
独立于 verify 内的固定基准，用于证明 DP 实现与穷举定义在随机数据上等价。
直接运行：python3 tests/stress_vs_bruteforce.py（需在仓库根目录）。
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rulelearn.api import _build_solver
from rulelearn.brute_force import brute_force_optimum, probe_queues_across_optima
from rulelearn.data import Case, CostMatrix, Plan, Predicate


def random_instance(rng, n_pred, n_cases, n_q):
    fields = ["f1", "f2", "f3", "age"]
    pred_specs = [
        ("f1", "eq", 1), ("f2", "eq", 1), ("f3", "eq", 0),
        ("age", "ge", 2), ("age", "le", 1),
    ]
    preds = [Predicate(f"p{i}", *pred_specs[i % len(pred_specs)]) for i in range(n_pred)]
    qnames = [f"q{j}" for j in range(n_q)]
    cost = {}
    for t in qnames:
        cost[t] = {q: (0 if q == t else rng.randint(1, 9)) for q in qnames}
    costs = CostMatrix(cost, tuple(qnames))
    cases = []
    for j in range(n_cases):
        vals = {
            "f1": rng.randint(0, 1), "f2": rng.randint(0, 1),
            "f3": rng.randint(0, 1), "age": rng.randint(0, 4),
        }
        cases.append(Case(f"c{j}", vals, rng.choice(qnames), rng.randint(1, 4)))
    plan = Plan(
        max_rules=rng.randint(0, n_pred),
        rule_penalty=rng.randint(0, 5),
        allow_default=tuple(rng.sample(qnames, k=rng.randint(1, n_q))),
    )
    probes = [
        {"id": f"pr{k}", "fields": {
            "f1": rng.randint(0, 1), "f2": rng.randint(0, 1),
            "f3": rng.randint(0, 1), "age": rng.randint(0, 4)}}
        for k in range(3)
    ]
    return preds, cases, costs, plan, probes


def main():
    grand_fails = 0
    for seed in (20261114, 7, 99, 4242, 31337):
        rng = random.Random(seed)
        fails = 0
        for trial in range(60):
            n_pred = rng.randint(1, 5)
            n_cases = rng.randint(1, 8)
            n_q = rng.randint(2, 3)
            preds, cases, costs, plan, probes = random_instance(rng, n_pred, n_cases, n_q)
            # 穷举只在 ≤4 谓词时可承受；n=5 仅做不变性与值核验
            bf = brute_force_optimum(preds, cases, costs, plan)
            solver = _build_solver(preds, cases, costs, plan)
            terms = solver.terminals()
            v = solver.optimum_value(terms)
            assert abs(v - bf["value"]) <= 1e-9, (seed, trial, v, bf["value"])

            if n_pred <= 4:
                bf_probes = probe_queues_across_optima(bf["winners"], probes)
                dp_probes = solver.probe_ambiguity(probes, terms, v)
                for a, b in zip(bf_probes, dp_probes):
                    if sorted(a["reachable_queues"]) != sorted(b["reachable_queues"]):
                        fails += 1
                        print("PROBE MISMATCH seed", seed, "trial", trial, a["probe_id"],
                              a["reachable_queues"], b["reachable_queues"])

            shuffled = list(reversed(cases))
            s2 = _build_solver(list(reversed(preds)), shuffled, costs, plan)
            t2 = s2.terminals()
            if abs(s2.optimum_value(t2) - v) > 1e-9:
                fails += 1
                print("ORDER MISMATCH seed", seed, "trial", trial)
        print(f"seed {seed}: 60 组，失败 {fails} 组")
        grand_fails += fails
    print(f"总计失败 {grand_fails} 组")
    return grand_fails


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
