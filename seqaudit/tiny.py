"""tiny 独立枚举验证：穷举零假设缺陷配置 × 全部抽查排列，核查风险上限。

与主审计算法相互独立：这里用逐步条件概率乘积形式重算证据，
对每条完整路径检查"任意时点是否曾达到接受阈值"（不只看末时点，
也不复用主算法的接受标记），再按等权排列汇总曾接受概率。
"""
from __future__ import annotations

import json
from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path

from .evidence import evidence_ge, evidence_walk_product_form, mixture_weights
from .io import InputError


def load_tiny_spec(path: str | Path) -> tuple[int, int, Fraction]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        n = int(raw["N"])
        d0 = int(raw["null_defects_at_least"])
        alpha = Fraction(str(raw["alpha"]))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise InputError(f"无法解析 tiny 规格: {exc}") from exc
    if not 1 <= d0 <= n:
        raise InputError(f"tiny: null_defects_at_least={d0} 必须落在 1..{n}")
    if not 0 < alpha < 1:
        raise InputError(f"tiny: alpha={alpha} 必须落在 (0,1) 内")
    if n > 10:
        raise InputError(f"tiny: N={n} 过大，枚举限于 N<=10")
    return n, d0, alpha


def _ever_accepted_on_path(n: int, d0: int, defects: list[int], weights, threshold) -> bool:
    """独立判定：沿完整揭示路径逐前缀重算证据，任一前缀达阈即视为曾接受。"""
    for ev in evidence_walk_product_form(n, d0, defects, weights):
        if evidence_ge(ev, threshold):
            return True
    return False


def enumerate_tiny(n: int, d0: int, alpha: Fraction) -> dict:
    weights = mixture_weights(d0)
    threshold = Fraction(1, 1) / alpha
    perms = list(permutations(range(n)))
    total_perms = len(perms)

    null_rows = []
    for size in range(d0, n + 1):
        for config in combinations(range(n), size):
            cfg = frozenset(config)
            ever = 0
            for perm in perms:
                defects = [1 if it in cfg else 0 for it in perm]
                if _ever_accepted_on_path(n, d0, defects, weights, threshold):
                    ever += 1
            prob = Fraction(ever, total_perms)
            null_rows.append(
                {
                    "defects_total": size,
                    "defect_positions": list(config),
                    "permutations": total_perms,
                    "ever_accepted": ever,
                    "probability_exact": str(prob),
                    "probability_float": float(prob),
                }
            )
    max_prob = max(Fraction(r["ever_accepted"], total_perms) for r in null_rows)

    # 合格配置（D < d0）的一致性：任何顺序下全量检查末件前必须曾接受。
    good_rows = []
    for size in range(0, d0):
        for config in combinations(range(n), size):
            cfg = frozenset(config)
            all_ok = all(
                _ever_accepted_on_path(
                    n, d0, [1 if it in cfg else 0 for it in perm], weights, threshold
                )
                for perm in perms
            )
            good_rows.append(
                {
                    "defects_total": size,
                    "defect_positions": list(config),
                    "accepted_on_every_order_by_full_inspection": all_ok,
                }
            )

    return {
        "schema": "seqaudit.tiny_enumeration/v1",
        "method": (
            "独立枚举全部零假设缺陷配置（D>=d0 的全部子集）与全部 N! 抽查排列；"
            "对每条路径用逐步条件概率乘积形式逐前缀重算证据，"
            "检查整条路径上是否曾达到接受阈值（不只看末时点，不复用主算法标记）；"
            "每个配置下排列等权，曾接受概率=曾接受排列数/N!。"
        ),
        "design": {
            "N": n,
            "null_hypothesis": f"D >= {d0}",
            "alpha": str(alpha),
            "evidence_threshold": str(threshold),
            "mixture_weights": f"uniform over d1 in {{0,...,{d0 - 1}}}, each 1/{d0}",
        },
        "null_configurations": null_rows,
        "max_false_accept_probability_exact": str(max_prob),
        "max_false_accept_probability_float": float(max_prob),
        "alpha_float": float(alpha),
        "risk_bound_respected": max_prob <= alpha,
        "good_configurations": good_rows,
        "all_good_configurations_accepted_by_full_inspection": all(
            r["accepted_on_every_order_by_full_inspection"] for r in good_rows
        ),
    }
