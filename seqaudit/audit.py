"""audit 入口的核心：资格门控 + 顺序证据 + 证书/诊断报告生成。"""
from __future__ import annotations

from fractions import Fraction

from . import __version__
from .eligibility import check_eligibility
from .evidence import (
    evidence_exact_str,
    evidence_float,
    evidence_ge,
    evidence_value,
    mixture_weights,
)
from .io import ObsRow, Plan

SCHEMA = "seqaudit.report/v1"

DECISION_ACCEPT = "ACCEPT"
DECISION_CONTINUE = "CONTINUE_INSUFFICIENT_EVIDENCE"
DECISION_DO_NOT_ACCEPT = "DO_NOT_ACCEPT"


def design_json(plan: Plan) -> dict:
    threshold = Fraction(1, 1) / plan.alpha
    return {
        "lot_size": plan.lot_size,
        "null_hypothesis": f"整批缺陷总数 D >= {plan.d0}（复合零假设）",
        "reject_if_defects_at_least": plan.d0,
        "alpha": plan.alpha_raw,
        "alpha_fraction": str(plan.alpha),
        "evidence_threshold": str(threshold),
        "evidence_process": (
            "混合似然比检验鞅：均匀混合备择 d1 in "
            f"{{0,...,{plan.d0 - 1}}}（各 1/{plan.d0}）对边界零假设 D={plan.d0}；"
            "无放回超几何似然，精确有理数计算"
        ),
        "mixture_weights": f"uniform over d1 in {{0,...,{plan.d0 - 1}}}, each 1/{plan.d0}",
        "design_frozen_before_outcomes": True,
        "draw_algorithm": plan.draw_algorithm,
        "draw_seed": plan.seed,
        "look_batches": list(plan.batches),
        "risk_guarantee": (
            f"对任意缺陷总数 D >= {plan.d0} 的批次，以及任意可依赖已揭示结果的"
            f"查看/停止规则，在任意查看时点曾被错误接受的概率 <= {plan.alpha_raw}"
            "（非负上鞅 + Ville 不等式）。"
        ),
    }


def run_audit(
    plan: Plan,
    lot_ids: list[str],
    obs_rows: list[ObsRow],
    inputs_meta: dict,
) -> tuple[dict, int]:
    """返回 (报告 dict, 退出码)。退出码 0=出具有效证书，2=资格失败仅诊断。"""
    elig = check_eligibility(plan, lot_ids, obs_rows)
    report = {
        "schema": SCHEMA,
        "certificate_valid": False,
        "tool": {"name": "seqaudit", "version": __version__},
        "inputs": inputs_meta,
        "design": design_json(plan),
        "eligibility": elig.to_json(),
        "audit": None,
    }
    if not elig.ok:
        # 资格失败：保留诊断，不得继续出具有效审计证书。
        report["note"] = "抽样资格失败：保留诊断，不出具有效审计证书，不给出统计结论。"
        return report, 2

    n, d0 = plan.lot_size, plan.d0
    defects = [int(r.defect_raw) for r in obs_rows]
    m = len(defects)
    threshold = Fraction(1, 1) / plan.alpha
    weights = mixture_weights(d0)

    # 逐前缀证据（只依赖观察前缀本身，与查看批次如何划分无关）。
    evs = []
    k = 0
    for x in defects:
        k += x
        evs.append(evidence_value(n, d0, len(evs) + 1, k, weights))

    path = []
    k = 0
    for t, (row, x) in enumerate(zip(obs_rows, defects), start=1):
        k += x
        path.append(
            {
                "draw_no": t,
                "item_id": row.item_id,
                "defect": x,
                "cum_defects": k,
                "remaining": n - t,
                "evidence_exact": evidence_exact_str(evs[t - 1]),
                "evidence_float": evidence_float(evs[t - 1]),
            }
        )

    # 查看点：plan 批次边界（<= m），加上观察终止点（若不在边界上）。
    look_draws = [b for b in plan.batches if b <= m]
    look_type = {b: "batch_boundary" for b in look_draws}
    if m not in look_type:
        look_draws.append(m)
        look_type[m] = "end_of_observations"

    looks = []
    first_accept = None
    for t in look_draws:
        cum_defects = sum(defects[:t])
        allowed = evidence_ge(evs[t - 1], threshold) if t >= 1 else False
        if allowed and first_accept is None:
            first_accept = t
        looks.append(
            {
                "look_type": look_type[t],
                "draw": t,
                "observed": t,
                "defects": cum_defects,
                "remaining": n - t,
                "evidence_exact": evidence_exact_str(evs[t - 1]) if t >= 1 else "1",
                "evidence_float": evidence_float(evs[t - 1]) if t >= 1 else 1.0,
                "threshold": str(threshold),
                "accept_allowed": allowed,
                "first_accept_draw_so_far": first_accept,
            }
        )

    first_crossing = next(
        (t for t in range(1, m + 1) if evidence_ge(evs[t - 1], threshold)), None
    )

    total_defects = sum(defects)
    if first_accept is not None:
        decision = DECISION_ACCEPT
        reason = (
            f"第 {first_accept} 抽查看点证据达到阈值 {threshold}，"
            "按顺序停止规则接受整批；此后若继续观察不影响本次结论。"
        )
    elif m == n:
        # 全量检查到末件仍未接受：真实缺陷总数已被完整揭示且 D >= d0。
        decision = DECISION_DO_NOT_ACCEPT
        reason = (
            f"已全量检查 {n} 件，真实缺陷总数 D={total_defects} >= {d0}，"
            "零假设成立，不得接受整批。"
        )
    else:
        peak = max(evs) if evs else Fraction(1)
        decision = DECISION_CONTINUE
        reason = (
            f"已查 {m}/{n} 件，证据峰值 {evidence_exact_str(peak)} 低于阈值 "
            f"{threshold}；证据不足，不接受整批，需继续抽查。"
        )

    audit_block = {
        "observations_reviewed": m,
        "threshold": str(threshold),
        "looks": looks,
        "path": path,
        "first_evidence_crossing_draw": first_crossing,
        "first_accept_draw": first_accept,
        "decision": decision,
        "decision_reason": reason,
    }
    if m == n:
        # 全量检查：累计缺陷即真实缺陷总数，结论与真实状态一致。
        audit_block["full_inspection"] = {
            "true_total_defects": total_defects,
            "null_hypothesis_true": total_defects >= d0,
            "decision_matches_truth": (total_defects < d0)
            == (decision == DECISION_ACCEPT),
        }
    report["audit"] = audit_block
    report["certificate_valid"] = True
    return report, 0
