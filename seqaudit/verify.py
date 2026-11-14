"""verify 入口：从原始输入独立重算，核对证书与风险保证。

复核内容：
  1. 输入文件哈希与证书记载一致；
  2. 抽样资格独立重审，结论与证书一致；
  3. 证据逐前缀重算（组合数形式），并用逐步条件乘积形式交叉复核；
  4. 批次不变性：更换查看批次重跑审计，同一观察前缀的证据不变；
  5. 判定逻辑：接受标志 == (证据 >= 阈值)、首次接受位置、结论自洽；
  6. 全量检查一致性：观察到末件时，结论与真实缺陷总数一致；
  7. 资格失败时证书必须无效且不携带统计结论。
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from .audit import DECISION_ACCEPT, run_audit
from .eligibility import check_eligibility
from .evidence import (
    INF,
    evidence_exact_str,
    evidence_ge,
    evidence_value,
    evidence_walk_product_form,
    mixture_weights,
)
from .io import ObsRow, Plan


def _parse_evidence(s: str) -> Fraction | float:
    return INF if s == "inf" else Fraction(s)


class _Recorder:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def ck(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(c["ok"] for c in self.checks)


def verify_report(
    plan: Plan,
    lot_ids: list[str],
    obs_rows: list[ObsRow],
    report: dict,
    inputs_meta: dict,
) -> dict:
    rec = _Recorder()

    # 1. 输入哈希一致
    for key in ("plan", "lot", "observations"):
        want = report.get("inputs", {}).get(key, {}).get("sha256")
        got = inputs_meta[key]["sha256"]
        rec.ck(
            f"input_hash_{key}",
            want == got,
            f"证书 sha256={want}，重算 sha256={got}",
        )

    # 2. 资格独立重审
    elig = check_eligibility(plan, lot_ids, obs_rows)
    rec.ck(
        "eligibility_recomputed_matches",
        elig.to_json() == report.get("eligibility"),
        f"重审 ok={elig.ok}，证书记载 ok={report.get('eligibility', {}).get('ok')}",
    )

    if not elig.ok:
        rec.ck(
            "no_valid_certificate_on_ineligible",
            report.get("certificate_valid") is False and report.get("audit") is None,
            "资格失败时报告必须 certificate_valid=false 且不含统计结论",
        )
        return _record(rec)

    rec.ck(
        "certificate_marked_valid",
        report.get("certificate_valid") is True,
        "资格通过时证书必须标记有效",
    )

    # 3. 证据逐前缀重算（组合数形式）+ 交叉实现（条件乘积形式）
    n, d0 = plan.lot_size, plan.d0
    defects = [int(r.defect_raw) for r in obs_rows]
    m = len(defects)
    weights = mixture_weights(d0)
    evs, k = [], 0
    for t, x in enumerate(defects, start=1):
        k += x
        evs.append(evidence_value(n, d0, t, k, weights))
    evs_cross = evidence_walk_product_form(n, d0, defects, weights)
    rec.ck(
        "evidence_cross_implementation",
        evs == evs_cross,
        "组合数形式与逐步条件乘积形式在每一前缀完全一致"
        if evs == evs_cross
        else "两种独立实现结果不一致",
    )

    audit = report.get("audit") or {}
    cert_path = audit.get("path") or []
    path_ok = len(cert_path) == m
    if path_ok:
        for i, entry in enumerate(cert_path):
            if (
                entry.get("draw_no") != i + 1
                or entry.get("item_id") != obs_rows[i].item_id
                or entry.get("defect") != defects[i]
                or entry.get("cum_defects") != sum(defects[: i + 1])
                or entry.get("evidence_exact") != evidence_exact_str(evs[i])
            ):
                path_ok = False
                break
    rec.ck(
        "evidence_path_recomputed",
        path_ok,
        f"逐抽路径 {m} 条与证书一致（含件号、缺陷、累计、证据精确值）",
    )

    # 4. 批次不变性：同一观察前缀的证据与查看批次划分无关
    alt_plan = replace(plan, batches=tuple(range(1, max(m, 1) + 1)))
    alt_report, _ = run_audit(alt_plan, lot_ids, obs_rows, inputs_meta)
    alt_path = (alt_report.get("audit") or {}).get("path") or []
    same_prefix = [e.get("evidence_exact") for e in alt_path] == [
        evidence_exact_str(e) for e in evs
    ]
    looks_ok = True
    for look in audit.get("looks") or []:
        t = look.get("draw")
        if not (1 <= t <= m) and t != 0:
            looks_ok = False
            break
        if t >= 1 and look.get("evidence_exact") != evidence_exact_str(evs[t - 1]):
            looks_ok = False
            break
    rec.ck(
        "batching_does_not_change_evidence",
        same_prefix and looks_ok,
        "改用逐抽查看重跑后，同一前缀证据不变；证书查看点取值等于该前缀证据",
    )

    # 5. 判定逻辑自洽
    threshold = Fraction(audit.get("threshold", "1"))
    looks = audit.get("looks") or []
    flags_ok = all(
        evidence_ge(_parse_evidence(l["evidence_exact"]), threshold)
        == l.get("accept_allowed")
        for l in looks
    )
    rec.ck(
        "accept_flags_match_threshold",
        flags_ok,
        "每个查看点的 accept_allowed 恰为 证据 >= 阈值",
    )
    computed_first = next((l["draw"] for l in looks if l.get("accept_allowed")), None)
    rec.ck(
        "first_accept_draw_consistent",
        computed_first == audit.get("first_accept_draw"),
        f"重算首次允许接受位置={computed_first}，证书记载={audit.get('first_accept_draw')}",
    )
    decision = audit.get("decision")
    if computed_first is not None:
        expect = DECISION_ACCEPT
    elif m == n:
        expect = "DO_NOT_ACCEPT"
    else:
        expect = "CONTINUE_INSUFFICIENT_EVIDENCE"
    rec.ck(
        "decision_consistent",
        decision == expect,
        f"结论应为 {expect}，证书记载 {decision}",
    )

    # 6. 全量检查一致性：观察到末件时结论与真实缺陷总数一致
    if m == n:
        total = sum(defects)
        rec.ck(
            "full_inspection_matches_truth",
            (total < d0) == (decision == DECISION_ACCEPT),
            f"全量检查真实缺陷总数 D={total}，d0={d0}，结论={decision}",
        )
        fi = audit.get("full_inspection") or {}
        rec.ck(
            "full_inspection_block_consistent",
            fi.get("true_total_defects") == total
            and fi.get("decision_matches_truth") is True,
            "证书 full_inspection 块与真实缺陷总数一致",
        )

    # 7. 设计参数确由 plan 冻结
    design = report.get("design") or {}
    rec.ck(
        "design_frozen_from_plan",
        design.get("reject_if_defects_at_least") == d0
        and design.get("alpha_fraction") == str(plan.alpha)
        and design.get("evidence_threshold") == str(Fraction(1, 1) / plan.alpha)
        and design.get("look_batches") == list(plan.batches)
        and design.get("design_frozen_before_outcomes") is True,
        "零假设边界、alpha、阈值、查看批次均取自 plan（读结果前冻结）",
    )
    return _record(rec)


def _record(rec: _Recorder) -> dict:
    return {
        "schema": "seqaudit.verification/v1",
        "checks": rec.checks,
        "overall": "PASS" if rec.passed else "FAIL",
    }
