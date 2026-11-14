"""Static report rendering and CSV/JSON artefacts."""
from __future__ import annotations

import csv
import json
import math
import os

import numpy as np

from .tool import (FitResult, Inputs, ReplicateResult, VarianceResult,
                   VerifyReport, check_feasibility)


def _fmt(x, nd: int = 8) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _write_csv(path: str, header: list[str], rows: list[list]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)


def write_artefacts(inputs: Inputs, A: np.ndarray, d: np.ndarray,
                    fit_result: FitResult, var: VarianceResult,
                    reps: list[ReplicateResult], verify_report: VerifyReport,
                    outdir: str) -> dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    paths: dict[str, str] = {}
    w = fit_result.weights

    rows = []
    for i, r in enumerate(inputs.respondents):
        rows.append([
            r.respondent_id, r.groups.get("age", ""), r.groups.get("region", ""),
            r.groups.get("employment", ""), f"{r.score:g}",
            f"{d[i]:.12g}", f"{fit_result.lower[i]:.12g}",
            f"{fit_result.upper[i]:.12g}", f"{w[i]:.12g}",
            int(fit_result.at_lower[i]), int(fit_result.at_upper[i]),
        ])
    paths["weights"] = os.path.join(outdir, "weights.csv")
    _write_csv(paths["weights"],
               ["respondent_id", "age", "region", "employment", "score",
                "base_weight", "lower_bound", "upper_bound", "adjusted_weight",
                "at_lower_bound", "at_upper_bound"], rows)

    mrows = []
    for i, m in enumerate(inputs.margins):
        c = next(c for c in verify_report.constraint_checks
                 if c["constraint_id"] == m.constraint_id)
        mrows.append([m.constraint_id, m.field, m.value, m.raw_total,
                      f"{m.total_float:.12g}", f"{c['lhs']:.12g}",
                      f"{c['residual']:.6e}", f"{c['tol']:.3e}",
                      int(c["independent"]), "ok" if c["ok"] else "FAIL"])
    paths["margins"] = os.path.join(outdir, "margin_residuals.csv")
    _write_csv(paths["margins"],
               ["constraint_id", "field", "value", "population_total",
                "target", "achieved", "residual", "tolerance", "independent",
                "status"], mrows)

    rrows = []
    for rr in reps:
        rrows.append([
            rr.replicate_id, rr.status,
            f"{rr.fit.max_residual:.6e}"
            if rr.fit is not None and math.isfinite(rr.fit.max_residual) else "",
            f"{rr.fit.iterations}" if rr.fit and rr.status == "feasible" else "",
            " | ".join(rr.diagnostics),
            *[f"{rr.estimates[dom['id']]:.12g}"
              if rr.status == "feasible" and dom["id"] in rr.estimates else ""
              for dom in inputs.plan.domains],
        ])
    paths["replicates"] = os.path.join(outdir, "replicates.csv")
    _write_csv(paths["replicates"],
               ["replicate_id", "status", "max_margin_residual", "iterations",
                "diagnostics", *[dom["id"] for dom in inputs.plan.domains]],
               rrows)

    summary = {
        "status": fit_result.status,
        "objective": fit_result.objective,
        "iterations": fit_result.iterations,
        "solver_converged": fit_result.solver_converged,
        "max_margin_residual": fit_result.max_residual,
        "max_margin_tolerance": fit_result.max_margin_tol,
        "max_interior_kkt": fit_result.max_kkt,
        "optimality_ok": fit_result.optimality_ok,
        "tolerances": fit_result.tolerance,
        "independent_constraints": [inputs.margins[i].constraint_id
                                    for i in fit_result.independent],
        "margins": [{"id": m.constraint_id, "target": str(m.total),
                     "raw": m.raw_total} for m in inputs.margins],
        "replicate_variance_coefficient": var.coefficient,
        "domains": {},
        "replicate_status": var.replicate_status,
        "replicate_diagnostics": [
            {"replicate_id": rr.replicate_id, "status": rr.status,
             "max_margin_residual": (rr.fit.max_residual
                                     if rr.fit is not None and
                                     math.isfinite(rr.fit.max_residual) else None),
             "diagnostics": rr.diagnostics,
             "estimates": (rr.estimates if rr.status == "feasible" else {})}
            for rr in reps],
        "nonresponse_assumption": inputs.plan.nonresponse_assumption,
        "verify_ok": verify_report.ok,
        "constraint_checks": verify_report.constraint_checks,
        "bound_checks": verify_report.bound_checks,
        "zero_checks": verify_report.zero_checks,
        "objective_check": verify_report.objective_check,
        "optimality_check": verify_report.optimality_check,
        "variance_recalc": verify_report.recalc_variance,
        "replicate_checks": verify_report.replicate_checks,
        "problems": verify_report.problems,
        "variance_notes": var.notes,
    }
    for dom in inputs.plan.domains:
        did = dom["id"]
        summary["domains"][did] = {
            "estimate": var.estimates.get(did, float("nan")),
            "variance": var.variances.get(did, float("nan")),
            "standard_error": var.standard_errors.get(did, float("nan")),
            "replicates_used": var.n_used.get(did, 0),
            "se_complete": bool(var.complete.get(did, False)),
        }
    paths["summary"] = os.path.join(outdir, "summary.json")
    with open(paths["summary"], "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, allow_nan=True)

    paths["report"] = os.path.join(outdir, "report.md")
    with open(paths["report"], "w", encoding="utf-8") as fh:
        fh.write(render_report(inputs, A, d, fit_result, var, reps,
                               verify_report))
    return paths


def render_report(inputs: Inputs, A: np.ndarray, d: np.ndarray,
                  fit_result: FitResult, var: VarianceResult | None,
                  reps: list[ReplicateResult] | None,
                  verify_report: VerifyReport | None) -> str:
    L: list[str] = []
    P = L.append
    P("# 有界熵补权与设计重复方差 — 静态报告\n")

    P("## 0. 状态\n")
    status_label = {"feasible": "可行 / 求解成功（基础数据）",
                    "infeasible": "不可行（基础数据）",
                    "failed": "求解未通过收敛/残差/最优性检查"}[fit_result.status]
    P(f"- 基础数据状态: **{fit_result.status.upper()}** — {status_label}")
    if fit_result.status in ("infeasible", "failed"):
        if fit_result.status == "infeasible":
            P("- 未输出任何调整权重，也未输出完整标准误；以下为可独立复核的"
              "冲突/上下界证据。\n")
        else:
            P("- 求解器未能满足收敛/残差/最优性检查；不输出完整标准误，"
              "以下为诊断。\n")
        P("### 预检/求解诊断")
        for msg in fit_result.diagnostics:
            P(f"- {msg}")
        fe = check_feasibility(inputs, A, d, inputs.plan.relative_lower,
                               inputs.plan.relative_upper)
        if fe.evidence:
            P("\n### 上下界/冲突证据（JSON 可逐项复核）")
            for ev in fe.evidence:
                P(f"- `{json.dumps(ev, ensure_ascii=False, default=str)}`")
        P("\n### 各约束在盒式界下的可达区间")
        P("| 约束 | 目标 | 盒式下界最小值 | 盒式上界最大值 | 结论 |")
        P("|---|---:|---:|---:|---|")
        lo = inputs.plan.relative_lower * d
        hi = inputs.plan.relative_upper * d
        for i, m in enumerate(inputs.margins):
            lb, ub = float(A[i] @ lo), float(A[i] @ hi)
            verdict = "可达" if lb <= m.total_float <= ub else \
                ("低于最小可达" if m.total_float < lb else "高于最大可达")
            P(f"| {m.constraint_id} | {m.total_float:.8g} | {lb:.8g} | "
              f"{ub:.8g} | {verdict} |")
        P("\n## 6. 非响应假设与解释边界\n")
        P(f"- 明确假设: {inputs.plan.nonresponse_assumption}")
        P("- 该假设无法由仅有响应者的数据本身检验。")
        P("- 边际配平只修正已知分组结构；它不等于消除所有非响应偏差。")
        P("- 未抽取新样本，未补造未响应者，未填补评分。")
        return "\n".join(L) + "\n"

    P(f"- 求解器收敛: {fit_result.solver_converged}; 迭代步数 "
      f"{fit_result.iterations}")
    P(f"- 目标值（熵距离）: {_fmt(fit_result.objective, 10)}")
    P(f"- 最大边际残差: {fit_result.max_residual:.3e} "
      f"(容差上界 {fit_result.max_margin_tol:.3e})")
    P(f"- 内部点最大 |KKT|: {fit_result.max_kkt:.3e}; "
      f"下界 kkt 最小 {fit_result.kkt_lower_min:.3e}; "
      f"上界 kkt 最大 {fit_result.kkt_upper_max:.3e}")
    P(f"- 最优性检查: {'通过' if fit_result.optimality_ok else '未通过'}")
    P(f"- 独立复核: {'PASS' if verify_report and verify_report.ok else 'FAIL'}")
    for msg in fit_result.diagnostics:
        P(f"- 诊断: {msg}")
    P("")

    P("## 1. 数值容差（相对人口总量，公开）\n")
    t = fit_result.tolerance
    P(f"- 约束容差: `REL_TOL={t['rel_tol']:.1e}` × max(|目标总量|, 1)，"
      f"逐约束计算（见第 2 节 tolerance 列）。")
    P(f"- 界容差: `BOUND_REL_TOL={t['bound_rel_tol']:.1e}` × max(上界, 1)，"
      f"逐响应者计算。")
    P(f"- 最优性容差: `OPT_TOL={t['opt_tol']:.1e}`；LP 预检 "
      f"`LP_FEAS_TOL={t['lp_feas_tol']:.1e}`。")
    P("- 容差随人口总量缩放；不使用大容差掩盖不收敛：任何一项失败都会把状态"
      "标为 `failed`，并不输出完整标准误。\n")

    P("## 2. 约束支持证据（含冗余约束的精确核对）\n")
    P("| 约束 | 目标 | 实现 | 残差 | 逐约束容差 | 独立/冗余 | 状态 |")
    P("|---|---:|---:|---:|---:|:--:|:--:|")
    for c in verify_report.constraint_checks:
        P(f"| {c['constraint_id']} | {c['rhs']:.10g} | {c['lhs']:.10g} | "
          f"{c['residual']:.2e} | {c['tol']:.2e} | "
          f"{'独立' if c['independent'] else '冗余'} | "
          f"{'ok' if c['ok'] else 'FAIL'} |")
    P("")
    fe = check_feasibility(inputs, A, d, inputs.plan.relative_lower,
                           inputs.plan.relative_upper)
    P("冗余边际以**精确有理数系数组合**核对其业务含义（不是直接删除）：")
    for cert in fe.certificates:
        combo = " + ".join(
            f"({c})×[{cid}]" for c, cid in
            zip(cert.coefficients, cert.independent_ids))
        P(f"- `{cert.dependent_id}` = {combo}; 精确隐含总量 "
          f"{cert.implied_total}，给定 {cert.stated_total} → "
          f"{'一致' if cert.consistent else '冲突'}")
    P("")

    P("## 3. 上下界命中\n")
    n_low = int(np.sum(fit_result.at_lower))
    n_up = int(np.sum(fit_result.at_upper))
    P(f"- 范围: `[{inputs.plan.relative_lower}×d, "
      f"{inputs.plan.relative_upper}×d]`；零基础权重固定为 0。")
    P(f"- 命中下界 {n_low} 人，命中上界 {n_up} 人，内部 "
      f"{int(np.sum((d > 0) & ~fit_result.at_lower & ~fit_result.at_upper))} 人。")
    for i, resp in enumerate(inputs.respondents):
        if d[i] == 0:
            P(f"- `{resp.respondent_id}` 基础权重 0 → 调整权重固定 0。")
    P("")

    P("## 4. 逐响应者权重与领域估计\n")
    P(f"- 逐响应者调整权重、上下界、是否命中界见 `weights.csv`"
      f"（{len(inputs.respondents)} 行，按 respondent_id 可追溯）。")
    P(f"- 每名响应者都带齐分组、评分与基础权重；本工具仅调整**已响应者**的权重，"
      f"不补造未响应者。\n")
    if var:
        P(f"- 方差系数（plan 供设计）: {var.coefficient:g}")
        P("| 领域 | 估计 | 方差 | 标准误 | 用到的重复数 | SE 完整 |")
        P("|---|---:|---:|---:|---:|:--:|")
        for dom in inputs.plan.domains:
            did = dom["id"]
            P(f"| {did} | {_fmt(var.estimates.get(did))} | "
              f"{_fmt(var.variances.get(did))} | "
              f"{_fmt(var.standard_errors.get(did))} | "
              f"{var.n_used.get(did, 0)}/{var.n_total} | "
              f"{'是' if var.complete.get(did) else '**否（不完整）**'} |")
        P(f"- 公式: {var.coefficient:g} × Σ_r (estimate_r − full_estimate)²，"
          f"r 仅为可用重复。")
        P("- 若某重复失败，该领域标准误按可用项合成并显式标为不完整；"
          "不跳过却仍给完整 SE。\n")

        P("### 逐重复诊断")
        P("| 重复 | 状态 | 最大边际残差 | 诊断 |")
        P("|---|:--:|---:|---|")
        for rr in reps or []:
            res = "n/a" if rr.fit is None or not math.isfinite(rr.fit.max_residual) \
                else f"{rr.fit.max_residual:.2e}"
            diag = "; ".join(rr.diagnostics) or "-"
            P(f"| {rr.replicate_id} | {rr.status} | {res} | {diag} |")
        P("")
        for note in var.notes:
            P(f"- {note}")
        P("")

    P("## 5. 独立复核（由权重重算，不复用求解器）\n")
    if verify_report:
        P(f"- 全部约束、界、零权重: "
          f"{'PASS' if verify_report.ok else 'FAIL'}")
        tc = verify_report.total_check
        if tc.get("present"):
            P(f"- 权重总和 {tc['sum_weights']:.10g} vs '*' 总量 "
              f"{tc['stated']:.10g}（差 {tc['gap']:.2e}）")
        P(f"- 目标函数重算差: {verify_report.objective_check['gap']:.2e}")
        oc = verify_report.optimality_check
        P(f"- KKT 重算: 内部 {oc['interior_max_abs']:.2e}，下界 {oc['lower_min']:.2e}，"
          f"上界 {oc['upper_max']:.2e}")
        if var:
            for did, rec in verify_report.recalc_variance.items():
                if isinstance(rec, dict) and "direct" in rec:
                    P(f"- 方差 {did}: 直接代入 = {rec['direct']:.10g}，"
                      f"报告 = {rec['reported']:.10g} → "
                      f"{'一致' if rec['ok'] else '不一致'}（n={rec['n']}）")
        if verify_report.replicate_checks:
            bad = [c for c in verify_report.replicate_checks if not c["ok"]]
            P(f"- 逐重复复核: {len(verify_report.replicate_checks)} 个重复，"
              f"{len(bad)} 个问题")
            for c in verify_report.replicate_checks:
                if c["status"] == "feasible":
                    P(f"  - {c['replicate_id']}: margins_ok={c['margins_ok']}, "
                      f"bounds_ok={c['bounds_ok']}, "
                      f"max_resid={c['max_margin_residual']:.2e}")
                else:
                    P(f"  - {c['replicate_id']}: {c['status']}, "
                      f"已由独立 LP 确认为不可行="
                      f"{c.get('confirmed_infeasible_by_lp')}")
        for p in verify_report.problems:
            P(f"- 问题: {p}")
    P("")

    P("## 6. 非响应假设与解释边界\n")
    P(f"- 明确假设: {inputs.plan.nonresponse_assumption}")
    P("- 该假设无法由仅有响应者的数据本身检验。")
    P("- 边际配平只修正已知分组结构（年龄/地区/就业及其交叠总量）；它**不等于**"
      "消除所有非响应偏差，也不保证组内评分分布相同。")
    P("- 未抽取新样本，未补造未响应者，未填补评分。")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Infeasible-path artefacts
# ---------------------------------------------------------------------------
def write_infeasible_artefacts(inputs: Inputs, A: np.ndarray, d: np.ndarray,
                               fit_result: FitResult, fe,
                               outdir: str) -> dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    paths: dict[str, str] = {}
    rows = []
    lo, hi = inputs.plan.relative_lower * d, inputs.plan.relative_upper * d
    for i, m in enumerate(inputs.margins):
        lb, ub = float(A[i] @ lo), float(A[i] @ hi)
        rows.append([m.constraint_id, m.field, m.value, m.raw_total,
                     f"{m.total_float:.12g}", f"{lb:.12g}", f"{ub:.12g}",
                     "ok" if lb <= m.total_float <= ub else "unreachable"])
    paths["bounds"] = os.path.join(outdir, "constraint_bounds.csv")
    _write_csv(paths["bounds"],
               ["constraint_id", "field", "value", "population_total",
                "target", "box_min", "box_max", "status"], rows)

    brows = []
    for i, r in enumerate(inputs.respondents):
        brows.append([r.respondent_id, f"{d[i]:.12g}", f"{lo[i]:.12g}",
                      f"{hi[i]:.12g}"])
    paths["bounds_per_respondent"] = os.path.join(outdir, "bounds.csv")
    _write_csv(paths["bounds_per_respondent"],
               ["respondent_id", "base_weight", "lower_bound", "upper_bound"],
               brows)

    summary = {
        "status": fit_result.status,
        "messages": fit_result.diagnostics,
        "evidence": fe.evidence,
        "minimal_conflict_set": fe.conflict_set,
        "independent_constraints": [inputs.margins[i].constraint_id
                                    for i in fit_result.independent],
        "redundancy_certificates": [
            {"dependent": c.dependent_id,
             "basis": c.independent_ids,
             "coefficients": [str(x) for x in c.coefficients],
             "implied_total": str(c.implied_total),
             "stated_total": str(c.stated_total),
             "consistent": c.consistent}
            for c in fe.certificates],
        "tolerances": fit_result.tolerance,
    }
    paths["summary"] = os.path.join(outdir, "summary.json")
    with open(paths["summary"], "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    paths["report"] = os.path.join(outdir, "report.md")
    with open(paths["report"], "w", encoding="utf-8") as fh:
        fh.write(render_report(inputs, A, d, fit_result, None, [], None))
    return paths
