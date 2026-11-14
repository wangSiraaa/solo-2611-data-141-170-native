"""端到端编排：装配输入、驱动搜索、独立重建核对并产出可执行工件。

独立重建（:func:`rebuild_and_verify`）刻意不接收搜索期任何矩阵对象，只从
最终候选 id 序列 + 原始声明重新构造设计矩阵，再逐条核对硬约束与可估计性，
避免"搜索时用一套矩阵、汇报时用另一套"的偏差。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .baseline import BlockDesign, build_baseline
from .constraints import Constraints, LockedRun, block_replicate_satisfied, load_locked_runs
from .criteria import (
    EPS_RANK, Quality, block_projector, correlation_diagnostics, d_efficiency, evaluate,
)
from .enumerate import EnumerationResult, count_multisets, enumerate_optimal
from .errors import ConfigError
from .factors import CandidateSet, FactorSet, build_candidate_set
from .model import BLOCK_PREFIX, Model
from .plan import Plan, StructuralEvidence, assert_structurally_feasible
from .search import SearchResult, coordinate_exchange


@dataclass
class DesignBundle:
    plan: Plan
    factor_set: FactorSet
    model: Model
    constraints: Constraints
    candidates: CandidateSet
    locked: list[LockedRun]
    evidence: StructuralEvidence
    result: SearchResult
    baseline_design: dict[str, BlockDesign]
    baseline_quality: Quality
    enumeration: EnumerationResult | None
    enumeration_note: str
    verification: dict[str, Any]


# ---------------------------------------------------------------------- #
# 装配
# ---------------------------------------------------------------------- #
def load_bundle(
    factors_path: str,
    model_path: str,
    constraints_path: str,
    plan_path: str,
) -> DesignBundle:
    """读取四份 JSON 并完成交叉校验（不执行搜索）。"""
    factor_set = FactorSet.from_json_file(factors_path)
    model = Model.from_json_file(factor_set, model_path)
    constraints = Constraints.from_json_file(constraints_path)
    plan = Plan.from_json_file(plan_path)

    candidates = build_candidate_set(factor_set, constraints.forbidden_rules)
    if candidates.n_candidates == 0:
        raise ConfigError("禁用规则剔除了全部候选，无合法组合可用")

    locked: list[LockedRun] = []
    if plan.locked_path:
        locked_path = _resolve_relative(plan_path, plan.locked_path)
        locked = load_locked_runs(
            locked_path, factor_set, candidates, block_ids=set(plan.block_ids)
        )
    evidence = assert_structurally_feasible(
        factor_set, candidates, model, plan, constraints, locked
    )
    # 占位字段在 run_design 中填充。
    return DesignBundle(
        plan=plan, factor_set=factor_set, model=model, constraints=constraints,
        candidates=candidates, locked=locked, evidence=evidence,
        result=None,  # type: ignore[arg-type]
        baseline_design={}, baseline_quality=None,  # type: ignore[arg-type]
        enumeration=None, enumeration_note="", verification={},
    )


def _resolve_relative(plan_path: str, maybe_relative: str) -> str:
    if os.path.isabs(maybe_relative):
        return maybe_relative
    # plan 可能位于 fixtures/ 或其任意子目录；从 plan 所在目录逐级向上，
    # 取第一个使相对路径真实存在的祖先目录，避免对路径层级做硬编码假设。
    plan_dir = os.path.dirname(os.path.abspath(plan_path))
    directory = plan_dir
    while True:
        candidate = os.path.normpath(os.path.join(directory, maybe_relative))
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    # 全部祖先都找不到时返回最贴近的路径，交由 open 抛出带路径的错误。
    return os.path.normpath(os.path.join(plan_dir, maybe_relative))


# ---------------------------------------------------------------------- #
# 搜索 + 基线 + 枚举 + 独立重建
# ---------------------------------------------------------------------- #
def run_design(
    bundle: DesignBundle,
    n_random_starts: int = 8,
    max_evaluations: int = 20000,
    time_budget_seconds: float = 120.0,
    enumeration_max_multisets: int = 5000,
    enumeration_max_total: int = 20000,
) -> DesignBundle:
    result = coordinate_exchange(
        bundle.plan, bundle.model, bundle.candidates, bundle.constraints,
        bundle.locked, n_random_starts=n_random_starts,
        max_evaluations=max_evaluations,
        time_budget_seconds=time_budget_seconds,
    )
    bundle.result = result

    bundle.baseline_design = build_baseline(
        bundle.plan, bundle.candidates, bundle.constraints, bundle.locked,
        model=bundle.model,
    )
    x_base_treat, x_base_full, _ = _matrices_of(
        bundle.baseline_design, bundle.plan, bundle.model, bundle.candidates
    )
    q = block_projector(_z_matrix(bundle.plan, bundle.model))
    bundle.baseline_quality = evaluate(
        x_base_treat, x_base_full, bundle.model.treatment_names, q,
        full_names=bundle.model.full_column_names(bundle.plan.block_ids),
        rank_tol=EPS_RANK,
    )

    # 微缩集枚举（超阈值则明确说明未做全枚举）。
    per_block_cap = 1
    for block in bundle.plan.blocks:
        free = block.runs - sum(
            1 for run in bundle.locked if run.block == block.block_id
        )
        per_block_cap = max(per_block_cap,
                            count_multisets(bundle.candidates.n_candidates, free))
    if per_block_cap <= enumeration_max_multisets:
        bundle.enumeration = enumerate_optimal(
            bundle.plan, bundle.model, bundle.candidates, bundle.constraints,
            bundle.locked, max_multisets_per_block=enumeration_max_multisets,
            max_total_designs=enumeration_max_total,
        )
        if bundle.enumeration is None:
            bundle.enumeration_note = (
                "逐块多重集在阈值内，但跨块组合总数超过 "
                f"{enumeration_max_total}，未执行全枚举"
            )
    else:
        bundle.enumeration = None
        bundle.enumeration_note = (
            f"单块自由多重集约 {per_block_cap} 个，超过阈值 "
            f"{enumeration_max_multisets}，跳过全枚举（搜索结论仅限已探索邻域）"
        )

    bundle.verification = rebuild_and_verify(bundle)
    return bundle


def _matrices_of(
    design: dict[str, BlockDesign], plan: Plan, model: Model,
    candidates: CandidateSet,
):
    combos: list[tuple] = []
    blocks: list[str] = []
    for bid in plan.block_ids:
        for cid in design[bid].candidate_ids:
            combos.append(candidates.combos[cid])
            blocks.append(bid)
    x_treat = model.treatment_matrix(combos)
    x_full = model.full_matrix(x_treat, blocks, plan.block_ids)
    return x_treat, x_full, blocks


def _z_matrix(plan: Plan, model: Model | None = None) -> np.ndarray:
    # 无块效应时返回 0 列矩阵，使 block_projector 得到单位阵 Q=I。
    if model is not None and not model.include_block_effect:
        return np.zeros((plan.total_runs, 0))
    z = np.zeros((plan.total_runs, len(plan.block_ids)))
    row = 0
    for j, bid in enumerate(plan.block_ids):
        quota = plan.quota(bid)
        z[row:row + quota, j] = 1.0
        row += quota
    return z


# ---------------------------------------------------------------------- #
# 独立重建核对
# ---------------------------------------------------------------------- #
def rebuild_and_verify(bundle: DesignBundle) -> dict[str, Any]:
    """只依赖最终 id 序列与原始声明，重建并核对全部硬约束与可估计对比。"""
    design = bundle.result.design
    checks: dict[str, Any] = {}

    # 1) 块名额
    quota_ok = all(
        len(design[b.block_id].candidate_ids) == b.runs for b in bundle.plan.blocks
    )
    checks["block_quotas_ok"] = quota_ok

    # 2) 锁定运行未移动/未改因素：锁定位置数量与顺序必须与 CSV 完全一致
    locked_checks = []
    for run in bundle.locked:
        bd = design[run.block]
        n_locked = len(bd.locked_positions)
        prefix_combos = [
            bundle.candidates.combos[cid]
            for cid in bd.candidate_ids[:n_locked]
        ]
        locked_checks.append({
            "run_id": run.run_id,
            "block": run.block,
            "combo_preserved": run.combo in prefix_combos,
            "locked_prefix_intact": bd.locked_positions == list(range(n_locked)),
        })
    checks["locked_runs_ok"] = all(item["combo_preserved"] for item in locked_checks)
    checks["locked_run_details"] = locked_checks

    # 3) 全部 id 合法（禁用组合物理不可达，再做一次显式断言）
    legal_ids = set(range(bundle.candidates.n_candidates))
    checks["all_candidates_legal"] = all(
        cid in legal_ids
        for bd in design.values() for cid in bd.candidate_ids
    )

    # 4) 块内重复
    checks["within_block_replicates_ok"] = {
        bid: block_replicate_satisfied(bd.candidate_ids)
        for bid, bd in design.items()
    }

    # 5) 从 id 序列独立重建矩阵并复核秩（不引用搜索期矩阵）
    x_treat, x_full, blocks = _matrices_of(
        design, bundle.plan, bundle.model, bundle.candidates
    )
    t_rank = int(np.linalg.matrix_rank(
        x_treat, tol=EPS_RANK * max(x_treat.shape)
    ))
    f_rank = int(np.linalg.matrix_rank(
        x_full, tol=EPS_RANK * max(x_full.shape)
    ))
    # 可估计性：处理矩阵自身满秩；对块效应调整后非截距目标列满秩；
    # 全矩阵达到调整后期望秩（截距与块 one-hot 共线属正常过参数化）。
    z = _z_matrix(bundle.plan, bundle.model)
    q = block_projector(z)
    has_blocks = bundle.model.include_block_effect
    target_cols = [
        i for i, name in enumerate(bundle.model.treatment_names)
        if not (has_blocks and name == "Intercept")
    ]
    qx = q @ x_treat[:, target_cols]
    target_rank = int(np.linalg.matrix_rank(qx, tol=EPS_RANK * max(qx.shape)))
    r_z = int(np.linalg.matrix_rank(z, tol=EPS_RANK * max(z.shape))) if has_blocks else 0
    expected_full = r_z + len(target_cols)
    checks["rebuilt_treatment_rank"] = t_rank
    checks["rebuilt_full_rank"] = f_rank
    checks["rebuilt_target_rank_adjusted"] = target_rank
    checks["expected_full_rank_adjusted"] = expected_full
    checks["n_target_columns"] = len(target_cols)
    checks["treatment_estimable"] = (
        t_rank == bundle.model.n_treatment
        and target_rank == len(target_cols)
    )
    checks["full_matrix_full_rank"] = f_rank == expected_full
    checks["intercept_block_alias_expected"] = has_blocks

    # 6) 与搜索期报告矩阵逐元素一致（确认汇报的就是实际运行表）
    checks["matrix_matches_search_object"] = bool(np.allclose(
        x_full, bundle.result.x_full, atol=1e-12
    ))

    # 7) 块内重复占用名额的事实统计
    checks["distinct_combinations"] = {
        bid: len(set(bd.candidate_ids)) for bid, bd in design.items()
    }
    checks["repeat_slots"] = {
        bid: len(bd.candidate_ids) - len(set(bd.candidate_ids))
        for bid, bd in design.items()
    }
    checks["all_passed"] = (
        quota_ok and checks["locked_runs_ok"] and checks["all_candidates_legal"]
        and all(checks["within_block_replicates_ok"].values())
        and checks["treatment_estimable"] and checks["full_matrix_full_rank"]
        and checks["matrix_matches_search_object"]
    )
    return checks


# ---------------------------------------------------------------------- #
# 运行表与随机顺序
# ---------------------------------------------------------------------- #
def run_table_rows(bundle: DesignBundle) -> list[dict[str, Any]]:
    """规范运行表：锁定运行带原 run_id 且位于块内锁定前缀，其余顺序为规范顺序。"""
    rows: list[dict[str, Any]] = []
    design = bundle.result.design
    locked_remaining = {bid: [r for r in bundle.locked if r.block == bid]
                        for bid in bundle.plan.block_ids}
    serial = 1
    for bid in bundle.plan.block_ids:
        bd = design[bid]
        for pos, cid in enumerate(bd.candidate_ids):
            combo = bundle.candidates.combos[cid]
            if pos in bd.locked_positions and locked_remaining[bid]:
                run = locked_remaining[bid].pop(0)
                run_id = run.run_id
                kind = "locked"
            else:
                run_id = f"{bid}-R{pos + 1}"
                kind = "designed"
            row = {
                "run_id": run_id,
                "block": bid,
                "position_in_block": pos + 1,
                "kind": kind,
            }
            for factor, value in zip(bundle.factor_set.factors, combo):
                row[factor.name] = value
            row["candidate_id"] = cid
            row["serial"] = serial
            serial += 1
            rows.append(row)
    return rows


def randomized_order(bundle: DesignBundle) -> list[dict[str, Any]]:
    """块内随机执行顺序。

    只对 *位置* 做置换，组合本身、锁定归属与可估计性不变；每个块用由
    plan.seed 与块序号派生的独立 RNG，结果可复现。另给出
    ``execution_order``（块内 1..k）与回查用的 ``position_in_block``。
    """
    rng = np.random.default_rng(bundle.plan.seed + 7_919)
    rows = run_table_rows(bundle)
    ordered: list[dict[str, Any]] = []
    for j, bid in enumerate(bundle.plan.block_ids):
        block_rows = [row for row in rows if row["block"] == bid]
        perm_rng = np.random.default_rng(rng.integers(0, 2**31 - 1) + j)
        perm = perm_rng.permutation(len(block_rows))
        for exec_pos, idx in enumerate(perm, start=1):
            row = dict(block_rows[int(idx)])
            row["execution_order_in_block"] = exec_pos
            ordered.append(row)
    return ordered


def design_matrix_records(bundle: DesignBundle) -> tuple[list[str], list[dict[str, Any]]]:
    """设计矩阵为显式数值表，列身份与 :class:`Model` 编译结果完全一致。"""
    names = bundle.model.full_column_names(bundle.plan.block_ids)
    mat = bundle.result.x_full
    records = [
        {"row": i + 1, **{name: float(mat[i, j]) for j, name in enumerate(names)}}
        for i in range(mat.shape[0])
    ]
    return names, records


def diagnostics_payload(bundle: DesignBundle) -> dict[str, Any]:
    """汇总准则、诊断、基线对照、搜索审计与枚举结论。"""
    res = bundle.result
    pairs, max_corr = correlation_diagnostics(
        res.x_treat, bundle.model.treatment_names
    )
    base = bundle.baseline_quality
    best = res.quality
    payload: dict[str, Any] = {
        "criterion": res.report.criterion,
        "selected_design": {
            "feasible_full_rank": best.feasible_rank,
            "log_det_conditional_information": best.log_det,
            "det_root_per_parameter": best.det_root,
            "d_efficiency_vs_n": d_efficiency(best, bundle.plan.total_runs),
            "min_eigenvalue_M_target": best.min_eigenvalue,
            "condition_number_M_target": best.condition_number,
            "treatment_rank": best.treatment_rank,
            "target_rank_adjusted_for_blocks": best.target_rank,
            "n_target_columns": best.n_target_columns,
            "full_rank": best.full_rank,
            "full_rank_expected_adjusted": best.full_rank_expected,
            "intercept_aliased_with_block_means": best.intercept_aliased_with_blocks,
            "max_abs_pairwise_correlation": max_corr,
            "top_absolute_correlations": [
                {"pair": pair, "correlation": corr}
                for pair, corr in pairs[:8]
            ],
        },
        "baseline_round_robin": {
            "feasible_full_rank": base.feasible_rank,
            "log_det_conditional_information":
                None if np.isneginf(base.log_det) else base.log_det,
            "det_root_per_parameter": base.det_root,
            "d_efficiency_vs_n": d_efficiency(base, bundle.plan.total_runs),
            "min_eigenvalue_M_target": base.min_eigenvalue,
            "condition_number_M_target":
                None if np.isinf(base.condition_number) else base.condition_number,
            "treatment_rank": base.treatment_rank,
            "target_rank_adjusted_for_blocks": base.target_rank,
            "full_rank": base.full_rank,
            "full_rank_expected_adjusted": base.full_rank_expected,
        },
        "improvement_over_baseline": _improvement(best, base),
        "search_report": asdict(res.report),
        "structural_evidence": asdict(bundle.evidence),
        "enumeration": _enumeration_payload(bundle),
        "independent_verification": bundle.verification,
        "coding_identity": {
            "numeric": "(raw - center) / scale，一次项列名=因素名，"
                       "p 次幂列名=因素名^p",
            "categorical": "treatment 虚拟编码，参考水平全 0，"
                           "非参考水平列名=因素名[水平]",
            "interaction": "两侧基础列逐元素乘积，列名 A*B",
            "block_columns": [
                name for name in bundle.model.full_column_names(bundle.plan.block_ids)
                if name.startswith(BLOCK_PREFIX)
            ],
            "treatment_columns": bundle.model.treatment_names,
        },
    }
    return payload


def _improvement(best: Quality, base: Quality) -> dict[str, Any]:
    if not base.feasible_rank or not best.feasible_rank:
        return {"note": "基线不满秩，无法做准则数值对比，仅对比可行性"}
    delta_logdet = best.log_det - base.log_det
    return {
        "delta_log_det": delta_logdet,
        "det_root_ratio": best.det_root / base.det_root if base.det_root else None,
        "d_efficiency_ratio": (
            (best.det_root / base.det_root) if base.det_root else None
        ),
    }


def _enumeration_payload(bundle: DesignBundle) -> dict[str, Any]:
    if bundle.enumeration is None:
        return {"exhaustive": False, "note": bundle.enumeration_note}
    enum = bundle.enumeration
    match = None
    if enum.best_objective is not None and bundle.result.quality.feasible_rank:
        match = abs(enum.best_objective - bundle.result.quality.log_det) <= 1e-8
    return {
        "exhaustive": True,
        "n_block_multisets": enum.n_block_multisets,
        "n_designs_evaluated": enum.n_designs_evaluated,
        "enumerated_best_objective": enum.best_objective,
        "search_matches_enumerated_optimum": match,
        "cap_reason": enum.cap_reason,
    }
