"""实验计划（块名额、预算、随机种子）与结构性可行性筛查。

可行性判定分三层，证据全部可独立复核：

1. 锁定冲突 —— 锁定运行数量超过块名额，或块内无法同时容纳锁定运行与重复要求；
2. 预算不足 —— 总名额少于全模型列数，秩必然不够（线性代数事实，与搜索无关）；
3. 结构混淆 —— 用 *全部合法候选*（每个候选都可任意次复用的最宽松条件）构造
   处理矩阵，若其秩已低于目标列数，则任何预算内设计都无法估计该效应，
   直接给出秩与零空间混淆证据。

三层都通过时仍可能搜索失败，那属于启发式预算问题，见 :mod:`expdesign.search`，
不会被当作不可行证明。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constraints import Constraints, LockedRun, block_replicate_satisfied
from .errors import ConfigError, InfeasibleError
from .factors import CandidateSet, FactorSet
from .model import Model


@dataclass
class Block:
    block_id: str
    runs: int


@dataclass
class Plan:
    blocks: list[Block]
    locked_path: str | None
    seed: int
    criterion_text: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def block_ids(self) -> list[str]:
        return [b.block_id for b in self.blocks]

    @property
    def total_runs(self) -> int:
        return sum(b.runs for b in self.blocks)

    def quota(self, block_id: str) -> int:
        for block in self.blocks:
            if block.block_id == block_id:
                return block.runs
        raise ConfigError(f"未知块 {block_id!r}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Plan":
        if not isinstance(payload, dict) or "blocks" not in payload:
            raise ConfigError("plan JSON 顶层必须是含 'blocks' 的对象")
        raw_blocks = payload["blocks"]
        if not isinstance(raw_blocks, list) or not raw_blocks:
            raise ConfigError("plan.blocks 必须是非空列表")
        blocks: list[Block] = []
        seen: set[str] = set()
        for item in raw_blocks:
            if not isinstance(item, dict):
                raise ConfigError("块声明必须是对象")
            bid = item.get("id")
            runs = item.get("runs")
            extra = set(item) - {"id", "runs"}
            if extra:
                raise ConfigError(f"块声明含未知键: {sorted(extra)}")
            if not isinstance(bid, str) or not bid:
                raise ConfigError("块 id 必须是非空字符串")
            if bid in seen:
                raise ConfigError(f"块 id 重复: {bid!r}")
            seen.add(bid)
            if not isinstance(runs, int) or isinstance(runs, bool) or runs <= 0:
                raise ConfigError(f"块 {bid!r} 的 runs 必须是正整数")
            blocks.append(Block(bid, runs))
        extra = set(payload) - {"blocks", "locked_runs", "criterion", "seed"}
        if extra:
            raise ConfigError(f"plan JSON 含未知键: {sorted(extra)}")
        seed = payload.get("seed")
        if seed is None:
            raise ConfigError("plan 必须声明 seed 以保证可复现")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ConfigError("seed 必须是整数")
        locked_path = payload.get("locked_runs")
        if locked_path is not None and not isinstance(locked_path, str):
            raise ConfigError("locked_runs 必须是 CSV 路径字符串")
        return cls(
            blocks=blocks,
            locked_path=locked_path,
            seed=seed,
            criterion_text=payload.get("criterion"),
            raw=dict(payload),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "Plan":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def matrix_rank(x: np.ndarray, tol: float | None = None) -> int:
    """SVD 数值秩，默认阈值与 ``numpy.linalg.matrix_rank`` 一致。"""
    if x.size == 0:
        return 0
    return int(np.linalg.matrix_rank(x, tol=tol))


@dataclass
class StructuralEvidence:
    """结构性可行性检查的可复核证据。"""

    n_legal_candidates: int
    n_treatment_columns: int
    n_full_columns: int
    treatment_rank: int
    full_rank: int
    total_runs: int
    expected_full_rank: int = 0
    null_vectors: list[dict[str, Any]] = field(default_factory=list)


def structural_evidence(
    factor_set: FactorSet,
    candidates: CandidateSet,
    model: Model,
    plan: Plan,
) -> StructuralEvidence:
    """以全部合法候选为行（最宽松可复用条件）计算秩与零空间证据。"""
    x_full = model.treatment_matrix(candidates.combos)
    t_rank = matrix_rank(x_full)
    block_ids = plan.block_ids
    if model.include_block_effect:
        # 最宽松条件：让 *每个块都包含全部合法候选*（候选可任意次复用），
        # 避免人为分块导致块内子空间退化而产生假阳性。
        blocks = [bid for bid in block_ids for _ in candidates.combos]
        x_rep = np.vstack([x_full for _ in block_ids])
        x_all = model.full_matrix(x_rep, blocks, block_ids)
    else:
        x_all = x_full
    f_rank = matrix_rank(x_all)
    n_block_cols = len(block_ids) if model.include_block_effect else 0
    # 截距与块 one-hot 共线，调整后期望秩 = 块列秩 + 非截距目标列数。
    n_target = model.n_treatment - (1 if model.include_block_effect else 0)
    expected_full_rank = n_block_cols + n_target
    evidence = StructuralEvidence(
        n_legal_candidates=len(candidates.combos),
        n_treatment_columns=model.n_treatment,
        n_full_columns=x_all.shape[1],
        treatment_rank=t_rank,
        full_rank=f_rank,
        total_runs=plan.total_runs,
    )
    evidence.expected_full_rank = expected_full_rank
    if t_rank < model.n_treatment:
        evidence.null_vectors = _null_space_report(x_full, model.treatment_names)
    return evidence


def assert_structurally_feasible(
    factor_set: FactorSet,
    candidates: CandidateSet,
    model: Model,
    plan: Plan,
    constraints: Constraints,
    locked: list[LockedRun],
) -> StructuralEvidence:
    """按锁定冲突 → 预算 → 结构混淆的顺序给出确定性的不可行结论。"""
    # 1) 锁定运行与块名额/重复要求的计数冲突
    for bid in plan.block_ids:
        quota = plan.quota(bid)
        n_locked = sum(1 for run in locked if run.block == bid)
        if n_locked > quota:
            raise InfeasibleError(
                InfeasibleError.REASON_LOCKED,
                f"块 {bid!r} 锁定运行 {n_locked} 个，超过名额 {quota}",
                {"block": bid, "locked": n_locked, "quota": quota},
            )
        free = quota - n_locked
        locked_ids = [run.candidate_id for run in locked if run.block == bid]
        if constraints.replicate_required and not block_replicate_satisfied(locked_ids):
            if free < 1:
                raise InfeasibleError(
                    InfeasibleError.REASON_LOCKED,
                    f"块 {bid!r} 已被锁定运行占满且内部无重复，无法满足块内重复要求",
                    {"block": bid, "locked": n_locked, "quota": quota},
                )

    # 2) 预算不足：行数少于调整后秩所需下界时可估计性不可能成立。
    #    有块效应时截距与块列共线，最小秩需求 = 块列数 + 非截距目标列数
    #    = n_block + (p_treatment - 1)；无块效应时为 p_treatment。
    n_block_cols = len(plan.block_ids) if model.include_block_effect else 0
    n_target_cols = model.n_treatment - (1 if model.include_block_effect else 0)
    min_runs = n_block_cols + n_target_cols
    if plan.total_runs < min_runs:
        raise InfeasibleError(
            InfeasibleError.REASON_BUDGET,
            f"总运行数 {plan.total_runs} 少于调整后秩所需最小行数 {min_runs}"
            f"（块列 {n_block_cols} + 非截距目标列 {n_target_cols}），"
            f"目标效应不可能可估计（预算不足，非搜索失败）",
            {
                "total_runs": plan.total_runs,
                "minimum_runs_for_rank": min_runs,
                "treatment_columns": model.n_treatment,
                "block_columns": n_block_cols,
                "note": "截距与块 one-hot 共线，不重复计数",
            },
        )

    # 3) 合法候选整体不支持某些模型项：给出秩与零空间证据
    evidence = structural_evidence(factor_set, candidates, model, plan)
    if evidence.treatment_rank < evidence.n_treatment_columns:
        raise InfeasibleError(
            InfeasibleError.REASON_STRUCTURAL_ALIAS,
            f"全部 {evidence.n_legal_candidates} 个合法候选的处理矩阵秩 "
            f"{evidence.treatment_rank} < 列数 {evidence.n_treatment_columns}，"
            f"目标效应在候选空间内本质混淆，任何设计都无法估计",
            {
                "treatment_rank": evidence.treatment_rank,
                "treatment_columns": evidence.n_treatment_columns,
                "full_rank": evidence.full_rank,
                "null_space": evidence.null_vectors,
            },
        )
    if evidence.full_rank < evidence.expected_full_rank:
        raise InfeasibleError(
            InfeasibleError.REASON_STRUCTURAL_ALIAS,
            f"含块效应全矩阵调整后秩 {evidence.full_rank} < 期望秩 "
            f"{evidence.expected_full_rank}（块列秩 + 非截距目标列数），"
            f"处理效应与块效应本质混淆",
            {
                "full_rank": evidence.full_rank,
                "expected_full_rank": evidence.expected_full_rank,
                "full_columns": evidence.n_full_columns,
                "treatment_rank": evidence.treatment_rank,
                "note": "全矩阵物理列数含与截距共线的块列，故期望秩小于列数",
            },
        )

    # 4) 重复约束感知的块内秩上界：每块强制至少一个重复，意味着块内不同
    #    组合数 d_b <= quota_b - 1（锁定已自带重复时除外）。在最宽松的
    #    "可自由挑选 d_b 个不同合法候选"条件下检验块内中心化行空间，
    #    若仍无法张满目标列，则任何满足重复约束的设计都不可估计。
    if constraints.replicate_required:
        rep_evidence = _replicate_feasibility(
            factor_set, candidates, model, plan, locked
        )
        if rep_evidence is not None:
            raise InfeasibleError(
                InfeasibleError.REASON_STRUCTURAL_ALIAS,
                "块内重复要求占用名额后，可用的不同组合数不足以张满目标效应"
                "子空间（重复约束下可证明的结构混淆，非搜索失败）",
                rep_evidence,
            )
    return evidence


def _replicate_feasibility(
    factor_set: FactorSet,
    candidates: CandidateSet,
    model: Model,
    plan: Plan,
    locked: list[LockedRun],
) -> dict | None:
    """返回证据字典表示重复约束下不可行；可行（或无法据此判定）返回 None。"""
    x_all = model.treatment_matrix(candidates.combos)
    block_ids = plan.block_ids
    per_block: list[dict] = []
    blocking = model.include_block_effect
    for bid in block_ids:
        quota = plan.quota(bid)
        locked_combos = [run.candidate_id for run in locked if run.block == bid]
        locked_distinct = len(set(locked_combos))
        # 锁定已自带重复时不额外损失一个不同组合名额。
        already_replicated = block_replicate_satisfied(locked_combos)
        max_distinct = quota - (0 if already_replicated else 1)
        max_distinct = max(max_distinct, locked_distinct)
        per_block.append({"block": bid, "quota": quota,
                          "max_distinct_combinations": max_distinct})

    if not blocking:
        # 无块（单块）：全部不同组合数上界 = 各块容量之和。用 QR 选主列在
        # 全部合法候选的处理行上判断：是否存在该上界数量内的行子集达到满列秩。
        total_distinct_cap = sum(b["max_distinct_combinations"] for b in per_block)
        n_need = model.n_treatment
        if total_distinct_cap < n_need:
            return {"reason": "distinct_combo_cap_below_columns",
                    "distinct_capacity": total_distinct_cap,
                    "required_treatment_columns": n_need,
                    "per_block": per_block}
        # QR 列主序选出达到满秩所需的最少行数，再与容量上界比较。
        x = x_all
        rank_full = matrix_rank(x)
        if rank_full < n_need:
            return {"reason": "treatment_rank_deficient",
                    "treatment_rank": rank_full,
                    "required_treatment_columns": n_need,
                    "per_block": per_block}
        # 选主列 QR：每次挑出使残差范数最大的行，统计达到满秩所需行数。
        min_rows = _minimum_rows_for_full_rank(x, n_need)
        if min_rows is not None and min_rows > total_distinct_cap:
            return {"reason": "no_full_rank_subset_within_capacity",
                    "minimum_distinct_rows_for_rank": min_rows,
                    "distinct_capacity": total_distinct_cap,
                    "required_treatment_columns": n_need,
                    "per_block": per_block}
        return None

    # 分块：目标效应由 *各块中心化行联合* 张成。最宽松上界是让每块都能使用
    # 全部合法候选（候选可复用），对全部候选做块内中心化后堆叠，检验联合
    # 行空间秩；再用全局不同组合容量做计数下界。这两个都是必要条件，
    # 不满足才可证明不可行，避免对受限子集做过度否定。
    target_names = [n for n in model.treatment_names if n != "Intercept"]
    target_idx = [i for i, n in enumerate(model.treatment_names)
                  if n != "Intercept"]
    centered_blocks = []
    for info in per_block:
        xb = x_all.copy()
        xb_c = xb - xb.mean(axis=0, keepdims=True)
        centered_blocks.append(xb_c[:, target_idx])
    x_centered = np.vstack(centered_blocks)
    rank_centered = matrix_rank(x_centered)
    if rank_centered < len(target_idx):
        return {
            "reason": "within_block_centered_rank_deficient",
            "within_block_centered_rank": rank_centered,
            "required_target_columns": len(target_idx),
            "target_columns": target_names,
            "per_block": per_block,
        }
    # 计数下界：联合张满 p 个目标列至少需要 p 个（跨块）线性无关的不同组合。
    total_distinct_cap = sum(b["max_distinct_combinations"] for b in per_block)
    if total_distinct_cap < len(target_idx):
        return {
            "reason": "distinct_combo_cap_below_target_rank",
            "distinct_capacity": total_distinct_cap,
            "required_target_columns": len(target_idx),
            "per_block": per_block,
        }
    return None


def _null_space_report(x: np.ndarray, col_names: list[str],
                       tol: float | None = None) -> list[dict[str, Any]]:
    """提取零空间基向量，并指出每个向量中主导的列，方便报告混淆关系。"""
    u, s, vh = np.linalg.svd(x, full_matrices=True)
    if tol is None:
        tol = max(x.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0)
    null_basis = vh.T[:, s < tol] if s.size else vh.T
    report: list[dict[str, Any]] = []
    for j in range(null_basis.shape[1]):
        vec = null_basis[:, j]
        order = np.argsort(-np.abs(vec))
        dominant = [
            {"column": col_names[k], "coefficient": float(vec[k])}
            for k in order if abs(vec[k]) > tol
        ]
        report.append({"null_vector": vec.tolist(), "dominant_columns": dominant})
    return report


def _minimum_rows_for_full_rank(
    x: np.ndarray, n_need: int, tol: float | None = None
) -> int | None:
    """贪心行选择：逐步挑对当前正交补投影最大的行，返回达满列秩所需行数。

    这是一个上界型构造证据：若贪心在容量内即可达到满秩，说明存在可行子集；
    若贪心所需行数超过容量上限，则容量内不存在满秩子集（该贪心对列空间
    张成是完备的——每步严格扩大张成空间，故结论可靠）。
    """
    n, p = x.shape
    if tol is None:
        tol = max(n, p) * np.finfo(float).eps
    basis: list[np.ndarray] = []  # 已选行张成空间的标准正交基（p 维）
    chosen = 0
    remaining = list(range(n))
    while len(basis) < n_need and remaining:
        # 当前张成空间的正交投影矩阵（作用于 R^p 中的行向量）
        q_basis = np.column_stack(basis) if basis else np.zeros((p, 0))
        proj = q_basis @ q_basis.T if basis else np.zeros((p, p))
        best_idx = None
        best_norm = -1.0
        for i in remaining:
            residual = x[i] - proj @ x[i]
            norm = float(np.linalg.norm(residual))
            if norm > best_norm:
                best_norm, best_idx = norm, i
        if best_norm <= tol:
            break
        # 对当前基正交化选中行并归一化后加入
        v = x[best_idx].astype(float).copy()
        for e in basis:
            v = v - float(e @ v) * e
        v = v / np.linalg.norm(v)
        basis.append(v)
        remaining.remove(best_idx)
        chosen += 1
    return chosen if len(basis) >= n_need else None
