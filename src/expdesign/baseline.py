"""朴素可行基线设计。

基线刻意不做坐标交换，只用一个简单、确定性的贪心规则构造 *可行* 设计，
作为与启发式搜索对照的真实参照（而不是只比行列式）：

1. 锁定运行按块预置并冻结；
2. 逐块、逐自由名额，从合法候选中按 round-robin 顺序轮流挑选，
   每步选使全局条件信息 ``M_t = X_t' Q X_t`` 最小特征值最大的候选，
   同分（含初始全零矩阵）按候选 id 顺序取——即"先保证可估计、再谈效率"；
3. 最后一个自由名额在需要时复制块内已有候选，满足块内重复要求
   （结构性检查已保证此时至少有一个自由名额）。

它只用合法候选、保持锁定运行、满足块名额与重复约束，且一旦候选空间结构上
支持目标效应，贪心即可得到满秩可行设计；信息效率通常明显低于坐标交换结果。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constraints import Constraints, LockedRun, block_replicate_satisfied
from .criteria import (
    EPS_RANK, block_projector, conditional_information, target_indices,
)
from .factors import CandidateSet
from .model import Model
from .plan import Plan


@dataclass
class BlockDesign:
    """一个块的候选 id 序列；``locked_positions`` 中的位置不可移动/交换。"""

    block_id: str
    candidate_ids: list[int]
    locked_positions: list[int]


def build_baseline(
    plan: Plan,
    candidates: CandidateSet,
    constraints: Constraints,
    locked: list[LockedRun],
    model: Model | None = None,
) -> dict[str, BlockDesign]:
    """确定性贪心可行基线。"""
    design = _seed_locked(plan, locked)
    if model is None:  # pragma: no cover - 防御性分支
        return _round_robin_fallback(plan, candidates, constraints, locked, design)

    block_ids = plan.block_ids
    t_idx = target_indices(
        model.treatment_names, model.include_block_effect
    )

    # 按块轮流填充自由名额，每步全局贪心。
    free_counters = {
        bid: plan.quota(bid) - len(design[bid].candidate_ids) for bid in block_ids
    }
    cursor = 0
    while any(v > 0 for v in free_counters.values()):
        for bid in block_ids:
            if free_counters[bid] <= 0:
                continue
            chosen = _greedy_pick(
                design, bid, candidates, model, block_ids, t_idx, cursor
            )
            design[bid].candidate_ids.append(chosen)
            cursor += 1
            free_counters[bid] -= 1

    for bid in block_ids:
        _ensure_replicate(design[bid], constraints)
    return design


def _greedy_pick(
    design: dict[str, BlockDesign], block_id: str,
    candidates: CandidateSet, model: Model, block_ids: list[str],
    target_idx: list[int], cursor: int,
) -> int:
    """两级贪心：先最大化块调整后条件秩，满秩后再最大化最小特征值。

    同分时按 round-robin 顺序取靠前候选，保证确定性与可解释性。
    """
    n_cand = candidates.n_candidates
    order = [(cursor + k) % n_cand for k in range(n_cand)]
    best_id = order[0]
    best_rank = -1
    best_min_eig = -np.inf
    for cid in order:
        trial_ids = {b: list(d.candidate_ids) for b, d in design.items()}
        trial_ids[block_id].append(cid)
        combos: list[tuple] = []
        blocks: list[str] = []
        for bid in block_ids:
            for cand_id in trial_ids[bid]:
                combos.append(candidates.combos[cand_id])
                blocks.append(bid)
        x = model.treatment_matrix(combos)
        # 无块效应时 Q=I；否则 Q 只依赖当前行的块归属，逐次按实际行数构造。
        if model.include_block_effect:
            z = np.zeros((len(blocks), len(block_ids)))
            for i, bid in enumerate(blocks):
                z[i, block_ids.index(bid)] = 1.0
        else:
            z = np.zeros((len(blocks), 0))
        q = block_projector(z)
        qx = q @ x[:, target_idx]
        rank = int(np.linalg.matrix_rank(qx, tol=EPS_RANK * max(qx.shape)))
        m = conditional_information(x, q)
        m_t = m[np.ix_(target_idx, target_idx)]
        min_eig = float(np.linalg.eigvalsh(m_t).min())
        if rank > best_rank:
            best_rank, best_min_eig, best_id = rank, min_eig, cid
        elif rank == best_rank and min_eig > best_min_eig + EPS_RANK:
            best_min_eig, best_id = min_eig, cid
    return best_id


def _seed_locked(plan: Plan, locked: list[LockedRun]) -> dict[str, BlockDesign]:
    design: dict[str, BlockDesign] = {}
    for block in plan.blocks:
        ids = [run.candidate_id for run in locked if run.block == block.block_id]
        design[block.block_id] = BlockDesign(
            block_id=block.block_id,
            candidate_ids=ids,
            locked_positions=list(range(len(ids))),
        )
    return design


def _ensure_replicate(bd: BlockDesign, constraints: Constraints) -> None:
    if constraints.replicate_required and not block_replicate_satisfied(
            bd.candidate_ids):
        bd.candidate_ids[-1] = bd.candidate_ids[0]


def _round_robin_fallback(  # pragma: no cover - 防御性分支
    plan: Plan, candidates: CandidateSet,
    constraints: Constraints, locked: list[LockedRun],
    design: dict[str, BlockDesign],
) -> dict[str, BlockDesign]:
    n_cand = candidates.n_candidates
    cursor = 0
    for block in plan.blocks:
        free = plan.quota(block.block_id) - len(design[block.block_id].candidate_ids)
        for _ in range(free):
            design[block.block_id].candidate_ids.append(cursor % n_cand)
            cursor += 1
        _ensure_replicate(design[block.block_id], constraints)
    return design
