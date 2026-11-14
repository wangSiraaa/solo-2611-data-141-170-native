"""微型候选集的完全枚举。

当候选数 ``C`` 与块名额都很小时，块内可行多重集（multiset）数量为
``C(C+1)...(C+k-1)/k!``，可以把 *所有* 满足锁定/名额/重复约束的设计逐一
枚举、逐一评估，得到该微缩问题的真实最优，用于核对坐标交换搜索：

* 枚举只在显式规模阈值内进行，超过阈值返回 ``None``，绝不抽样冒充枚举；
* 枚举与搜索走同一套矩阵/准则代码，差别只在覆盖范围是全集还是邻域；
* 报告枚举总数与最优目标，搜索若达到相同目标即说明在该微缩集上取到最优，
  不会对未探索组合声称更差。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .baseline import BlockDesign
from .constraints import Constraints, LockedRun, block_replicate_satisfied
from .criteria import EPS_RANK, Quality, block_projector, evaluate
from .factors import CandidateSet
from .model import Model
from .plan import Plan


@dataclass
class EnumerationResult:
    exhaustive: bool
    n_block_multisets: int
    n_designs_evaluated: int
    best_objective: float | None
    best_quality: Quality | None
    best_design: dict[str, BlockDesign] | None
    cap_reason: str | None


def count_multisets(n_candidates: int, slots: int) -> int:
    """可重复组合数 C(n_candidates + slots - 1, slots)。"""
    if slots <= 0:
        return 1
    return int(_comb(n_candidates + slots - 1, slots))


def _comb(n: int, k: int) -> int:
    k = min(k, n - k)
    out = 1
    for i in range(1, k + 1):
        out = out * (n - k + i) // i
    return out


def enumerate_optimal(
    plan: Plan,
    model: Model,
    candidates: CandidateSet,
    constraints: Constraints,
    locked: list[LockedRun],
    max_multisets_per_block: int = 5000,
    max_total_designs: int = 20000,
) -> EnumerationResult | None:
    """可行时返回穷举结果；超出规模阈值返回 ``None``（调用方需如实说明）。"""
    n_cand = candidates.n_candidates

    # 先逐块枚举可行多重集（含锁定位置的固定前缀，自由位为非降候选序列）。
    per_block: dict[str, list[tuple[int, ...]]] = {}
    n_multisets = 1
    for block in plan.blocks:
        locked_ids = tuple(
            run.candidate_id for run in locked if run.block == block.block_id
        )
        free = block.runs - len(locked_ids)
        cap = count_multisets(n_cand, free)
        if cap > max_multisets_per_block:
            return None
        feasible: list[tuple[int, ...]] = []
        for tail in itertools.combinations_with_replacement(range(n_cand), free):
            ids = locked_ids + tail
            if not constraints.replicate_required or block_replicate_satisfied(
                    list(ids)):
                feasible.append(ids)
        if not feasible:
            return EnumerationResult(
                exhaustive=True, n_block_multisets=0, n_designs_evaluated=0,
                best_objective=None, best_quality=None, best_design=None,
                cap_reason=None,
            )
        per_block[block.block_id] = feasible
        n_multisets *= cap

    total = 1
    for ids_list in per_block.values():
        total *= len(ids_list)
    if total > max_total_designs:
        return None

    z = _z_matrix(plan)
    q = block_projector(z)
    best_obj = -np.inf
    best_quality: Quality | None = None
    best_design: dict[str, BlockDesign] | None = None
    evaluated = 0
    block_ids = plan.block_ids
    for combo_of_lists in itertools.product(*(per_block[bid] for bid in block_ids)):
        design: dict[str, BlockDesign] = {}
        combos: list[tuple] = []
        blocks: list[str] = []
        for bid, ids in zip(block_ids, combo_of_lists):
            # 锁定运行构成块内固定前缀，锁定位置即前 n_locked 个。
            n_locked = sum(1 for run in locked if run.block == bid)
            design[bid] = BlockDesign(
                block_id=bid,
                candidate_ids=list(ids),
                locked_positions=list(range(n_locked)),
            )
            for cid in ids:
                combos.append(candidates.combos[cid])
                blocks.append(bid)
        x_treat = model.treatment_matrix(combos)
        x_full = model.full_matrix(x_treat, blocks, block_ids)
        quality = evaluate(
            x_treat, x_full, model.treatment_names, q,
            full_names=model.full_column_names(block_ids), rank_tol=EPS_RANK,
        )
        evaluated += 1
        obj = quality.objective()
        if obj > best_obj + 1e-12 or (
            abs(obj - best_obj) <= 1e-12 and best_quality is not None
            and quality.condition_number < best_quality.condition_number
        ):
            best_obj = obj
            best_quality = quality
            best_design = design

    return EnumerationResult(
        exhaustive=True,
        n_block_multisets=n_multisets,
        n_designs_evaluated=evaluated,
        best_objective=None if np.isneginf(best_obj) else float(best_obj),
        best_quality=best_quality,
        best_design=best_design,
        cap_reason=None,
    )


def _z_matrix(plan: Plan) -> np.ndarray:
    n = plan.total_runs
    z = np.zeros((n, len(plan.block_ids)))
    row = 0
    for j, bid in enumerate(plan.block_ids):
        quota = plan.quota(bid)
        z[row:row + quota, j] = 1.0
        row += quota
    return z
