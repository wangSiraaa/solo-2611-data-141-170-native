"""候选坐标交换（candidate coordinate exchange）有限搜索。

方法（公开、可复核）：

* 设计表示为每块一个候选 id 序列；锁定位置冻结，自由位置只允许换成
  *合法候选集* 内的 id——禁用组合不在候选集中，物理上无法被重新引入。
* 一次 sweep 依次扫描每个自由位置，尝试全部合法候选，保留使全局条件 D 目标
  （秩优先、其次 ``log det(X'QX)``）提升最大的交换。
* 一个起点连续两整轮 sweep 无严格改进即停（局部最优）；多起点各自独立搜索，
  保留全局最优。起点包括 3 个确定性结构化起点和 ``n_random_starts`` 个
  随机起点（种子派生 RNG，完全可复现）。
* 预算耗尽（评估次数/时间）即停。搜索结束仍未找到满秩可行方案时抛
  :class:`SearchFailure`——这只是有限搜索的结论，不是不可行证明。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .baseline import BlockDesign
from .constraints import Constraints, LockedRun, block_replicate_satisfied
from .criteria import EPS_RANK, Quality, block_projector, evaluate
from .errors import SearchFailure
from .factors import CandidateSet
from .model import Model
from .plan import Plan

#: 目标值改进小于该阈值视为数值噪声，不计为严格改进。
IMPROVEMENT_TOL = 1e-10


@dataclass
class SearchReport:
    """搜索过程的完整审计轨迹。"""

    criterion: str
    start_points: list[str]
    stop_reason: str
    n_objective_evaluations: int
    n_sweeps: int
    n_exchange_moves: int
    n_distinct_designs_checked: int
    elapsed_seconds: float
    max_evaluations: int
    time_budget_seconds: float
    found_feasible: bool
    best_objective: float
    per_start: list[dict] = field(default_factory=list)


@dataclass
class SearchResult:
    design: dict[str, BlockDesign]
    quality: Quality
    report: SearchReport
    x_treat: np.ndarray
    x_full: np.ndarray
    flat_blocks: list[str]


class _State:
    """搜索期可变状态，集中管理矩阵重建与评估计数。"""

    def __init__(self, plan: Plan, model: Model, candidates: CandidateSet,
                 constraints: Constraints, locked: list[LockedRun]):
        self.plan = plan
        self.model = model
        self.candidates = candidates
        self.constraints = constraints
        self.locked = locked
        self.n_cand = candidates.n_candidates
        self.block_ids = plan.block_ids
        self.q = block_projector(self._z_matrix())
        self.evaluations = 0
        self.exchange_moves = 0
        self.designs_seen: set[tuple[tuple[str, tuple[int, ...]], ...]] = set()
        self.x_full_names = model.full_column_names(self.block_ids)

    def _z_matrix(self) -> np.ndarray:
        n = self.plan.total_runs
        # 无块效应时不构造任何噪声块列，Q=I，截距正常可估计。
        if not self.model.include_block_effect:
            return np.zeros((n, 0))
        z = np.zeros((n, len(self.block_ids)))
        row = 0
        for bid in self.block_ids:
            quota = self.plan.quota(bid)
            j = self.block_ids.index(bid)
            z[row:row + quota, j] = 1.0
            row += quota
        return z

    # ------------------------------------------------------------------ #
    def build_matrices(self, design: dict[str, BlockDesign]):
        combos: list[tuple] = []
        blocks: list[str] = []
        for bid in self.block_ids:
            for cid in design[bid].candidate_ids:
                combos.append(self.candidates.combos[cid])
                blocks.append(bid)
        x_treat = self.model.treatment_matrix(combos)
        x_full = self.model.full_matrix(x_treat, blocks, self.block_ids)
        return x_treat, x_full, blocks

    def assess(self, design: dict[str, BlockDesign],
               count_evaluation: bool = True) -> Quality:
        x_treat, x_full, _ = self.build_matrices(design)
        if count_evaluation:
            self.evaluations += 1
        return evaluate(
            x_treat, x_full, self.model.treatment_names, self.q,
            full_names=self.x_full_names, rank_tol=EPS_RANK,
        )

    def signature(self, design: dict[str, BlockDesign]) -> tuple:
        return tuple(
            (bid, tuple(design[bid].candidate_ids)) for bid in self.block_ids
        )

    def register_design(self, design: dict[str, BlockDesign]) -> bool:
        """:returns: 该设计是否为首次见到（重复设计不重复计数方向）。"""
        sig = self.signature(design)
        if sig in self.designs_seen:
            return False
        self.designs_seen.add(sig)
        return True

    # ------------------------------------------------------------------ #
    def replicate_ok(self, block_id: str, ids: list[int]) -> bool:
        if not self.constraints.replicate_required:
            return True
        return block_replicate_satisfied(ids)

    def legal_moves(self, block_id: str, position: int,
                    ids: list[int]) -> list[int]:
        """该位置允许换成的候选 id（保持块内重复约束；当前值也在其中）。"""
        moves: list[int] = []
        current = ids[position]
        for cid in range(self.n_cand):
            trial = list(ids)
            trial[position] = cid
            if cid == current or self.replicate_ok(block_id, trial):
                moves.append(cid)
        return moves


def coordinate_exchange(
    plan: Plan,
    model: Model,
    candidates: CandidateSet,
    constraints: Constraints,
    locked: list[LockedRun],
    n_random_starts: int = 8,
    max_evaluations: int = 20000,
    time_budget_seconds: float = 120.0,
    seed: int | None = None,
) -> SearchResult:
    """多起点坐标交换主入口。"""
    rng = np.random.default_rng(
        plan.seed if seed is None else seed
    )
    state = _State(plan, model, candidates, constraints, locked)
    starts = _structured_starts(plan, candidates, constraints, locked)
    start_labels = [f"structured:{name}" for name in
                    ("round_robin", "spread", "locked_anchor")]
    for k in range(n_random_starts):
        starts.append(_random_start(plan, candidates, constraints, locked, rng))
        start_labels.append(f"random:{k + 1}")

    t0 = time.perf_counter()
    best_design: dict[str, BlockDesign] | None = None
    best_quality: Quality | None = None
    best_obj = -np.inf
    n_sweeps_total = 0
    per_start: list[dict] = []
    stop_reason = "all_starts_exhausted"

    for label, design in zip(start_labels, starts):
        if state.evaluations >= max_evaluations:
            stop_reason = "max_evaluations_reached"
            break
        if time.perf_counter() - t0 > time_budget_seconds:
            stop_reason = "time_budget_reached"
            break

        state.register_design(design)
        quality = state.assess(design)
        obj = quality.objective()
        sweeps_here = 0
        moves_here = 0
        stale_rounds = 0
        start_obj = obj

        while stale_rounds < 2:
            if state.evaluations >= max_evaluations:
                break
            if time.perf_counter() - t0 > time_budget_seconds:
                break
            sweeps_here += 1
            improved = False
            for bid in plan.block_ids:
                block = design[bid]
                free_positions = [
                    pos for pos in range(len(block.candidate_ids))
                    if pos not in set(block.locked_positions)
                ]
                for pos in free_positions:
                    if state.evaluations >= max_evaluations:
                        break
                    if time.perf_counter() - t0 > time_budget_seconds:
                        break
                    local_best_id = block.candidate_ids[pos]
                    local_best_obj = obj
                    local_best_quality = quality
                    for cid in state.legal_moves(bid, pos, block.candidate_ids):
                        if cid == local_best_id:
                            continue
                        trial = {b: BlockDesign(
                            block_id=b,
                            candidate_ids=list(d.candidate_ids),
                            locked_positions=list(d.locked_positions),
                        ) for b, d in design.items()}
                        trial[bid].candidate_ids[pos] = cid
                        state.register_design(trial)
                        q_trial = state.assess(trial)
                        o_trial = q_trial.objective()
                        if o_trial > local_best_obj + IMPROVEMENT_TOL:
                            local_best_obj = o_trial
                            local_best_id = cid
                            local_best_quality = q_trial
                    if local_best_id != block.candidate_ids[pos]:
                        block.candidate_ids[pos] = local_best_id
                        obj = local_best_obj
                        quality = local_best_quality
                        improved = True
                        moves_here += 1
                        state.exchange_moves += 1
            n_sweeps_total += 1
            stale_rounds = 0 if improved else stale_rounds + 1

        per_start.append({
            "start": label,
            "starting_objective": None if np.isneginf(start_obj) else start_obj,
            "final_objective": None if np.isneginf(obj) else obj,
            "sweeps": sweeps_here,
            "moves": moves_here,
            "feasible": quality.feasible_rank,
        })
        if _prefer(obj, quality, best_obj, best_quality):
            best_obj = obj
            best_quality = quality
            best_design = {b: BlockDesign(
                block_id=b,
                candidate_ids=list(d.candidate_ids),
                locked_positions=list(d.locked_positions),
            ) for b, d in design.items()}

    elapsed = time.perf_counter() - t0
    found = best_quality is not None and best_quality.feasible_rank

    # 秩救援：所有常规起点都未达到满秩时，用 QR 选主行确定性地构造一个
    # 可行起点，再对它做坐标交换。结构检查已证明可行时，这一步保证启发式
    # 不会把"可行但没搜到"误报成搜索失败；仍失败才是真正的预算耗尽。
    if not found and state.evaluations < max_evaluations:
        rescue = _rank_rescue_start(plan, candidates, constraints, locked, state)
        if rescue is not None:
            state.register_design(rescue)
            r_quality = state.assess(rescue)
            r_obj = r_quality.objective()
            rescue_label = "structured:rank_rescue_qr"
            start_labels.append(rescue_label)
            stale_rounds = 0
            sweeps_here = 0
            moves_here = 0
            while stale_rounds < 2:
                if state.evaluations >= max_evaluations:
                    stop_reason = "max_evaluations_reached"
                    break
                if time.perf_counter() - t0 > time_budget_seconds:
                    stop_reason = "time_budget_reached"
                    break
                sweeps_here += 1
                improved = False
                for bid in plan.block_ids:
                    block = rescue[bid]
                    free_positions = [
                        pos for pos in range(len(block.candidate_ids))
                        if pos not in set(block.locked_positions)
                    ]
                    for pos in free_positions:
                        if state.evaluations >= max_evaluations:
                            break
                        local_best_id = block.candidate_ids[pos]
                        local_best_obj = r_obj
                        local_best_quality = r_quality
                        for cid in state.legal_moves(bid, pos, block.candidate_ids):
                            if cid == local_best_id:
                                continue
                            trial = {b: BlockDesign(
                                block_id=b,
                                candidate_ids=list(d.candidate_ids),
                                locked_positions=list(d.locked_positions),
                            ) for b, d in rescue.items()}
                            trial[bid].candidate_ids[pos] = cid
                            state.register_design(trial)
                            q_trial = state.assess(trial)
                            o_trial = q_trial.objective()
                            if o_trial > local_best_obj + IMPROVEMENT_TOL:
                                local_best_obj = o_trial
                                local_best_id = cid
                                local_best_quality = q_trial
                        if local_best_id != block.candidate_ids[pos]:
                            block.candidate_ids[pos] = local_best_id
                            r_obj = local_best_obj
                            r_quality = local_best_quality
                            improved = True
                            moves_here += 1
                            state.exchange_moves += 1
                n_sweeps_total += 1
                stale_rounds = 0 if improved else stale_rounds + 1
            per_start.append({
                "start": rescue_label,
                "starting_objective":
                    None if np.isneginf(r_obj) else r_obj,
                "final_objective": None if np.isneginf(r_obj) else r_obj,
                "sweeps": sweeps_here, "moves": moves_here,
                "feasible": r_quality.feasible_rank,
            })
            if _prefer(r_obj, r_quality, best_obj, best_quality):
                best_obj, best_quality = r_obj, r_quality
                best_design = {b: BlockDesign(
                    block_id=b,
                    candidate_ids=list(d.candidate_ids),
                    locked_positions=list(d.locked_positions),
                ) for b, d in rescue.items()}
            found = best_quality.feasible_rank

    report = SearchReport(
        criterion=(
            "conditional D: maximize log det(X' Q X), Q = I - Z(Z'Z)^-1 Z' "
            "block-within projector; full-rank feasibility required, ties kept "
            "by smallest condition number"
        ),
        start_points=start_labels,
        stop_reason=stop_reason,
        n_objective_evaluations=state.evaluations,
        n_sweeps=n_sweeps_total,
        n_exchange_moves=state.exchange_moves,
        n_distinct_designs_checked=len(state.designs_seen),
        elapsed_seconds=elapsed,
        max_evaluations=max_evaluations,
        time_budget_seconds=time_budget_seconds,
        found_feasible=found,
        best_objective=None if np.isneginf(best_obj) else float(best_obj),
        per_start=per_start,
    )
    if not found:
        raise SearchFailure(
            f"有限搜索（{state.evaluations} 次评估、{len(start_labels)} 个起点、"
            f"{elapsed:.2f}s）未找到满秩可行设计；这不是不可行证明，"
            f"可提高 max_evaluations/n_random_starts 后重试"
        )
    x_treat, x_full, flat_blocks = state.build_matrices(best_design)
    return SearchResult(best_design, best_quality, report, x_treat, x_full,
                        flat_blocks)


def _prefer(obj: float, quality: Quality, best_obj: float,
            best_quality: Quality | None) -> bool:
    if np.isneginf(obj):
        return False
    if best_quality is None or obj > best_obj + IMPROVEMENT_TOL:
        return True
    if abs(obj - best_obj) <= IMPROVEMENT_TOL:
        # 同分时选条件数更小（更稳定）的方案。
        return quality.condition_number < best_quality.condition_number
    return False


# ---------------------------------------------------------------------- #
# 起点构造
# ---------------------------------------------------------------------- #
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


def _structured_starts(
    plan: Plan, candidates: CandidateSet,
    constraints: Constraints, locked: list[LockedRun],
) -> list[dict[str, BlockDesign]]:
    n_cand = candidates.n_candidates
    designs: list[dict[str, BlockDesign]] = []

    # 起点 1：全局 round-robin（与基线同型，但作为交换起点继续优化）。
    d1 = _seed_locked(plan, locked)
    cursor = 0
    for block in plan.blocks:
        free = plan.quota(block.block_id) - len(d1[block.block_id].candidate_ids)
        for _ in range(free):
            d1[block.block_id].candidate_ids.append(cursor % n_cand)
            cursor += 1
        _ensure_replicate(d1[block.block_id], constraints)
    designs.append(d1)

    # 起点 2：每块内部跨候选均匀铺开（块内 spread），鼓励覆盖不同方向。
    d2 = _seed_locked(plan, locked)
    for block in plan.blocks:
        bd = d2[block.block_id]
        free = plan.quota(block.block_id) - len(bd.candidate_ids)
        offset = (plan.block_ids.index(block.block_id) * 2) % max(n_cand, 1)
        for k in range(free):
            bd.candidate_ids.append((offset + k) % n_cand)
        _ensure_replicate(bd, constraints)
    designs.append(d2)

    # 起点 3：围绕锁定运行的邻域锚点（锁定候选 ±1），保持与锁定方向接近。
    d3 = _seed_locked(plan, locked)
    for block in plan.blocks:
        bd = d3[block.block_id]
        free = plan.quota(block.block_id) - len(bd.candidate_ids)
        anchor = bd.candidate_ids[0] if bd.candidate_ids else 0
        for k in range(free):
            bd.candidate_ids.append((anchor + 1 + k) % n_cand)
        _ensure_replicate(bd, constraints)
    designs.append(d3)
    return designs


def _random_start(
    plan: Plan, candidates: CandidateSet,
    constraints: Constraints, locked: list[LockedRun], rng: np.random.Generator,
) -> dict[str, BlockDesign]:
    design = _seed_locked(plan, locked)
    n_cand = candidates.n_candidates
    for block in plan.blocks:
        bd = design[block.block_id]
        free = plan.quota(block.block_id) - len(bd.candidate_ids)
        for _ in range(free):
            bd.candidate_ids.append(int(rng.integers(0, n_cand)))
        _ensure_replicate(bd, constraints)
    return design


def _ensure_replicate(bd: BlockDesign, constraints: Constraints) -> None:
    """填充完成后兜底块内重复要求（自由名额在结构检查中已保证 >=1）。"""
    if constraints.replicate_required and not block_replicate_satisfied(
            bd.candidate_ids):
        bd.candidate_ids[-1] = bd.candidate_ids[0]


def _rank_rescue_start(
    plan: Plan, candidates: CandidateSet,
    constraints: Constraints, locked: list[LockedRun],
    state: "_State",
) -> dict[str, BlockDesign] | None:
    """全局 QR 选主行确定性构造可行起点。

    先在全部合法候选的处理行（分块时用块内中心化目标行）上用贪心正交基
    选出能张满目标列空间的候选 id 序列，再把这些高跨度候选按块轮流填入
    自由名额，锁定前缀保持不动，最后兜底块内重复。结构检查已证明可行时，
    该构造能达到满秩；构造后仍不满秩则返回 None，保留 SearchFailure 语义。
    """
    design = _seed_locked(plan, locked)
    n_cand = candidates.n_candidates
    target_idx = [
        i for i, n in enumerate(state.model.treatment_names)
        if not (state.model.include_block_effect and n == "Intercept")
    ]
    # 选主行：在用于可估计性判定的行空间上做 Gram-Schmidt 贪心。
    if state.model.include_block_effect:
        rows = []
        for cid in range(n_cand):
            x = state.model.treatment_matrix([candidates.combos[cid]])
            rows.append(x[0, target_idx])
        row_mat = np.vstack(rows)
        # 中心化使块间均值差异不被误计为处理信息。
        row_mat = row_mat - row_mat.mean(axis=0, keepdims=True)
    else:
        row_mat = state.model.treatment_matrix(candidates.combos)
    ordered_ids = _greedy_full_rank_order(row_mat, len(target_idx))

    # 按块轮流分配高跨度候选，填满各块自由名额（可循环复用）。
    cursor = 0
    for bid in plan.block_ids:
        bd = design[bid]
        free = plan.quota(bid) - len(bd.candidate_ids)
        pool = ordered_ids if ordered_ids else list(range(n_cand))
        for _ in range(free):
            bd.candidate_ids.append(pool[cursor % len(pool)])
            cursor += 1
        _ensure_replicate(bd, constraints)
    quality = state.assess(design, count_evaluation=False)
    return design if quality.feasible_rank else None


def _greedy_full_rank_order(x: np.ndarray, n_need: int,
                            tol: float = 1e-12) -> list[int]:
    """Gram-Schmidt 贪心返回使行空间最快张满的候选行 id 顺序。"""
    n, p = x.shape
    basis: list[np.ndarray] = []
    order: list[int] = []
    remaining = list(range(n))
    while len(basis) < min(n_need, p) and remaining:
        q_basis = np.column_stack(basis) if basis else np.zeros((p, 0))
        proj = q_basis @ q_basis.T if basis else np.zeros((p, p))
        best_idx, best_norm = None, -np.inf
        for i in remaining:
            residual = x[i] - proj @ x[i]
            norm = float(np.linalg.norm(residual))
            if norm > best_norm:
                best_norm, best_idx = norm, i
        if best_norm <= tol:
            break
        v = x[best_idx].astype(float).copy()
        for e in basis:
            v = v - float(e @ v) * e
        v = v / np.linalg.norm(v)
        basis.append(v)
        order.append(best_idx)
        remaining.remove(best_idx)
    # 未选满时把其余候选按原顺序补在后面，保证填充池覆盖全部候选。
    order.extend(remaining)
    return order
