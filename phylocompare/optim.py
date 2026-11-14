"""枝长重拟合：在公开界内对全部边做多起点约束优化。

公开规则（同时写入报告的 run_parameters，不靠隐式默认值）：

* 边界 ``[1e-5, 1.5]`` 来自 plan.json，对所有候选、所有数据集、
  每次 bootstrap 重采样都相同；Newick 里的枝长只是初值。
* 多起点共 4 组：Newick 初值（先裁到界内）、统一 0.02、统一 0.2、
  统一 0.6。每组都从可行点出发，全部结果纳入报告。
* 优化器为 SciPy L-BFGS-B（解析梯度难以稳定时的标准选择），
  收敛规则：``gtol=1e-8``、``ftol=1e-11``、单次最多 200 次迭代；
  终止且无 NaN/Inf 即视为该起点数值收敛，最终取起点中似然最高者。
* 若某起点报告失败仍如实记录；只有所有起点都失败，该候选才标记
  ``failed``，由上层保留证据，绝不用“最差数”填充后假装完成比较。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .model import SitePatterns, log_likelihood
from .tree import UnrootedTree

DEFAULT_BOUNDS = (1e-5, 1.5)
STARTS = (0.02, 0.2, 0.6)
GTOL = 1e-8
FTOL = 1e-11
MAX_ITER = 200
BOUNDARY_TOL = 1e-6  # 距边界多近即记录为“触界参数”


@dataclass
class StartRecord:
    label: str
    initial: list[float]
    success: bool
    message: str
    iterations: int
    evaluations: int
    final_loglik: float | None


@dataclass
class FitResult:
    ok: bool
    loglik: float
    lengths: list[float]
    boundary_edges: list[dict]
    starts: list[StartRecord] = field(default_factory=list)
    failure_reason: str | None = None

    @property
    def converged(self) -> bool:
        return self.ok


def starting_points(tree: UnrootedTree, bounds: tuple[float, float]) -> list[tuple[str, np.ndarray]]:
    """生成公开的四组起点，保证全部落在界内。"""
    lo, hi = bounds
    clipped = np.array(tree.clipped_initial_lengths(bounds), dtype=np.float64)
    points: list[tuple[str, np.ndarray]] = [("newick_initial_clipped", clipped)]
    for value in STARTS:
        value = min(hi, max(lo, value))
        points.append((f"uniform_{value:g}", np.full(tree.n_edges, value)))
    return points


def fit_branch_lengths(
    tree: UnrootedTree,
    patterns: SitePatterns,
    bounds: tuple[float, float] = DEFAULT_BOUNDS,
) -> FitResult:
    """在固定拓扑下优化全部枝长，返回完整的多起点证据。"""
    lo, hi = bounds
    if not (0.0 < lo < hi):
        raise ValueError(f"非法枝长边界: {bounds}")
    scipy_bounds = [(lo, hi)] * tree.n_edges
    records: list[StartRecord] = []
    best: tuple[float, np.ndarray, StartRecord] | None = None

    for label, x0 in starting_points(tree, bounds):
        def neg_ll(x: np.ndarray) -> float:
            value = log_likelihood(tree, patterns, x)
            return -value if np.isfinite(value) else 1e300

        try:
            opt = minimize(
                neg_ll,
                x0,
                method="L-BFGS-B",
                bounds=scipy_bounds,
                options={"gtol": GTOL, "ftol": FTOL, "maxiter": MAX_ITER},
            )
            final_ll = log_likelihood(tree, patterns, np.asarray(opt.x, dtype=np.float64))
            success = bool(opt.success) and np.isfinite(final_ll)
            record = StartRecord(
                label=label,
                initial=[float(v) for v in x0],
                success=success,
                message=str(opt.message),
                iterations=int(opt.nit),
                evaluations=int(opt.nfev),
                final_loglik=float(final_ll) if np.isfinite(final_ll) else None,
            )
        except Exception as exc:  # 数值异常也要留证据，不能静默吞掉
            record = StartRecord(
                label=label,
                initial=[float(v) for v in x0],
                success=False,
                message=f"{type(exc).__name__}: {exc}",
                iterations=0,
                evaluations=0,
                final_loglik=None,
            )
        records.append(record)

        if record.success and (best is None or record.final_loglik > best[0]):
            best = (record.final_loglik, np.asarray(opt.x, dtype=np.float64), record)

    if best is None:
        return FitResult(
            ok=False,
            loglik=float("-inf"),
            lengths=[],
            boundary_edges=[],
            starts=records,
            failure_reason="全部优化起点均未收敛",
        )

    best_ll, best_x, _ = best
    boundary_edges = []
    for edge, value in zip(tree.edges, best_x):
        if abs(value - lo) <= BOUNDARY_TOL:
            boundary_edges.append({"edge_key": list(edge.key), "side": "lower", "value": float(value)})
        elif abs(value - hi) <= BOUNDARY_TOL:
            boundary_edges.append({"edge_key": list(edge.key), "side": "upper", "value": float(value)})

    return FitResult(
        ok=True,
        loglik=float(best_ll),
        lengths=[float(v) for v in best_x],
        boundary_edges=boundary_edges,
        starts=records,
    )
