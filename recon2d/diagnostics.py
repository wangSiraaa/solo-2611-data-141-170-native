"""可辨识性与稳定性诊断。

求解器返回 success 只说明找到了一个最优的平滑解，不代表密度被数据
唯一确定。这里从算子本身出发给出：
1. 每格射线支持（射线条数、总路径长度）；
2. 加权前向算子的秩、奇异值与近零空间；
3. 近零空间能量在网格上的分布（哪些格的变化数据看不出来）；
4. 正则强度扫描下的逐格变异（哪些区域随先验摆动、不稳定）；
5. 有限角度探测：行向/列向零和模式在加权算子下的可见比；
6. 射线角度覆盖。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .regularize import support_from_operator


def cell_support(A: sparse.csr_matrix) -> dict:
    """每格的射线支持：射线条数与该格内累计路径长度。"""
    counts = support_from_operator(A)
    path_sum = np.asarray(A.sum(axis=0)).ravel()
    return {"ray_count_per_cell": counts,
            "path_length_per_cell": path_sum,
            "unsupported_cell_count": int(np.sum(counts == 0)),
            "supported_cell_count": int(np.sum(counts > 0))}


def weighted_svd(A_valid: sparse.csr_matrix, sigma_valid: np.ndarray,
                 mass_mu: float) -> dict:
    """对加权前向算子 W*mu*A 做完整 SVD（规模上限 144 列，稠密可接受）。

    必须用 full_matrices=True：紧 SVD 只返回 rank 个右奇异向量，
    会把右零空间整个丢掉，无法做“哪些格变化数据看不出来”的分析。
    完整 Vt 为 n_cols x n_cols，零奇异值对应的行即零空间的正交基。
    """
    B = (mass_mu * A_valid.toarray()) / sigma_valid[:, None]
    u, s, vt = np.linalg.svd(B, full_matrices=True)
    # 完整 SVD 的 vt 为 n_cols 行，但 s 长度只有 min(m,n)；
    # 缺失的右奇异向量对应精确零奇异值，显式补零以保持索引一致
    if s.size < vt.shape[0]:
        s = np.concatenate([s, np.zeros(vt.shape[0] - s.size)])
    s_max = s[0] if s.size else 0.0
    tol_rank = max(B.shape) * np.finfo(float).eps * s_max
    rank = int(np.sum(s > tol_rank))
    # 工程阈值：相对最大奇异值 1e-8 以下视为近零空间
    near_tol = 1e-8 * s_max
    near_null = s < near_tol
    return {"singular_values": s, "vt": vt, "rank": rank,
            "numeric_tol": tol_rank, "near_null_tol": near_tol,
            "near_null_count": int(np.sum(near_null)),
            "n_rows": int(B.shape[0]), "n_cols": int(B.shape[1])}


def nullspace_energy_map(vt: np.ndarray, singular_values: np.ndarray,
                         nx: int, ny: int, relative: bool = True
                         ) -> np.ndarray:
    """近零空间右奇异向量在每个单元上的能量（平方分量之和）。

    完整 SVD 下 Vt 为正交方阵，每列平方和为 1，因此近零空间分量的
    平方和本身就是“该格变化落在数据不可约束方向上的比例”，取值 [0,1]。
    relative=False 时返回原始能量（正交基下数值相同，保留接口）。
    """
    s_max = singular_values[0] if singular_values.size else 0.0
    near = singular_values < (1e-8 * s_max)
    if not np.any(near):
        return np.zeros(ny * nx)
    v_null = vt[near, :]                      # k x n
    energy = np.sum(v_null ** 2, axis=0)
    if relative:
        total = np.sum(vt ** 2, axis=0)
        total[total == 0] = 1.0
        energy = energy / total
    return energy


def stability_sweep(reconstruct_fn, lambdas: np.ndarray) -> dict:
    """在多个正则强度下重复求解，返回逐格变异与各解。

    reconstruct_fn(lambda_) 需返回带 .density 与 .objective 的结果。
    """
    densities, objectives, successes = [], [], []
    for lam in lambdas:
        res = reconstruct_fn(lam)
        densities.append(res.density)
        objectives.append(res.objective)
        successes.append(res.success)
    X = np.asarray(densities)                     # k x n
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    rel = np.divide(std, np.maximum(mean, 1e-12))
    # 解之间的逐对最大相对差：直接刻画“换正则解差多少”
    spread = (X.max(axis=0) - X.min(axis=0))
    rel_spread = np.divide(spread, np.maximum(np.abs(mean), 1e-12))
    return {"lambdas": np.asarray(lambdas),
            "densities": X,
            "mean_density": mean,
            "std_density": std,
            "cv_density": rel,
            "relative_spread": rel_spread,
            "objectives": objectives,
            "successes": successes}


def angle_coverage(endpoints: np.ndarray) -> dict:
    """射线方向角（度，模 180）分布与有限角度判定。"""
    d = endpoints[:, 2:4] - endpoints[:, 0:2]
    angles = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0
    # 直方图，10 度一个扇区
    bins = np.arange(0.0, 181.0, 10.0)
    hist, _ = np.histogram(angles, bins=bins)
    occupied = int(np.sum(hist > 0))
    # 最大无射线角度空隙（循环）
    centers = (bins[:-1] + bins[1:]) / 2.0
    empty_gap = 0
    run = 0
    seq = list(hist > 0) * 2
    for ok in seq:
        run = run + 1 if not ok else 0
        empty_gap = max(empty_gap, run)
    return {"angles_deg_mod180": angles,
            "bin_edges_deg": bins,
            "histogram_10deg": hist,
            "occupied_sectors": occupied,
            "max_empty_sector_run": empty_gap * 10,
            "finite_angle_suspect": occupied <= 3 or empty_gap * 10 >= 60}


def direction_probes(nx: int, ny: int) -> dict:
    """构造两类数据可能无法区分的空间变化模式。

    - row_checker: 每行内零和的交替模式（纯水平射线的精确零空间）；
    - col_checker: 每列内零和的交替模式（纯垂直射线的精确零空间）。
    """
    row_pattern = np.zeros((ny, nx))
    col_pattern = np.zeros((ny, nx))
    for r in range(ny):
        for c in range(nx):
            row_pattern[r, c] = 1.0 if c % 2 == 0 else -1.0
            col_pattern[r, c] = 1.0 if r % 2 == 0 else -1.0
    # 去掉偶数列/行时的非零均值，保证严格零和
    row_pattern -= row_pattern.mean(axis=1, keepdims=True)
    col_pattern -= col_pattern.mean(axis=0, keepdims=True)
    return {"row_checker": row_pattern.ravel(),
            "col_checker": col_pattern.ravel()}


def probe_visibility(A_valid: sparse.csr_matrix, sigma_valid: np.ndarray,
                     mass_mu: float, probes: dict) -> list[dict]:
    """模式 v 的可见比 = ||B v|| / ||v||；0 表示数据完全无法区分该变化。"""
    B = (mass_mu * A_valid.toarray()) / sigma_valid[:, None]
    out = []
    for name, v in probes.items():
        norm_v = np.linalg.norm(v)
        vis = float(np.linalg.norm(B @ v) / norm_v) if norm_v else 0.0
        out.append({"pattern": name, "l2_norm": float(norm_v),
                    "visibility_ratio": vis,
                    "invisible": vis < 1e-8})
    return out
