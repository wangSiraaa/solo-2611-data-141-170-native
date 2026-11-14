"""信息准则与混淆/相关诊断。

公开选用的准则（唯一主准则，全程不变）：

* **条件 D 准则（D-optimal for treatment effects conditional on blocks）**
  对带块 one-hot 的模型 ``y = X*beta + Z*gamma + eps``，块效应是噪声参数，
  令 ``Q = I - Z (Z'Z)^-1 Z'`` 为块内中心化投影。处理效应的条件信息矩阵为
  ``M = X' Q X``。

  注意截距列满足 ``Q*1 = 0``：在分块设计中截距（总均值）与块均值本质共线，
  不可能与块效应分别估计——这是分块模型的固有过参数化，不是设计缺陷。
  因此目标参数取 *非截距处理列*（无块效应时则含截距的全部列），记为
  ``X_t``，准则与特征值诊断都在 ``M_t = X_t' Q X_t`` 上进行：

      目标 = log det(M_t)，并报告 det(M_t)^(1/p_t) 与相对 D 效率。

  可估计性条件 = ``M_t`` 正定（等价于 ``rank(Q X_t) = p_t``），
  且全矩阵满足调整后秩 ``rank([X Z]) = rank(Z) + p_t``。
  条件 D 与 Atkinson/Donev/Tobias 的分块设计准则一致，避免把只在块间变化的
  信息误算成处理效应信息。

辅助诊断（不参与优化，仅披露）：秩、最小特征值、条件数、设计列两两相关。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import BLOCK_PREFIX, INTERCEPT_NAME

#: 秩亏时的哨兵目标值，保证任何满秩方案都更优。
NEG_INF = -np.inf
#: 判定数值秩/零特征值的相对阈值。
EPS_RANK = 1e-10


@dataclass
class Quality:
    """一次设计评估的完整质量画像。"""

    feasible_rank: bool
    log_det: float
    det_root: float           # det(M_t)^(1/p_t)
    min_eigenvalue: float
    condition_number: float
    treatment_rank: int            # rank(X)，全部处理列（含截距）
    target_rank: int               # rank(Q X_t)，可估计性实际依据
    full_rank: int                 # rank([X Z])
    full_rank_expected: int        # rank(Z) + p_t，调整后应达到的秩
    n_treatment_columns: int       # p（含截距）
    n_target_columns: int          # p_t
    n_full_columns: int
    intercept_aliased_with_blocks: bool
    max_abs_correlation: float
    correlation_pairs: list[tuple[str, float]]

    def objective(self) -> float:
        """坐标交换实际优化的标量目标：秩优先，其次 log det。"""
        if not self.feasible_rank:
            return NEG_INF
        return self.log_det


def block_projector(z: np.ndarray) -> np.ndarray:
    """``Q = I - Z (Z'Z)^-1 Z'``；无块列时返回单位阵。"""
    n = z.shape[0]
    if z.shape[1] == 0:
        return np.eye(n)
    # 块 one-hot 时 Z'Z 是对角阵，直接逐块做均值投影，数值稳定。
    counts = z.sum(axis=0)
    if np.any(counts == 0):
        # 退化块：改用伪逆，保持通用。
        ztz_pinv = np.linalg.pinv(z.T @ z)
        return np.eye(n) - z @ ztz_pinv @ z.T
    z_scaled = z / counts[np.newaxis, :]
    return np.eye(n) - z_scaled @ z.T


def conditional_information(x_treat: np.ndarray, q: np.ndarray) -> np.ndarray:
    """条件处理信息矩阵 ``M = X' Q X``（对称化消除浮点不对称）。"""
    m = x_treat.T @ q @ x_treat
    return (m + m.T) / 2.0


def target_indices(treatment_names: list[str], has_blocks: bool) -> list[int]:
    """目标参数列下标：有块效应时剔除与块均值共线的截距列。"""
    if not has_blocks:
        return list(range(len(treatment_names)))
    return [i for i, name in enumerate(treatment_names) if name != INTERCEPT_NAME]


def _block_submatrix(x_full: np.ndarray, treatment_names: list[str],
                     full_names: list[str]) -> np.ndarray:
    block_cols = [
        j for j, name in enumerate(full_names)
        if name.startswith(BLOCK_PREFIX)
    ]
    if not block_cols:
        return np.empty((x_full.shape[0], 0))
    return x_full[:, block_cols]


def evaluate(
    x_treat: np.ndarray,
    x_full: np.ndarray,
    treatment_names: list[str],
    q: np.ndarray,
    full_names: list[str] | None = None,
    rank_tol: float = EPS_RANK,
) -> Quality:
    """计算设计的全部质量指标。"""
    n, p = x_treat.shape
    z = _block_submatrix(x_full, treatment_names, full_names or [])
    has_blocks = z.shape[1] > 0
    t_idx = target_indices(treatment_names, has_blocks)
    p_t = len(t_idx)

    m_full = conditional_information(x_treat, q)
    m_target = m_full[np.ix_(t_idx, t_idx)] if p_t else np.empty((0, 0))

    t_rank = int(np.linalg.matrix_rank(x_treat, tol=rank_tol * max(n, p)))
    qx_target = q @ x_treat[:, t_idx]
    target_rank = int(np.linalg.matrix_rank(
        qx_target, tol=rank_tol * max(n, max(p_t, 1))
    ))
    f_rank = int(np.linalg.matrix_rank(
        x_full, tol=rank_tol * max(x_full.shape)
    ))
    r_z = int(np.linalg.matrix_rank(z, tol=rank_tol * max(z.shape))) if has_blocks else 0
    full_expected = r_z + p_t

    eig = np.linalg.eigvalsh(m_target) if p_t else np.array([1.0])
    target_pd = p_t > 0 and bool(np.all(eig > EPS_RANK))
    feasible = (
        t_rank == p
        and target_rank == p_t
        and target_pd
        and f_rank == full_expected
    )

    if feasible:
        sign, logdet = np.linalg.slogdet(m_target)
        log_det = float(logdet) if sign > 0 else NEG_INF
        det_root = float(np.exp(log_det / p_t))
        min_eig = float(eig.min())
        cond = float(eig.max() / eig.min())
    else:
        log_det = NEG_INF
        det_root = 0.0
        min_eig = float(eig.min()) if p_t else 0.0
        cond = float("inf")

    corr_pairs, max_corr = correlation_diagnostics(x_treat, treatment_names)
    return Quality(
        feasible_rank=feasible,
        log_det=log_det,
        det_root=det_root,
        min_eigenvalue=min_eig,
        condition_number=cond,
        treatment_rank=t_rank,
        target_rank=target_rank,
        full_rank=f_rank,
        full_rank_expected=full_expected,
        n_treatment_columns=p,
        n_target_columns=p_t,
        n_full_columns=x_full.shape[1],
        intercept_aliased_with_blocks=has_blocks,
        max_abs_correlation=max_corr,
        correlation_pairs=corr_pairs,
    )


def correlation_diagnostics(
    x_treat: np.ndarray, col_names: list[str]
) -> tuple[list[tuple[str, float]], float]:
    """列两两（皮尔逊）相关，返回降序绝对相关对与最大值；常数列跳过。"""
    n, p = x_treat.shape
    centered = x_treat - x_treat.mean(axis=0, keepdims=True)
    std = np.sqrt((centered ** 2).sum(axis=0))
    pairs: list[tuple[str, float]] = []
    max_corr = 0.0
    for i in range(p):
        for j in range(i + 1, p):
            if std[i] == 0.0 or std[j] == 0.0:
                continue
            corr = float(centered[:, i] @ centered[:, j] / (std[i] * std[j]))
            corr = min(1.0, max(-1.0, corr))
            label = f"{col_names[i]} ~ {col_names[j]}"
            pairs.append((label, corr))
            max_corr = max(max_corr, abs(corr))
    pairs.sort(key=lambda item: -abs(item[1]))
    return pairs, max_corr


def d_efficiency(quality: Quality, n_runs: int) -> float:
    """相对 D 效率：``det(M_t/n)^(1/p_t)``，满秩且 n>0 时可横比不同规模设计。"""
    if not quality.feasible_rank or n_runs <= 0 or quality.n_target_columns == 0:
        return 0.0
    return float(
        np.exp(quality.log_det / quality.n_target_columns) / n_runs
    )
