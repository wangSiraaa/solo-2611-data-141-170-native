"""非负误差加权最小二乘求解。

目标量（全部在同一最小二乘里求解，单位为无量纲衰减与无量纲差分）：
    minimize_x>=0  J(x) = J_data + lambda * J_smooth + J_prior
      J_data   = sum_r ((atten_r - mu*(A x)_r) / sigma_r)^2
      J_smooth = ||D x||^2                邻格平滑（每对相邻格一行）
      J_prior  = ||w_prior P (x - x_prior)||^2
等价的增广线性残差系统 r(x) = M x - z：
      [ W mu A ]       [ W atten ]
      [ sqrt(l) D ] x = [   0     ]
      [ w_p P   ]       [ w_p P x0]
其中 W = diag(1/sigma)。仅纳入与区域相交的有效射线；完全不相交的
射线（A 全零行）不进入数据项，避免被当成一条穿满区域的观测。

误差加权：每条观测按 1/sigma 加权，J_data 即卡方量；
正则尺度：lambda 作用在“单位邻格差分平方”上，诊断里同时报告
归一化强度 lambda_eff = lambda * n_data / n_edges，便于跨网格比较。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear


@dataclass
class ReconResult:
    density: np.ndarray                 # 扁平解向量（含平滑补出的格）
    predicted: np.ndarray               # 全部射线的预测衰减（不相交射线为 0）
    residual: np.ndarray                # 全部射线的 atten - predicted
    weighted_residual: np.ndarray       # 仅有效射线上的 (atten-pred)/sigma
    success: bool
    status: int
    message: str
    n_iter: int
    objective: dict                     # 目标量分解与正则尺度
    lambda_: float
    prior_weight: float
    prior_value: float


def build_augmented(A_valid: sparse.csr_matrix, atten_valid: np.ndarray,
                    sigma_valid: np.ndarray, mass_mu: float,
                    D: sparse.csr_matrix, P: sparse.csr_matrix,
                    lambda_: float, prior_weight: float,
                    prior_value: float) -> tuple[sparse.csr_matrix, np.ndarray]:
    """组装增广系统 (M, z)。"""
    n_data = A_valid.shape[0]
    inv_sigma = 1.0 / sigma_valid
    W = sparse.diags(inv_sigma)
    blocks_m = [W @ (mass_mu * A_valid)]
    blocks_z = [atten_valid * inv_sigma]

    if lambda_ < 0:
        raise ValueError(f"正则强度必须非负，得到 {lambda_}")
    if lambda_ > 0 and D.shape[0] > 0:
        blocks_m.append(np.sqrt(lambda_) * D)
        blocks_z.append(np.zeros(D.shape[0]))

    if prior_weight > 0 and P.shape[0] > 0:
        blocks_m.append(prior_weight * P)
        blocks_z.append(np.full(P.shape[0], prior_weight * prior_value))

    M = sparse.vstack(blocks_m, format="csr")
    z = np.concatenate(blocks_z)
    return M, z


def reconstruct(A: sparse.csr_matrix, zero_row: np.ndarray,
                attenuation: np.ndarray, sigma: np.ndarray, mass_mu: float,
                D: sparse.csr_matrix, P: sparse.csr_matrix, lambda_: float,
                prior_weight: float = 0.0, prior_value: float = 0.0
                ) -> ReconResult:
    """在非负密度约束下求解一次重建。"""
    valid = ~zero_row
    A_valid = A[valid]
    atten_v = attenuation[valid]
    sigma_v = sigma[valid]
    n_data = A_valid.shape[0]
    n_cells = A.shape[1]

    M, z = build_augmented(A_valid, atten_v, sigma_v, mass_mu, D, P,
                           lambda_, prior_weight, prior_value)
    # trf 支持稀疏矩阵与显式界；lb=0 即非负密度约束
    sol = lsq_linear(M, z, bounds=(0.0, np.inf), method="trf",
                     tol=1e-12, max_iter=2000, verbose=0)
    x = sol.x

    predicted_all = mass_mu * (A @ x)
    residual_all = attenuation - predicted_all
    wres = residual_all[valid] / sigma[valid]

    j_data = float(np.dot(wres, wres))
    dx = D @ x
    j_smooth = float(np.dot(dx, dx))
    if P.shape[0] > 0:
        px = P @ x - prior_value
        j_prior = float(prior_weight**2 * np.dot(px, px))
    else:
        j_prior = 0.0
    n_edges = max(D.shape[0], 1)
    objective = {
        "J_data_chi2": j_data,
        "J_smooth_unit_diff2": j_smooth,
        "J_prior": j_prior,
        "J_total": j_data + lambda_ * j_smooth + j_prior,
        "lambda_raw": lambda_,
        "lambda_eff_per_data_edge": lambda_ * n_data / n_edges,
        "n_data_rays": int(n_data),
        "n_smooth_edges": int(D.shape[0]),
        "reduced_chi2": j_data / n_data if n_data else None,
        "rms_weighted_residual": float(np.sqrt(j_data / n_data))
                                 if n_data else None,
        "rms_residual_attenuation": float(np.sqrt(
            np.mean(residual_all[valid] ** 2))) if n_data else None,
        "data_norm_weighted": float(np.linalg.norm(atten_v / sigma_v)),
        "regularization_to_data_ratio": float(
            (lambda_ * j_smooth + j_prior) / j_data) if j_data > 0 else None,
    }
    return ReconResult(
        density=x, predicted=predicted_all, residual=residual_all,
        weighted_residual=wres, success=bool(sol.success),
        status=int(sol.status), message=str(sol.message),
        n_iter=int(sol.nit), objective=objective, lambda_=lambda_,
        prior_weight=prior_weight, prior_value=prior_value)
