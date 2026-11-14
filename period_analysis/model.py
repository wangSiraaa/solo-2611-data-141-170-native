"""加权谐波周期模型与加权最小二乘拟合。

模型（相对时间原点 t0，固定取观测时间均值，仅用于相位数值稳定）::

    mu(t) = c + sum_{k=1..K} [ a_k cos(2*pi*k*f*(t-t0)) + b_k sin(...) ]

K 取 0、1、2；K=0 即纯常数模型。所有权重 w_i = 1 / sigma_i^2，
拟合为普通（非迭代）加权最小二乘，beta_hat = (X' W X)^-1 X' W y。
所有数值均可由设计矩阵独立复算，本模块不依赖任何 FFT。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_HARMONICS = 2


def design_matrix(t: np.ndarray, freq: float, n_harmonics: int, t0: float) -> np.ndarray:
    """在少数频率上可直接独立构建的设计矩阵。"""
    if n_harmonics < 0 or n_harmonics > MAX_HARMONICS:
        raise ValueError(f"n_harmonics 必须在 0..{MAX_HARMONICS} 之间，收到 {n_harmonics}")
    phase = 2.0 * np.pi * freq * (t - t0)
    cols = [np.ones_like(t)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(k * phase))
        cols.append(np.sin(k * phase))
    return np.column_stack(cols)


@dataclass(frozen=True)
class HarmonicTerm:
    order: int
    amplitude: float          # sqrt(a_k^2 + b_k^2)，亮度单位
    amplitude_err: float      # 由 (X'WX)^-1 线性传播的标准误
    phase_rad: float          # atan2(-b, a)，使项写作 A cos(2 pi k f (t-t0) + phi)
    phase_err_rad: float
    cos_coef: float
    sin_coef: float

    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "amplitude": self.amplitude,
            "amplitude_err": self.amplitude_err,
            "phase_rad": self.phase_rad,
            "phase_err_rad": self.phase_err_rad,
            "cos_coef": self.cos_coef,
            "sin_coef": self.sin_coef,
        }


@dataclass
class FitResult:
    freq: float
    n_harmonics: int
    t0: float
    coefs: np.ndarray            # [c, a1, b1, (a2, b2)]
    cov: np.ndarray              # (X'WX)^-1，单位含亮度^2；误差口径即 sqrt(diag)
    chi2: float
    dof: int
    reduced_chi2: float
    residual: np.ndarray
    predicted: np.ndarray

    @property
    def n_params(self) -> int:
        return 1 + 2 * self.n_harmonics

    @property
    def constant(self) -> float:
        return float(self.coefs[0])

    def harmonic_terms(self) -> list[HarmonicTerm]:
        terms: list[HarmonicTerm] = []
        for k in range(1, self.n_harmonics + 1):
            ia, ib = 2 * k - 1, 2 * k
            a, b = float(self.coefs[ia]), float(self.coefs[ib])
            amp = float(np.hypot(a, b))
            # phi = atan2(-b, a)；梯度传播 Var(phi)
            var_a = float(self.cov[ia, ia])
            var_b = float(self.cov[ib, ib])
            cov_ab = float(self.cov[ia, ib])
            denom = a * a + b * b
            var_phi = (b * b * var_a + a * a * var_b + 2.0 * a * b * cov_ab) / denom**2
            var_amp = (a * a * var_a + b * b * var_b + 2.0 * a * b * cov_ab) / denom if denom > 0 else np.nan
            terms.append(
                HarmonicTerm(
                    order=k,
                    amplitude=amp,
                    amplitude_err=float(np.sqrt(max(var_amp, 0.0))),
                    phase_rad=float(np.arctan2(-b, a)),
                    phase_err_rad=float(np.sqrt(max(var_phi, 0.0))),
                    cos_coef=a,
                    sin_coef=b,
                )
            )
        return terms

    def predict(self, t: np.ndarray) -> np.ndarray:
        return design_matrix(t, self.freq, self.n_harmonics, self.t0) @ self.coefs


def weighted_fit(
    t: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    freq: float,
    n_harmonics: int,
    t0: float | None = None,
) -> FitResult:
    """在给定频率上做加权最小二乘；返回 chi2、参数协方差与残差。"""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if t0 is None:
        t0 = float(np.mean(t))
    if n_harmonics == 0:
        w = 1.0 / sigma**2
        c = np.sum(w * y) / np.sum(w)
        var_c = 1.0 / np.sum(w)
        pred = np.full_like(t, c)
        resid = y - pred
        chi2 = float(np.sum(w * resid**2))
        dof = t.size - 1
        return FitResult(0.0, 0, t0, np.array([c]), np.array([[var_c]]),
                         chi2, dof, chi2 / dof, resid, pred)
    X = design_matrix(t, freq, n_harmonics, t0)
    w = 1.0 / sigma**2
    Xw = X * w[:, None]
    XtWX = Xw.T @ X
    XtWy = Xw.T @ y
    cov = np.linalg.inv(XtWX)
    coefs = cov @ XtWy
    pred = X @ coefs
    resid = y - pred
    chi2 = float(np.sum(w * resid**2))
    dof = t.size - X.shape[1]
    return FitResult(float(freq), n_harmonics, t0, coefs, cov,
                     chi2, dof, chi2 / dof, resid, pred)


def bic(chi2: float, n_points: int, n_params: int) -> float:
    """高斯似然下 BIC = n ln(chi2/n) + k ln n（与 sigma 已知的常数项无关）。"""
    return n_points * np.log(chi2 / n_points) + n_params * np.log(n_points)
