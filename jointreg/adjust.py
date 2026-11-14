"""联合稳健平差（bundle adjustment）。

参数化（保证 R 恒属于 SO(3)，反射矩阵在参数空间中根本不存在）：

* 非锚站：旋转向量 r（3）+ 平移 t（3），R = SO3_exp(r)；
* 锚站：固定 r=0, t=0（世界基准，同时消除整体 6 自由度基准漂移）；
* 标靶：世界坐标 X（每个被至少一条拟合观测看到的标靶 3 个参数）。

观测方程::

    e_i = (R_s @ local_i + t_s) - X_t          （米，回到原观测坐标系）

稳健策略：IRLS + Huber 损失，f_scale=2 sigma。错配离群点不被删除，
而是按规则降权并在输出中给出依据；另跑一遍普通最小二乘作为对照，
让"降权 vs 不处理"的差别可审计。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd

from .geometry import SO3_exp, SO3_log

HUBER_K_SIGMA = 2.0          # Huber 拐点：2 倍测量标准差
IRLS_MAX_ITER = 50
IRLS_TOL_COST = 1e-10
IRLS_TOL_DX = 1e-11
# 可识别性 SVD：相对最大奇异值小于该值的方向视为不可识别/弱约束
NULL_RATIO = 1e-8
WEAK_RATIO = 1e-3


def _skew_symmetric_cross(v: np.ndarray) -> np.ndarray:
    """标准叉积矩阵 [v]_×，满足 [v]_× u = v × u。"""
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )


def so3_left_jacobian(r: np.ndarray) -> np.ndarray:
    """SO(3) 左雅可比 Jr(r)，由 BCH 恒等式
    exp(r+δ) = exp(Jr(r) δ) exp(r) 定义（已用有限差分核对符号），
    闭式（θ=|r|, u=r/θ, [u]_× 为标准叉积矩阵）::

        Jr = (sin θ / θ) I + (1 - sin θ/θ) u uᵀ + ((1-cos θ)/θ) [u]_×
    """
    angle = float(np.linalg.norm(r))
    if angle < 1e-10:
        return np.eye(3) - 0.5 * _skew_symmetric_cross(r)
    u = r / angle
    K = _skew_symmetric_cross(u)
    a = np.sin(angle) / angle
    b = (1.0 - np.cos(angle)) / angle
    return a * np.eye(3) + (1.0 - a) * np.outer(u, u) + b * K


def rotation_action_jacobian(r: np.ndarray, local: np.ndarray) -> np.ndarray:
    """d[SO3_exp(r) @ local] / d r（完整 3x3，任意 r 成立，有限差分核对过）。

    由 BCH：exp(r+δ) l = exp(Jr(r)δ) R l ≈ Rl + [Jr(r)δ]_× (Rl)
                                       = Rl - [Rl]_× Jr(r) δ，故::

        d(R l)/d r = -[R l]_× Jr(r)
    """
    R = SO3_exp(r)
    return -_skew_symmetric_cross(R @ local) @ so3_left_jacobian(r)


@dataclass
class ParamLayout:
    stations: list[str]
    station_r0: dict[str, int]      # 旋转向量起始下标，锚站为 -1
    station_t0: dict[str, int]
    targets: list[str]
    target_x0: dict[str, int]
    n_params: int


class JointModel:
    """固定锚站的联合模型。断开分量由调用方分别建模（各自临时锚）。"""

    def __init__(
        self,
        fit_obs,                       # list[Observation]
        anchor_station: str,
        R0: dict,
        t0: dict,
        X0: dict,
    ):
        self.obs = fit_obs
        self.anchor = anchor_station
        stations = sorted({o.station for o in fit_obs})
        targets = sorted({o.target for o in fit_obs})

        idx = 0
        r0, tt0 = {}, {}
        for s in stations:
            if s == anchor_station:
                r0[s] = tt0[s] = -1
            else:
                r0[s] = idx
                idx += 3
                tt0[s] = idx
                idx += 3
        x0_map = {}
        for tg in targets:
            x0_map[tg] = idx
            idx += 3
        self.layout = ParamLayout(stations, r0, tt0, targets, x0_map, idx)

        self.n_obs = 3 * len(fit_obs)
        self.sigma = np.array([o.sigma for o in fit_obs])
        self.local = np.array([o.local for o in fit_obs])
        self.station_of = [o.station for o in fit_obs]
        self.target_of = [o.target for o in fit_obs]

        self.x = np.zeros(idx)
        for s in stations:
            if s != anchor_station:
                self.x[r0[s]:r0[s] + 3] = SO3_log(R0[s])
                self.x[tt0[s]:tt0[s] + 3] = t0[s]
        for tg in targets:
            self.x[x0_map[tg]:x0_map[tg] + 3] = X0[tg]

    # ---------- 前向模型 ----------
    def station_Rt(self, s: str, x: np.ndarray | None = None):
        x = self.x if x is None else x
        if s == self.anchor:
            return np.eye(3), np.zeros(3)
        r = x[self.layout.station_r0[s]:self.layout.station_r0[s] + 3]
        t = x[self.layout.station_t0[s]:self.layout.station_t0[s] + 3]
        return SO3_exp(r), t

    def target_X(self, tg: str, x: np.ndarray | None = None) -> np.ndarray:
        x = self.x if x is None else x
        i = self.layout.target_x0[tg]
        return x[i:i + 3]

    def residuals(self, x: np.ndarray | None = None) -> np.ndarray:
        x = self.x if x is None else x
        out = np.empty(len(self.obs) * 3)
        for k, o in enumerate(self.obs):
            R, t = self.station_Rt(o.station, x)
            pred = R @ o.local + t
            X = self.target_X(o.target, x)
            out[3 * k:3 * k + 3] = pred - X
        return out

    def jacobian(self, x: np.ndarray | None = None) -> np.ndarray:
        """解析雅可比（残差对全部自由参数）。

        e = R(r) l + t - X:

        * d e / d t = I3
        * d e / d r = -[R l]_x （链式法则 d(Rv)/dr = -[Rv]_x）
        * d e / d X = -I3
        """
        x = self.x if x is None else x
        J = np.zeros((self.n_obs, self.layout.n_params))
        for k, o in enumerate(self.obs):
            row = 3 * k
            R, _ = self.station_Rt(o.station, x)
            if o.station != self.anchor:
                # 完整旋转雅可比（小角时退化为 -[R l]_×）
                r_vec = x[self.layout.station_r0[o.station]:
                          self.layout.station_r0[o.station] + 3]
                Jr = rotation_action_jacobian(r_vec, o.local)
                i_r = self.layout.station_r0[o.station]
                i_t = self.layout.station_t0[o.station]
                J[row:row + 3, i_r:i_r + 3] = Jr
                J[row:row + 3, i_t:i_t + 3] = np.eye(3)
            i_x = self.layout.target_x0[o.target]
            J[row:row + 3, i_x:i_x + 3] = -np.eye(3)
        return J

    # ---------- 求解 ----------
    def solve_plain(self) -> dict:
        """普通加权最小二乘（无稳健损失），作为对照解。"""
        return self._solve_gn(huber=False, label="plain_weighted_lsq")

    def solve_robust(self) -> dict:
        """Huber IRLS 稳健解。"""
        return self._solve_gn(huber=True, label="huber_irls")

    def _solve_gn(self, huber: bool, label: str) -> dict:
        x = self.x.copy()
        # 每条观测的 3 个分量共享一个标量 sigma（题给数据如此），
        # 归一化残差按分量计算；Huber 权重按分量取，观测级权重取三者均值。
        sigma_rep = np.repeat(self.sigma, 3)
        history = []
        for it in range(IRLS_MAX_ITER):
            e = self.residuals(x)
            u = e / sigma_rep
            if huber:
                # Huber: rho'(u)/u -> w；|u|<=k 权重 1，否则 k/|u|
                w_comp = np.where(
                    np.abs(u) <= HUBER_K_SIGMA,
                    1.0,
                    HUBER_K_SIGMA / np.maximum(np.abs(u), 1e-12),
                )
            else:
                w_comp = np.ones_like(u)
            cost = float(np.sum(w_comp * u**2))
            J = self.jacobian(x)
            W_sqrt = np.sqrt(w_comp) / sigma_rep
            A = (J * W_sqrt[:, None]).T @ (J * W_sqrt[:, None])
            g = (J * W_sqrt[:, None]).T @ (e * W_sqrt)
            # 秩亏时（共线/断开）法方程有零空间：pinv 给出的最小范数解在
            # 不同点序下仍可能沿零空间漂移。这里显式做零空间投影——
            # 增量只允许落在可识别子空间，零空间方向严格保持初始化代表值，
            # 从而输出对记录顺序确定，且不掩盖任何不可识别自由度。
            sA = np.linalg.svd(A, compute_uv=False)
            if sA[-1] < NULL_RATIO * sA[0]:
                evals, evecs = np.linalg.eigh(A)
                keep = evals > NULL_RATIO * evals[-1]
                P_ident = evecs[:, keep] @ evecs[:, keep].T
                dx = P_ident @ (-np.linalg.pinv(A, rcond=NULL_RATIO) @ g)
            else:
                dx = np.linalg.solve(A, -g)
            x = x + dx
            history.append({"iter": it, "cost": cost, "dx_norm": float(np.linalg.norm(dx))})
            if np.linalg.norm(dx) < IRLS_TOL_DX and it > 0:
                break
            if it > 0 and abs(history[-2]["cost"] - cost) < IRLS_TOL_COST * max(1.0, cost):
                break
        e = self.residuals(x)
        u = e / sigma_rep
        if huber:
            w_comp = np.where(
                np.abs(u) <= HUBER_K_SIGMA,
                1.0,
                HUBER_K_SIGMA / np.maximum(np.abs(u), 1e-12),
            )
        else:
            w_comp = np.ones_like(u)
        return {
            "label": label,
            "x": x,
            "iterations": len(history),
            "history": history,
            "component_weights": w_comp,          # 每个残差分量的 IRLS 权重
            "normalized_residuals": u,
            "residuals_m": e,
        }

    # ---------- 可识别性：加权雅可比 SVD ----------
    def identifiability(self, x: np.ndarray, length_scale: float) -> dict:
        """在解点对归一化雅可比做 SVD，找零空间/弱方向并映射回物理自由度。

        平移与标靶参数以 length_scale 无量纲化，使所有奇异值可直接比较。
        收敛只说明优化器停下，不说明解唯一——唯一性由这里的零空间判定。
        """
        J = self.jacobian(x)
        sigma_rep = np.repeat(self.sigma, 3)
        Jn = J / sigma_rep[:, None]
        scale = np.ones(self.layout.n_params)
        for s in self.layout.stations:
            if s != self.anchor:
                scale[self.layout.station_t0[s]:self.layout.station_t0[s] + 3] = (
                    1.0 / length_scale
                )
        for tg in self.layout.targets:
            i = self.layout.target_x0[tg]
            scale[i:i + 3] = 1.0 / length_scale
        Js = Jn * scale[None, :]
        U, S, Vt = svd(Js, full_matrices=True)
        smax = S[0] if len(S) and S[0] > 0 else 1.0
        null_dirs, weak_dirs = [], []
        for i, sval in enumerate(S):
            ratio = sval / smax
            if ratio < NULL_RATIO:
                null_dirs.append((i, float(sval), Vt[i]))
            elif ratio < WEAK_RATIO:
                weak_dirs.append((i, float(sval), Vt[i]))

        def describe(vec: np.ndarray) -> list[dict]:
            items = []
            for s in self.layout.stations:
                if s == self.anchor:
                    continue
                ir = self.layout.station_r0[s]
                it = self.layout.station_t0[s]
                vr, vt = vec[ir:ir + 3], vec[it:it + 3]
                if np.linalg.norm(vr) > 0.1:
                    items.append({"kind": "station_rotation", "station": s,
                                  "vector": vr.tolist(), "norm": float(np.linalg.norm(vr))})
                if np.linalg.norm(vt) > 0.1:
                    items.append({"kind": "station_translation", "station": s,
                                  "vector": vt.tolist(), "norm": float(np.linalg.norm(vt))})
            for tg in self.layout.targets:
                i = self.layout.target_x0[tg]
                v = vec[i:i + 3]
                if np.linalg.norm(v) > 0.1:
                    items.append({"kind": "target_position", "target": tg,
                                  "vector": v.tolist(), "norm": float(np.linalg.norm(v))})
            return items

        return {
            "singular_values": [float(s) for s in S],
            "n_params": self.layout.n_params,
            "n_observations": self.n_obs,
            "null_directions": [
                {"singular_value": sv, "ratio": sv / smax, "parameter_mix": describe(v)}
                for _, sv, v in null_dirs
            ],
            "weak_directions": [
                {"singular_value": sv, "ratio": sv / smax, "parameter_mix": describe(v)}
                for _, sv, v in weak_dirs
            ],
            "length_scale_m": float(length_scale),
        }
