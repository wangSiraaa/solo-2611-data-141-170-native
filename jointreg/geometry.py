"""几何核心：SO(3) 指数/对数映射、加权 Kabsch、局部几何支撑诊断。

所有点均为 numpy 形状 (n, 3) 的数组，单位米。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 奇异值比值阈值：小于该值视为精确缺秩；1e-3 以下视为近退化（弱约束）。
EXACT_RANK_RATIO = 1e-9
WEAK_RANK_RATIO = 1e-3


def hat(w: np.ndarray) -> np.ndarray:
    """标准反对称（叉积）矩阵 [w]_×，满足 hat(w) @ u == cross(w, u)。"""
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
    )


def SO3_exp(rotvec: np.ndarray) -> np.ndarray:
    """旋转向量（李代数 so(3)）到 SO(3) 矩阵，Rodrigues 公式。

    输出恒为正交矩阵；det 的符号由公式本身保证为 +1（不含反射分量）。
    """
    w = np.asarray(rotvec, dtype=float).reshape(3)
    angle = float(np.linalg.norm(w))
    if angle < 1e-12:
        # 小角展开，避免角度接近 0 时除以角度
        K = hat(w)
        return np.eye(3) + K + 0.5 * K @ K
    axis = w / angle
    K = hat(axis)
    # Rodrigues: R = I + sin(θ) K + (1-cos(θ)) K²，K=[u]_×
    return (
        np.eye(3)
        + np.sin(angle) * K
        + (1.0 - np.cos(angle)) * (K @ K)
    )


def SO3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) 矩阵到旋转向量，与 :func:`SO3_exp` 互逆。"""
    R = np.asarray(R, dtype=float)
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-12:
        # [R32-R23, R13-R31, R21-R12] / 2 的一阶近似
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        )
    if abs(angle - np.pi) < 1e-9:
        # 接近 pi 时改用对称形式，数值更稳
        M = (R - np.eye(3)) / 2.0
        # 取对角最大的列恢复转轴
        i = int(np.argmax(np.diag(M) + 1.0))
        u = np.zeros(3)
        u[i] = np.sqrt(max(M[i, i] + 1.0, 0.0))
        for j in range(3):
            if j != i:
                u[j] = M[i, j] / u[i] if u[i] > 0 else 0.0
        return angle * u / np.linalg.norm(u)
    return (
        angle
        / (2.0 * np.sin(angle))
        * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        )
    )


@dataclass
class KabschResult:
    R: np.ndarray
    t: np.ndarray
    rms_m: float
    reflection_corrected: bool          # 最优无约束解为反射、已强制翻回 SO(3)
    singular_values: np.ndarray         # 加权协方差 H 的奇异值
    rotational_rank: int                # H 的有效秩（1=共线,2=共面,3=完整）


def weighted_kabsch(
    local: np.ndarray, world: np.ndarray, sigma: np.ndarray | None = None
) -> KabschResult:
    """加权 Kabsch：求 world = R @ local + t 的最小二乘 R,t。

    权重取 w_i = 1/sigma_i^2（各点标量不确定度）。
    反射防护：若无约束最优解 det<0，则翻转最小奇异方向后重做，保证 det(R)=+1，
    并置 reflection_corrected=True——反射矩阵不得以小残差冒充旋转。
    点集缺秩时仍返回一个 R，但 rotational_rank 会如实标注，调用方必须检查。
    """
    P = np.asarray(local, dtype=float).reshape(-1, 3)
    Q = np.asarray(world, dtype=float).reshape(-1, 3)
    if P.shape != Q.shape or P.shape[0] < 1:
        raise ValueError("Kabsch 需要相同数量、至少 1 个对应点")
    if sigma is None:
        w = np.ones(P.shape[0])
    else:
        s = np.asarray(sigma, dtype=float).reshape(-1)
        if np.any(s <= 0):
            raise ValueError("sigma 必须为正数")
        w = 1.0 / s**2
    w = w / w.sum()

    p_bar = w @ P
    q_bar = w @ Q
    Pc = P - p_bar
    H = (Pc * w[:, None]).T @ (Q - q_bar)
    U, S, Vt = np.linalg.svd(H)

    rank = int(np.sum(S > EXACT_RANK_RATIO * (S[0] if S[0] > 0 else 1.0)))
    U, Vt = _canonicalize_svd_bases(U, Vt, rank)

    d = np.sign(np.linalg.det(Vt.T @ U.T))
    reflection_corrected = d < 0
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    t = q_bar - R @ p_bar
    # 残差回到原观测：e = R @ local + t - world（米）
    resid = (P @ R.T + t) - Q
    rms = float(np.sqrt(np.average(np.sum(resid**2, axis=1), weights=w * len(w))))
    return KabschResult(R, t, rms, reflection_corrected, S, rank)


def _canonicalize_svd_bases(U: np.ndarray, Vt: np.ndarray, rank: int):
    """缺秩时把 SVD 的零空间基确定化，使代表姿态与点序/数值微扰无关。

    满秩（rank=3）时 SVD 基由数据唯一确定（至多整体符号），无需处理；
    rank=1/2 时零空间内的旋转/符号是任意的，这里用主轴叉积构造
    确定性右手基：可识别的主轴方向保持不变，缺秩轴按确定性规则补齐。
    注意：这只固定"代表姿态"，并不恢复任何缺失的物理自由度——
    可识别性结论仍由 support_report / 联合雅可比 SVD 给出。
    """
    if rank == 3:
        return U, Vt
    if rank == 2:
        u1, u2 = U[:, 0], U[:, 1]
        v1, v2 = Vt[0], Vt[1]
        u3 = np.cross(u1, u2)
        v3 = np.cross(v1, v2)
        Uc = np.column_stack([u1, u2, u3])
        Vc = np.column_stack([v1, v2, v3])
        return Uc, Vc.T
    if rank == 1:
        # 选与主轴最大分量正交的两个确定性世界基向量补齐
        u1 = U[:, 0].copy()
        v1 = Vt[0].copy()
        e = np.zeros(3)
        e[int(np.argmin(np.abs(u1)))] = 1.0
        u2 = np.cross(e, u1)
        u2 /= np.linalg.norm(u2)
        u3 = np.cross(u1, u2)
        e2 = np.zeros(3)
        e2[int(np.argmin(np.abs(v1)))] = 1.0
        v2 = np.cross(e2, v1)
        v2 /= np.linalg.norm(v2)
        v3 = np.cross(v1, v2)
        Uc = np.column_stack([u1, u2, u3])
        Vc = np.column_stack([v1, v2, v3])
        return Uc, Vc.T
    # rank 0：无方向信息，单位基即可（旋转完全不可识别）
    return np.eye(3), np.eye(3)


@dataclass
class SupportReport:
    n_points: int
    singular_values: np.ndarray         # 加权中心化点坐标的奇异值（米）
    rank: int                           # 精确秩
    rotational_dof_free: int            # 未被约束的旋转自由度数
    weak_axes: list[int]                # 近退化主轴编号（奇异值比 < 1e-3）
    line_axis: np.ndarray | None        # 共线时的直线方向（局部系）
    plane_normal: np.ndarray | None     # 共面时的平面法向（局部系）
    verdict: str                        # full / planar / collinear / single_point / empty

    def as_dict(self) -> dict:
        return {
            "n_points": self.n_points,
            "singular_values_m": [float(x) for x in self.singular_values],
            "exact_rank": self.rank,
            "rotational_dof_unconstrained": self.rotational_dof_free,
            "weak_axes": self.weak_axes,
            "line_axis": None if self.line_axis is None else [float(x) for x in self.line_axis],
            "plane_normal": None if self.plane_normal is None else [float(x) for x in self.plane_normal],
            "verdict": self.verdict,
        }


def support_report(points: np.ndarray, sigma: np.ndarray | None = None) -> SupportReport:
    """单站局部点集的几何支撑诊断（与任何世界系无关）。

    自由度结论（刚体配准中平移只要有 >=1 个对应点即可确定）：

    * 完整 3D 支撑：旋转 3 自由度全部可识别；
    * 共面（rank 2）：两个主轴确定姿态，第三轴由右手定则唯一补齐，旋转仍唯一；
    * 共线（rank 1）：绕直线轴的旋转完全不可识别（1 个自由旋转 DOF）；
    * 单点（rank 0）：只有平移可识别，旋转 3 自由度全自由。

    近退化（不精确共线但扁平/细长）单独通过 weak_axes 报告，不与精确缺秩混为一谈。
    """
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    n = P.shape[0]
    if n == 0:
        return SupportReport(0, np.zeros(3), 0, 3, [], None, None, "empty")
    if sigma is None:
        w = np.ones(n)
    else:
        w = 1.0 / np.asarray(sigma, dtype=float).reshape(-1) ** 2
    w = w / w.sum()
    p_bar = w @ P
    # 注意：np.linalg.svd 返回的右奇异向量是 Vt 的行（Vt[i]），不是 U 的列；
    # 点少于 3 个时奇异值个数 < 3，补零到 3 以便统一按主轴编号报告
    U, S_full, Vt = np.linalg.svd(np.sqrt(w)[:, None] * (P - p_bar))
    S = np.zeros(3)
    S[: len(S_full)] = S_full
    s1 = S[0] if S[0] > 0 else 1.0
    rank = int(np.sum(S > EXACT_RANK_RATIO * s1))
    weak = [i for i in range(3) if S[i] / s1 < WEAK_RANK_RATIO]

    line_axis = plane_normal = None
    if rank == 1:
        line_axis = Vt[0]
        rot_free, verdict = 1, "collinear"
    elif rank == 0:
        rot_free, verdict = 3, "single_point"
    elif rank == 2:
        plane_normal = Vt[2]
        rot_free, verdict = 0, "planar"
    else:
        rot_free, verdict = 0, "full"
    return SupportReport(n, S, rank, rot_free, weak, line_axis, plane_normal, verdict)
