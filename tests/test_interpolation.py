"""时间包围、线性插值与单位四元数短弧 slerp 的测试。"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sounding.interpolation import (
    EXACT,
    GAP_TOO_LARGE,
    INTERPOLATED,
    NO_DATA,
    OUT_OF_RANGE,
    find_bracket,
    interpolate_quaternion,
    linear_interpolate,
    slerp_wxyz,
)

GAP = 1500.0
TIMES = np.array([0.0, 1000.0, 2000.0, 7000.0])


def test_exact_sample_uses_record_directly():
    b = find_bracket(TIMES, 1000.0, GAP)
    assert b.status == EXACT
    assert b.left_index == 1 and b.right_index == 1
    vals = np.array([10.0, 20.0, 30.0, 40.0])
    out, err = linear_interpolate(vals, b)
    assert err is None and out == 20.0


def test_interpolated_midpoint_linear():
    b = find_bracket(TIMES, 500.0, GAP)
    assert b.status == INTERPOLATED
    assert (b.left_index, b.right_index) == (0, 1)
    assert b.alpha == pytest.approx(0.5)
    vals = np.array([0.0, 10.0, 20.0, 70.0])
    out, _ = linear_interpolate(vals, b)
    assert out == pytest.approx(5.0)


def test_gap_too_large_rejected_not_nearest():
    # 4500 落在 2000~7000 之间，间隔 5000 > 门槛 1500
    b = find_bracket(TIMES, 4500.0, GAP)
    assert b.status == GAP_TOO_LARGE
    assert b.gap_ms == 5000.0
    vals = np.arange(4.0)
    out, err = linear_interpolate(vals, b)
    assert out is None and err == GAP_TOO_LARGE


def test_out_of_range_never_uses_nearest():
    for t in (-1.0, 7001.0, 100000.0):
        b = find_bracket(TIMES, t, GAP)
        assert b.status == OUT_OF_RANGE
        assert b.left_index is None and b.right_index is None


def test_empty_series():
    b = find_bracket(np.array([]), 10.0, GAP)
    assert b.status == NO_DATA


def test_exact_sample_ignores_gap_limit():
    # 精确命中 7000，即使与前一点间隔 5000 也直接采用
    b = find_bracket(TIMES, 7000.0, GAP)
    assert b.status == EXACT


def test_quat_identity_slerp():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    # 绕 y 轴转 2*0.04=0.08 rad 的四元数
    ang = 0.08
    q1 = np.array([np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0])
    q_mid = slerp_wxyz(q0, q1, 0.5)
    expected_ang = ang / 2
    assert q_mid[0] == pytest.approx(np.cos(expected_ang / 2), abs=1e-12)
    assert q_mid[2] == pytest.approx(np.sin(expected_ang / 2), abs=1e-12)
    # 结果仍为单位四元数
    assert np.linalg.norm(q_mid) == pytest.approx(1.0)


def test_slerp_sign_inversion_gives_same_rotation():
    """q 与 -q 是同一姿态：翻转符号不得改变插值姿态。"""
    ang = 0.5
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0])
    alpha = 0.37
    qa = slerp_wxyz(q0, q1, alpha)
    qb = slerp_wxyz(q0, -q1, alpha)  # 右端取反
    # 同一姿态（四元数本身可能整体差一个符号）
    assert np.allclose(qa, qb, atol=1e-12) or np.allclose(qa, -qb, atol=1e-12)


def test_slerp_never_collapses_to_zero():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = -q0  # 与 q0 同一姿态但符号相反
    q = slerp_wxyz(q0, q1, 0.5)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    assert np.allclose(q, q0)


def test_interpolate_quaternion_exact_and_bad_norm():
    ang = 0.4
    good = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0],
        ]
    )
    b_exact = find_bracket(np.array([0.0, 1000.0]), 0.0, GAP)
    q, err = interpolate_quaternion(good, b_exact)
    assert err is None and np.allclose(q, good[0])

    bad = good.copy()
    bad[1] = bad[1] * 0.5  # 范数 0.5，非单位四元数
    b_interp = find_bracket(np.array([0.0, 1000.0]), 500.0, GAP)
    q2, err2 = interpolate_quaternion(bad, b_interp)
    assert q2 is None and err2.startswith("attitude_not_unit_quaternion")


def test_2d_linear_interpolation():
    values = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    b = find_bracket(np.array([0.0, 1000.0]), 250.0, GAP)
    out, err = linear_interpolate(values, b)
    assert err is None
    assert np.allclose(out, [0.5, 1.0, 1.5])


def test_slerp_matches_scipy_reference_on_random_rotations():
    """自研短弧 slerp 与 scipy Rotation 的旋转组合参考在机器精度内一致。"""
    rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(100):
        r0 = Rotation.random(random_state=rng)
        r1 = Rotation.random(random_state=rng)
        q0 = np.roll(r0.as_quat(), 1)  # scipy 用 (x,y,z,w)，转 (w,x,y,z)
        q1 = np.roll(r1.as_quat(), 1)
        alpha = float(rng.random())

        mine = slerp_wxyz(q0, q1, alpha)

        # 参考：r0 * exp(alpha * log(r0^-1 r1))，即短弧球面插值
        delta = r0.inv() * r1
        ref = np.roll(
            (r0 * Rotation.from_rotvec(delta.as_rotvec() * alpha)).as_quat(), 1
        )
        err = min(np.linalg.norm(mine - ref), np.linalg.norm(mine + ref))
        max_err = max(max_err, err)
    assert max_err < 1e-12
