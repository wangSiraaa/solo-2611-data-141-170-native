"""姿态几何测试：旋转矩阵、杆臂、波束方向与 q/-q 等价性。"""

from __future__ import annotations

import numpy as np
import pytest

from sounding.geometry import (
    body_beam_direction,
    is_downward,
    quat_wxyz_to_matrix,
    rotate_body_to_world,
    transducer_position_ned,
)


def test_identity_quaternion_matrix():
    r = quat_wxyz_to_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(r, np.eye(3))


def test_quaternion_sign_inversion_same_matrix():
    ang = 0.6
    q = np.array([np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0])
    r1 = quat_wxyz_to_matrix(q)
    r2 = quat_wxyz_to_matrix(-q)
    assert np.allclose(r1, r2)


def test_pitch_rotation_maps_body_axes():
    # 绕 body y 轴（右舷）旋转 theta：FRD->NED，down 轴向 north 方向倾斜
    theta = 0.1
    q = np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])
    r = quat_wxyz_to_matrix(q)
    down_world = r @ np.array([0.0, 0.0, 1.0])
    # 绕 +y 右手旋转：z 轴 -> (sin theta, 0, cos theta)
    assert down_world == pytest.approx(
        [np.sin(theta), 0.0, np.cos(theta)], abs=1e-12
    )


def test_body_beam_direction_unit_and_angle():
    d = body_beam_direction(np.deg2rad(30.0))
    assert np.linalg.norm(d) == pytest.approx(1.0)
    assert d[1] == pytest.approx(0.5)
    assert d[2] == pytest.approx(np.sqrt(3) / 2)
    assert d[0] == 0.0


def test_lever_arm_rotated_before_ray():
    theta = 0.25
    q = np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])
    r = quat_wxyz_to_matrix(q)
    pos = np.array([10.0, 5.0, 2.0])
    lever = np.array([0.3, 0.1, 1.0])
    xducer = transducer_position_ned(pos, r, lever)
    assert np.allclose(xducer, pos + r @ lever)


def test_is_downward_classification():
    assert is_downward(np.array([0.0, 0.0, 0.5]))
    assert not is_downward(np.array([0.0, 0.0, 0.0]))
    assert not is_downward(np.array([0.0, 0.0, -1.0]))
    # 接近水平的向下波束仍判向下
    assert is_downward(np.array([1.0, 0.0, 1e-9]))


def test_rotated_non_downward_beam_detected():
    # 船首上仰 90 度后，body down 轴指向 world north（水平）
    theta = np.pi / 2
    q = np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])
    r = quat_wxyz_to_matrix(q)
    vertical_beam = body_beam_direction(0.0)
    world_dir = rotate_body_to_world(r, vertical_beam)
    assert not is_downward(world_dir)
