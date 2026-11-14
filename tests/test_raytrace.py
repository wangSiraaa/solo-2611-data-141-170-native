"""Snell 分层射线追踪测试：解析核对、跨层折射、临界角、超剖面。"""

from __future__ import annotations

import numpy as np
import pytest

from sounding.io_data import SoundSpeedLayer
from sounding.raytrace import (
    STATUS_BEYOND_PROFILE,
    STATUS_CRITICAL,
    STATUS_OK,
    STATUS_START_BELOW_PROFILE,
    trace_ray,
)

C = 1500.0
ONE_LAYER = [SoundSpeedLayer(0.0, 100.0, C)]
DOWN = np.array([0.0, 0.0, 1.0])


def test_vertical_constant_speed_analytic():
    t_one = 0.01
    res = trace_ray(ONE_LAYER, 0.0, DOWN, C, t_one)
    assert res.status == STATUS_OK
    assert res.end_depth_m == pytest.approx(C * t_one)
    assert res.total_horizontal_north_m == 0.0
    assert res.total_horizontal_east_m == 0.0
    assert len(res.segments) == 1
    seg = res.segments[0]
    assert seg.travel_time_s == pytest.approx(t_one)
    assert seg.path_length_m == pytest.approx(C * t_one)
    assert seg.incidence_angle_rad == 0.0


def test_oblique_constant_speed_analytic():
    angle = np.deg2rad(30.0)
    direction = np.array([0.0, np.sin(angle), np.cos(angle)])
    t_one = 0.02
    res = trace_ray(ONE_LAYER, 0.0, direction, C, t_one)
    assert res.status == STATUS_OK
    path_len = C * t_one  # 30 m
    assert res.end_depth_m == pytest.approx(path_len * np.cos(angle))
    assert res.total_horizontal_east_m == pytest.approx(
        path_len * np.sin(angle)
    )
    assert res.total_horizontal_north_m == pytest.approx(0.0, abs=1e-15)
    # 单段时间即总单程时间
    assert sum(s.travel_time_s for s in res.segments) == pytest.approx(t_one)


def test_ray_ending_inside_layer_uses_remaining_time():
    # 层厚 15m，竖直射线单程 0.1s 只会走 15m 中的 ... 0.005s -> 7.5m
    layers = [SoundSpeedLayer(0.0, 15.0, C)]
    res = trace_ray(layers, 0.0, DOWN, C, 0.005)
    assert res.status == STATUS_OK
    assert res.end_depth_m == pytest.approx(7.5)
    assert len(res.segments) == 1


def test_snell_refraction_across_interface():
    """跨界面 sin(theta)/c 守恒，层内分段时间之和等于单程时间。"""
    layers = [
        SoundSpeedLayer(0.0, 15.0, 1450.0),
        SoundSpeedLayer(15.0, 35.0, 1500.0),
        SoundSpeedLayer(35.0, 70.0, 1540.0),
    ]
    angle = np.deg2rad(25.0)
    direction = np.array([0.0, np.sin(angle), np.cos(angle)])
    t_one = 0.015
    res = trace_ray(layers, 0.0, direction, 1450.0, t_one)
    assert res.status == STATUS_OK
    assert len(res.segments) >= 2

    p = np.sin(angle) / 1450.0
    assert res.snell_p == pytest.approx(p)
    for seg in res.segments:
        # 每层 Snell 关系
        assert np.sin(seg.incidence_angle_rad) == pytest.approx(
            p * seg.speed_m_s, abs=1e-12
        )
        # 段几何自洽：path = speed * time
        assert seg.path_length_m == pytest.approx(
            seg.speed_m_s * seg.travel_time_s, rel=1e-12
        )
        # 竖直与水平分量勾股
        assert np.hypot(seg.vertical_length_m, seg.horizontal_length_m) \
            == pytest.approx(seg.path_length_m, rel=1e-12)

    # 总时间守恒
    assert sum(s.travel_time_s for s in res.segments) == pytest.approx(t_one)
    # 总水平偏移等于各段之和
    assert sum(s.horizontal_offset_east_m for s in res.segments) \
        == pytest.approx(res.total_horizontal_east_m, rel=1e-12)


def test_critical_angle_returns_undetermined():
    # 80 度在 1450->1500 界面：sin(80)*1500/1450 ≈ 1.019 > 1，全反射
    layers = [
        SoundSpeedLayer(0.0, 15.0, 1450.0),
        SoundSpeedLayer(15.0, 35.0, 1500.0),
        SoundSpeedLayer(35.0, 70.0, 1540.0),
    ]
    angle = np.deg2rad(80.0)
    direction = np.array([0.0, np.sin(angle), np.cos(angle)])
    res = trace_ray(layers, 0.0, direction, 1450.0, 5.0)
    assert res.status == STATUS_CRITICAL
    # 临界前的路径段仍然保留：只走完第一层
    assert len(res.segments) == 1
    assert res.segments[-1].exit_depth_m == pytest.approx(15.0)


def test_beyond_known_profile_returns_undetermined():
    # 时间长到足以穿出 100m 层底
    res = trace_ray(ONE_LAYER, 0.0, DOWN, C, 1.0)
    assert res.status == STATUS_BEYOND_PROFILE
    assert res.end_depth_m == pytest.approx(100.0)
    # 已完成的层段保留，剩余时间在原因中报告
    assert "无速度可用" in res.reason


def test_start_below_profile_rejected():
    res = trace_ray(ONE_LAYER, 100.0, DOWN, C, 0.01)
    assert res.status == STATUS_START_BELOW_PROFILE
    assert res.segments == []


def test_start_inside_deeper_layer():
    layers = [
        SoundSpeedLayer(0.0, 15.0, 1450.0),
        SoundSpeedLayer(15.0, 35.0, 1500.0),
    ]
    res = trace_ray(layers, 20.0, DOWN, 1500.0, 0.01)
    assert res.status == STATUS_OK
    assert res.segments[0].layer_index == 1
    assert res.start_depth_m == 20.0


def test_segment_depths_and_chain_continuity():
    layers = [
        SoundSpeedLayer(0.0, 15.0, 1450.0),
        SoundSpeedLayer(15.0, 35.0, 1500.0),
    ]
    angle = np.deg2rad(20.0)
    direction = np.array([0.0, np.sin(angle), np.cos(angle)])
    res = trace_ray(layers, 0.0, direction, 1450.0, 0.02)
    assert res.status == STATUS_OK
    # 段间深度首尾相接
    for a, b in zip(res.segments, res.segments[1:]):
        assert a.exit_depth_m == pytest.approx(b.entry_depth_m, abs=1e-12)
        assert b.entry_depth_m == pytest.approx(a.bottom_depth_m, abs=1e-12)
    assert res.segments[0].entry_depth_m == 0.0
    assert res.segments[-1].exit_depth_m == pytest.approx(res.end_depth_m)
