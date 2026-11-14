"""端到端流水线测试：解析核对、姿态/杆臂、四元数符号反转、缺口与失败分类。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sounding.config import load_config
from sounding.pipeline import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_UNDETERMINED,
    run_pipeline,
)


def _run(make_files, cfg_overrides=None, lever=(0.0, 0.0, 0.0), gap_ms=1500, **kw):
    paths = make_files(lever=lever, gap_ms=gap_ms, **kw)
    config = load_config(paths["config"])
    return run_pipeline(
        beams_path=paths["beams"],
        nav_path=paths["nav"],
        tide_path=paths["tide"],
        speed_path=paths["speed"],
        config=config,
        output_dir=paths["beams"].parent / "out",
    )


def _by_id(result):
    return {r.beam_id: r for r in result["results"]}


def test_analytic_vertical_and_oblique_constant_speed(make_files):
    result = _run(
        make_files,
        beam_rows=[
            ("V", 0, 0, 0.02),
            ("A", 500, 30, 0.04),
        ],
        layer_rows=[(0, 100, 1500)],
    )
    beams = _by_id(result)
    v, a = beams["V"], beams["A"]
    assert v.status == STATUS_OK and a.status == STATUS_OK

    # 竖直：depth = c * t_one
    assert v.end_depth_below_surface_m == pytest.approx(1500 * 0.01)
    assert v.sounding_point_ned_m[2] == pytest.approx(15.0)

    # 30 度斜束：path=30m, depth=30 cos30, east=30 sin30
    assert a.end_depth_below_surface_m == pytest.approx(
        30 * np.cos(np.deg2rad(30))
    )
    assert a.sounding_point_ned_m[1] == pytest.approx(
        30 * np.sin(np.deg2rad(30))
    )
    assert a.sounding_point_ned_m[0] == pytest.approx(0.0, abs=1e-12)


def test_lever_arm_and_tide_chain(make_files):
    """杆臂随姿态旋转、潮位决定水面，坐标链 NED = 换能器 + 水平偏移。"""
    result = _run(
        make_files,
        lever=(0.3, 0.1, 1.0),
        nav_rows=[
            (0, 0, 0, 0.5, 1, 0, 0, 0),
            (1000, 1, 0.2, 0.5, 1, 0, 0, 0),
        ],
        tide_rows=[(0, 0.2), (1000, 0.25)],
        beam_rows=[("V", 0, 0, 0.012)],
        layer_rows=[(0, 100, 1500)],
    )
    b = _by_id(result)["V"]
    assert b.status == STATUS_OK
    # 单位姿态：换能器 = 位置 + 杆臂
    assert b.transducer_ned_m == pytest.approx([0.3, 0.1, 1.5])
    # 水面 down=-0.2，起点深度 1.5-(-0.2)=1.7
    assert b.start_depth_below_surface_m == pytest.approx(1.7)
    # 终点深度 = 1.7 + 1500*0.006
    assert b.end_depth_below_surface_m == pytest.approx(1.7 + 9.0)
    # NED down = 水面 down + 水面下深度
    assert b.sounding_point_ned_m[2] == pytest.approx(-0.2 + 10.7)
    assert b.sounding_point_ned_m[:2] == pytest.approx([0.3, 0.1])


def test_attitude_changes_initial_direction_not_post_rotation(make_files):
    """倾斜姿态必须改变射线初始方向与测点，而非事后旋转常速结果。"""
    theta = np.deg2rad(10.0)
    qw, qy = np.cos(theta / 2), np.sin(theta / 2)
    result = _run(
        make_files,
        nav_rows=[
            (0, 0, 0, 0, qw, 0, qy, 0),
            (1000, 0, 0, 0, qw, 0, qy, 0),
        ],
        beam_rows=[("P", 0, 0, 0.02)],  # 船体竖直向下波束
        layer_rows=[(0, 100, 1500)],
    )
    b = _by_id(result)["P"]
    assert b.status == STATUS_OK
    # 世界系初始方向即绕 y 旋转后的 down 轴
    assert b.beam_direction_world == pytest.approx(
        [np.sin(theta), 0.0, np.cos(theta)], abs=1e-12
    )
    path = 1500 * 0.01
    assert b.sounding_point_ned_m[0] == pytest.approx(path * np.sin(theta))
    assert b.end_depth_below_surface_m == pytest.approx(
        path * np.cos(theta)
    )


def test_quaternion_sign_flip_identical_sounding(make_files):
    """导航四元数整体取反（同一姿态）必须给出完全相同的测点。"""
    theta = np.deg2rad(12.0)
    qw, qy = np.cos(theta / 2), np.sin(theta / 2)
    common = dict(
        beam_rows=[("S", 500, 20, 0.03)],
        layer_rows=[(0, 100, 1500)],
        lever=(0.2, -0.1, 0.8),
    )
    r_pos = _run(
        make_files,
        nav_rows=[
            (0, 0, 0, 0.5, qw, 0, qy, 0),
            (1000, 2, 0.4, 0.5, qw, 0, qy, 0),
        ],
        **common,
    )
    r_neg = _run(
        make_files,
        nav_rows=[
            (0, 0, 0, 0.5, -qw, 0, -qy, 0),
            (1000, 2, 0.4, 0.5, -qw, 0, -qy, 0),
        ],
        **common,
    )
    p1 = _by_id(r_pos)["S"].sounding_point_ned_m
    p2 = _by_id(r_neg)["S"].sounding_point_ned_m
    assert np.allclose(p1, p2, atol=1e-12)


def test_time_gap_and_out_of_range(make_files):
    nav_rows = [
        (0, 0, 0, 0, 1, 0, 0, 0),
        (2000, 1, 0, 0, 1, 0, 0, 0),
        (7000, 2, 0, 0, 1, 0, 0, 0),
    ]
    tide_rows = [(0, 0.0), (2000, 0.1), (7000, 0.2)]
    result = _run(
        make_files,
        nav_rows=nav_rows,
        tide_rows=tide_rows,
        beam_rows=[
            ("GAP", 4500, 0, 0.01),   # 2000~7000 间隔超门槛
            ("LATE", 8000, 0, 0.01),  # 超出末条记录
        ],
    )
    beams = _by_id(result)
    gap, late = beams["GAP"], beams["LATE"]
    assert gap.status == STATUS_FAILED
    assert any("gap_too_large" in r for r in gap.failure_reasons)
    assert gap.attitude_interp["status"] == "gap_too_large"
    assert late.status == STATUS_FAILED
    assert any("out_of_range" in r for r in late.failure_reasons)
    assert late.sounding_point_ned_m is None  # 不悄悄给坐标


def test_zero_and_negative_time_distinct(make_files):
    result = _run(
        make_files,
        beam_rows=[
            ("ZERO", 0, 0, 0.0),
            ("NEG", 0, 0, -0.01),
        ],
    )
    beams = _by_id(result)
    assert beams["ZERO"].status == STATUS_FAILED
    assert "zero_two_way_time" in beams["ZERO"].failure_reasons[0]
    assert beams["NEG"].status == STATUS_FAILED
    assert "negative_two_way_time" in beams["NEG"].failure_reasons[0]


def test_non_downward_beam_after_attitude(make_files):
    # 船首上仰 90 度：body down 轴变成水平 north
    q = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]
    result = _run(
        make_files,
        nav_rows=[
            (0, 0, 0, 0, *q),
            (1000, 0, 0, 0, *q),
        ],
        beam_rows=[("UP", 0, 0, 0.02)],
    )
    b = _by_id(result)["UP"]
    assert b.status == STATUS_FAILED
    assert any("non_downward_beam" in r for r in b.failure_reasons)


def test_layered_critical_and_beyond_profile(make_files):
    result = _run(
        make_files,
        beam_rows=[
            ("CRIT", 0, 80, 0.5),
            ("LONG", 0, 0, 1.0),
        ],
        layer_rows=[
            (0, 15, 1450),
            (15, 35, 1500),
            (35, 70, 1540),
        ],
    )
    beams = _by_id(result)
    assert beams["CRIT"].status == STATUS_UNDETERMINED
    assert any("critical_propagation" in r for r in beams["CRIT"].failure_reasons)
    assert beams["LONG"].status == STATUS_UNDETERMINED
    assert any(
        "beyond_known_sound_speed" in r for r in beams["LONG"].failure_reasons
    )
    # 未确定波束仍保留已追踪路径段作为证据
    assert len(beams["LONG"].segments) >= 1


def test_illegal_layer_boundaries_recorded(make_files):
    result = _run(
        make_files,
        layer_rows=[
            (0, 15, 1450),
            (16, 35, 1500),   # 层间不连续
        ],
    )
    b = _by_id(result)["V"]
    assert b.status == STATUS_FAILED
    assert any("sound_speed_layers_invalid" in r for r in b.failure_reasons)


def test_outputs_written_with_full_evidence(make_files):
    result = _run(make_files)
    detail = json.loads(result["detail_path"].read_text(encoding="utf-8"))
    beam = detail["beams"][0]
    # 支持证据字段齐全，不只一个 XYZ
    for key in (
        "position_interp",
        "attitude_interp",
        "tide_interp",
        "quaternion_wxyz",
        "beam_direction_world",
        "transducer_ned_m",
        "segments",
    ):
        assert key in beam
    assert result["summary_path"].exists()
    csv_text = result["summary_path"].read_text(encoding="utf-8")
    assert "beam_id" in csv_text and "depth_below_surface_m" in csv_text
