"""单波束归位流水线与整批处理。

每个波束独立处理并保留完整支持证据：导航/姿态/潮位的插值包围记录、
换能器位置与初始方向、逐层路径段与失败原因。失败不抛异常中断整批，
而是在该波束记录上标注 failed/undetermined 状态与原因。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .geometry import (
    body_beam_direction,
    is_downward,
    quat_wxyz_to_matrix,
    rotate_body_to_world,
    transducer_position_ned,
)
from .interpolation import (
    find_bracket,
    interpolate_quaternion,
    linear_interpolate,
)
from .io_data import (
    BeamRecord,
    NavRecord,
    SoundSpeedLayer,
    TideRecord,
    load_beams,
    load_layers,
    load_navigation,
    load_tide,
)
from .raytrace import trace_ray

STATUS_OK = "ok"
STATUS_UNDETERMINED = "undetermined"
STATUS_FAILED = "failed"


@dataclass
class BeamOutput:
    """单波束归位结果（成功、未确定、失败共用同一结构）。"""

    beam_id: str
    time_ms: float
    status: str
    failure_reasons: list[str] = field(default_factory=list)
    one_way_time_s: float | None = None
    # 插值支持证据
    position_interp: dict | None = None
    attitude_interp: dict | None = None
    tide_interp: dict | None = None
    # 姿态几何
    quaternion_wxyz: list[float] | None = None
    beam_direction_body: list[float] | None = None
    beam_direction_world: list[float] | None = None
    transducer_ned_m: list[float] | None = None
    surface_down_m: float | None = None
    start_depth_below_surface_m: float | None = None
    # 射线追踪
    snell_p: float | None = None
    ray_horizontal_heading_rad: float | None = None
    end_depth_below_surface_m: float | None = None
    sounding_point_ned_m: list[float] | None = None
    segments: list[dict] = field(default_factory=list)


def _evidence(bracket, interp_value=None) -> dict:
    """把一次插值的包围信息整理成可序列化证据。"""
    ev = {
        "status": bracket.status,
        "left_index": bracket.left_index,
        "right_index": bracket.right_index,
        "left_time_ms": _clean(bracket.left_time_ms),
        "right_time_ms": _clean(bracket.right_time_ms),
        "gap_ms": _clean(bracket.gap_ms),
        "alpha": _clean(bracket.alpha),
    }
    if interp_value is not None:
        ev["interpolated_value"] = [
            float(v) for v in np.atleast_1d(interp_value)
        ]
    return ev


def _clean(x):
    """NaN 转 None，保证 JSON 合法。"""
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else x


def process_beam(
    beam: BeamRecord,
    nav: NavRecord,
    tide: TideRecord,
    layers: list[SoundSpeedLayer] | None,
    layer_error: str | None,
    config: Config,
) -> BeamOutput:
    """处理单个波束，返回带完整证据的 :class:`BeamOutput`。"""
    out = BeamOutput(beam_id=beam.beam_id, time_ms=beam.time_ms, status=STATUS_OK)

    # 1) 时间合法性：零或负往返时间分别定位
    if beam.two_way_time_s < 0.0:
        out.status = STATUS_FAILED
        out.failure_reasons.append(
            f"negative_two_way_time: {beam.two_way_time_s} s"
        )
        return out
    if beam.two_way_time_s == 0.0:
        out.status = STATUS_FAILED
        out.failure_reasons.append("zero_two_way_time")
        return out
    out.one_way_time_s = beam.two_way_time_s / 2.0

    t = beam.time_ms
    gap_limit = config.max_interpolation_gap_ms

    # 2) 导航位置：线性插值，缺口/越界保留原因，绝不用最近邻
    pos_bracket = find_bracket(nav.time_ms, t, gap_limit)
    position, pos_err = linear_interpolate(
        np.column_stack([nav.north_m, nav.east_m, nav.down_m]), pos_bracket
    )
    out.position_interp = _evidence(pos_bracket, interp_value=position)
    if pos_err is not None:
        out.failure_reasons.append(f"position_unavailable:{pos_err}")

    # 3) 姿态：单位四元数短弧插值（q 与 -q 同姿态，先选半球再 slerp）
    att_bracket = find_bracket(nav.time_ms, t, gap_limit)
    quat, att_err = interpolate_quaternion(nav.quat_wxyz, att_bracket)
    out.attitude_interp = _evidence(att_bracket)
    if att_err is not None:
        out.failure_reasons.append(f"attitude_unavailable:{att_err}")

    # 4) 潮位：线性插值；水面 NED down = -tide
    tide_bracket = find_bracket(tide.time_ms, t, gap_limit)
    tide_val, tide_err = linear_interpolate(tide.tide_m, tide_bracket)
    out.tide_interp = _evidence(tide_bracket, interp_value=tide_val)
    if tide_err is not None:
        out.failure_reasons.append(f"tide_unavailable:{tide_err}")

    # 三类时间支持任一不可用都无法归位；证据已全部保留，统一结束
    if pos_err is not None or att_err is not None or tide_err is not None:
        out.status = STATUS_FAILED
        return out

    # 5) 姿态先于射线进入几何：旋转矩阵、杆臂、波束初始方向
    rot = quat_wxyz_to_matrix(quat)
    out.quaternion_wxyz = [float(v) for v in quat]
    dir_body = body_beam_direction(beam.angle_rad)
    dir_world = rotate_body_to_world(rot, dir_body)
    out.beam_direction_body = [float(v) for v in dir_body]
    out.beam_direction_world = [float(v) for v in dir_world]

    if not is_downward(dir_world):
        out.status = STATUS_FAILED
        out.failure_reasons.append(
            "non_downward_beam: 波束世界方向 down 分量 "
            f"{dir_world[2]:.9f} <= 0"
        )
        return out

    xducer = transducer_position_ned(position, rot, config.lever_arm)
    out.transducer_ned_m = [float(v) for v in xducer]

    # 6) 瞬时水面定义分层深度
    surface_down = -float(np.atleast_1d(tide_val)[0])
    out.surface_down_m = surface_down
    start_depth = xducer[2] - surface_down  # 水面以下为正
    out.start_depth_below_surface_m = float(start_depth)
    if start_depth < 0.0:
        out.status = STATUS_FAILED
        out.failure_reasons.append(
            f"transducer_above_surface: 相对水面深度 {start_depth:.6f} m < 0"
        )
        return out

    # 7) 声速层合法性（非法层界单独定位）
    if layer_error is not None or layers is None:
        out.status = STATUS_FAILED
        out.failure_reasons.append(f"sound_speed_layers_invalid: {layer_error}")
        return out

    start_layer = _layer_index_at(layers, start_depth)
    if start_layer >= len(layers):
        out.status = STATUS_UNDETERMINED
        out.failure_reasons.append(
            "start_below_known_profile_undetermined: 换能器深度 "
            f"{start_depth:.6f} m 超出已知声速最大深度 "
            f"{layers[-1].bottom_depth_m:.6f} m"
        )
        return out

    # 8) Snell 逐层追踪（姿态影响已在初始方向中，事后不再旋转结果）
    trace = trace_ray(
        layers=layers,
        start_depth_m=start_depth,
        initial_direction_world=dir_world,
        initial_speed_m_s=layers[start_layer].speed_m_s,
        one_way_time_s=out.one_way_time_s,
    )
    out.snell_p = float(trace.snell_p)
    out.ray_horizontal_heading_rad = float(trace.ray_horizontal_heading_rad)
    out.segments = [asdict(seg) for seg in trace.segments]

    if trace.status != "ok":
        out.status = STATUS_UNDETERMINED
        out.failure_reasons.append(f"{trace.status}: {trace.reason}")
        out.end_depth_below_surface_m = (
            None if trace.end_depth_m is None else float(trace.end_depth_m)
        )
        if trace.end_depth_m is not None:
            # 未确定也给出已追踪到的位置，但明确不是合格测点
            out.sounding_point_ned_m = [
                float(xducer[0] + trace.total_horizontal_north_m),
                float(xducer[1] + trace.total_horizontal_east_m),
                float(surface_down + trace.end_depth_m),
            ]
        return out

    # 9) 合格测点：世界 NED 坐标 + 水面下深度
    out.end_depth_below_surface_m = float(trace.end_depth_m)
    out.sounding_point_ned_m = [
        float(xducer[0] + trace.total_horizontal_north_m),
        float(xducer[1] + trace.total_horizontal_east_m),
        float(surface_down + trace.end_depth_m),
    ]
    return out


def _layer_index_at(layers: list[SoundSpeedLayer], depth: float) -> int:
    """深度所在层下标；超出最后一层底时返回 len(layers)。"""
    for k, layer in enumerate(layers):
        if depth < layer.bottom_depth_m:
            return k
    return len(layers)


def run_pipeline(
    beams_path: str | Path,
    nav_path: str | Path,
    tide_path: str | Path,
    speed_path: str | Path,
    config: Config,
    output_dir: str | Path,
) -> dict:
    """整批归位并写出 CSV 汇总 + JSON 全量证据。

    Returns:
        含 outputs 与统计信息的 dict（同时序列化为 JSON）。
    """
    beams = load_beams(beams_path)
    nav = load_navigation(nav_path)
    tide = load_tide(tide_path)

    layers: list[SoundSpeedLayer] | None
    try:
        layers = load_layers(speed_path)
        layer_error: str | None = None
    except ValueError as exc:
        layers = None
        layer_error = str(exc)

    results = [
        process_beam(beam, nav, tide, layers, layer_error, config)
        for beam in beams
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "soundings.csv"
    detail_path = output_dir / "soundings_detail.json"

    _write_summary(results, summary_path)
    payload = {
        "coordinate_convention": {
            "body_frame": "FRD (x forward, y starboard, z down)",
            "world_frame": "NED (north, east, down; down positive)",
            "tide_positive_up": True,
            "surface_ned_down": "-tide_m",
            "layer_depth_reference": "instantaneous water surface, downward positive",
        },
        "summary": _stats(results),
        "beams": [asdict(r) for r in results],
    }
    with open(detail_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    return {"results": results, "summary_path": summary_path,
            "detail_path": detail_path, "stats": payload["summary"]}


def _stats(results: list[BeamOutput]) -> dict:
    counts = {STATUS_OK: 0, STATUS_UNDETERMINED: 0, STATUS_FAILED: 0}
    for r in results:
        counts[r.status] += 1
    return {"total": len(results), **counts}


_SUMMARY_COLUMNS = [
    "beam_id",
    "time_ms",
    "status",
    "two_way_used_one_way_s",
    "north_m",
    "east_m",
    "down_m",
    "depth_below_surface_m",
    "n_segments",
    "failure_reasons",
]


def _write_summary(results: list[BeamOutput], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_SUMMARY_COLUMNS)
        for r in results:
            # 只有 ok 波束才在汇总表给测点坐标；未确定波束的已追踪位置
            # 属于支持证据，仅出现在 JSON 中，避免被误读为合格测点
            point = r.sounding_point_ned_m if r.status == STATUS_OK else None
            depth = (
                r.end_depth_below_surface_m if r.status == STATUS_OK else None
            )
            writer.writerow(
                [
                    r.beam_id,
                    r.time_ms,
                    r.status,
                    "" if r.one_way_time_s is None else r.one_way_time_s,
                    "" if point is None else point[0],
                    "" if point is None else point[1],
                    "" if point is None else point[2],
                    "" if depth is None else depth,
                    len(r.segments),
                    " | ".join(r.failure_reasons),
                ]
            )
