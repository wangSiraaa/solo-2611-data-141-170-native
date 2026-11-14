"""水平分层介质中的 Snell 射线追踪。

模型假设（见 config.json）：

* 声速层水平、层内分段常数，层内无梯度，射线在层内走直线；
* 单程时间 = 往返时间 / 2，发射期间换能器位置不变；
* 跨界面遵循 Snell 关系 ``sin(theta) / c = p``（theta 为射线与竖直
  down 方向的夹角），水平方位在水平分层中保持不变；
* 射线可能在当前层内结束，也可能跨多层；
* 达到临界传播条件（sin(theta) >= 1，全反射/掠射）或剩余时间超出
  已知声速深度时，返回未确定状态并保留已走完的路径段，绝不虚构速度。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .io_data import SoundSpeedLayer

# 物理判定容差：sin(theta) 达到 1 的临界判定
CRITICAL_SIN_TOL = 1e-12
# 界面深度重合的相对容差（避免界面处因舍入产生零厚度段）
DEPTH_EPS = 1e-10
# 竖直射线判定：水平分量小于该值即按竖直处理
VERTICAL_SIN_TOL = 1e-15

STATUS_OK = "ok"
STATUS_CRITICAL = "critical_propagation_undetermined"
STATUS_BEYOND_PROFILE = "beyond_known_sound_speed_undetermined"
STATUS_START_BELOW_PROFILE = "start_below_known_profile_undetermined"


@dataclass(frozen=True)
class PathSegment:
    """一条层内直线段的完整支持证据。"""

    layer_index: int
    top_depth_m: float
    bottom_depth_m: float
    speed_m_s: float
    incidence_angle_rad: float       # 与竖直 down 方向夹角
    entry_depth_m: float
    exit_depth_m: float
    vertical_length_m: float         # 沿 down 的竖直长度（>=0）
    horizontal_length_m: float
    path_length_m: float
    travel_time_s: float
    horizontal_offset_north_m: float  # 本段水平位移（世界 NED）
    horizontal_offset_east_m: float


@dataclass
class TraceResult:
    """射线追踪结果。未确定时仍保留已完成的路径段与失败原因。"""

    status: str
    reason: str | None
    one_way_time_s: float
    start_depth_m: float
    end_depth_m: float | None
    total_horizontal_north_m: float
    total_horizontal_east_m: float
    ray_horizontal_heading_rad: float
    snell_p: float
    segments: list[PathSegment] = field(default_factory=list)


def _layer_at_depth(layers: list[SoundSpeedLayer], depth: float) -> int:
    """返回包含该深度的层下标；深度恰在界面上取界面下方那一层。"""
    for k, layer in enumerate(layers):
        if depth < layer.bottom_depth_m - DEPTH_EPS:
            return k
        if abs(depth - layer.bottom_depth_m) <= DEPTH_EPS:
            if k + 1 < len(layers):
                return k + 1
            return k
    return len(layers)  # 已在剖面之外


def trace_ray(
    layers: list[SoundSpeedLayer],
    start_depth_m: float,
    initial_direction_world: np.ndarray,
    initial_speed_m_s: float,
    one_way_time_s: float,
) -> TraceResult:
    """逐层追踪一条向下射线。

    Args:
        layers: 已通过合法性校验的水平常速层，层深自瞬时水面向下。
        start_depth_m: 射线起点（换能器）相对瞬时水面的向下深度。
        initial_direction_world: 世界 NED 下的单位初始方向，down 分量须为正。
        initial_speed_m_s: 起点所在层声速。
        one_way_time_s: 单程传播时间（往返时间的一半），须为正。

    Returns:
        :class:`TraceResult`；``status`` 为 ok 或两类未确定状态。
    """
    direction = np.asarray(initial_direction_world, dtype=float)
    d_n, d_e, d_d = direction
    sin0 = float(np.hypot(d_n, d_e))
    # Snell 不变量：全程守恒
    snell_p = sin0 / float(initial_speed_m_s)
    heading = float(np.arctan2(d_e, d_n))  # 水平方位，水平分层中不变

    last_bottom = layers[-1].bottom_depth_m
    if start_depth_m >= last_bottom - DEPTH_EPS:
        return TraceResult(
            status=STATUS_START_BELOW_PROFILE,
            reason=(
                f"换能器深度 {start_depth_m:.6f} m 已达到/超过已知声速"
                f"最大深度 {last_bottom:.6f} m，无法定义传播速度"
            ),
            one_way_time_s=one_way_time_s,
            start_depth_m=start_depth_m,
            end_depth_m=None,
            total_horizontal_north_m=0.0,
            total_horizontal_east_m=0.0,
            ray_horizontal_heading_rad=heading,
            snell_p=snell_p,
        )

    k = _layer_at_depth(layers, start_depth_m)
    depth = float(start_depth_m)
    time_remaining = float(one_way_time_s)
    total_hn = 0.0
    total_he = 0.0
    segments: list[PathSegment] = []

    while True:
        layer = layers[k]
        sin_theta = snell_p * layer.speed_m_s

        # 临界传播：全反射或掠射，不能继续按向下射线虚构
        if sin_theta >= 1.0 - CRITICAL_SIN_TOL:
            return TraceResult(
                status=STATUS_CRITICAL,
                reason=(
                    f"在第 {k} 层（深度 {layer.top_depth_m}-"
                    f"{layer.bottom_depth_m} m，声速 {layer.speed_m_s} m/s）"
                    f"达到临界传播条件 sin(theta)={min(sin_theta, 1.0):.9f}"
                ),
                one_way_time_s=one_way_time_s,
                start_depth_m=start_depth_m,
                end_depth_m=depth,
                total_horizontal_north_m=total_hn,
                total_horizontal_east_m=total_he,
                ray_horizontal_heading_rad=heading,
                snell_p=snell_p,
                segments=segments,
            )

        sin_theta = min(sin_theta, 1.0)
        cos_theta = float(np.sqrt(max(1.0 - sin_theta * sin_theta, 0.0)))
        theta = float(np.arcsin(sin_theta))

        thickness = max(layer.bottom_depth_m - depth, 0.0)
        if sin_theta <= VERTICAL_SIN_TOL:
            # 竖直射线：水平偏移为零
            time_to_boundary = thickness / layer.speed_m_s
            if time_remaining <= time_to_boundary + DEPTH_EPS / layer.speed_m_s:
                vert = layer.speed_m_s * time_remaining
                _append_segment(
                    segments, k, layer, theta, depth, depth + vert,
                    vert, 0.0, vert, time_remaining, 0.0, 0.0,
                )
                depth += vert
                time_remaining = 0.0
                break
            # 穿越整层
            _append_segment(
                segments, k, layer, theta, depth, layer.bottom_depth_m,
                thickness, 0.0, thickness, time_to_boundary, 0.0, 0.0,
            )
            depth = layer.bottom_depth_m
            time_remaining -= time_to_boundary
        else:
            tan_theta = sin_theta / cos_theta
            time_to_boundary = (
                thickness / (layer.speed_m_s * cos_theta)
            )
            if time_remaining <= time_to_boundary + DEPTH_EPS / layer.speed_m_s:
                # 射线在当前层内结束
                path_len = layer.speed_m_s * time_remaining
                vert = path_len * cos_theta
                horiz = path_len * sin_theta
                h_n = horiz * np.cos(heading)
                h_e = horiz * np.sin(heading)
                _append_segment(
                    segments, k, layer, theta, depth, depth + vert,
                    vert, horiz, path_len, time_remaining, h_n, h_e,
                )
                depth += vert
                total_hn += h_n
                total_he += h_e
                time_remaining = 0.0
                break

            # 穿越整层到下一界面
            path_len = thickness / cos_theta
            horiz = thickness * tan_theta
            seg_time = path_len / layer.speed_m_s
            h_n = horiz * np.cos(heading)
            h_e = horiz * np.sin(heading)
            _append_segment(
                segments, k, layer, theta, depth, layer.bottom_depth_m,
                thickness, horiz, path_len, seg_time, h_n, h_e,
            )
            depth = layer.bottom_depth_m
            total_hn += h_n
            total_he += h_e
            time_remaining -= seg_time

        # 已越过最后一个已知层但仍有剩余时间：不虚构更深层速度
        if k + 1 >= len(layers):
            return TraceResult(
                status=STATUS_BEYOND_PROFILE,
                reason=(
                    f"射线在 {depth:.6f} m 处耗尽已知声速剖面"
                    f"（最大深度 {last_bottom:.6f} m），仍有 "
                    f"{time_remaining:.9e} s 单程时间无速度可用"
                ),
                one_way_time_s=one_way_time_s,
                start_depth_m=start_depth_m,
                end_depth_m=depth,
                total_horizontal_north_m=total_hn,
                total_horizontal_east_m=total_he,
                ray_horizontal_heading_rad=heading,
                snell_p=snell_p,
                segments=segments,
            )
        k += 1

    return TraceResult(
        status=STATUS_OK,
        reason=None,
        one_way_time_s=one_way_time_s,
        start_depth_m=start_depth_m,
        end_depth_m=depth,
        total_horizontal_north_m=total_hn,
        total_horizontal_east_m=total_he,
        ray_horizontal_heading_rad=heading,
        snell_p=snell_p,
        segments=segments,
    )


def _append_segment(
    segments: list[PathSegment],
    k: int,
    layer: SoundSpeedLayer,
    theta: float,
    entry: float,
    exit_: float,
    vert: float,
    horiz: float,
    path_len: float,
    seg_time: float,
    h_n: float,
    h_e: float,
) -> None:
    segments.append(
        PathSegment(
            layer_index=k,
            top_depth_m=layer.top_depth_m,
            bottom_depth_m=layer.bottom_depth_m,
            speed_m_s=layer.speed_m_s,
            incidence_angle_rad=theta,
            entry_depth_m=entry,
            exit_depth_m=exit_,
            vertical_length_m=vert,
            horizontal_length_m=horiz,
            path_length_m=path_len,
            travel_time_s=seg_time,
            horizontal_offset_north_m=h_n,
            horizontal_offset_east_m=h_e,
        )
    )
