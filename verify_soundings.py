#!/usr/bin/env python3
"""测深结果独立复核脚本。

本脚本**不导入 sounding 包、不重新跑归位**，只读取流水线落盘的
``soundings_detail.json``，用其中记录的路径段与支持证据独立重算并核对：

1. 每段 path_length == speed * travel_time，且满足勾股
   (vertical, horizontal)；
2. 各段时间之和 == 声明的单程时间（往返时间的一半）；
3. 段间深度首尾相接，各段 Snell 量 sin(theta)/speed 全程守恒，
   并与声速界面两侧的折射关系一致；
4. 段水平分量按方位角分解后累加，与声明总偏移一致；
5. 坐标链：测点 NED = 换能器 NED + 总水平偏移，
   测点 down = 水面 down + 终止深度，水面 down = -潮位；
6. failed/undetermined 波束不得带合格测点证据冲突。

用法::

    python verify_soundings.py output/soundings_detail.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

TOL = 1e-9


class VerificationError(AssertionError):
    """单波束复核失败。"""


def _approx(a: float, b: float, rel: float = TOL, abs_tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


def verify_beam(beam: dict) -> list[str]:
    """复核一个波束记录，返回检查说明列表；失败抛 VerificationError。"""
    bid = beam["beam_id"]
    notes: list[str] = []

    if beam["status"] == "failed":
        if not beam["failure_reasons"]:
            raise VerificationError(f"{bid}: failed 但没有 failure_reasons")
        if beam["sounding_point_ned_m"] is not None:
            raise VerificationError(f"{bid}: failed 波束不应给出测点坐标")
        notes.append("失败波束，原因已记录，无测点坐标")
        return notes

    segments = beam["segments"]
    one_way = beam["one_way_time_s"]
    if one_way is None or one_way <= 0:
        raise VerificationError(f"{bid}: 单程时间非法 {one_way}")

    # ---- 1) 段内几何与时间 ----
    total_time = 0.0
    sum_h_n = 0.0
    sum_h_e = 0.0
    snell_values: list[float] = []
    prev_exit: float | None = None
    for k, seg in enumerate(segments):
        # path = speed * time
        expect_path = seg["speed_m_s"] * seg["travel_time_s"]
        if not _approx(seg["path_length_m"], expect_path, rel=1e-10):
            raise VerificationError(
                f"{bid} 段{k}: 路径长度 {seg['path_length_m']} != "
                f"speed*time {expect_path}"
            )
        # 勾股
        hyp = math.hypot(seg["vertical_length_m"], seg["horizontal_length_m"])
        if not _approx(hyp, seg["path_length_m"], rel=1e-10):
            raise VerificationError(f"{bid} 段{k}: 勾股关系不成立")
        # 深度与竖直长度
        if not _approx(
            seg["exit_depth_m"] - seg["entry_depth_m"],
            seg["vertical_length_m"],
            rel=1e-10,
        ):
            raise VerificationError(f"{bid} 段{k}: 进出深度差与竖直长度不符")
        # 段间首尾相接
        if prev_exit is not None and not _approx(prev_exit, seg["entry_depth_m"]):
            raise VerificationError(f"{bid} 段{k}: 与上一段深度不连续")
        prev_exit = seg["exit_depth_m"]

        total_time += seg["travel_time_s"]
        sum_h_n += seg["horizontal_offset_north_m"]
        sum_h_e += seg["horizontal_offset_east_m"]

        theta = seg["incidence_angle_rad"]
        # 水平分量 = path * sin(theta)，竖直 = path * cos(theta)
        if not _approx(
            seg["horizontal_length_m"],
            seg["path_length_m"] * math.sin(theta),
            rel=1e-9,
        ):
            raise VerificationError(f"{bid} 段{k}: 水平分量与入射角不符")
        if not _approx(
            seg["vertical_length_m"],
            seg["path_length_m"] * math.cos(theta),
            rel=1e-9,
        ):
            raise VerificationError(f"{bid} 段{k}: 竖直分量与入射角不符")
        # 方位角分解
        heading = beam["ray_horizontal_heading_rad"]
        if not _approx(
            seg["horizontal_offset_north_m"],
            seg["horizontal_length_m"] * math.cos(heading),
            rel=1e-9,
        ):
            raise VerificationError(f"{bid} 段{k}: 北向水平分解与方位角不符")
        if not _approx(
            seg["horizontal_offset_east_m"],
            seg["horizontal_length_m"] * math.sin(heading),
            rel=1e-9,
        ):
            raise VerificationError(f"{bid} 段{k}: 东向水平分解与方位角不符")
        snell_values.append(math.sin(theta) / seg["speed_m_s"])

    # ---- 2) 时间守恒 ----
    if beam["status"] == "ok":
        if not _approx(total_time, one_way, rel=1e-9, abs_tol=1e-12):
            raise VerificationError(
                f"{bid}: 段时间和 {total_time} != 单程时间 {one_way}"
            )
    else:
        # 未确定：段时间不得超过单程时间
        if total_time > one_way + 1e-12:
            raise VerificationError(f"{bid}: 未确定波束段时间超过单程时间")
    notes.append(f"段时间合计 {total_time:.9f} s（单程 {one_way:.9f} s）")

    # ---- 3) Snell 守恒与界面折射 ----
    if not snell_values:
        raise VerificationError(f"{bid}: 没有任何路径段")
    p_declared = beam["snell_p"]
    for k, p in enumerate(snell_values):
        if not _approx(p, p_declared, rel=1e-9, abs_tol=1e-12):
            raise VerificationError(
                f"{bid} 段{k}: Snell 量 {p} 与声明 {p_declared} 不一致"
            )
    # 界面折射：sin(theta_i)/c_i == sin(theta_j)/c_j
    for a, b in zip(segments, segments[1:]):
        pa = math.sin(a["incidence_angle_rad"]) / a["speed_m_s"]
        pb = math.sin(b["incidence_angle_rad"]) / b["speed_m_s"]
        if not _approx(pa, pb, rel=1e-9, abs_tol=1e-12):
            raise VerificationError(
                f"{bid}: 界面折射不满足 Snell 关系 ({pa} != {pb})"
            )
    notes.append(f"Snell 守恒量 p={p_declared:.9e} s/m，界面折射核对通过")

    # ---- 4) 水平偏移累加与 N/E 坐标链 ----
    point = beam["sounding_point_ned_m"]
    xducer = beam["transducer_ned_m"]
    if point is None or xducer is None:
        raise VerificationError(f"{bid}: 缺少换能器/测点坐标")
    if not _approx(point[0] - xducer[0], sum_h_n, rel=1e-9, abs_tol=1e-12):
        raise VerificationError(f"{bid}: 北向坐标链不闭合")
    if not _approx(point[1] - xducer[1], sum_h_e, rel=1e-9, abs_tol=1e-12):
        raise VerificationError(f"{bid}: 东向坐标链不闭合")
    notes.append("水平偏移累加与 N/E 坐标链闭合")

    # ---- 5) 深度链：down = -tide + depth_below_surface ----
    tide_value = beam["tide_interp"]["interpolated_value"][0]
    surface_down = -tide_value
    if not _approx(surface_down, beam["surface_down_m"], rel=1e-10):
        raise VerificationError(f"{bid}: 水面 down 与 -tide 不符")
    end_depth = beam["end_depth_below_surface_m"]
    if end_depth is None:
        raise VerificationError(f"{bid}: 缺少终止深度")
    if not _approx(segments[-1]["exit_depth_m"], end_depth, rel=1e-10):
        raise VerificationError(f"{bid}: 末段退出深度与终止深度不符")
    if not _approx(point[2], surface_down + end_depth, rel=1e-10):
        raise VerificationError(
            f"{bid}: down 坐标链不闭合 ({point[2]} != {surface_down}+{end_depth})"
        )
    # 起点深度 = 换能器 down - 水面 down
    start_depth = beam["start_depth_below_surface_m"]
    if not _approx(start_depth, xducer[2] - surface_down, rel=1e-10):
        raise VerificationError(f"{bid}: 起点深度链不闭合")
    if not _approx(segments[0]["entry_depth_m"], start_depth, rel=1e-10):
        raise VerificationError(f"{bid}: 首段入口深度与换能器深度不符")
    notes.append("深度链（潮位->水面->换能器->测点）闭合")

    if beam["status"] == "undetermined":
        notes.append(f"未确定波束：{'; '.join(beam['failure_reasons'])}")
    return notes


def main(argv: list[str] | None = None) -> int:
    path = Path(argv[0] if argv else sys.argv[1])
    payload = json.loads(path.read_text(encoding="utf-8"))
    beams = payload["beams"]
    n_fail = 0
    for beam in beams:
        try:
            notes = verify_beam(beam)
        except VerificationError as exc:
            n_fail += 1
            print(f"[FAIL] {exc}")
            continue
        tag = beam["status"].upper()
        print(f"[PASS] {beam['beam_id']:>4} [{tag:<12}] " + "；".join(notes))

    print(
        f"\n复核 {len(beams)} 个波束，失败 {n_fail} 个"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
