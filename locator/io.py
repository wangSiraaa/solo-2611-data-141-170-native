"""输入读取与校验。

校验规则(全部不满足即拒绝,抛 InputError):
- 台站 id 唯一,坐标为有限值;
- 到时引用的台站必须存在;相位必须在 plan 的速度表中;
- 到时、标准差为有限值,标准差必须为正;
- pick_id 唯一;同一台站同一相位只允许一条 pick(重复 pick 身份拒绝);
- 每个相位的波速为正的有限值,且不同相位不得共用同一波速;
- 搜索盒各维下界 < 上界,网格步长与代价差为正。
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


class InputError(ValueError):
    """输入数据不满足模型假设。"""


@dataclass(frozen=True)
class Station:
    station_id: str
    x_km: float
    y_km: float
    depth_km: float


@dataclass(frozen=True)
class Arrival:
    pick_id: str
    station_id: str
    phase: str
    time_s: float
    sigma_s: float


@dataclass(frozen=True)
class Plan:
    velocities: dict[str, float]          # phase -> km/s
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    depth_bounds: tuple[float, float]
    origin_time_bounds: tuple[float, float]
    profile_grid_step_km: float
    profile_score_delta: float


def _finite(value: float, what: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise InputError(f"{what} 不是有限值: {value!r}")
    return v


def _bounds(pair, name: str) -> tuple[float, float]:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise InputError(f"{name} 应为 [下界, 上界]")
    lo = _finite(pair[0], f"{name} 下界")
    hi = _finite(pair[1], f"{name} 上界")
    if not lo < hi:
        raise InputError(f"{name} 下界必须小于上界: [{lo}, {hi}]")
    return (lo, hi)


def load_plan(path: str | Path) -> Plan:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"plan 不是合法 JSON: {exc}") from exc

    vel_raw = raw.get("velocity_km_s")
    if not isinstance(vel_raw, dict) or not vel_raw:
        raise InputError("plan.velocity_km_s 应为非空的 {相位: 波速} 表")
    velocities: dict[str, float] = {}
    for phase, v in vel_raw.items():
        phase = str(phase).strip()
        if not phase:
            raise InputError("相位名不能为空")
        v = _finite(v, f"相位 {phase} 波速")
        if v <= 0:
            raise InputError(f"相位 {phase} 波速必须为正: {v}")
        if phase in velocities:
            raise InputError(f"相位 {phase} 在速度表中重复")
        velocities[phase] = v
    # 不同相位不可共用一个波速
    seen: dict[float, str] = {}
    for phase, v in velocities.items():
        if v in seen:
            raise InputError(
                f"相位 {seen[v]} 与 {phase} 共用同一波速 {v} km/s,违反一相位一波速"
            )
        seen[v] = phase

    pb = raw.get("position_bounds_km")
    if not isinstance(pb, dict):
        raise InputError("plan.position_bounds_km 缺失")
    step = _finite(raw.get("profile_grid_step_km"), "profile_grid_step_km")
    if step <= 0:
        raise InputError("profile_grid_step_km 必须为正")
    delta = _finite(raw.get("profile_score_delta"), "profile_score_delta")
    if delta <= 0:
        raise InputError("profile_score_delta 必须为正")

    return Plan(
        velocities=velocities,
        x_bounds=_bounds(pb.get("x"), "position_bounds_km.x"),
        y_bounds=_bounds(pb.get("y"), "position_bounds_km.y"),
        depth_bounds=_bounds(pb.get("depth"), "position_bounds_km.depth"),
        origin_time_bounds=_bounds(raw.get("origin_time_bounds_seconds"),
                                   "origin_time_bounds_seconds"),
        profile_grid_step_km=step,
        profile_score_delta=delta,
    )


def load_stations(path: str | Path) -> dict[str, Station]:
    stations: dict[str, Station] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"station_id", "x_km", "y_km", "depth_km"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise InputError(f"stations 表缺少列,需要 {sorted(required)}")
        for lineno, row in enumerate(reader, start=2):
            sid = (row["station_id"] or "").strip()
            if not sid:
                raise InputError(f"stations 第 {lineno} 行: station_id 为空")
            if sid in stations:
                raise InputError(f"stations 第 {lineno} 行: 台站 {sid!r} 重复定义")
            try:
                st = Station(
                    station_id=sid,
                    x_km=_finite(row["x_km"], f"台站 {sid} x_km"),
                    y_km=_finite(row["y_km"], f"台站 {sid} y_km"),
                    depth_km=_finite(row["depth_km"], f"台站 {sid} depth_km"),
                )
            except ValueError as exc:
                raise InputError(f"stations 第 {lineno} 行: {exc}") from exc
            stations[sid] = st
    if not stations:
        raise InputError("stations 表为空")
    return stations


def load_arrivals(path: str | Path,
                  stations: dict[str, Station],
                  velocities: dict[str, float]) -> list[Arrival]:
    arrivals: list[Arrival] = []
    seen_pick_ids: set[str] = set()
    seen_station_phase: set[tuple[str, str]] = set()
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"pick_id", "station_id", "phase", "time_seconds", "sigma_seconds"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise InputError(f"arrivals 表缺少列,需要 {sorted(required)}")
        for lineno, row in enumerate(reader, start=2):
            pid = (row["pick_id"] or "").strip()
            sid = (row["station_id"] or "").strip()
            phase = (row["phase"] or "").strip()
            if not pid:
                raise InputError(f"arrivals 第 {lineno} 行: pick_id 为空")
            if pid in seen_pick_ids:
                raise InputError(f"arrivals 第 {lineno} 行: pick_id {pid!r} 重复")
            if (sid, phase) in seen_station_phase:
                raise InputError(
                    f"arrivals 第 {lineno} 行: 台站 {sid!r} 相位 {phase!r} 的 pick 重复"
                )
            if sid not in stations:
                raise InputError(
                    f"arrivals 第 {lineno} 行: 引用了不存在的台站 {sid!r}"
                )
            if phase not in velocities:
                raise InputError(
                    f"arrivals 第 {lineno} 行: 未知相位 {phase!r}"
                    f"(速度表只有 {sorted(velocities)})"
                )
            try:
                t = _finite(row["time_seconds"], f"pick {pid} 到时")
                sigma = _finite(row["sigma_seconds"], f"pick {pid} 标准差")
            except ValueError as exc:
                raise InputError(f"arrivals 第 {lineno} 行: {exc}") from exc
            if sigma <= 0:
                raise InputError(
                    f"arrivals 第 {lineno} 行: pick {pid} 标准差必须为正: {sigma}"
                )
            seen_pick_ids.add(pid)
            seen_station_phase.add((sid, phase))
            arrivals.append(Arrival(pid, sid, phase, t, sigma))
    if not arrivals:
        raise InputError("arrivals 表为空")
    return arrivals
