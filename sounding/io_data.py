"""CSV 输入读取。

所有数值列在读取时即检查有限性；时间列要求单调不减（重复时间允许，
插值模块会把重复采样当作精确命中处理）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _read_csv(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """读取带表头的 CSV，返回 (表头, 字符串行矩阵)。

    数值转换由各加载器自行完成，以便兼容 beam_id 这类字符串列。
    """
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"{path} 缺少数据行")
    header = lines[0].split(",")
    rows = [ln.split(",") for ln in lines[1:]]
    if any(len(r) != len(header) for r in rows):
        raise ValueError(f"{path} 存在列数与表头不一致的数据行")
    return header, rows


def _numeric(data: list[list[str]], columns: list[int], path: str | Path) -> np.ndarray:
    """把指定列转为 float 数组，并检查有限性。"""
    arr = np.asarray(
        [[float(r[c]) for c in columns] for r in data], dtype=float
    )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path} 含 NaN/Inf 或无法解析的数值")
    return arr


def _check_time_column(path: str | Path, times: np.ndarray) -> None:
    if np.any(np.diff(times) < 0):
        raise ValueError(f"{path} 的时间列必须单调不减")


@dataclass(frozen=True)
class BeamRecord:
    """beams.csv 的一行。"""

    beam_id: str
    time_ms: float
    angle_rad: float
    two_way_time_s: float


def load_beams(path: str | Path) -> list[BeamRecord]:
    """读取波束观测表。"""
    header, rows = _read_csv(path)
    required = ["beam_id", "time_ms", "across_track_angle_deg", "two_way_time_s"]
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"{path} 缺少列: {missing}")
    idx = {c: header.index(c) for c in required}
    id_col = idx["beam_id"]
    ids = [ln[id_col] for ln in rows]

    num = _numeric(
        rows,
        [idx["time_ms"], idx["across_track_angle_deg"], idx["two_way_time_s"]],
        path,
    )
    times = num[:, 0]
    # 波束为观测记录，时间允许任意顺序；单调性只要求于导航/潮位查找表
    return [
        BeamRecord(
            beam_id=bid,
            time_ms=float(t),
            angle_rad=float(np.deg2rad(a)),
            two_way_time_s=float(twt),
        )
        for bid, t, a, twt in zip(ids, times, num[:, 1], num[:, 2])
    ]


@dataclass(frozen=True)
class NavRecord:
    """navigation.csv 的一行：位置与姿态四元数 (w,x,y,z)。"""

    time_ms: np.ndarray
    north_m: np.ndarray
    east_m: np.ndarray
    down_m: np.ndarray
    quat_wxyz: np.ndarray  # shape (n, 4)


def load_navigation(path: str | Path) -> NavRecord:
    """读取导航/姿态表。"""
    header, rows = _read_csv(path)
    required = ["time_ms", "north_m", "east_m", "down_m", "qw", "qx", "qy", "qz"]
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"{path} 缺少列: {missing}")
    idx = {c: header.index(c) for c in required}
    num = _numeric(rows, [idx[c] for c in required], path)
    # num 列顺序与 required 一致
    times = num[:, 0]
    _check_time_column(path, times)
    quat = num[:, 4:8]
    return NavRecord(
        time_ms=times,
        north_m=num[:, 1],
        east_m=num[:, 2],
        down_m=num[:, 3],
        quat_wxyz=quat,
    )


@dataclass(frozen=True)
class TideRecord:
    """tide.csv 的一行，潮位向上为正。"""

    time_ms: np.ndarray
    tide_m: np.ndarray


def load_tide(path: str | Path) -> TideRecord:
    """读取潮位表。"""
    header, rows = _read_csv(path)
    for col in ("time_ms", "tide_m"):
        if col not in header:
            raise ValueError(f"{path} 缺少列: {col}")
    ti, vi = header.index("time_ms"), header.index("tide_m")
    num = _numeric(rows, [ti, vi], path)
    times = num[:, 0]
    _check_time_column(path, times)
    return TideRecord(time_ms=times, tide_m=num[:, 1])


@dataclass(frozen=True)
class SoundSpeedLayer:
    """一个水平常速层，深度自瞬时水面向下计（米）。"""

    top_depth_m: float
    bottom_depth_m: float
    speed_m_s: float


def load_layers(path: str | Path) -> list[SoundSpeedLayer]:
    """读取声速分层表，并检查层界合法性。

    非法情形（空表、非正厚度、非正声速、层间不连续/重叠/顺序颠倒）
    以 :class:`ValueError` 抛出，错误信息逐项列出，供流水线转成
    ``sound_speed_layers_invalid`` 失败原因。
    """
    header, rows = _read_csv(path)
    for col in ("top_depth_m", "bottom_depth_m", "speed_m_s"):
        if col not in header:
            raise ValueError(f"{path} 缺少列: {col}")
    ti = header.index("top_depth_m")
    bi = header.index("bottom_depth_m")
    si = header.index("speed_m_s")
    num = _numeric(rows, [ti, bi, si], path)

    layers: list[SoundSpeedLayer] = []
    errors: list[str] = []
    prev_bottom: float | None = None
    for row in range(num.shape[0]):
        top, bottom, speed = (
            float(num[row, 0]),
            float(num[row, 1]),
            float(num[row, 2]),
        )
        if not (bottom > top):
            errors.append(
                f"第 {row} 层层界非法: top={top}, bottom={bottom}，要求 bottom>top"
            )
        if speed <= 0:
            errors.append(f"第 {row} 层声速非正: speed={speed}")
        if prev_bottom is not None and abs(top - prev_bottom) > 1e-9:
            errors.append(
                f"第 {row} 层顶 {top} 与上一层底 {prev_bottom} 不连续"
            )
        prev_bottom = bottom
        layers.append(SoundSpeedLayer(top, bottom, speed))

    if not layers:
        errors.append("声速分层表为空")
    if errors:
        raise ValueError("; ".join(errors))
    return layers
