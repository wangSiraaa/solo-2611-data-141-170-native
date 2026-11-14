"""pytest 共享夹具：合成导航/潮位/波束/声速文件与配置。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sounding.config import Config


@pytest.fixture
def config_zero_lever() -> Config:
    return Config(
        lever_arm_body_m=np.zeros(3), max_interpolation_gap_ms=1500.0
    )


@pytest.fixture
def config_with_lever() -> Config:
    return Config(
        lever_arm_body_m=np.array([0.3, 0.1, 1.0]),
        max_interpolation_gap_ms=1500.0,
    )


def _write_csv(path: Path, header: list[str], rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    return path


@pytest.fixture
def make_files(tmp_path: Path):
    """生成一组输入文件的工厂。"""

    def _make(
        nav_rows=None,
        tide_rows=None,
        beam_rows=None,
        layer_rows=None,
        lever=(0.0, 0.0, 0.0),
        gap_ms=1500,
    ):
        nav_rows = (
            [
                (0, 0, 0, 0, 1, 0, 0, 0),
                (1000, 0, 0, 0, 1, 0, 0, 0),
            ]
            if nav_rows is None
            else nav_rows
        )
        tide_rows = (
            [(0, 0.0), (1000, 0.0)] if tide_rows is None else tide_rows
        )
        beam_rows = (
            [("V", 0, 0, 0.02)] if beam_rows is None else beam_rows
        )
        layer_rows = (
            [(0, 100, 1500)] if layer_rows is None else layer_rows
        )
        paths = {
            "nav": _write_csv(
                tmp_path / "navigation.csv",
                ["time_ms", "north_m", "east_m", "down_m",
                 "qw", "qx", "qy", "qz"],
                nav_rows,
            ),
            "tide": _write_csv(
                tmp_path / "tide.csv", ["time_ms", "tide_m"], tide_rows
            ),
            "beams": _write_csv(
                tmp_path / "beams.csv",
                ["beam_id", "time_ms", "across_track_angle_deg",
                 "two_way_time_s"],
                beam_rows,
            ),
            "speed": _write_csv(
                tmp_path / "sound_speed.csv",
                ["top_depth_m", "bottom_depth_m", "speed_m_s"],
                layer_rows,
            ),
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "lever_arm_body_m": list(lever),
                    "max_interpolation_gap_ms": gap_ms,
                }
            ),
            encoding="utf-8",
        )
        paths["config"] = cfg_path
        return paths

    return _make
