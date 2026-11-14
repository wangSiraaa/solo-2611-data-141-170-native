"""配置加载与校验。

只承载数值型/结构性参数；``config.json`` 中的文字协议说明字段会被忽略。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    """运行配置。

    Attributes:
        lever_arm_body_m: 换能器相对船体参考点的固定杆臂，body-FRD 米。
        max_interpolation_gap_ms: 合法包围记录的最大时间间隔（毫秒）。
            精确命中采样时刻不受该门槛限制；只有需要插值时才检查间隔。
    """

    lever_arm_body_m: np.ndarray
    max_interpolation_gap_ms: float

    @property
    def lever_arm(self) -> np.ndarray:
        """杆臂向量副本，shape (3,)。"""
        return np.array(self.lever_arm_body_m, dtype=float)


def load_config(path: str | Path) -> Config:
    """读取并校验 ``config.json``。

    Raises:
        ValueError: 缺少必要字段或字段不合法。
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if "lever_arm_body_m" not in raw:
        raise ValueError("config.json 缺少 lever_arm_body_m")
    lever = np.asarray(raw["lever_arm_body_m"], dtype=float)
    if lever.shape != (3,):
        raise ValueError(
            f"lever_arm_body_m 必须是长度 3 的数组，实际 shape={lever.shape}"
        )
    if not np.all(np.isfinite(lever)):
        raise ValueError("lever_arm_body_m 含非有限数值")

    gap = raw.get("max_interpolation_gap_ms")
    if gap is None:
        raise ValueError("config.json 缺少 max_interpolation_gap_ms")
    gap = float(gap)
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError("max_interpolation_gap_ms 必须为正数")

    return Config(lever_arm_body_m=lever, max_interpolation_gap_ms=gap)
