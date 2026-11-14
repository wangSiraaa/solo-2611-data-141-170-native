"""jointreg: 多站位已知对应标靶的联合刚体配准工具。

约定（见 fixtures/anchor.json）::

    world_point = R @ local_point + t,   R 正交且 det(R) = +1

不做点云识别 / ICP / SLAM / 非刚体形变。
"""

from .geometry import (
    SO3_exp,
    SO3_log,
    weighted_kabsch,
    support_report,
)
from .pipeline import run_pipeline
from .io import load_observations, load_anchor

__all__ = [
    "SO3_exp",
    "SO3_log",
    "weighted_kabsch",
    "support_report",
    "run_pipeline",
    "load_observations",
    "load_anchor",
]
