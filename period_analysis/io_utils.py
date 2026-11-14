"""CSV 读取与数据质量校验。

重复 observation_id、非有限数、非正误差都按行定位（文件名 + 行号 + 编号 + 字段），
不静默丢弃；分析仅在通过校验的干净数据上进行。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field

import numpy as np

REQUIRED_COLUMNS = ("observation_id", "night_id", "time_day", "brightness", "sigma")
NUMERIC_COLUMNS = ("time_day", "brightness", "sigma")


@dataclass
class DataIssue:
    csv_row: int          # CSV 物理行号（含表头，首数据行为 2）
    observation_id: str
    night_id: str
    column: str
    kind: str             # duplicate_id | non_finite | non_positive_sigma
    detail: str

    def to_dict(self) -> dict:
        return {
            "csv_row": self.csv_row,
            "observation_id": self.observation_id,
            "night_id": self.night_id,
            "column": self.column,
            "kind": self.kind,
            "detail": self.detail,
        }


@dataclass
class LightCurve:
    source_csv: str
    observation_id: np.ndarray
    night_id: np.ndarray
    time: np.ndarray
    brightness: np.ndarray
    sigma: np.ndarray
    issues: list[DataIssue] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.issues

    @property
    def n_points(self) -> int:
        return self.time.size

    @property
    def time_span(self) -> float:
        return float(self.time.max() - self.time.min())

    def night_groups(self) -> dict[str, np.ndarray]:
        groups: dict[str, list[int]] = {}
        for i, nid in enumerate(self.night_id):
            groups.setdefault(str(nid), []).append(i)
        return {k: np.array(v, dtype=int) for k, v in groups.items()}


def load_lightcurve(path: str) -> LightCurve:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: 缺少必需列 {missing}")
        raw = list(reader)

    ids: list[str] = []
    nights: list[str] = []
    times: list[float] = []
    bright: list[float] = []
    sigmas: list[float] = []
    issues: list[DataIssue] = []
    seen_ids: set[str] = set()

    for offset, row in enumerate(raw):
        csv_row = offset + 2  # 表头占第 1 行
        oid = (row.get("observation_id") or "").strip()
        nid = (row.get("night_id") or "").strip()
        if oid in seen_ids:
            issues.append(DataIssue(csv_row, oid, nid, "observation_id",
                                    "duplicate_id", "observation_id 在文件中重复出现"))
        seen_ids.add(oid)
        parsed: dict[str, float] = {}
        for col in NUMERIC_COLUMNS:
            try:
                val = float(row[col])
            except (TypeError, ValueError):
                val = math.nan
            parsed[col] = val
            if not math.isfinite(val):
                issues.append(DataIssue(csv_row, oid, nid, col,
                                        "non_finite", f"值 {row.get(col)!r} 不是有限数"))
        if math.isfinite(parsed["sigma"]) and parsed["sigma"] <= 0.0:
            issues.append(DataIssue(csv_row, oid, nid, "sigma",
                                    "non_positive_sigma", f"sigma={parsed['sigma']} 必须为正"))
        ids.append(oid)
        nights.append(nid)
        times.append(parsed["time_day"])
        bright.append(parsed["brightness"])
        sigmas.append(parsed["sigma"])

    lc = LightCurve(
        source_csv=path,
        observation_id=np.array(ids, dtype=object),
        night_id=np.array(nights, dtype=object),
        time=np.array(times, dtype=float),
        brightness=np.array(bright, dtype=float),
        sigma=np.array(sigmas, dtype=float),
        issues=issues,
    )
    return lc
