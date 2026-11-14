"""工件落盘：运行表 CSV、随机顺序 CSV、设计矩阵 CSV 与诊断 JSON。"""

from __future__ import annotations

import csv
import json
import os
from typing import Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        raise ValueError(f"拒绝写空 CSV: {path}")
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return fieldnames


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_json_default)
        fh.write("\n")


def _json_default(value: Any) -> Any:
    import numpy as np

    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"无法序列化类型 {type(value).__name__}")
