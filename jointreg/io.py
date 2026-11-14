"""输入读取与数据结构。记录顺序不参与任何身份判定——身份只由 station_id/target_id 决定。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass

import numpy as np


@dataclass
class Observation:
    station: str
    target: str
    local: np.ndarray
    sigma: float
    holdout: bool
    row: int  # 原始行号（含表头从 1 起），仅用于报告追溯


@dataclass
class Dataset:
    observations: list[Observation]
    anchor_station: str
    units: str
    convention: str

    @property
    def stations(self) -> list[str]:
        return sorted({o.station for o in self.observations})

    @property
    def targets(self) -> list[str]:
        return sorted({o.target for o in self.observations})

    def fit_observations(self) -> list[Observation]:
        return [o for o in self.observations if not o.holdout]

    def holdout_observations(self) -> list[Observation]:
        return [o for o in self.observations if o.holdout]

    def by_station(self, station: str, holdouts: bool = False) -> list[Observation]:
        return [
            o
            for o in self.observations
            if o.station == station and (holdouts or not o.holdout)
        ]


def _parse_bool(text: str) -> bool:
    return text.strip().lower() in ("true", "1", "yes", "y")


def load_observations(csv_path: str) -> list[Observation]:
    out: list[Observation] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"station_id", "target_id", "x_m", "y_m", "z_m", "sigma_m", "holdout"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} 缺少列: {sorted(missing)}")
        for i, r in enumerate(reader, start=2):
            sigma = float(r["sigma_m"])
            if sigma <= 0:
                raise ValueError(f"第 {i} 行 sigma_m 必须为正数")
            out.append(
                Observation(
                    station=r["station_id"].strip(),
                    target=r["target_id"].strip(),
                    local=np.array(
                        [float(r["x_m"]), float(r["y_m"]), float(r["z_m"])]
                    ),
                    sigma=sigma,
                    holdout=_parse_bool(r["holdout"]),
                    row=i,
                )
            )
    if not out:
        raise ValueError(f"{csv_path} 不含任何观测")
    return out


def load_anchor(json_path: str) -> dict:
    with open(json_path, encoding="utf-8") as fh:
        anchor = json.load(fh)
    if "anchor_station" not in anchor:
        raise ValueError("anchor.json 缺少 anchor_station")
    return anchor


def load_dataset(csv_path: str, json_path: str) -> Dataset:
    obs = load_observations(csv_path)
    anchor = load_anchor(json_path)
    anchor_station = anchor["anchor_station"]
    stations = {o.station for o in obs}
    if anchor_station not in stations:
        raise ValueError(f"锚站 {anchor_station} 不在观测数据中")
    return Dataset(
        observations=obs,
        anchor_station=anchor_station,
        units=anchor.get("units", "metres"),
        convention=anchor.get(
            "transform_convention",
            "world_point = R @ local_point + t",
        ),
    )
