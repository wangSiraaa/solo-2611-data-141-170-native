"""Synthetic fixture generators for tests (numpy only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile


def disc(shape, cx, cy, r=4.5):
    """Integer-pixel disc label mask as a boolean array."""
    ys, xs = np.indices(shape)
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r


def ellipse(shape, cx, cy, ax, ay):
    ys, xs = np.indices(shape)
    return ((xs - cx) / ax) ** 2 + ((ys - cy) / ay) ** 2 <= 1.0


def write_series(tmp_path: Path, frames_masks, times, pixel_size=(0.5, 0.5),
                 speed=2.5, ratio=(0.65, 1.6), shape=(96, 128)):
    """Write frames_masks (list of dict label->bool mask) as TIFFs + series.json."""
    tmp_path = Path(tmp_path)
    entries = []
    for i, (masks, t) in enumerate(zip(frames_masks, times)):
        arr = np.zeros(shape, dtype=np.uint16)
        for lab, mask in masks.items():
            arr[mask] = lab
        fn = f"frame_{i:02d}.tif"
        tifffile.imwrite(tmp_path / fn, arr)
        entries.append({"frame_id": f"f{i}", "file": fn, "time_minutes": t})
    series = {
        "pixel_size_um": list(pixel_size),
        "background": 0,
        "frames": entries,
        "max_centroid_speed_um_per_min": speed,
        "division_area_ratio_range": list(ratio),
    }
    (tmp_path / "series.json").write_text(json.dumps(series), encoding="utf-8")
    return tmp_path / "series.json"
