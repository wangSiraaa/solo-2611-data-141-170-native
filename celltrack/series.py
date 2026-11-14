"""series.json parsing, frame loading and input validation.

The loader is deliberately strict about *reporting* anomalies and relaxed about
*recovering* from them: every problem is surfaced as an :class:`Anomaly`
record while the frames that are individually valid continue to be processed.
Frame ordering is derived from ``time_minutes`` (ties broken deterministically by
``frame_id``), so reordering the ``frames`` array or the directory listing does
not change any conclusion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


@dataclass
class Anomaly:
    kind: str            # stable machine-readable code
    message: str         # human-readable explanation
    frame_id: str | None = None
    severity: str = "warning"   # "warning" | "error"


@dataclass
class Frame:
    frame_id: str
    file: str
    time_minutes: float
    index: int                  # position in time-sorted order
    labels: np.ndarray          # 2-D integer label image (0 = background)
    anomalies: list[Anomaly] = field(default_factory=list)

    @property
    def label_ids(self) -> np.ndarray:
        u = np.unique(self.labels)
        return u[u != 0]


@dataclass
class Series:
    frames: list[Frame]
    pixel_size_um: tuple[float, float]
    background: int
    max_centroid_speed_um_per_min: float
    division_area_ratio_range: tuple[float, float]
    raw: dict[str, Any]
    anomalies: list[Anomaly] = field(default_factory=list)

    @property
    def image_shape(self) -> tuple[int, int] | None:
        return self.frames[0].labels.shape if self.frames else None


def _load_label_image(path: Path) -> tuple[np.ndarray | None, list[Anomaly]]:
    """Read a label TIFF, returning (array, anomalies). Array is None on failure."""
    anomalies: list[Anomaly] = []
    if not path.exists():
        anomalies.append(Anomaly("missing_file", f"file not found: {path}"))
        return None, anomalies
    try:
        arr = tifffile.imread(path)
    except Exception as exc:  # noqa: BLE001 - surface any reader failure
        anomalies.append(Anomaly("unreadable_file", f"cannot read {path}: {exc}"))
        return None, anomalies

    if arr.ndim != 2:
        anomalies.append(
            Anomaly("illegal_ndim", f"{path.name} has {arr.ndim} dimensions, expected 2")
        )
        return None, anomalies
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        anomalies.append(Anomaly("illegal_dimensions", f"{path.name} has zero extent {arr.shape}"))
        return None, anomalies
    if not np.issubdtype(arr.dtype, np.integer):
        anomalies.append(
            Anomaly("non_integer_labels", f"{path.name} dtype {arr.dtype} is not integer")
        )
        return None, anomalies
    if arr.min() < 0:
        anomalies.append(
            Anomaly("negative_labels", f"{path.name} contains negative label values")
        )
        return None, anomalies
    return arr.astype(np.int64, copy=False), anomalies


def parse_series(series_path: str | Path, base_dir: str | Path | None = None) -> Series:
    series_path = Path(series_path)
    base = Path(base_dir) if base_dir is not None else series_path.parent
    anomalies: list[Anomaly] = []

    with open(series_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    px = raw.get("pixel_size_um", [1.0, 1.0])
    if not (isinstance(px, (list, tuple)) and len(px) == 2 and all(float(v) > 0 for v in px)):
        anomalies.append(Anomaly("illegal_pixel_size", f"pixel_size_um invalid: {px!r}", severity="error"))
        px = [1.0, 1.0]
    background = int(raw.get("background", 0))
    if background != 0:
        anomalies.append(
            Anomaly("background_nonzero", f"background is {background}; only 0 is supported")
        )
    speed = float(raw.get("max_centroid_speed_um_per_min", 2.5))
    if speed <= 0:
        anomalies.append(Anomaly("illegal_speed", f"max_centroid_speed must be > 0, got {speed}", severity="error"))
        speed = 2.5
    ratio = raw.get("division_area_ratio_range", [0.65, 1.6])
    if not (isinstance(ratio, (list, tuple)) and len(ratio) == 2 and 0 < ratio[0] <= ratio[1]):
        anomalies.append(Anomaly("illegal_area_ratio", f"division_area_ratio_range invalid: {ratio!r}", severity="error"))
        ratio = [0.65, 1.6]

    entries = raw.get("frames", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("series.json has no frames")

    # --- duplicate frame identity / duplicate file detection -----------
    seen_ids: dict[str, int] = {}
    seen_files: dict[str, int] = {}
    parsed_entries: list[tuple[float, str, str, int]] = []  # (time, frame_id, file, src_index)
    for i, e in enumerate(entries):
        fid = str(e.get("frame_id", f"frame_{i}"))
        fname = str(e.get("file", ""))
        t = e.get("time_minutes")
        if t is None:
            anomalies.append(Anomaly("missing_time", f"frame entry {i} has no time_minutes", frame_id=fid, severity="error"))
            continue
        t = float(t)
        if fid in seen_ids:
            anomalies.append(
                Anomaly("duplicate_frame_id",
                        f"frame_id {fid!r} appears more than once (entries {seen_ids[fid]}, {i})",
                        frame_id=fid, severity="error")
            )
            continue
        if fname in seen_files:
            anomalies.append(
                Anomaly("duplicate_frame_file",
                        f"file {fname!r} referenced by multiple frame entries ({seen_files[fname]}, {i})",
                        frame_id=fid, severity="error")
            )
            continue
        seen_ids[fid] = i
        seen_files[fname] = i
        parsed_entries.append((t, fid, fname, i))

    # Deterministic ordering by time, then frame_id; listing order is irrelevant.
    parsed_entries.sort(key=lambda r: (r[0], r[1]))

    times = [r[0] for r in parsed_entries]
    for a, b in zip(times, times[1:]):
        if b == a:
            anomalies.append(
                Anomaly("duplicate_frame_time",
                        f"two frames share time {a} min; they cannot be ordered", severity="error")
            )

    frames: list[Frame] = []
    expected_shape: tuple[int, int] | None = None
    for idx, (t, fid, fname, src_i) in enumerate(parsed_entries):
        arr, load_anoms = _load_label_image(base / fname)
        for a in load_anoms:
            a.frame_id = fid
            anomalies.append(a)
        if arr is None:
            continue
        if expected_shape is None:
            expected_shape = arr.shape
        elif arr.shape != expected_shape:
            anomalies.append(
                Anomaly("illegal_dimensions",
                        f"{fname} shape {arr.shape} != first frame shape {expected_shape}",
                        frame_id=fid, severity="error")
            )
            continue
        frame = Frame(frame_id=fid, file=fname, time_minutes=t, index=idx, labels=arr)
        if frame.label_ids.size == 0:
            anomalies.append(
                Anomaly("empty_frame", f"{fid} ({fname}) contains no objects", frame_id=fid)
            )
        frames.append(frame)

    if not frames:
        detail = "; ".join(f"{a.kind}: {a.message}" for a in anomalies) or "no frame entries"
        raise ValueError(f"no valid frames could be loaded ({detail})")

    # Re-index densely after any drops so Frame.index matches list position.
    for i, fr in enumerate(frames):
        fr.index = i

    return Series(
        frames=frames,
        pixel_size_um=(float(px[0]), float(px[1])),
        background=background,
        max_centroid_speed_um_per_min=speed,
        division_area_ratio_range=(float(ratio[0]), float(ratio[1])),
        raw=raw,
        anomalies=anomalies,
    )
