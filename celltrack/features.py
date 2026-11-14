"""Per-object feature extraction with full provenance.

Each connected label in a frame becomes one observation.  The feature vector
contains position, area and interpretable morphology, all in physical units
(minutes are handled at the frame level, distances/areas in um).  Every
observation keeps ``frame_id`` and the original ``label`` so it can always be
traced back to the exact pixels in the source TIFF.

Morphology evidence used by the tracker
---------------------------------------
* ``area_um2``            -- pixel count * pixel area (calibrated)
* ``equiv_diameter_um``   -- 2*sqrt(area/pi), a size summary
* ``major_axis_um`` / ``minor_axis_um`` -- 4*sqrt(covariance eigenvalues);
  aspect ratio distinguishes elongated (dividing) shapes from round ones
* ``eccentricity``        -- elongation in [0, 1)
* ``circularity``         -- 4*pi*area/perimeter^2 in (0, 1]; drops for
  irregular/elongated shapes
* ``solidity``            -- area / convex-hull area; drops for concave shapes
* ``orientation_deg``     -- major-axis angle in image coordinates
* ``perimeter_um``        -- staircase perimeter (4-connected boundary edges)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .series import Series


@dataclass
class ObjectRecord:
    obs_id: str                 # stable within a series: "f<index>:l<label>"
    frame_id: str
    frame_index: int
    time_minutes: float
    file: str
    label: int
    area_px: int
    area_um2: float
    centroid_x_px: float
    centroid_y_px: float
    centroid_x_um: float
    centroid_y_um: float
    bbox_min_x: int
    bbox_min_y: int
    bbox_max_x: int
    bbox_max_y: int
    equiv_diameter_um: float
    major_axis_um: float
    minor_axis_um: float
    eccentricity: float
    circularity: float
    solidity: float
    orientation_deg: float
    perimeter_um: float
    pixel_size_um: tuple[float, float]
    warnings: list[str] = field(default_factory=list)

    # pixel provenance (row, col) -- kept in memory for audit / export
    pixels_rc: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def position_um(self) -> np.ndarray:
        return np.array([self.centroid_x_um, self.centroid_y_um], dtype=float)

    def pixel_lookup(self) -> np.ndarray:
        """Return the original (row, col) pixels occupied by this object."""
        if self.pixels_rc is None:
            raise ValueError("pixel provenance not retained")
        return self.pixels_rc


def _staircase_perimeter(mask: np.ndarray) -> int:
    """Number of 4-neighbour boundary edges of a binary mask (in pixels)."""
    perim = 0
    # add a 1-pixel background frame so image-border edges are counted
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    core = padded[1:-1, 1:-1]
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = padded[1 + dy:padded.shape[0] - 1 + dy, 1 + dx:padded.shape[1] - 1 + dx]
        perim += int(np.sum(core & ~shifted))
    return perim


def _shape_descriptors(mask: np.ndarray, area_um2: float, px: tuple[float, float]) -> dict:
    ys, xs = np.nonzero(mask)
    # Covariance in physical um; eigvals give axis variances.
    x_um = xs.astype(float) * px[0]
    y_um = ys.astype(float) * px[1]
    if x_um.size >= 3:
        cov = np.cov(np.vstack([x_um, y_um]))
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, 0.0, None)
        minor = float(4.0 * np.sqrt(evals[0]))
        major = float(4.0 * np.sqrt(evals[1]))
        if major > 1e-9:
            ecc = float(np.sqrt(max(0.0, 1.0 - (evals[0] / evals[1]))) ) if evals[1] > 1e-9 else 0.0
        else:
            ecc = 0.0
        # orientation of the major axis (eigenvector for largest eigenvalue)
        vy, vx = evecs[:, 1]
        orientation = float(np.degrees(np.arctan2(vy * px[1], vx * px[0]))) % 180.0
    else:
        minor = major = float(np.sqrt(max(area_um2, 0.0)))
        ecc = 0.0
        orientation = 0.0

    perim_px = _staircase_perimeter(mask)
    perim_um = float(perim_px * 0.5 * (px[0] + px[1]))
    if perim_um > 0:
        circularity = float(4.0 * np.pi * area_um2 / (perim_um ** 2))
        circularity = min(circularity, 1.0)
    else:
        circularity = 0.0

    # Solidity via convex hull of pixel centres (physical coordinates).
    solidity = 1.0
    if xs.size >= 3:
        pts = np.column_stack([x_um, y_um])
        try:
            hull = ConvexHull(pts)
            solidity = float(min(1.0, area_um2 / hull.volume)) if hull.volume > 1e-9 else 1.0
        except QhullError:
            solidity = 1.0
    return {
        "minor_axis_um": minor,
        "major_axis_um": major,
        "eccentricity": ecc,
        "orientation_deg": orientation,
        "perimeter_um": perim_um,
        "circularity": circularity,
        "solidity": solidity,
    }


def extract_features(series: Series, min_reliable_area_um2: float = 1.0) -> list[ObjectRecord]:
    px = series.pixel_size_um
    pixel_area = px[0] * px[1]
    records: list[ObjectRecord] = []
    for frame in series.frames:
        labels = frame.label_ids
        # Slicing by label directly (labels are small integers in fixtures, but a
        # generic bincount/unique path keeps this O(pixels) regardless).
        for lab in labels:
            mask = frame.labels == lab
            area_px = int(mask.sum())
            if area_px == 0:
                continue
            ys, xs = np.nonzero(mask)
            cx_px = float(xs.mean())
            cy_px = float(ys.mean())
            area_um2 = float(area_px * pixel_area)
            shape = _shape_descriptors(mask, area_um2, px)
            warnings: list[str] = []
            if area_um2 < min_reliable_area_um2:
                warnings.append("sliver_object")
            if area_px < 5:
                warnings.append("tiny_object")
            rec = ObjectRecord(
                obs_id=f"f{frame.index}:l{int(lab)}",
                frame_id=frame.frame_id,
                frame_index=frame.index,
                time_minutes=frame.time_minutes,
                file=frame.file,
                label=int(lab),
                area_px=area_px,
                area_um2=area_um2,
                centroid_x_px=cx_px,
                centroid_y_px=cy_px,
                centroid_x_um=cx_px * px[0],
                centroid_y_um=cy_px * px[1],
                bbox_min_x=int(xs.min()),
                bbox_min_y=int(ys.min()),
                bbox_max_x=int(xs.max()),
                bbox_max_y=int(ys.max()),
                equiv_diameter_um=float(2.0 * np.sqrt(area_um2 / np.pi)),
                pixel_size_um=px,
                pixels_rc=np.column_stack([ys, xs]).astype(np.int32),
                **shape,
            )
            records.append(rec)
    return records


def group_by_frame(records: list[ObjectRecord]) -> dict[int, list[ObjectRecord]]:
    out: dict[int, list[ObjectRecord]] = {}
    for r in records:
        out.setdefault(r.frame_index, []).append(r)
    for k in out:
        out[k].sort(key=lambda r: r.label)
    return out
