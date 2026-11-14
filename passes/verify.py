"""Independent scalar model used to cross-check the vectorised chain.

This module deliberately re-implements the same physics *differently*:

* a fresh ``Satrec`` is propagated one timestamp at a time with ``sgp4.jd/fr``;
* GMST, the WGS84 site vector, the rotation and the ENU projection are written
  out explicitly rather than as matrix products;
* the periodic horizon is interpolated with a hand-written bracketing loop
  instead of ``numpy.interp``.

Agreement between the two implementations is evidence that the published
numbers do not depend on one code path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import brentq
from sgp4.api import Satrec

from .constants import (
    JD_J2000,
    SECONDS_PER_DAY,
    WGS84_A_KM,
    WGS84_E2,
    WGS84_F,
)
from .io import HorizonMask, Station, TleRecord
from .timeutils import iso_from_seconds


@dataclass
class ScalarGeometry:
    seconds: float
    jd_whole: float
    jd_fraction: float
    gmst_deg: float
    teme_km: tuple[float, float, float]
    ecef_km: tuple[float, float, float]
    enu_km: tuple[float, float, float]
    range_km: float
    azimuth_deg: float
    elevation_deg: float
    threshold_deg: float

    @property
    def gap_deg(self) -> float:
        return self.elevation_deg - self.threshold_deg

    @property
    def visible(self) -> bool:
        return self.gap_deg >= 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "seconds": self.seconds,
            "jd": self.jd_whole + self.jd_fraction,
            "gmst_deg": self.gmst_deg,
            "teme_km": list(self.teme_km),
            "ecef_km": list(self.ecef_km),
            "enu_km": list(self.enu_km),
            "range_km": self.range_km,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "threshold_deg": self.threshold_deg,
            "gap_deg": self.gap_deg,
            "visible": self.visible,
        }


def _gmst_deg(jd_whole: float, jd_fraction: float) -> float:
    d = (jd_whole - JD_J2000) + jd_fraction
    t = d / 36525.0
    theta = (
        280.46061837
        + 360.98564736629 * d
        + 0.000387933 * t * t
        - (t * t * t) / 38710000.0
    )
    theta = math.fmod(theta, 360.0)
    if theta < 0.0:
        theta += 360.0
    return theta


def _wgs84_site(lat_deg: float, lon_deg: float, height_km: float) -> tuple[float, ...]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + height_km) * cos_lat * cos_lon
    y = (n + height_km) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + height_km) * sin_lat
    return x, y, z, sin_lat, cos_lat, sin_lon, cos_lon


def _horizon_threshold(az_deg: float, mask: HorizonMask) -> float:
    """Hand-written periodic linear interpolation, no numpy.interp."""
    az = math.fmod(az_deg, 360.0)
    if az < 0.0:
        az += 360.0
    nodes = mask.azimuth_deg
    values = mask.minimum_elevation_deg
    # Search the closed interval [node_i, node_{i+1}] over the extended list.
    for i in range(len(nodes) - 1):
        left, right = nodes[i], nodes[i + 1]
        if left <= az < right:
            span = right - left
            return values[i] + (values[i + 1] - values[i]) * (az - left) / span
    # Last segment wraps through 360 to the first node.
    last = nodes[-1]
    span = 360.0 - last + nodes[0]
    if az >= last:
        frac = (az - last) / span
        return values[-1] + (values[0] - values[-1]) * frac
    frac = (az + 360.0 - last) / span
    return values[-1] + (values[0] - values[-1]) * frac


class IndependentModel:
    """Scalar reference model bound to the same inputs."""

    def __init__(self, tle: TleRecord, station: Station, mask: HorizonMask, start_jd, start_utc):
        # Re-parse from strings so the reference does not reuse the solver's
        # Satrec object.
        self.satrec = Satrec.twoline2rv(tle.line1, tle.line2)
        self.mask = mask
        self.start_jd = (float(start_jd[0]), float(start_jd[1]))
        self.start_utc = start_utc
        self.site = _wgs84_site(
            station.latitude_deg, station.longitude_deg_east, station.ellipsoid_height_m / 1000.0
        )
        self.station = station

    def evaluate(self, seconds: float) -> ScalarGeometry:
        frac = self.start_jd[1] + float(seconds) / SECONDS_PER_DAY
        carry = math.floor(frac)
        whole = self.start_jd[0] + carry
        fraction = frac - carry

        error, position, _ = self.satrec.sgp4(whole, fraction)
        if error != 0:
            raise RuntimeError(
                f"independent SGP4 error code {error} at {iso_from_seconds(self.start_utc, seconds)}"
            )
        x, y, z = (float(value) for value in position)

        theta = math.radians(_gmst_deg(whole, fraction))
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        ecef_x = cos_t * x + sin_t * y
        ecef_y = -sin_t * x + cos_t * y
        ecef_z = z

        sx, sy, sz, sin_lat, cos_lat, sin_lon, cos_lon = self.site
        dx = ecef_x - sx
        dy = ecef_y - sy
        dz = ecef_z - sz

        # Explicit local basis rows.
        east = (-sin_lon) * dx + (cos_lon) * dy + 0.0 * dz
        north = (
            (-sin_lat * cos_lon) * dx
            + (-sin_lat * sin_lon) * dy
            + (cos_lat) * dz
        )
        up = (cos_lat * cos_lon) * dx + (cos_lat * sin_lon) * dy + sin_lat * dz

        rng = math.sqrt(dx * dx + dy * dy + dz * dz)
        azimuth = math.degrees(math.atan2(east, north))
        azimuth = math.fmod(azimuth, 360.0)
        if azimuth < 0.0:
            azimuth += 360.0
        elevation = math.degrees(math.atan2(up, math.hypot(east, north)))
        threshold = _horizon_threshold(azimuth, self.mask)

        return ScalarGeometry(
            seconds=float(seconds),
            jd_whole=whole,
            jd_fraction=fraction,
            gmst_deg=_gmst_deg(whole, fraction),
            teme_km=(x, y, z),
            ecef_km=(ecef_x, ecef_y, ecef_z),
            enu_km=(east, north, up),
            range_km=rng,
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            threshold_deg=threshold,
        )


def enu_axis_checks(station: Station, scale_km: float = 1000.0) -> dict[str, Any]:
    """Verify the ENU basis using known station axial vectors.

    A displacement of ``scale_km`` along up/east/north from the site must map
    to (0,0,s)/(s,0,0)/(0,s,0) in ENU.  The ellipsoid normal at the site must
    equal the up vector, and east x north must equal up.
    """
    site = station.site_ecef_km
    basis = station.enu_basis

    def enu_of(point: np.ndarray) -> list[float]:
        return [float(v) for v in basis @ (point - site)]

    up_point = site + scale_km * station.up
    east_point = site + scale_km * station.east
    north_point = site + scale_km * station.north

    # Ellipsoid normal at the *surface point below the station*: gradient of
    # x^2/a^2 + y^2/a^2 + z^2/b^2 = 1, i.e. (x0/a^2, y0/a^2, z0/b^2), with
    # b = a(1-f).  Evaluating at the elevated site would tilt the gradient by
    # ~1.5e-7 for a 300 m height, so use the h=0 construction explicitly.
    sin_lat = math.sin(math.radians(station.latitude_deg))
    cos_lat = math.cos(math.radians(station.latitude_deg))
    sin_lon = math.sin(math.radians(station.longitude_deg_east))
    cos_lon = math.cos(math.radians(station.longitude_deg_east))
    b_km = WGS84_A_KM * (1.0 - WGS84_F)
    n_surface = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x0 = n_surface * cos_lat * cos_lon
    y0 = n_surface * cos_lat * sin_lon
    z0 = n_surface * (1.0 - WGS84_E2) * sin_lat
    normal = np.array(
        [
            x0 / (WGS84_A_KM * WGS84_A_KM),
            y0 / (WGS84_A_KM * WGS84_A_KM),
            z0 / (b_km * b_km),
        ]
    )
    normal /= np.linalg.norm(normal)

    checks = {
        "site_ecef_km": [float(v) for v in site],
        "site_norm_km": float(np.linalg.norm(site)),
        "east_axis_ecef": [float(v) for v in station.east],
        "north_axis_ecef": [float(v) for v in station.north],
        "up_axis_ecef": [float(v) for v in station.up],
        "orthonormality_max_abs_error": float(np.max(np.abs(basis @ basis.T - np.eye(3)))),
        "up_displacement_enu": enu_of(up_point),
        "east_displacement_enu": enu_of(east_point),
        "north_displacement_enu": enu_of(north_point),
        "east_cross_north_minus_up": [float(v) for v in np.cross(station.east, station.north) - station.up],
        "ellipsoid_normal_minus_up": [float(v) for v in normal - station.up],
        "up_dot_radial_positive": bool(float(np.dot(station.up, site / np.linalg.norm(site))) > 0.0),
        "east_has_zero_z": bool(abs(float(station.east[2])) <= 1e-15),
    }
    # Convert vector residual lists to a tidy scalar where useful.
    checks["east_cross_north_minus_up_norm"] = float(
        np.linalg.norm(np.cross(station.east, station.north) - station.up)
    )
    return checks


def independent_root(
    model: "IndependentModel", lo: float, hi: float, xtol: float = 1e-6
) -> float:
    """Re-solve a visibility crossing using only the scalar reference model."""
    f_lo = model.evaluate(lo).gap_deg
    f_hi = model.evaluate(hi).gap_deg
    if f_lo == 0.0:
        return float(lo)
    if f_hi == 0.0:
        return float(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(f"no sign change in bracket [{lo}, {hi}]")
    return float(
        brentq(
            lambda s: model.evaluate(float(s)).gap_deg,
            float(lo),
            float(hi),
            xtol=xtol,
            rtol=4.0 * np.finfo(float).eps,
            maxiter=200,
        )
    )


def compare_scalar_to_vector(
    reference: IndependentModel,
    ephemeris,
    times: Sequence[float],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Compare independent scalar results with the vectorised kernel."""
    rows = []
    maxima = {
        "teme_km": 0.0,
        "ecef_km": 0.0,
        "enu_km": 0.0,
        "range_km": 0.0,
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "threshold_deg": 0.0,
    }
    mismatches: list[dict[str, Any]] = []
    for seconds in times:
        ref = reference.evaluate(float(seconds))
        vec = ephemeris.sample(float(seconds))
        az_diff = abs(((ref.azimuth_deg - vec.azimuth_deg + 180.0) % 360.0) - 180.0)
        diffs = {
            "teme_km": float(np.linalg.norm(np.array(ref.teme_km) - vec.teme_km)),
            "ecef_km": float(np.linalg.norm(np.array(ref.ecef_km) - vec.ecef_km)),
            "enu_km": float(np.linalg.norm(np.array(ref.enu_km) - vec.enu_km)),
            "range_km": abs(ref.range_km - vec.range_km),
            "azimuth_deg": az_diff,
            "elevation_deg": abs(ref.elevation_deg - vec.elevation_deg),
            "threshold_deg": abs(ref.threshold_deg - vec.threshold_deg),
        }
        for key, value in diffs.items():
            maxima[key] = max(maxima[key], value)
        row = {
            "seconds": float(seconds),
            "utc": iso_from_seconds(reference.start_utc, float(seconds)),
            "jd": ref.jd_whole + ref.jd_fraction,
            "vector_azimuth_deg": vec.azimuth_deg,
            "scalar_azimuth_deg": ref.azimuth_deg,
            "vector_elevation_deg": vec.elevation_deg,
            "scalar_elevation_deg": ref.elevation_deg,
            "vector_threshold_deg": vec.threshold_deg,
            "scalar_threshold_deg": ref.threshold_deg,
            "vector_visible": vec.visible,
            "scalar_visible": ref.visible,
            "diffs": diffs,
        }
        rows.append(row)
        if ref.visible != vec.visible:
            mismatches.append({"seconds": float(seconds), "type": "visibility_mismatch"})
        for key, limit in thresholds.items():
            if diffs[key] > limit:
                mismatches.append(
                    {"seconds": float(seconds), "type": "tolerance_exceeded", "metric": key, "value": diffs[key], "limit": limit}
                )
    return {"rows": rows, "max_abs_diffs": maxima, "mismatches": mismatches, "ok": not mismatches}
