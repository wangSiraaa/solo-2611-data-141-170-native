"""Consistent TEME -> ECEF -> ENU -> (azimuth, elevation) geometry chain.

The model intentionally follows the task README:

* ``r_teme`` is SGP4 output in kilometres.
* ``theta = GMST(UTC)`` with UT1=UTC and zero polar motion.
* ECEF rotation::

      x_ecef =  cos(theta) * x_teme + sin(theta) * y_teme
      y_ecef = -sin(theta) * x_teme + cos(theta) * y_teme
      z_ecef =  z_teme

* ENU = R_enu @ (r_ecef - site_ecef), with the WGS84 site vector.
* Azimuth is degrees clockwise from true north, elevation is the geometric
  ENU elevation, and neither uses refraction.

All distances in this module are kilometres.  The station height conversion
from metres happens once in :mod:`passes.io`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sgp4.api import SatrecArray

from .errors import PropagationError
from .io import HorizonMask, Station
from .timeutils import gmst_degrees, iso_from_seconds, split_jd, wrap360

# Keep temporary arrays bounded; a day at 0.02 s is ~4.3M samples and would
# otherwise allocate several hundred MB.
_CHUNK = 200_000


@dataclass
class Geometry:
    """Per-sample geometry.  Arrays share one leading time dimension."""

    seconds: np.ndarray
    jd_whole: np.ndarray
    jd_fraction: np.ndarray
    jd: np.ndarray
    gmst_deg: np.ndarray
    teme_km: np.ndarray
    ecef_km: np.ndarray
    enu_km: np.ndarray
    range_km: np.ndarray
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    threshold_deg: np.ndarray
    gap_deg: np.ndarray  # elevation - threshold

    @property
    def visible(self) -> np.ndarray:
        return self.gap_deg >= 0.0

    def index(self, i: int) -> dict[str, Any]:
        return {
            "seconds": float(self.seconds[i]),
            "jd": float(self.jd[i]),
            "gmst_deg": float(self.gmst_deg[i]),
            "teme_km": [float(v) for v in self.teme_km[i]],
            "ecef_km": [float(v) for v in self.ecef_km[i]],
            "enu_km": [float(v) for v in self.enu_km[i]],
            "range_km": float(self.range_km[i]),
            "azimuth_deg": float(self.azimuth_deg[i]),
            "elevation_deg": float(self.elevation_deg[i]),
            "threshold_deg": float(self.threshold_deg[i]),
            "gap_deg": float(self.gap_deg[i]),
            "visible": bool(self.gap_deg[i] >= 0.0),
        }


@dataclass
class Sample:
    """Scalar geometry at one instant."""

    seconds: float
    jd: float
    gmst_deg: float
    teme_km: np.ndarray
    ecef_km: np.ndarray
    enu_km: np.ndarray
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
            "jd": self.jd,
            "gmst_deg": self.gmst_deg,
            "teme_km": [float(v) for v in self.teme_km],
            "ecef_km": [float(v) for v in self.ecef_km],
            "enu_km": [float(v) for v in self.enu_km],
            "range_km": self.range_km,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "threshold_deg": self.threshold_deg,
            "gap_deg": self.gap_deg,
            "visible": self.visible,
        }


class Ephemeris:
    """Vectorised SGP4 + rotation chain bound to one station and time range."""

    def __init__(self, satrec, station: Station, horizon: HorizonMask, start_datetime):
        self.satrec = satrec
        self.station = station
        self.horizon = horizon
        self.array = SatrecArray([satrec])
        self.jd0 = None  # set by bind_epoch
        self.start_datetime = start_datetime

    # ------------------------------------------------------------------ setup
    def bind_epoch(self, jd_whole: float, jd_fraction: float) -> None:
        self.jd0 = (float(jd_whole), float(jd_fraction))

    def _require_epoch(self):
        if self.jd0 is None:
            raise RuntimeError("Ephemeris.bind_epoch() must be called before evaluation")

    # -------------------------------------------------------------- propagation
    def _propagate(self, whole, fraction, seconds) -> np.ndarray:
        """Run SGP4 and locate every non-zero error code.

        SGP4 failures are never treated as invisible samples: the first
        offending timestamps are reported with both UTC and seconds-from-start
        so the caller can point at the exact instant.
        """
        seconds_arr = np.asarray(seconds, dtype=float)
        n = seconds_arr.size
        whole_arr = np.asarray(whole, dtype=float)
        fraction_arr = np.asarray(fraction, dtype=float)
        # SatrecArray requires broadcastable arrays.
        if whole_arr.ndim == 0:
            whole_arr = np.full(n, float(whole_arr))
        if fraction_arr.ndim == 0:
            fraction_arr = np.full(n, float(fraction_arr))

        positions = np.empty((n, 3), dtype=float)
        failures: list[dict[str, Any]] = []
        for begin in range(0, n, _CHUNK):
            end = min(begin + _CHUNK, n)
            errors, r, _ = self.array.sgp4(whole_arr[begin:end], fraction_arr[begin:end])
            codes = np.asarray(errors)[0]
            chunk_r = np.asarray(r)[0]
            bad = np.flatnonzero(codes != 0)
            for offset in bad:
                index = begin + int(offset)
                if len(failures) < 64:
                    failures.append(
                        {
                            "seconds_from_start_s": float(seconds_arr[index]),
                            "utc": iso_from_seconds(self.start_datetime, seconds_arr[index]),
                            "jd": float(whole_arr[index]) + float(fraction_arr[index]),
                            "sgp4_error_code": int(codes[offset]),
                        }
                    )
            positions[begin:end] = chunk_r
        if failures:
            raise PropagationError("SGP4 propagation failed", failures)
        if not np.all(np.isfinite(positions)):
            bad = np.flatnonzero(~np.all(np.isfinite(positions), axis=1))
            located = [
                {
                    "seconds_from_start_s": float(seconds_arr[i]),
                    "utc": iso_from_seconds(self.start_datetime, seconds_arr[i]),
                    "jd": float(whole_arr[i]) + float(fraction_arr[i]),
                    "sgp4_error_code": 0,
                    "reason": "non-finite position",
                }
                for i in bad[:64]
            ]
            raise PropagationError("SGP4 returned a non-finite position", located)
        return positions

    @property
    def start_datetime_utc(self):
        return self.start_datetime

    # -------------------------------------------------------------- evaluation
    def evaluate(self, seconds) -> Geometry:
        self._require_epoch()
        seconds_arr = np.atleast_1d(np.asarray(seconds, dtype=float))
        whole, fraction = split_jd(self.jd0[0], self.jd0[1], seconds_arr)
        whole = np.asarray(whole, dtype=float)
        fraction = np.asarray(fraction, dtype=float)
        jd = whole + fraction
        theta_deg = np.asarray(gmst_degrees(whole, fraction), dtype=float)
        theta = np.radians(theta_deg)

        teme = self._propagate(whole, fraction, seconds_arr)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x, y, z = teme[:, 0], teme[:, 1], teme[:, 2]
        ecef = np.empty_like(teme)
        ecef[:, 0] = cos_t * x + sin_t * y
        ecef[:, 1] = -sin_t * x + cos_t * y
        ecef[:, 2] = z

        delta = ecef - self.station.site_ecef_km[None, :]
        enu = delta @ self.station.enu_basis.T
        east, north, up = enu[:, 0], enu[:, 1], enu[:, 2]
        horizontal = np.hypot(east, north)
        range_km = np.sqrt(horizontal * horizontal + up * up)
        azimuth = wrap360(np.degrees(np.arctan2(east, north)))
        elevation = np.degrees(np.arctan2(up, horizontal))
        threshold = np.asarray(self.horizon.threshold_deg(azimuth), dtype=float)
        gap = elevation - threshold

        return Geometry(
            seconds=seconds_arr,
            jd_whole=whole,
            jd_fraction=fraction,
            jd=jd,
            gmst_deg=theta_deg,
            teme_km=teme,
            ecef_km=ecef,
            enu_km=enu,
            range_km=range_km,
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            threshold_deg=threshold,
            gap_deg=gap,
        )

    def sample(self, seconds: float) -> Sample:
        geometry = self.evaluate([float(seconds)])
        i = 0
        return Sample(
            seconds=float(geometry.seconds[i]),
            jd=float(geometry.jd[i]),
            gmst_deg=float(geometry.gmst_deg[i]),
            teme_km=geometry.teme_km[i].copy(),
            ecef_km=geometry.ecef_km[i].copy(),
            enu_km=geometry.enu_km[i].copy(),
            range_km=float(geometry.range_km[i]),
            azimuth_deg=float(geometry.azimuth_deg[i]),
            elevation_deg=float(geometry.elevation_deg[i]),
            threshold_deg=float(geometry.threshold_deg[i]),
        )

    def gap(self, seconds: float) -> float:
        """Scalar signed margin ``elevation - threshold`` at one instant."""
        return float(self.sample(seconds).gap_deg)

    def elevation(self, seconds: float) -> float:
        return float(self.sample(seconds).elevation_deg)

    def azimuth(self, seconds: float) -> float:
        return float(self.sample(seconds).azimuth_deg)
