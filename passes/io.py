"""Load and validate the fixture inputs.

Every input is checked for shape and physical plausibility *before* any
propagation, so a malformed horizon cannot silently change the answer.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sgp4.api import Satrec

from .constants import (
    DEFAULT_BOUNDARY_TOLERANCE_S,
    DEFAULT_OBSERVATION_STEP_S,
    WGS84_A_KM,
    WGS84_E2,
)
from .errors import InputError
from .timeutils import jd_parts_from_datetime, parse_utc_iso


@dataclass(frozen=True)
class TleRecord:
    name: str
    line1: str
    line2: str
    satrec: Satrec = field(repr=False, compare=False)
    epoch_utc: str = ""


@dataclass(frozen=True)
class Station:
    station_id: str
    latitude_deg: float
    longitude_deg_east: float
    ellipsoid_height_m: float
    datum: str
    # Derived geometry, all kilometres / unit vectors.
    site_ecef_km: np.ndarray = field(repr=False)
    east: np.ndarray = field(repr=False)
    north: np.ndarray = field(repr=False)
    up: np.ndarray = field(repr=False)
    enu_basis: np.ndarray = field(repr=False)  # rows: east, north, up

    @property
    def ellipsoid_height_km(self) -> float:
        return self.ellipsoid_height_m / 1000.0


@dataclass(frozen=True)
class HorizonMask:
    azimuth_deg: np.ndarray = field(repr=False)
    minimum_elevation_deg: np.ndarray = field(repr=False)
    # Extended nodes: original nodes plus first node + 360 degrees.
    extended_azimuth_deg: np.ndarray = field(repr=False)
    extended_elevation_deg: np.ndarray = field(repr=False)

    def threshold_deg(self, azimuth_deg):
        """Periodic piecewise-linear threshold, vectorised over azimuth."""
        azimuth = np.mod(np.asarray(azimuth_deg, dtype=float), 360.0)
        values = np.interp(azimuth, self.extended_azimuth_deg, self.extended_elevation_deg)
        if np.ndim(values) == 0:
            return float(values)
        return values


@dataclass(frozen=True)
class Plan:
    start_utc: str
    end_utc: str
    start: Any  # datetime
    end: Any  # datetime
    start_jd: tuple[float, float]
    end_jd: tuple[float, float]
    duration_s: float
    boundary_tolerance_s: float
    observation_step_s: float
    azimuth_convention: str
    elevation_convention: str
    raw: dict


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise InputError(f"{name} must be finite, got {value!r}")
    return number


def load_tle(path: Path) -> TleRecord:
    try:
        lines = [line.rstrip("\n\r") for line in path.read_text(encoding="utf-8").splitlines()]
    except OSError as exc:
        raise InputError(f"cannot read TLE file {path}: {exc}") from exc
    lines = [line for line in lines if line.strip()]
    if len(lines) < 2:
        raise InputError(f"{path}: expected a name line plus TLE line 1 and line 2")
    # Accept either name + two lines, or just two lines.
    if len(lines) >= 3 and lines[1].startswith("1 "):
        name, line1, line2 = lines[0].strip(), lines[1], lines[2]
    elif lines[0].startswith("1 "):
        name, line1, line2 = "", lines[0], lines[1]
    else:
        raise InputError(f"{path}: could not identify TLE line 1")
    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise InputError(f"{path}: malformed TLE lines")
    if len(line1) != 69 or len(line2) != 69:
        raise InputError(
            f"{path}: TLE lines must be 69 characters, got {len(line1)} and {len(line2)}"
        )
    try:
        satrec = Satrec.twoline2rv(line1, line2)
    except Exception as exc:  # sgp4 raises a variety of errors
        raise InputError(f"{path}: sgp4 rejected the TLE: {exc}") from exc
    # Epoch as a readable UTC string for evidence.
    epoch_utc = _tle_epoch_string(satrec)
    return TleRecord(name=name, line1=line1, line2=line2, satrec=satrec, epoch_utc=epoch_utc)


def _tle_epoch_string(satrec: Satrec) -> str:
    """Render the SGP4 TLE epoch (UTC by task convention) as ISO-8601."""
    epoch_days = float(satrec.epochdays)
    year = int(satrec.epochyr)
    year += 1900 if year > 56 else 2000
    base = parse_utc_iso(f"{year:04d}-01-01T00:00:00Z")
    import datetime as _dt

    value = base + _dt.timedelta(days=epoch_days - 1.0)
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond:06d}Z"


def station_geometry(
    latitude_deg: float, longitude_deg_east: float, height_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """WGS84 site ECEF (km) and the exact ECEF->ENU basis rows.

    ``height_m`` is ellipsoidal height in metres; it is converted to km here.
    The basis is orthonormal, so ENU = R @ (r_ecef - site_ecef).
    """
    if not -90.0 <= latitude_deg <= 90.0:
        raise InputError("latitude_deg must lie in [-90, 90]")
    if not -360.0 <= longitude_deg_east <= 360.0:
        raise InputError("longitude_deg_east must lie in [-360, 360]")
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg_east)
    height_km = height_m / 1000.0

    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    n = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    site = np.array(
        [
            (n + height_km) * cos_lat * cos_lon,
            (n + height_km) * cos_lat * sin_lon,
            (n * (1.0 - WGS84_E2) + height_km) * sin_lat,
        ],
        dtype=float,
    )
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    basis = np.vstack([east, north, up])
    return site, east, north, up, basis


def load_station(path: Path) -> Station:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read station file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise InputError("station file must contain a JSON object")
    datum = raw.get("datum", "WGS84")
    if str(datum).upper() != "WGS84":
        raise InputError(f"only WGS84 is supported, got datum={datum!r}")
    lat = _require_number(raw.get("latitude_deg"), "latitude_deg")
    lon = _require_number(raw.get("longitude_deg_east"), "longitude_deg_east")
    height = _require_number(raw.get("ellipsoid_height_m"), "ellipsoid_height_m")
    site, east, north, up, basis = station_geometry(lat, lon, height)
    return Station(
        station_id=str(raw.get("station_id", "station")),
        latitude_deg=lat,
        longitude_deg_east=lon,
        ellipsoid_height_m=height,
        datum="WGS84",
        site_ecef_km=site,
        east=east,
        north=north,
        up=up,
        enu_basis=basis,
    )


def load_horizon(path: Path) -> HorizonMask:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise InputError("horizon CSV has no header")
            fields = {name.strip().lower(): name for name in reader.fieldnames}
            az_field = fields.get("azimuth_deg")
            el_field = fields.get("minimum_elevation_deg")
            if az_field is None or el_field is None:
                raise InputError(
                    "horizon CSV needs azimuth_deg and minimum_elevation_deg columns"
                )
            azimuth, elevation = [], []
            for row_number, row in enumerate(reader, start=2):
                try:
                    az_text = (row[az_field] or "").strip()
                    el_text = (row[el_field] or "").strip()
                except KeyError as exc:
                    raise InputError(f"horizon row {row_number}: {exc}") from exc
                if not az_text and not el_text:
                    continue
                try:
                    az = float(az_text)
                    el = float(el_text)
                except ValueError as exc:
                    raise InputError(
                        f"horizon row {row_number}: non-numeric value ({az_text!r}, {el_text!r})"
                    ) from exc
                if not (math.isfinite(az) and math.isfinite(el)):
                    raise InputError(f"horizon row {row_number}: non-finite value")
                azimuth.append(az)
                elevation.append(el)
    except OSError as exc:
        raise InputError(f"cannot read horizon file {path}: {exc}") from exc

    if len(azimuth) < 2:
        raise InputError("horizon needs at least two azimuth nodes")
    az = np.asarray(azimuth, dtype=float)
    el = np.asarray(elevation, dtype=float)

    # Normalise to [0, 360) and require a strictly increasing sequence.  A
    # final 360.0 node is accepted as a duplicate of 0.0 and dropped.
    az = np.mod(az, 360.0)
    order = np.argsort(az, kind="stable")
    if not np.array_equal(order, np.arange(len(az))):
        # Sorting is safe only if there are no duplicate azimuths, which the
        # check below enforces.
        az = az[order]
        el = el[order]
    if np.any(np.diff(az) <= 0.0):
        raise InputError("horizon azimuth nodes must be strictly increasing in [0,360)")
    if el.min() <= -90.0 or el.max() > 90.0:
        raise InputError("horizon elevation thresholds must lie in (-90, 90]")

    extended_az = np.concatenate([az, [az[0] + 360.0]])
    extended_el = np.concatenate([el, [el[0]]])
    return HorizonMask(
        azimuth_deg=az,
        minimum_elevation_deg=el,
        extended_azimuth_deg=extended_az,
        extended_elevation_deg=extended_el,
    )


def load_plan(path: Path) -> Plan:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read plan file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise InputError("plan file must contain a JSON object")

    start = parse_utc_iso(raw.get("start_utc", ""))
    end = parse_utc_iso(raw.get("end_utc", ""))
    duration = (end - start).total_seconds()
    if duration <= 0.0:
        raise InputError("plan end_utc must be strictly after start_utc")

    tolerance = _require_number(
        raw.get("boundary_tolerance_seconds", DEFAULT_BOUNDARY_TOLERANCE_S),
        "boundary_tolerance_seconds",
    )
    if tolerance <= 0.0 or tolerance > 60.0:
        raise InputError("boundary_tolerance_seconds must lie in (0, 60]")

    observation_step = _require_number(
        raw.get("observation_step_s", DEFAULT_OBSERVATION_STEP_S), "observation_step_s"
    )
    if observation_step <= 0.0:
        raise InputError("observation_step_s must be positive")

    if raw.get("refraction", "none") not in ("none", None):
        raise InputError("this solver only supports refraction=none")
    if str(raw.get("earth_rotation", "")).lower().find("gmst") < 0:
        raise InputError("plan must specify the README GMST earth-rotation model")

    return Plan(
        start_utc=raw.get("start_utc"),
        end_utc=raw.get("end_utc"),
        start=start,
        end=end,
        start_jd=jd_parts_from_datetime(start),
        end_jd=jd_parts_from_datetime(end),
        duration_s=duration,
        boundary_tolerance_s=tolerance,
        observation_step_s=observation_step,
        azimuth_convention=str(raw.get("azimuth_convention", "")),
        elevation_convention=str(raw.get("elevation_convention", "")),
        raw=raw,
    )
