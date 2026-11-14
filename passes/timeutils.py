"""UTC <-> Julian-date helpers.

SGP4's ``jday`` supplies the two-part Julian date used by the propagator.
Keeping the integer/half-integer part separate avoids losing tens of
microseconds when seconds-of-day are added to a ~2.46e6 day number.
"""

from __future__ import annotations

import datetime as _dt
from typing import Union

import numpy as np
from sgp4.api import jday

from .constants import JD_J2000, SECONDS_PER_DAY
from .errors import InputError


def parse_utc_iso(text: str) -> _dt.datetime:
    """Parse an ISO-8601 timestamp as UTC.

    A trailing ``Z`` and explicit offsets are accepted.  A naive timestamp is
    interpreted as UTC because every input in this task is UTC.
    """
    if not isinstance(text, str) or not text.strip():
        raise InputError("empty or non-string UTC timestamp")
    raw = text.strip()
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    try:
        value = _dt.datetime.fromisoformat(raw)
    except ValueError as exc:
        raise InputError(f"invalid ISO-8601 timestamp {text!r}: {exc}") from exc
    if value.tzinfo is None:
        value = value.replace(tzinfo=_dt.timezone.utc)
    return value.astimezone(_dt.timezone.utc)


def format_utc(value: _dt.datetime) -> str:
    value = value.astimezone(_dt.timezone.utc)
    if value.microsecond:
        return value.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def jd_parts_from_datetime(value: _dt.datetime) -> tuple[float, float]:
    value = value.astimezone(_dt.timezone.utc)
    second = (
        value.second
        + value.microsecond / 1_000_000.0
        + (value.minute * 60.0)
        + (value.hour * 3600.0)
    )
    # jday takes hour/minute separately; passing all seconds in one field is
    # also supported and is the least lossy route.
    whole, fraction = jday(value.year, value.month, value.day, 0, 0, second)
    return float(whole), float(fraction)


def datetime_from_seconds(start: _dt.datetime, seconds: float) -> _dt.datetime:
    """Add a (possibly fractional) number of UTC seconds.

    Rounded to integer microseconds, the resolution of ``datetime``.
    """
    microseconds = int(round(float(seconds) * 1_000_000.0))
    return start + _dt.timedelta(microseconds=microseconds)


def iso_from_seconds(start: _dt.datetime, seconds: float) -> str:
    return format_utc(datetime_from_seconds(start, seconds))


def split_jd(
    start_whole: float,
    start_fraction: float,
    seconds: Union[float, np.ndarray],
) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Return the two-part Julian date at ``seconds`` after the start epoch."""
    seconds_arr = np.asarray(seconds, dtype=float)
    frac = start_fraction + seconds_arr / SECONDS_PER_DAY
    carry = np.floor(frac)
    whole = start_whole + carry
    fraction = frac - carry
    if seconds_arr.ndim == 0:
        return float(whole), float(fraction)
    return whole, fraction


def full_jd(
    start_whole: float,
    start_fraction: float,
    seconds: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    whole, fraction = split_jd(start_whole, start_fraction, seconds)
    return np.asarray(whole, dtype=float) + np.asarray(fraction, dtype=float)


def gmst_degrees(jd_whole: Union[float, np.ndarray], jd_fraction: Union[float, np.ndarray]):
    """README GMST polynomial, degrees in [0, 360).

    D is formed from the two JD parts to retain precision:
    ``D = (jd_whole - 2451545.0) + jd_fraction``.
    """
    whole = np.asarray(jd_whole, dtype=float)
    fraction = np.asarray(jd_fraction, dtype=float)
    d = (whole - JD_J2000) + fraction
    t = d / 36525.0
    theta = (
        280.46061837
        + 360.98564736629 * d
        + 0.000387933 * t * t
        - (t * t * t) / 38710000.0
    )
    theta = np.mod(theta, 360.0)
    if np.ndim(theta) == 0:
        return float(theta)
    return theta


def wrap180(angle_deg: Union[float, np.ndarray]):
    """Map an angle to [-180, 180)."""
    return (np.asarray(angle_deg, dtype=float) + 180.0) % 360.0 - 180.0


def wrap360(angle_deg: Union[float, np.ndarray]):
    """Map an angle to [0, 360)."""
    return np.mod(np.asarray(angle_deg, dtype=float), 360.0)
