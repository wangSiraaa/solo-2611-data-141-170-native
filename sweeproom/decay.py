"""Band-limited reverse-energy decay, gating, noise compensation and T20/T10.

The decay curve is the Schroeder reverse integral of the *noise-gated,
band-limited squared impulse response*:

    E(t)  = sum_{t'=t}^{t_cut} (e(t') - Nf)          (clamped at >= 0)
    L(t)  = 10 log10( E(t) / E(onset) )

Key choices, all reported per band:

* ``t_cut`` is the last time whose smoothed band energy is at least
  ``noise_gate_db`` above the deconvolved noise floor.  We never integrate the
  noise plateau beyond it.
* The reverse integral subtracts the constant noise energy density ``Nf``
  over the remaining samples.  This removes the plateau instead of integrating
  it; the resulting curve is bounded.
* A straight line is fitted over -5 .. -25 dB (T20) *only* when the cumulative
  noise clearance at the -25 dB end exceeds ``t20_clearance_min_db``.  The
  T20 number extrapolates that slope by a factor 2.4, so it needs a large
  margin.  Otherwise the reported fallback is the **observed** time for the
  -5 .. -15 dB drop (a 10 dB decay time, *not* extrapolated), and only when
  its clearance exceeds ``t10_clearance_min_db``.  Otherwise the band is
  reported as unavailable.  We never fit a line through a noise plateau.
* The residuals of every fit are reported.  A large residual flags a
  non-exponential decay and downgrades the verdict to "qualified".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import Params
from .noise import bandpass
from .sweep import InverseFilter


@dataclass
class DecayCurve:
    band: Tuple[float, float]
    times_s: np.ndarray
    level_db: np.ndarray  # noise-compensated reverse integral, 0 dB at onset
    raw_energy: np.ndarray  # noise-gated squared band response, for audit

    def decimation_indices(self, max_points: int = 4000) -> np.ndarray:
        n = self.times_s.size
        if n <= max_points:
            return np.arange(n)
        step = int(np.ceil(n / max_points))
        idx = np.arange(0, n, step)
        if idx[-1] != n - 1:
            idx = np.concatenate([idx, [n - 1]])
        return idx

    def decimated(self, max_points: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
        idx = self.decimation_indices(max_points)
        return self.times_s[idx], self.level_db[idx]


@dataclass
class BandDecay:
    band: Tuple[float, float]
    status: str  # "credible_t20" | "qualified_short_range" | "unavailable"
    status_reason: str
    noise_floor_rms: float
    noise_clearance_db_at_minus_5: Optional[float]
    noise_clearance_db_at_minus_15: Optional[float]
    noise_clearance_db_at_minus_25: Optional[float]
    dynamic_range_db: float
    time_origin_s: float  # absolute recording time of t=0 for this band (direct arrival)
    analysis_window_start_s: float
    analysis_window_end_s: float
    integration_cut_time_s: float
    gate_db: float
    t5_s: Optional[float]
    t15_s: Optional[float]
    t25_s: Optional[float]
    t20_s: Optional[float]
    t10_s: Optional[float]
    t20_fit_start_s: Optional[float]
    t20_fit_end_s: Optional[float]
    t20_fit_points: Optional[int]
    t20_residual_db: Optional[float]
    t20_slope_db_per_s: Optional[float]
    t10_fit_start_s: Optional[float]
    t10_fit_end_s: Optional[float]
    t10_residual_db: Optional[float]
    t10_slope_db_per_s: Optional[float]
    noise_sensitivity_t20_range_s: Optional[Tuple[float, float]]
    curve: DecayCurve

    def to_dict(self) -> dict:
        return {
            "band_hz": [float(self.band[0]), float(self.band[1])],
            "status": self.status,
            "status_reason": self.status_reason,
            "noise_floor_rms": float(self.noise_floor_rms),
            "noise_sensitivity_range_s": (
                None
                if self.noise_sensitivity_t20_range_s is None
                else [float(self.noise_sensitivity_t20_range_s[0]),
                      float(self.noise_sensitivity_t20_range_s[1])]
            ),
            "noise_sensitivity_statistic": (
                "T20" if self.status == "credible_t20"
                else "observed T10" if self.status == "qualified_short_range"
                else None
            ),
            "noise_clearance_db": {
                "at_minus_5_db_level": _maybe(self.noise_clearance_db_at_minus_5),
                "at_minus_15_db_level": _maybe(self.noise_clearance_db_at_minus_15),
                "at_minus_25_db_level": _maybe(self.noise_clearance_db_at_minus_25),
            },
            "dynamic_range_db": float(self.dynamic_range_db),
            "time_origin_s": float(self.time_origin_s),
            "analysis_window": {
                "start_s_relative_to_direct": float(self.analysis_window_start_s),
                "end_s_relative_to_direct": float(self.analysis_window_end_s),
                "integration_cut_time_s_relative_to_direct": float(self.integration_cut_time_s),
                "note": "band-decay times are relative to the direct arrival",
                "gate_db": float(self.gate_db),
            },
            "crossing_times_s": {
                "minus_5_db": _maybe(self.t5_s),
                "minus_15_db": _maybe(self.t15_s),
                "minus_25_db": _maybe(self.t25_s),
            },
            "t20": {
                "value_s": _maybe(self.t20_s),
                "fit_start_s": _maybe(self.t20_fit_start_s),
                "fit_end_s": _maybe(self.t20_fit_end_s),
                "fit_points": self.t20_fit_points,
                "residual_rms_db": _maybe(self.t20_residual_db),
                "slope_db_per_s": _maybe(self.t20_slope_db_per_s),
            },
            "t10_observed": {
                "value_s": _maybe(self.t10_s),
                "definition": "observed time for the -5..-15 dB drop (10 dB decay time, not extrapolated)",
                "fit_start_s": _maybe(self.t10_fit_start_s),
                "fit_end_s": _maybe(self.t10_fit_end_s),
                "residual_rms_db": _maybe(self.t10_residual_db),
                "slope_db_per_s": _maybe(self.t10_slope_db_per_s),
            },
        }


def _maybe(value: Optional[float]):
    return None if value is None else float(value)


def _first_crossing(level_db: np.ndarray, threshold: float) -> Optional[int]:
    idx = np.where(level_db <= threshold)[0]
    return int(idx[0]) if idx.size else None


def _fit_line(
    level_db: np.ndarray,
    start: int,
    stop: int,
    sample_rate: int,
    extrapolation_db: float,
) -> Optional[dict]:
    if stop <= start:
        return None
    t = np.arange(start, stop, dtype=np.float64) / sample_rate
    y = level_db[start:stop]
    if t.size < 3:
        return None
    design = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ coef
    residual = float(np.sqrt(np.mean((y - prediction) ** 2)))
    slope = float(coef[0])
    if slope >= 0.0:
        return {
            "slope": slope,
            "residual": residual,
            "start_s": float(t[0]),
            "end_s": float(t[-1]),
            "points": int(t.size),
            "value_s": None,
        }
    return {
        "slope": slope,
        "residual": residual,
        "start_s": float(t[0]),
        "end_s": float(t[-1]),
        "points": int(t.size),
        "value_s": float(-extrapolation_db / slope),
    }


def _clearance(
    remaining_signal: float, remaining_samples: int, noise_energy: float
) -> Optional[float]:
    if remaining_samples <= 0:
        return None
    remaining_noise = noise_energy * remaining_samples
    if remaining_noise <= 0.0:
        return None
    if remaining_signal <= 0.0:
        return float("-inf")
    return float(10.0 * np.log10(remaining_signal / remaining_noise))


def _reverse_integral(
    energy: np.ndarray,
    onset: int,
    floor: float,
    params: Params,
    sr: int,
    n_valid: int,
):
    """Core reverse-integral + gate for a pre-filtered band energy array.

    Returns a dict with the analysis segment, level curve and cut index so the
    main analysis and the noise-floor sensitivity sweep share one implementation.
    """
    smoothing = max(1, int(round(params.smoothing_ms * 1e-3 * sr)))
    if smoothing > 1:
        smooth = np.convolve(energy, np.ones(smoothing) / smoothing, mode="same")
    else:
        smooth = energy.copy()

    gate_factor = 10.0 ** (params.noise_gate_db / 10.0)
    threshold = floor * gate_factor
    above = np.where(smooth[onset:n_valid] > threshold)[0]
    cut = onset + int(above[-1]) if above.size else onset
    min_span = max(1, int(round(params.min_fit_span_s * sr)))
    cut = min(max(cut, onset + min_span), n_valid - 1)

    segment = energy[onset : cut + 1].copy()
    remaining_counts = np.arange(segment.size, 0, -1, dtype=np.float64)
    raw_reverse = np.cumsum(segment[::-1])[::-1]
    compensated = raw_reverse - floor * remaining_counts
    tiny = raw_reverse[0] * 1e-14 if raw_reverse[0] > 0 else 1e-300
    compensated = np.maximum(compensated, tiny)
    total = float(compensated[0])
    level = 10.0 * np.log10(compensated / total) if total > 0 else np.full(segment.size, -np.inf)
    return {
        "smooth": smooth,
        "cut": cut,
        "segment": segment,
        "compensated": compensated,
        "level": level,
    }


def _decay_numbers(energy, onset, floor, params, sr, n_valid) -> dict:
    core = _reverse_integral(energy, onset, floor, params, sr, n_valid)
    level = core["level"]
    compensated = core["compensated"]
    segment = core["segment"]

    t5_idx = _first_crossing(level, params.t20_interval_db[0])
    t15_idx = _first_crossing(level, params.t10_interval_db[1])
    t25_idx = _first_crossing(level, params.t20_interval_db[1])
    t35_idx = _first_crossing(level, -35.0)

    def clearance_at(idx):
        if idx is None:
            return None
        return _clearance(float(compensated[idx]), int(segment.size - idx), floor)

    return {
        "core": core,
        "t5_idx": t5_idx,
        "t15_idx": t15_idx,
        "t25_idx": t25_idx,
        "t35_idx": t35_idx,
        "clearance5": clearance_at(t5_idx),
        "clearance15": clearance_at(t15_idx),
        "clearance25": clearance_at(t25_idx),
        "clearance35": clearance_at(t35_idx),
    }


def analyze_band_decay(
    d: np.ndarray,
    band: Tuple[float, float],
    onset_lag: int,
    direct_lag: int,
    valid_lag: int,
    noise_band_energy: float,
    inv: InverseFilter,
    params: Params,
) -> BandDecay:
    sr = inv.sample_rate
    n_valid = max(1, min(valid_lag, d.size - 1) + 1)
    band_signal = bandpass(d[:n_valid], band, sr)
    energy = band_signal * band_signal

    onset = max(0, min(onset_lag, n_valid - 1))
    floor = float(max(noise_band_energy, 0.0))

    # All reported times are relative to the direct arrival so the decay curve,
    # the fit interval and the annotation share one unambiguous origin.
    time_origin = int(direct_lag)

    numbers = _decay_numbers(energy, onset, floor, params, sr, n_valid)
    core = numbers["core"]
    smooth = core["smooth"]
    cut = core["cut"]
    segment = core["segment"]
    level = core["level"]
    t5_idx = numbers["t5_idx"]
    t15_idx = numbers["t15_idx"]
    t25_idx = numbers["t25_idx"]
    clearance5 = numbers["clearance5"]
    clearance15 = numbers["clearance15"]
    clearance25 = numbers["clearance25"]

    # Instantaneous dynamic range: strongest smoothed band energy near onset
    # against the deconvolved noise floor.
    peak_end = min(n_valid, onset + int(0.02 * sr) + 1)
    peak_energy = float(np.max(smooth[onset:peak_end])) if onset < n_valid else 0.0
    dynamic_range = (
        10.0 * np.log10(peak_energy / floor)
        if peak_energy > 0 and floor > 0
        else float("inf")
    )

    # Fit spans measured in samples; require a real, non-degenerate span.
    min_span_s = params.min_fit_span_s
    t20_fit = None
    if (
        t5_idx is not None
        and t25_idx is not None
        and (t25_idx - t5_idx) > 0
        and (t25_idx - t5_idx) / sr >= min_span_s
    ):
        t20_fit = _fit_line(level, t5_idx, t25_idx + 1, sr, 60.0)

    t10_fit = None
    if (
        t5_idx is not None
        and t15_idx is not None
        and (t15_idx - t5_idx) > 0
        and (t15_idx - t5_idx) / sr >= min_span_s
    ):
        t10_fit = _fit_line(level, t5_idx, t15_idx + 1, sr, 30.0)

    # Decide the verdict.  T20 needs the *cumulative noise clearance* at -25 dB
    # to exceed the published threshold; otherwise it is not a credible
    # extrapolation and we must not quote it.
    have_t20_clearance = (
        clearance25 is not None and np.isfinite(clearance25)
        and clearance25 >= params.t20_clearance_min_db
    )
    have_t20_fit = t20_fit is not None and t20_fit["value_s"] is not None
    have_t20_residual = (
        t20_fit["residual"] <= params.fit_resid_max_db if t20_fit is not None else False
    )

    have_t10_clearance = (
        clearance15 is not None and np.isfinite(clearance15)
        and clearance15 >= params.t10_clearance_min_db
    )
    have_t10_fit = (
        t5_idx is not None
        and t15_idx is not None
        and (t15_idx - t5_idx) / sr >= params.min_fit_span_s
    )
    # The short-range value is the observed -5..-15 dB time; a linear fit of
    # that span is only used to check it is locally exponential.
    have_t10_residual = (
        t10_fit["residual"] <= params.fit_resid_max_db if t10_fit is not None else False
    )

    if have_t20_clearance and have_t20_fit and have_t20_residual:
        status = "credible_t20"
        reason = (
            "cumulative noise clearance at -25 dB level is "
            f"{clearance25:.1f} dB >= {params.t20_clearance_min_db:.1f} dB; "
            f"-5..-25 dB fit residual {t20_fit['residual']:.2f} dB <= "
            f"{params.fit_resid_max_db:.2f} dB"
        )
    elif have_t10_clearance and have_t10_fit and have_t10_residual:
        status = "qualified_short_range"
        reasons = []
        if not have_t20_clearance:
            reasons.append(
                "T20 not credible: cumulative noise clearance at -25 dB is "
                + ("undefined" if clearance25 is None else f"{clearance25:.1f} dB")
                + f" (< {params.t20_clearance_min_db:.1f} dB required by the 2.4x "
                "extrapolation)"
            )
        if not have_t20_fit:
            reasons.append("T20 fit interval (-5..-25 dB) not reached")
        if not have_t20_residual and t20_fit is not None:
            reasons.append("T20 fit residual exceeds threshold")
        obs = (t15_idx - t5_idx) / sr
        reasons.append(
            f"reported value is the observed -5..-15 dB decay time {obs:.3f} s "
            f"(not extrapolated), clearance {clearance15:.1f} dB, residual "
            f"{t10_fit['residual']:.2f} dB"
        )
        reason = "; ".join(reasons)
    else:
        status = "unavailable"
        reasons = []
        if clearance25 is None:
            reasons.append("no -25 dB crossing inside the noise-gated window")
        elif not have_t20_clearance:
            reasons.append(
                f"-25 dB clearance {clearance25:.1f} dB < "
                f"{params.t20_clearance_min_db:.1f} dB"
            )
        if clearance15 is None:
            reasons.append("no -15 dB crossing inside the noise-gated window")
        elif not have_t10_clearance:
            reasons.append(
                f"-15 dB clearance {clearance15:.1f} dB < "
                f"{params.t10_clearance_min_db:.1f} dB"
            )
        if t20_fit is None and t10_fit is None:
            reasons.append("no valid fit interval")
        if not reasons:
            reasons.append("insufficient decay range for any reported decay time")
        reason = "; ".join(reasons)

    # Do not expose T20 values unless they are credible; keep the fit for the
    # audit trail but mark the reported value as None.
    reported_t20 = t20_fit["value_s"] if status == "credible_t20" else None
    observed_t10 = (
        (t15_idx - t5_idx) / sr
        if status in ("credible_t20", "qualified_short_range")
        and t5_idx is not None
        and t15_idx is not None
        else None
    )

    # Noise-floor sensitivity: recompute the reported statistic with the floor
    # scaled by 0.5x and 2x.  This is NOT a statistical confidence interval; it
    # is a robustness read-out showing how strongly the reported value leans on
    # the floor estimate, and therefore how much the credible interval widens
    # under high noise.  Reported for credible T20 (T20 statistic) and for the
    # qualified short-range case (observed T10 statistic).
    sensitivity: Optional[Tuple[float, float]] = None
    if status in ("credible_t20", "qualified_short_range"):
        vals = []
        for factor in (0.5, 2.0):
            alt = _decay_numbers(energy, onset, floor * factor, params, sr, n_valid)
            if status == "credible_t20":
                a5, a25 = alt["t5_idx"], alt["t25_idx"]
                value = None
                if a5 is not None and a25 is not None and a25 > a5:
                    alt_fit = _fit_line(alt["core"]["level"], a5, a25 + 1, sr, 60.0)
                    if alt_fit is not None:
                        value = alt_fit["value_s"]
            else:
                a5, a15 = alt["t5_idx"], alt["t15_idx"]
                value = (a15 - a5) / sr if a5 is not None and a15 is not None and a15 > a5 else None
            if value is not None:
                vals.append(value)
        if len(vals) == 2:
            sensitivity = (min(vals), max(vals))
        elif vals:
            sensitivity = (min(vals), max(vals))

    # Curve samples are indexed from `onset`; convert to time since the direct
    # arrival so the curve, fit intervals and annotations share one origin.
    offset_s = (onset - time_origin) / sr
    times = np.arange(segment.size, dtype=np.float64) / sr + offset_s

    def rel_time(idx: Optional[int]) -> Optional[float]:
        if idx is None:
            return None
        return idx / sr + offset_s

    curve = DecayCurve(
        band=band, times_s=times, level_db=level, raw_energy=segment
    )

    return BandDecay(
        band=band,
        status=status,
        status_reason=reason,
        noise_floor_rms=float(np.sqrt(floor)),
        noise_clearance_db_at_minus_5=clearance5,
        noise_clearance_db_at_minus_15=clearance15,
        noise_clearance_db_at_minus_25=clearance25,
        dynamic_range_db=float(dynamic_range),
        time_origin_s=time_origin / sr,
        analysis_window_start_s=rel_time(onset),
        analysis_window_end_s=rel_time(cut),
        integration_cut_time_s=rel_time(cut),
        gate_db=float(params.noise_gate_db),
        t5_s=rel_time(t5_idx),
        t15_s=rel_time(t15_idx),
        t25_s=rel_time(t25_idx),
        t20_s=reported_t20,
        t10_s=observed_t10,
        t20_fit_start_s=(t20_fit["start_s"] + offset_s) if t20_fit else None,
        t20_fit_end_s=(t20_fit["end_s"] + offset_s) if t20_fit else None,
        t20_fit_points=t20_fit["points"] if t20_fit else None,
        t20_residual_db=t20_fit["residual"] if t20_fit else None,
        t20_slope_db_per_s=t20_fit["slope"] if t20_fit else None,
        t10_fit_start_s=(t10_fit["start_s"] + offset_s) if t10_fit else None,
        t10_fit_end_s=(t10_fit["end_s"] + offset_s) if t10_fit else None,
        t10_residual_db=t10_fit["residual"] if t10_fit else None,
        t10_slope_db_per_s=t10_fit["slope"] if t10_fit else None,
        noise_sensitivity_t20_range_s=sensitivity,
        curve=curve,
    )
