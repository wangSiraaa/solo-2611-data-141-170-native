"""Plan / parameter models for the offline swept-sine room measurement.

Only the single case described by ``fixtures/plan.json`` is supported:
one known exponential sine sweep, one single-channel PCM16 recording,
fixed analysis bands, offline processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import json


@dataclass(frozen=True)
class SweepDefinition:
    kind: str
    duration_seconds: float
    start_hz: float
    end_hz: float
    fade_seconds: float
    amplitude: float

    @property
    def exponent_k(self) -> float:
        """Time constant K of the log sweep: f(t) = f1 * exp(t / K)."""
        import math

        return self.duration_seconds / math.log(self.end_hz / self.start_hz)


@dataclass(frozen=True)
class Plan:
    sample_rate: int
    sweep: SweepDefinition
    bands_hz: Tuple[Tuple[float, float], ...]
    recording_description: str = ""
    reference_delay_description: str = ""
    limits: str = ""
    raw: dict = field(default_factory=dict, compare=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        sr = int(data["sample_rate"])
        s = data["sweep"]
        sweep = SweepDefinition(
            kind=str(s["kind"]),
            duration_seconds=float(s["duration_seconds"]),
            start_hz=float(s["start_hz"]),
            end_hz=float(s["end_hz"]),
            fade_seconds=float(s["fade_seconds"]),
            amplitude=float(s["amplitude"]),
        )
        if sweep.kind not in ("exponential_sine", "log_sine", "logarithmic_sine_sweep"):
            raise ValueError(f"unsupported sweep kind: {sweep.kind!r}")
        if not (0 < sweep.start_hz < sweep.end_hz):
            raise ValueError("sweep must satisfy 0 < start_hz < end_hz")
        if sweep.end_hz >= sr / 2:
            raise ValueError("sweep end_hz must stay below Nyquist")
        bands = tuple((float(b[0]), float(b[1])) for b in data["bands_hz"])
        for lo, hi in bands:
            if not (0 < lo < hi):
                raise ValueError(f"invalid analysis band {lo}-{hi}")
        return cls(
            sample_rate=sr,
            sweep=sweep,
            bands_hz=bands,
            recording_description=str(data.get("recording", "")),
            reference_delay_description=str(data.get("reference_delay", "")),
            limits=str(data.get("limits", "")),
            raw=data,
        )

    @classmethod
    def load(cls, path: str) -> "Plan":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


@dataclass(frozen=True)
class Params:
    """All published decision thresholds live here.

    Every threshold is copied into the result JSON so a reviewer can see the
    exact rule that produced a credible / insufficient verdict.
    """

    # --- inverse filter / localisation -----------------------------------
    # Guard half-width added before the detected direct-arrival onset when the
    # analysis window opens.  The inverse-filter self-response has finite
    # width; the direct path must not be clipped.
    analysis_guard_ms: float = 5.0
    # Peak must exceed the strongest competitor by this much (outside the
    # direct cluster and outside predicted harmonic image lags).
    direct_peak_dominance_db: float = 6.0
    # Direct arrival must be this far above the deconvolved noise floor.
    direct_peak_snr_min_db: float = 20.0
    # Half-width of the window around each predicted harmonic-image lag that
    # is excluded from the linear window.  Harmonic images of a log sweep are
    # chirp-compressed and spread over a few ms, so this is wider than the
    # inverse filter's own self-response.
    harmonic_exclusion_halfwidth_ms: float = 5.0
    # Onset search: backward from the global peak until the envelope has been
    # below this level (relative to peak) for ``onset_hold_ms``.
    onset_threshold_db: float = -30.0
    onset_hold_ms: float = 1.0

    # --- noise floor ------------------------------------------------------
    # Control window as a fraction of the valid lag range, sampled from the
    # late tail where the room has decayed.
    noise_control_frac: Tuple[float, float] = (0.72, 0.97)
    # A sub-window is only accepted as the noise control window if its first
    # and second halves agree within this many dB.
    noise_stationarity_max_db: float = 3.0
    # Propaged-noise model seed count (deterministic seeds).
    noise_propagation_seeds: int = 5
    # If empirical and propagated floors disagree by more than this, warn.
    noise_floor_agreement_max_db: float = 6.0

    # --- decay / gating ---------------------------------------------------
    smoothing_ms: float = 10.0
    # t_cut: last time whose smoothed band energy is above the noise floor by
    # this gate.  We never integrate the noise plateau.
    noise_gate_db: float = 10.0
    # Cumulative noise clearance required at the weak end of the fit interval.
    # T20 extrapolates the -5..-25 dB slope by 2.4x, so it needs a large
    # margin: 20 dB means the remaining signal energy at -25 dB is 100x the
    # integrated noise energy over the rest of the gated window, i.e. noise
    # contributes <1% there.  The short-range -5..-15 dB fallback only
    # reports the *observed* 10 dB decay time, so it needs less margin.
    t20_clearance_min_db: float = 20.0
    t10_clearance_min_db: float = 12.0
    # T20 fit is taken over -5 .. -25 dB; the short-range fallback is the
    # observed time for the -5 .. -15 dB drop (not extrapolated).
    t20_interval_db: Tuple[float, float] = (-5.0, -25.0)
    t10_interval_db: Tuple[float, float] = (-5.0, -15.0)
    min_fit_span_s: float = 0.02
    # Residual of the straight-line fit; above this the curve is not a clean
    # exponential and the verdict becomes "qualified".
    fit_resid_max_db: float = 3.0

    # --- diagnostics ------------------------------------------------------
    # PCM16 sample magnitude at which we call a sample digitally full scale.
    full_scale_threshold: float = 32767.0 / 32768.0
    # Minimum fraction of sweep power that must fall in a requested band.
    min_band_excitation_fraction: float = 0.01
    # Band edges must be inside the excited sweep range by this margin (Hz).
    band_inside_tolerance_hz: float = 0.0

    # --- sweep-model cross-check -----------------------------------------
    sweep_model_min_correlation: float = 0.99

    def to_dict(self) -> dict:
        return {
            "analysis_guard_ms": self.analysis_guard_ms,
            "direct_peak_dominance_db": self.direct_peak_dominance_db,
            "direct_peak_snr_min_db": self.direct_peak_snr_min_db,
            "harmonic_exclusion_halfwidth_ms": self.harmonic_exclusion_halfwidth_ms,
            "onset_threshold_db": self.onset_threshold_db,
            "onset_hold_ms": self.onset_hold_ms,
            "noise_control_frac": list(self.noise_control_frac),
            "noise_stationarity_max_db": self.noise_stationarity_max_db,
            "noise_propagation_seeds": self.noise_propagation_seeds,
            "noise_floor_agreement_max_db": self.noise_floor_agreement_max_db,
            "smoothing_ms": self.smoothing_ms,
            "noise_gate_db": self.noise_gate_db,
            "t20_clearance_min_db": self.t20_clearance_min_db,
            "t10_clearance_min_db": self.t10_clearance_min_db,
            "t20_interval_db": list(self.t20_interval_db),
            "t10_interval_db": list(self.t10_interval_db),
            "min_fit_span_s": self.min_fit_span_s,
            "fit_resid_max_db": self.fit_resid_max_db,
            "full_scale_threshold": self.full_scale_threshold,
            "min_band_excitation_fraction": self.min_band_excitation_fraction,
            "band_inside_tolerance_hz": self.band_inside_tolerance_hz,
            "sweep_model_min_correlation": self.sweep_model_min_correlation,
        }
