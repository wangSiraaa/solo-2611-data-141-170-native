"""Top-level analysis orchestration.

Given the plan, the emitted sweep, and one recording, this module produces:

* the linear impulse response in the lag domain (sampled time base);
* per-band decay curves and T20/T10 verdicts with evidence;
* diagnostics for full scale, excitation coverage and localisation ambiguity;
* a single JSON-serialisable result object used by the CLI and by ``verify``.

Everything here is deterministic and operates only on the supplied inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Params, Plan
from . import decay as decay_mod
from .io import Audio, describe_signal
from .localise import Localisation, localise
from .noise import NoiseModel, estimate_noise
from .sweep import (
    InverseFilter,
    build_inverse_filter,
    deconvolve,
    excitation_band_fraction,
    generate_sweep,
    sweep_phase_note,
    valid_lag_limit,
)


@dataclass
class SweepModelCheck:
    provided_matches_generated: bool
    correlation: float
    max_abs_diff: float
    rms_diff: float
    note: str

    def to_dict(self) -> dict:
        return {
            "provided_matches_generated": bool(self.provided_matches_generated),
            "correlation": float(self.correlation),
            "max_abs_diff": float(self.max_abs_diff),
            "rms_diff": float(self.rms_diff),
            "note": self.note,
        }


@dataclass
class Diagnostics:
    recording: dict
    excitation: dict
    sweep_model_check: SweepModelCheck
    full_scale_ok: bool
    full_scale_detail: str
    band_coverage: Dict[Tuple[float, float], dict]
    band_coverage_ok: bool
    localisation: dict
    localisation_ok: bool
    global_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "recording": self.recording,
            "excitation": self.excitation,
            "sweep_model_check": self.sweep_model_check.to_dict(),
            "full_scale_ok": bool(self.full_scale_ok),
            "full_scale_detail": self.full_scale_detail,
            "band_coverage": {
                f"{lo:g}-{hi:g}Hz": value for (lo, hi), value in self.band_coverage.items()
            },
            "band_coverage_ok": bool(self.band_coverage_ok),
            "localisation": self.localisation,
            "localisation_ok": bool(self.localisation_ok),
            "global_reasons": list(self.global_reasons),
            "warnings": list(self.warnings),
        }


@dataclass
class AnalysisResult:
    plan: Plan
    params: Params
    label: str
    recording_path: str
    recording_samples: np.ndarray
    deconvolution: np.ndarray
    inverse_filter: InverseFilter
    localisation: Localisation
    noise: NoiseModel
    bands: List[decay_mod.BandDecay]
    diagnostics: Diagnostics
    overall_status: str
    overall_reason: str

    def to_dict(self, include_curves: bool = False) -> dict:
        sr = self.plan.sample_rate
        out = {
            "label": self.label,
            "recording_path": self.recording_path,
            "plan": {
                "sample_rate": sr,
                "sweep": sweep_phase_note(self.plan.sweep, sr),
                "bands_hz": [list(b) for b in self.plan.bands_hz],
            },
            "params": self.params.to_dict(),
            "overall_status": self.overall_status,
            "overall_reason": self.overall_reason,
            "time_base": {
                "lag_zero_sample": 0,
                "samples_per_second": sr,
                "lag_time_s": [0.0, (self.localisation.valid_lag) / sr],
                "valid_lag_samples": int(self.localisation.valid_lag),
                "note": (
                    "Deconvolution is indexed by lag in samples from the start of "
                    "the recording. Lag 0 is the first recorded sample; the direct "
                    "arrival is at direct_lag_samples. Only lags up to "
                    "valid_lag_samples are fully supported by the excitation."
                ),
            },
            "diagnostics": self.diagnostics.to_dict(),
            "noise": self.noise.to_dict(),
            "bands": [b.to_dict() for b in self.bands],
        }
        if include_curves:
            out["curves"] = {
                f"{b.band[0]:g}-{b.band[1]:g}Hz": {
                    "time_s": b.curve.times_s.tolist(),
                    "level_db": b.curve.level_db.tolist(),
                }
                for b in self.bands
            }
        return out

    def linear_ir(self, normalise_peak: bool = False) -> Tuple[np.ndarray, float]:
        """Return the linear IR (valid range) and the scale applied.

        By default the physical deconvolution gain is preserved (reference
        recording -> 0.7 impulse).  ``normalise_peak`` is only for plotting
        and never used for the band analysis.
        """
        ir = self.deconvolution[: self.localisation.valid_lag + 1].astype(np.float64)
        if not normalise_peak:
            return ir, 1.0
        peak = float(np.max(np.abs(ir)))
        if peak <= 0:
            return ir, 1.0
        return ir / peak, peak


def _check_sweep_model(plan: Plan, excitation: Audio) -> SweepModelCheck:
    generated = generate_sweep(plan.sweep, plan.sample_rate, excitation.n_samples)
    provided = excitation.samples
    n = min(generated.size, provided.size)
    if n == 0:
        return SweepModelCheck(False, 0.0, 0.0, 0.0, "empty excitation")
    a = generated[:n]
    b = provided[:n]
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    corr = float(np.sum(a * b) / denom) if denom > 0 else 0.0
    max_diff = float(np.max(np.abs(a - b)))
    rms_diff = float(np.sqrt(np.mean((a - b) ** 2)))
    ok = corr >= Params().sweep_model_min_correlation
    note = (
        "provided sweep matches the plan's exponential-sine formula; inverse "
        "filter is derived from the provided waveform"
        if ok
        else "provided sweep deviates from the plan formula; inverse filter still uses the provided waveform"
    )
    return SweepModelCheck(ok, corr, max_diff, rms_diff, note)


def _overall_status(bands: List[decay_mod.BandDecay], diagnostics: Diagnostics) -> Tuple[str, str]:
    if diagnostics.global_reasons:
        return "diagnostic_failure", "; ".join(diagnostics.global_reasons)
    statuses = [b.status for b in bands]
    if all(s == "credible_t20" for s in statuses):
        return "ok", "all requested bands have credible T20 estimates"
    if all(s == "unavailable" for s in statuses):
        return "unavailable", "no band has a credible decay-time estimate"
    counts = {s: statuses.count(s) for s in set(statuses)}
    detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return "partial", f"mixed band verdicts: {detail}"


def analyze_recording(
    plan: Plan,
    excitation: Audio,
    recording: Audio,
    params: Optional[Params] = None,
    label: str = "recording",
) -> AnalysisResult:
    params = params or Params()
    sr = plan.sample_rate

    inv = build_inverse_filter(excitation.samples, sr, plan.sweep)
    d = deconvolve(recording.samples, inv)
    valid = valid_lag_limit(recording.n_samples, excitation.n_samples)

    band_coverage: Dict[Tuple[float, float], dict] = {}
    band_coverage_ok = True
    for lo, hi in plan.bands_hz:
        fraction = excitation_band_fraction(excitation.samples, sr, lo, hi)
        inside = (
            lo >= plan.sweep.start_hz - params.band_inside_tolerance_hz
            and hi <= plan.sweep.end_hz + params.band_inside_tolerance_hz
        )
        ok = fraction >= params.min_band_excitation_fraction and inside
        if not ok:
            band_coverage_ok = False
        band_coverage[(lo, hi)] = {
            "excitation_energy_fraction": float(fraction),
            "inside_swept_range": bool(inside),
            "ok": bool(ok),
        }

    rec_desc = describe_signal(recording, params.full_scale_threshold)
    exc_desc = describe_signal(excitation, params.full_scale_threshold)
    full_scale_ok = (
        rec_desc["full_scale_samples"] == 0 and exc_desc["full_scale_samples"] == 0
    )
    full_scale_detail = (
        "no samples at digital full scale"
        if full_scale_ok
        else (
            f"recording has {rec_desc['full_scale_samples']} samples and excitation "
            f"has {exc_desc['full_scale_samples']} samples at or beyond "
            f"{params.full_scale_threshold:.6f}; clipped samples are a diagnostic, "
            "not silently discarded"
        )
    )

    # Localisation uses an independent robust noise estimate (late tail) before
    # the full noise model exists.
    local = localise(d, inv, valid, params)
    noise = estimate_noise(recording.samples, d, local.onset_lag, valid, inv, plan.bands_hz, params)

    bands: List[decay_mod.BandDecay] = []
    for lo, hi in plan.bands_hz:
        if not band_coverage[(lo, hi)]["ok"]:
            # No usable excitation in this band: do not fabricate a decay.
            fake = _unavailable_band(
                (lo, hi),
                reason=(
                    "requested band has insufficient excitation energy "
                    f"({band_coverage[(lo, hi)]['excitation_energy_fraction']:.4f}) or lies "
                    "outside the swept range"
                ),
            )
            bands.append(fake)
            continue
        band_noise = noise.bands[(lo, hi)]
        bands.append(
            decay_mod.analyze_band_decay(
                d,
                (lo, hi),
                local.onset_lag,
                local.direct_lag,
                valid,
                band_noise.floor_energy,
                inv,
                params,
            )
        )

    reasons: List[str] = []
    soft_reasons: List[str] = []
    if not full_scale_ok:
        reasons.append("digital full scale detected; response may be clipped")
    if not band_coverage_ok:
        reasons.append("at least one requested band lacks usable excitation")
    if local.ambiguity:
        reasons.append(f"localisation ambiguity: {local.ambiguity_reason}")
    if local.onset_lag > local.direct_lag:
        soft_reasons.append("onset later than direct arrival; check localisation")
    for band_noise in noise.bands.values():
        if not band_noise.stationarity_ok:
            soft_reasons.append(
                f"noise control window non-stationary in band "
                f"{band_noise.band[0]:g}-{band_noise.band[1]:g} Hz "
                f"({band_noise.half_ratio_db:+.1f} dB half-to-half)"
            )
        if not band_noise.agreement_ok:
            soft_reasons.append(
                f"empirical vs propagated noise floor disagree in band "
                f"{band_noise.band[0]:g}-{band_noise.band[1]:g} Hz "
                f"({band_noise.agreement_db:+.1f} dB)"
            )

    sweep_check = _check_sweep_model(plan, excitation)
    if not sweep_check.provided_matches_generated:
        soft_reasons.append("provided sweep does not match the plan formula")

    diagnostics = Diagnostics(
        recording=rec_desc,
        excitation=exc_desc,
        sweep_model_check=sweep_check,
        full_scale_ok=full_scale_ok,
        full_scale_detail=full_scale_detail,
        band_coverage=band_coverage,
        band_coverage_ok=band_coverage_ok,
        localisation=local.to_dict(),
        localisation_ok=not local.ambiguity,
        global_reasons=[],
        warnings=soft_reasons,
    )
    # Only hard failures stop decay reporting; soft warnings are recorded but
    # do not erase otherwise credible bands.
    diagnostics.global_reasons = list(reasons)

    overall, overall_reason = _overall_status(bands, diagnostics)
    return AnalysisResult(
        plan=plan,
        params=params,
        label=label,
        recording_path=recording.path,
        recording_samples=recording.samples,
        deconvolution=d,
        inverse_filter=inv,
        localisation=local,
        noise=noise,
        bands=bands,
        diagnostics=diagnostics,
        overall_status=overall,
        overall_reason=overall_reason,
    )


def _unavailable_band(band: Tuple[float, float], reason: str) -> decay_mod.BandDecay:
    empty = np.zeros(1)
    return decay_mod.BandDecay(
        band=band,
        status="unavailable",
        status_reason=reason,
        noise_floor_rms=float("nan"),
        noise_clearance_db_at_minus_5=None,
        noise_clearance_db_at_minus_15=None,
        noise_clearance_db_at_minus_25=None,
        dynamic_range_db=float("nan"),
        time_origin_s=float("nan"),
        analysis_window_start_s=float("nan"),
        analysis_window_end_s=float("nan"),
        integration_cut_time_s=float("nan"),
        gate_db=0.0,
        t5_s=None,
        t15_s=None,
        t25_s=None,
        t20_s=None,
        t10_s=None,
        t20_fit_start_s=None,
        t20_fit_end_s=None,
        t20_fit_points=None,
        t20_residual_db=None,
        t20_slope_db_per_s=None,
        t10_fit_start_s=None,
        t10_fit_end_s=None,
        t10_residual_db=None,
        t10_slope_db_per_s=None,
        noise_sensitivity_t20_range_s=None,
        curve=decay_mod.DecayCurve(
            band=band, times_s=np.zeros(1), level_db=empty, raw_energy=empty
        ),
    )
