"""Background-noise estimation and propagation through the inverse filter.

The recording contains stationary added background noise.  After
deconvolution the noise is no longer white: the inverse filter colours it.  A
naive analysis that integrates the raw squared response over the whole
recording would therefore integrate a *noise plateau* and report a
meaningless, enormous decay time.  This module produces the per-band
deconvolved noise power used to gate the integration.

Two independent estimates are computed per band:

1. **empirical tail**: the deconvolved band-power in a late control window
   (by construction after the room has decayed), checked for stationarity
   between its first and second halves;
2. **propagated model**: the raw pre-onset noise PSD is estimated with
   Welch, synthetic coloured noise with that PSD is pushed through the
   actual inverse filter, and the resulting per-band power is measured over
   the valid lag range.

We take the **larger** of the two as the floor, so the reported dynamic
range is conservative.  If the two disagree beyond the published tolerance
we flag it; the number is still used but the disagreement is surfaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import signal

from .config import Params
from .sweep import InverseFilter, deconvolve


@dataclass
class BandNoise:
    band: Tuple[float, float]
    empirical_energy: float
    propagated_energy: float
    floor_energy: float  # max(empirical, propagated)
    half_ratio_db: float  # first-half vs second-half of control window
    agreement_db: float  # empirical / propagated, in dB
    psd_source_samples: int
    stationarity_ok: bool
    agreement_ok: bool

    def to_dict(self) -> dict:
        return {
            "band_hz": [float(self.band[0]), float(self.band[1])],
            "empirical_rms": float(np.sqrt(self.empirical_energy)),
            "propagated_rms": float(np.sqrt(self.propagated_energy)),
            "floor_rms": float(np.sqrt(self.floor_energy)),
            "first_vs_second_half_db": float(self.half_ratio_db),
            "empirical_vs_propagated_db": float(self.agreement_db),
            "control_psd_source_samples": int(self.psd_source_samples),
            "stationarity_ok": bool(self.stationarity_ok),
            "agreement_ok": bool(self.agreement_ok),
        }


@dataclass
class NoiseModel:
    bands: Dict[Tuple[float, float], BandNoise]
    raw_noise_rms: float
    raw_noise_samples: int
    raw_noise_window: Tuple[int, int]
    propagation_seeds: int

    def to_dict(self) -> dict:
        return {
            "raw_noise_rms": float(self.raw_noise_rms),
            "raw_noise_samples": int(self.raw_noise_samples),
            "raw_noise_window_samples": [int(self.raw_noise_window[0]), int(self.raw_noise_window[1])],
            "propagation_seeds": int(self.propagation_seeds),
            "bands": [b.to_dict() for b in self.bands.values()],
        }


def bandpass(x: np.ndarray, band: Tuple[float, float], sample_rate: int, order: int = 4) -> np.ndarray:
    lo, hi = band
    nyq = sample_rate / 2.0
    if hi >= nyq:
        hi = nyq * 0.999
    if lo <= 0.0:
        lo = nyq * 0.001
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, x)


def _raw_noise_window(
    recording: np.ndarray, onset_raw: int, sample_rate: int
) -> Tuple[int, int]:
    """Pick samples that precede the sweep and contain noise only.

    Uses the first half of the pre-onset region (leaving a 20 ms guard
    before the sweep) so that even a mis-detected onset cannot leak sweep
    energy into the noise estimate.
    """
    guard = int(0.02 * sample_rate)
    end = max(0, onset_raw - guard)
    start = 0
    if end - start < int(0.05 * sample_rate):
        # Fall back to a short window; still noise-only by construction as
        # long as the onset detector fired near the true start.
        end = max(0, min(onset_raw, recording.size))
        start = max(0, end - int(0.05 * sample_rate))
    return start, end


def estimate_raw_noise_psd(
    recording: np.ndarray, window: Tuple[int, int], sample_rate: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    start, end = window
    seg = recording[start:end]
    if seg.size < 64:
        # Not enough samples for a spectral estimate; use a white floor
        # consistent with the sample variance.
        var = float(np.var(seg)) if seg.size else 0.0
        nper = 64
        nf = np.fft.rfftfreq(nper, 1.0 / sample_rate)
        return nf, np.full_like(nf, max(var / (sample_rate / 2.0), 0.0)), var
    nper = min(4096, (seg.size // 2) * 2)
    nper = max(64, nper)
    nf, psd = signal.welch(
        seg, fs=sample_rate, nperseg=nper, noverlap=nper // 2, detrend="constant"
    )
    return nf, psd, float(np.var(seg))


def propagate_noise_to_bands(
    nf: np.ndarray,
    psd: np.ndarray,
    raw_var: float,
    recording_len: int,
    valid_lag: int,
    inv: InverseFilter,
    bands,
    sample_rate: int,
    seeds: int,
) -> Dict[Tuple[float, float], float]:
    """Push synthetic coloured noise through the inverse filter.

    Returns median band energy over deterministic seeds, measured on the
    fully-valid lag range only.
    """
    ff = np.fft.rfftfreq(recording_len, 1.0 / sample_rate)
    interp = np.interp(ff, nf, psd, left=psd[0], right=psd[-1])
    interp = np.maximum(interp, 0.0)
    values: Dict[Tuple[float, float], list] = {b: [] for b in bands}
    for seed in range(seeds):
        rng = np.random.default_rng(7919 + seed)
        white = rng.standard_normal(recording_len)
        coloured = np.fft.irfft(np.fft.rfft(white) * np.sqrt(interp), recording_len)
        cur = float(np.sqrt(np.mean(coloured * coloured)))
        if cur > 0 and raw_var > 0:
            coloured *= np.sqrt(raw_var) / cur
        d = deconvolve(coloured, inv)[: valid_lag + 1]
        for band in bands:
            filtered = bandpass(d, band, sample_rate)
            values[band].append(float(np.mean(filtered * filtered)))
    return {b: float(np.median(v)) for b, v in values.items()}


def estimate_noise(
    recording: np.ndarray,
    d: np.ndarray,
    onset_raw: int,
    valid_lag: int,
    inv: InverseFilter,
    bands,
    params: Params,
) -> NoiseModel:
    sr = inv.sample_rate
    window = _raw_noise_window(recording, onset_raw, sr)
    nf, psd, raw_var = estimate_raw_noise_psd(recording, window, sr)
    propagated = propagate_noise_to_bands(
        nf,
        psd,
        raw_var,
        recording.size,
        valid_lag,
        inv,
        bands,
        sr,
        params.noise_propagation_seeds,
    )

    band_models: Dict[Tuple[float, float], BandNoise] = {}
    for band in bands:
        filtered = bandpass(d[: valid_lag + 1], band, sr)
        energy = filtered * filtered
        lo = int(params.noise_control_frac[0] * valid_lag)
        hi = int(params.noise_control_frac[1] * valid_lag)
        lo = max(0, min(lo, energy.size - 1))
        hi = max(lo + 1, min(hi, energy.size))
        control = energy[lo:hi]
        empirical = float(np.median(control)) if control.size else 0.0
        half = control.size // 2
        if half >= 8:
            first = float(np.mean(control[:half]))
            second = float(np.mean(control[half:]))
            half_ratio = (
                10.0 * np.log10(max(first, 1e-300) / max(second, 1e-300))
                if first > 0 and second > 0
                else 0.0
            )
        else:
            half_ratio = 0.0
        floor = max(empirical, propagated[band])
        agreement = (
            10.0 * np.log10(max(empirical, 1e-300) / max(propagated[band], 1e-300))
            if empirical > 0 and propagated[band] > 0
            else 0.0
        )
        band_models[band] = BandNoise(
            band=band,
            empirical_energy=empirical,
            propagated_energy=propagated[band],
            floor_energy=floor,
            half_ratio_db=float(half_ratio),
            agreement_db=float(agreement),
            psd_source_samples=int(window[1] - window[0]),
            stationarity_ok=bool(abs(half_ratio) <= params.noise_stationarity_max_db),
            agreement_ok=bool(abs(agreement) <= params.noise_floor_agreement_max_db),
        )

    return NoiseModel(
        bands=band_models,
        raw_noise_rms=float(np.sqrt(max(raw_var, 0.0))),
        raw_noise_samples=int(window[1] - window[0]),
        raw_noise_window=window,
        propagation_seeds=int(params.noise_propagation_seeds),
    )
