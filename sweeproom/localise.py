"""Direct-arrival localisation for the lag-domain deconvolution.

Two facts drive this module, both specific to logarithmic sweeps:

1. **Unknown playback delay.**  The recording starts at an unknown offset, so
   the direct arrival sits at an unknown lag ``tau_1``.  We locate it as the
   global peak of ``|deconvolution|`` restricted to the *fully valid* lag
   range, after suppressing a small guard around lag 0 (the inverse filter
   has zero-lag content) and any predicted harmonic images.

2. **Harmonic pre-arrival images.**  A memoryless nonlinearity in front of
   the room produces harmonics ``k*f(t)``.  Correlating those with the
   fundamental inverse filter places their images at

       tau_k = tau_1 + K * ln(1/k) = tau_1 - K * ln(k)

   i.e. *earlier* than the direct arrival for k >= 2.  In this recording the
   k=2 image lands at lag ``tau_1 - K ln 2`` (hundreds of samples before the
   direct sound if the delay is short) and higher orders wrap below lag 0 and
   are unobservable in the causal range.  We therefore:

   * record the predicted offsets so the operator can see them;
   * exclude windows around *in-causal* images when choosing the main peak;
   * keep the analysis window starting at the direct arrival, so pre-arrival
     harmonic energy is by construction *outside* the linear-response window;
   * report the energy we excluded in each excluded window.

This is deliberately not a general nonlinear-system identifier: only the
Farina harmonic-image positions are used, as the task allows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import Params
from .sweep import InverseFilter


@dataclass
class HarmonicImage:
    order: int
    predicted_lag: float  # relative to the recording start (samples)
    in_causal_range: bool
    window_start: int  # sample index within recording
    window_end: int
    energy: float  # sum of squares in the window
    peak_abs: float
    snr_db: Optional[float]


@dataclass
class Localisation:
    direct_lag: int
    direct_time_s: float
    direct_amplitude: float
    onset_lag: int
    onset_time_s: float
    valid_lag: int
    valid_time_s: float
    peak_dominance_db: float
    peak_snr_db: float
    harmonic_images: List[HarmonicImage]
    excluded_energy: float  # energy before the analysis window (pre-onset)
    total_energy_in_window: float
    harmonic_image_energy: float  # energy inside predicted harmonic windows
    ambiguity: bool
    ambiguity_reason: Optional[str]

    def to_dict(self) -> dict:
        return {
            "direct_lag_samples": int(self.direct_lag),
            "direct_time_s": float(self.direct_time_s),
            "direct_amplitude": float(self.direct_amplitude),
            "onset_lag_samples": int(self.onset_lag),
            "onset_time_s": float(self.onset_time_s),
            "valid_lag_samples": int(self.valid_lag),
            "valid_time_s": float(self.valid_time_s),
            "peak_dominance_db": float(self.peak_dominance_db),
            "peak_snr_db": float(self.peak_snr_db),
            "excluded_energy_before_analysis_window": float(self.excluded_energy),
            "harmonic_image_energy": float(self.harmonic_image_energy),
            "total_energy_in_valid_window": float(self.total_energy_in_window),
            "ambiguity": bool(self.ambiguity),
            "ambiguity_reason": self.ambiguity_reason,
            "harmonic_images": [
                {
                    "order": h.order,
                    "predicted_lag_samples": float(h.predicted_lag),
                    "in_causal_range": bool(h.in_causal_range),
                    "window_start_sample": int(h.window_start),
                    "window_end_sample": int(h.window_end),
                    "energy": float(h.energy),
                    "peak_abs": float(h.peak_abs),
                    "snr_db": None if h.snr_db is None else float(h.snr_db),
                }
                for h in self.harmonic_images
            ],
        }


def _smoothed_energy(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x * x
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x * x, kernel, mode="same")


def _noise_floor_estimate(d: np.ndarray, valid: int, params: Params) -> float:
    """Robust late-tail energy density of the deconvolved signal."""
    lo = int(params.noise_control_frac[0] * valid)
    hi = int(params.noise_control_frac[1] * valid)
    lo = max(0, min(lo, d.size - 1))
    hi = max(lo + 1, min(hi, d.size))
    seg = d[lo:hi]
    return float(np.median(seg * seg)) if seg.size else 0.0


def localise(
    d: np.ndarray,
    inv: InverseFilter,
    valid: int,
    params: Params,
    noise_floor_energy: Optional[float] = None,
) -> Localisation:
    """Locate the direct arrival and enumerate harmonic images."""
    sr = inv.sample_rate
    search_end = min(valid, d.size - 1)
    if search_end <= 2:
        raise ValueError("recording too short for a valid deconvolution window")

    if noise_floor_energy is None:
        noise_floor_energy = _noise_floor_estimate(d, valid, params)
    noise_rms = float(np.sqrt(noise_floor_energy))

    # Predicted harmonic image lags use the *unknown* direct lag tau_1, which
    # we do not know yet.  HARMONIC_OFfset_k = -K ln(k) is independent of it,
    # so we first find the peak, then report images relative to it.
    env = np.abs(d[: search_end + 1])

    # Exclusion around lag 0: the inverse filter is anti-causal and leaves a
    # small zero-lag residual; do not let it masquerade as the direct sound.
    guard0 = max(
        int(round(inv.self_response_width_to_minus20db_s * sr * 4)),
        int(round(params.analysis_guard_ms * 1e-3 * sr)),
    )
    guard0 = min(guard0, search_end // 4)

    # First-pass peak ignoring lag-0 guard.
    if search_end <= guard0 + 2:
        direct = int(np.argmax(env))
    else:
        direct = guard0 + int(np.argmax(env[guard0:]))

    # Harmonic-image windows depend on the direct lag.  If the first-pass peak
    # were itself a harmonic image, the predicted positions would be wrong, so
    # iterate to a fixed point: recompute the harmonics from the current direct
    # lag, exclude them, re-select the peak.  It converges in one or two steps
    # for this class of signals.
    exclusion_halfwidth = max(
        int(round(params.harmonic_exclusion_halfwidth_ms * 1e-3 * sr)),
        int(round(inv.self_response_width_to_minus20db_s * sr * 4)),
        2,
    )
    harmonic_orders = range(2, 7)
    for _ in range(4):
        mask = np.ones(search_end + 1, dtype=bool)
        mask[: guard0 + 1] = False
        for order in harmonic_orders:
            predicted = direct + inv.harmonic_offset_samples(order)
            if 0 <= predicted < search_end:
                a = max(0, int(round(predicted)) - exclusion_halfwidth)
                b = min(search_end + 1, int(round(predicted)) + exclusion_halfwidth + 1)
                mask[a:b] = False
        candidates = np.where(mask, env, -np.inf)
        new_direct = int(np.argmax(candidates)) if np.any(mask) else direct
        if new_direct == direct:
            break
        direct = new_direct

    # Enumerate/report harmonic images at the converged direct lag.
    harmonic_images: List[HarmonicImage] = []
    for order in harmonic_orders:
        offset = inv.harmonic_offset_samples(order)
        predicted = direct + offset
        in_range = predicted >= 0 and predicted < search_end
        start = int(round(predicted)) - exclusion_halfwidth
        end = int(round(predicted)) + exclusion_halfwidth + 1
        if in_range:
            a = max(0, start)
            b = min(search_end, end)
            seg = d[a:b]
            energy = float(np.sum(seg * seg))
            peak = float(np.max(np.abs(seg))) if seg.size else 0.0
            snr = (
                20.0 * np.log10(peak / noise_rms)
                if noise_rms > 0 and peak > 0
                else None
            )
        else:
            energy = 0.0
            peak = 0.0
            snr = None
        harmonic_images.append(
            HarmonicImage(
                order=order,
                predicted_lag=float(predicted),
                in_causal_range=bool(in_range),
                window_start=start,
                window_end=end,
                energy=energy,
                peak_abs=peak,
                snr_db=snr,
            )
        )

    # Competitor strength: strongest peak outside the direct cluster (define
    # cluster as +-one inverse-filter width around the direct lag), also
    # ignoring the harmonic-image windows.
    cluster_half = max(1, int(round(inv.self_response_width_to_minus20db_s * sr * 4)))
    competitor_mask = np.ones(search_end + 1, dtype=bool)
    competitor_mask[: guard0 + 1] = False
    for h in harmonic_images:
        if h.in_causal_range:
            a = max(0, h.window_start)
            b = min(search_end + 1, h.window_end)
            competitor_mask[a:b] = False
    a = max(0, direct - cluster_half)
    b = min(search_end + 1, direct + cluster_half + 1)
    competitor_mask[a:b] = False
    if np.any(competitor_mask):
        competitor = float(np.max(np.where(competitor_mask, env, 0.0)))
    else:
        competitor = 0.0
    direct_amp = float(env[direct])
    dominance_db = (
        20.0 * np.log10(direct_amp / competitor) if competitor > 0 else float("inf")
    )
    snr_db = (
        20.0 * np.log10(direct_amp / noise_rms)
        if noise_rms > 0 and direct_amp > 0
        else float("inf")
    )

    # Onset: walk backwards from the direct peak until the smoothed envelope
    # has stayed below a threshold relative to the peak for a hold time.
    smooth = _smoothed_energy(d[: search_end + 1], max(1, int(0.0005 * sr)))
    peak_energy = float(smooth[direct]) if direct < smooth.size else 0.0
    threshold = peak_energy * 10.0 ** (params.onset_threshold_db / 10.0)
    hold = max(1, int(round(params.onset_hold_ms * 1e-3 * sr)))
    onset = direct
    below_run = 0
    for i in range(direct, max(-1, guard0 - 1), -1):
        if smooth[i] <= threshold:
            below_run += 1
            if below_run >= hold:
                onset = min(direct, i + below_run)
                break
        else:
            below_run = 0
    onset = min(onset, direct)

    # Energy excluded before the analysis window, and energy sitting in the
    # predicted harmonic-image windows.  These are reported separately so the
    # audit trail is not double-counted: the pre-window total already contains
    # any in-causal harmonic image.
    pre_window_energy = float(np.sum(d[: max(0, onset)] ** 2))
    harmonic_energy = float(
        sum(h.energy for h in harmonic_images if h.in_causal_range)
    )
    total_energy = float(np.sum(d[: search_end + 1] ** 2))

    ambiguity = False
    reason: Optional[str] = None
    if dominance_db < params.direct_peak_dominance_db:
        ambiguity = True
        reason = (
            f"direct peak dominance {dominance_db:.1f} dB below threshold "
            f"{params.direct_peak_dominance_db:.1f} dB"
        )
    elif snr_db < params.direct_peak_snr_min_db:
        ambiguity = True
        reason = (
            f"direct peak SNR {snr_db:.1f} dB below threshold "
            f"{params.direct_peak_snr_min_db:.1f} dB"
        )

    return Localisation(
        direct_lag=direct,
        direct_time_s=direct / sr,
        direct_amplitude=direct_amp,
        onset_lag=onset,
        onset_time_s=onset / sr,
        valid_lag=search_end,
        valid_time_s=search_end / sr,
        peak_dominance_db=float(dominance_db),
        peak_snr_db=float(snr_db),
        harmonic_images=harmonic_images,
        excluded_energy=pre_window_energy,
        total_energy_in_window=total_energy,
        harmonic_image_energy=harmonic_energy,
        ambiguity=ambiguity,
        ambiguity_reason=reason,
    )
