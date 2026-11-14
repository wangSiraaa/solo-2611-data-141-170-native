"""Swept-sine excitation model and its inverse filter.

The excitation is a *known* exponential (logarithmic) sine sweep:

    f(t) = f1 * exp(t / K),     K = T / ln(f2 / f1)
    s(t) = A * sin(2*pi*f1*K*(exp(t/K) - 1))

The inverse filter for such a sweep is the time-reversed sweep with an
exponential amplitude envelope (a Farina-style filter).  Choosing the envelope
``exp(-n/(sr*K))`` on the time-reversed sweep flattens ``|S(f) H(f)|`` across
the swept band: this is what keeps the *relative* amplitude of different
analysis bands intact after deconvolution.

Important: the inverse filter is normalised so that the deconvolution of the
*emitted sweep itself* has a unit peak at lag 0.  That makes the deconvolved
recording carry the physical channel gain directly: a recording that is the
sweep delayed by d and scaled by g deconvolves to an impulse of height g at lag
d (this is exercised by ``verify`` against ``reference_delay.wav``).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SweepDefinition


def generate_sweep(sweep: SweepDefinition, sample_rate: int, n_samples: int) -> np.ndarray:
    """Regenerate the nominal sweep from the plan.

    Matches the stated excitation: amplitude A, instantaneous frequency
    f1 -> f2 over the duration, with linear fades of ``fade_seconds``.
    """
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    total = n_samples / sample_rate
    phase = 2.0 * np.pi * sweep.start_hz * sweep.exponent_k * (
        np.exp(t / sweep.exponent_k) - 1.0
    )
    out = sweep.amplitude * np.sin(phase)
    fade = int(round(sweep.fade_seconds * sample_rate))
    if fade > 0 and 2 * fade <= n_samples:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]
    # The generated array's own duration may differ slightly from the plan's
    # nominal duration; record the effective one for the caller.
    _ = total
    return out


def sweep_phase_note(sweep: SweepDefinition, sample_rate: int) -> dict:
    return {
        "kind": sweep.kind,
        "duration_seconds": sweep.duration_seconds,
        "start_hz": sweep.start_hz,
        "end_hz": sweep.end_hz,
        "fade_seconds": sweep.fade_seconds,
        "amplitude": sweep.amplitude,
        "exponent_k_seconds": sweep.exponent_k,
        "instantaneous_frequency_hz_at": {
            "start": sweep.start_hz,
            "mid": float(sweep.start_hz * np.exp(sweep.duration_seconds / (2 * sweep.exponent_k))),
            "end": sweep.end_hz,
        },
        "sample_rate": sample_rate,
        "nyquist_hz": sample_rate / 2.0,
    }


@dataclass
class InverseFilter:
    """Linear inverse filter plus the metadata needed to interpret lags.

    ``filter`` is the time-domain FIR used with full linear convolution.
    ``direct_lag`` is the sample index of the unit self-response peak
    (== excitation length - 1 for the zero-lag convention used here).
    ``harmonic_lag_offsets`` maps harmonic order k to the *predicted* lag
    offset relative to the direct arrival: tau_k - tau_1 = -K * ln(k).
    """

    filter: np.ndarray
    excitation: np.ndarray
    sample_rate: int
    exponent_k: float
    self_response_peak_val: float
    self_response_peak_index: int
    self_response_width_to_minus20db_s: float

    @property
    def zero_lag_index(self) -> int:
        """Index in a full linear-convolution result that corresponds to lag 0."""
        return int(self.excitation.size - 1)

    def harmonic_offset_samples(self, k: int) -> float:
        """Predicted lag offset of the k-th harmonic image, relative to direct."""
        if k < 1:
            raise ValueError("harmonic order must be >= 1")
        return -self.exponent_k * np.log(float(k)) * self.sample_rate


def build_inverse_filter(
    excitation: np.ndarray, sample_rate: int, sweep: SweepDefinition
) -> InverseFilter:
    """Build the farina-style inverse filter for the *measured* excitation.

    The filter is derived from the actually supplied excitation waveform
    (not just from the plan formula) so that small generator differences in
    phase or fades do not bias the deconvolution.
    """
    n = excitation.size
    k = sweep.exponent_k
    time_index = np.arange(n, dtype=np.float64)
    envelope = np.exp(-time_index / (sample_rate * k))
    inv = excitation[::-1] * envelope

    # Full linear self-convolution: the unit response.  Its peak defines the
    # zero-lag reference and the normalisation.
    self_conv = np.convolve(excitation, inv, mode="full")
    peak_index = int(np.argmax(np.abs(self_conv)))
    peak_val = float(self_conv[peak_index])
    if not np.isfinite(peak_val) or peak_val == 0.0:
        raise ValueError("inverse-filter self-response is degenerate")
    inv = inv / peak_val

    # Measure the width of the *normalised* unit response around its peak,
    # leaving the peak out (the instant of the peak is the direct arrival).
    unit = np.convolve(excitation, inv, mode="full")
    centre = int(np.argmax(np.abs(unit)))
    centre = min(centre, unit.size - 1)
    env = np.abs(unit)
    target = env[centre] * 10.0 ** (-20.0 / 20.0)
    width = 0
    for offset in range(0, min(env.size // 2, centre + 1)):
        left = centre - offset
        if left >= 0 and env[left] > target:
            width = offset
        if offset < env.size - centre and env[centre + offset] > target:
            width = max(width, offset)
        if offset > 0 and env[centre + offset] <= target and env[left] <= target:
            break
    return InverseFilter(
        filter=inv,
        excitation=excitation,
        sample_rate=sample_rate,
        exponent_k=k,
        self_response_peak_val=peak_val,
        self_response_peak_index=peak_index,
        self_response_width_to_minus20db_s=width / sample_rate,
    )


def deconvolve(recording: np.ndarray, inv: InverseFilter) -> np.ndarray:
    """Return the linear deconvolution indexed by lag.

    The full linear convolution is computed, then the *valid causal* portion
    starting at ``zero_lag_index`` is returned.  That portion has the same
    length as the recording and covers lags ``0 .. len(recording)-1``.

    Callers must remember that the final ``len(excitation)-1`` lags are only
    *partially* covered by the excitation support; the analysis restricts
    itself to lags ``<= len(recording) - len(excitation)`` where every
    contributing excitation sample exists.
    """
    full = np.convolve(recording, inv.filter, mode="full")
    start = inv.zero_lag_index
    return full[start : start + recording.size]


def valid_lag_limit(recording_len: int, excitation_len: int) -> int:
    """Largest lag index for which the deconvolution is fully supported.

    For a recording that contains the full 2 s sweep, lag L is fully valid
    while L <= len(recording) - len(excitation).  Larger lags would only be
    supported by part of the excitation and must not be interpreted.
    """
    return max(0, recording_len - excitation_len)


def excitation_band_fraction(
    excitation: np.ndarray, sample_rate: int, lo: float, hi: float
) -> float:
    """Fraction of the excitation's energy that falls inside a band."""
    nfft = 1 << int(np.ceil(np.log2(max(2, excitation.size * 4))))
    spec = np.fft.rfft(excitation, nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    power = spec.real ** 2 + spec.imag ** 2
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    inside = float(np.sum(power[(freqs >= lo) & (freqs <= hi)]))
    return inside / total
