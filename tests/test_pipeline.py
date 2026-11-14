"""Tests that validate the measurement math with independent, analytically
known inputs.  No generator ground-truth files are read here; the synthetic
checks construct their own known references.
"""

from __future__ import annotations

import os
import wave

import numpy as np
import pytest

from sweeproom.analysis import analyze_recording
from sweeproom.config import Params, Plan
from sweeproom.decay import _fit_line
from sweeproom.io import Audio, AudioFormatError, read_wav
from sweeproom.report import write_band_csv
from sweeproom.sweep import (
    build_inverse_filter,
    deconvolve,
    excitation_band_fraction,
    generate_sweep,
    valid_lag_limit,
)


FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
CASES = os.path.join(os.path.dirname(__file__), "..", "cases")


@pytest.fixture(scope="session")
def plan() -> Plan:
    return Plan.load(os.path.join(FIXTURES, "plan.json"))


@pytest.fixture(scope="session")
def excitation(plan) -> Audio:
    return read_wav(os.path.join(FIXTURES, "sweep.wav"), expected_sr=plan.sample_rate)


def _wav_header(path, nchannels=1, sampwidth=2, sr=16000):
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == nchannels
        assert wf.getsampwidth() == sampwidth
        assert wf.getframerate() == sr
        return wf.getnframes()


def test_input_wav_format(plan, excitation):
    assert _wav_header(os.path.join(FIXTURES, "recording.wav")) > 0
    assert _wav_header(os.path.join(FIXTURES, "reference_delay.wav")) > 0
    assert _wav_header(os.path.join(CASES, "noisy_recording.wav")) > 0
    assert excitation.n_samples == int(2.0 * plan.sample_rate)


def test_mono_pcm16_is_enforced(tmp_path):
    # Stereo file must be rejected rather than silently averaged.
    import scipy.io.wavfile as sw

    sw.write(str(tmp_path / "stereo.wav"), 16000, np.zeros((100, 2), dtype="<i2"))
    with pytest.raises(AudioFormatError):
        read_wav(str(tmp_path / "stereo.wav"))


def test_generated_sweep_matches_fixture(plan, excitation):
    gen = generate_sweep(plan.sweep, plan.sample_rate, excitation.n_samples)
    corr = np.corrcoef(gen, excitation.samples)[0, 1]
    assert corr > 0.999


def test_inverse_filter_recovers_reference_delay(plan, excitation):
    """Deconvolving the plan's reference must give 0.050 s / 0.7 gain."""
    reference = read_wav(os.path.join(FIXTURES, "reference_delay.wav"))
    inv = build_inverse_filter(excitation.samples, plan.sample_rate, plan.sweep)
    d = deconvolve(reference.samples, inv)
    valid = valid_lag_limit(reference.n_samples, excitation.n_samples)
    start = max(1, int(round(inv.self_response_width_to_minus20db_s * plan.sample_rate * 4)))
    direct = start + int(np.argmax(np.abs(d[start : valid + 1])))
    assert abs(direct - 0.050 * plan.sample_rate) <= 1.0
    assert abs(abs(d[direct]) - 0.7) <= 0.05


def test_excitation_band_fraction(plan, excitation):
    lo, hi = plan.bands_hz[0]
    frac = excitation_band_fraction(excitation.samples, plan.sample_rate, lo, hi)
    assert 0.0 < frac < 1.0


def test_synthetic_exponential_slope_recovered(plan, excitation, tmp_path):
    """A generated IR with a known RT60 must yield the same T20 slope."""
    params = Params()
    sr = plan.sample_rate
    target_rt60 = 0.6
    length = int(1.2 * sr)
    n = np.arange(length, dtype=float)
    delay = int(0.05 * sr)
    env = 10.0 ** (-3.0 * n / (target_rt60 * sr))
    rng = np.random.default_rng(11)
    ir = rng.standard_normal(length) * env
    ir[:delay] = 0.0
    ir[delay] += 1.0
    wet = np.convolve(excitation.samples, ir, mode="full")[: excitation.n_samples + length]
    wet = wet + rng.standard_normal(wet.size) * 1e-4 * np.max(np.abs(wet))
    synth = Audio(samples=wet, sample_rate=sr, path=str(tmp_path / "synth.wav"))
    result = analyze_recording(plan, excitation, synth, params=params, label="synth")
    measured = [b.t20_s for b in result.bands if b.t20_s is not None]
    assert measured, "no credible T20 recovered from the synthetic decay"
    best = min(measured, key=lambda v: abs(v - target_rt60))
    assert abs(best - target_rt60) / target_rt60 < 0.15


def test_strong_noise_does_not_report_t20(plan, excitation):
    """Raise the noise until the -25 dB point loses its margin; T20 must vanish."""
    params = Params()
    sr = plan.sample_rate
    base = analyze_recording(
        plan,
        excitation,
        read_wav(os.path.join(FIXTURES, "recording.wav")),
        params=params,
        label="base",
    )
    # Synthesize a much noisier recording by matching the base IR exactly and
    # adding a large stationary noise floor.
    ir, _ = base.linear_ir()
    wet = np.convolve(excitation.samples, ir, mode="full")[: len(base.recording_samples)]
    rng = np.random.default_rng(3)
    sigma = 20.0 * np.sqrt(np.mean(base.recording_samples[:5000] ** 2))
    noisy = Audio(
        samples=wet + rng.standard_normal(wet.size) * sigma,
        sample_rate=sr,
        path="<synth-noisy>",
    )
    result = analyze_recording(plan, excitation, noisy, params=params, label="synth-noisy")
    assert any(b.t20_s is None for b in result.bands), (
        "heavily noisy synthesis must not claim a T20"
    )


def test_decay_reverse_integral_linear_slope():
    """The reverse-integral fit must recover a known linear dB slope."""
    sr = 16000
    n = 4000
    slope_db_per_s = -100.0
    t = np.arange(n) / sr
    level = slope_db_per_s * t
    fit = _fit_line(level, 200, 1400, sr, 60.0)
    assert fit is not None
    assert abs(fit["slope"] - slope_db_per_s) < 1e-6
    assert abs(fit["value_s"] - 0.6) < 1e-6
    assert fit["residual"] < 1e-9


def test_band_csv_has_all_evidence(plan, excitation):
    result = analyze_recording(
        plan,
        excitation,
        read_wav(os.path.join(FIXTURES, "recording.wav")),
        label="base",
    )
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bands.csv")
        write_band_csv(result, path)
        with open(path, encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
        for required in (
            "status",
            "clearance_at_minus_25_db",
            "t20_fit_start_s",
            "t20_fit_end_s",
            "t20_residual_db",
            "integration_cut_time_s",
        ):
            assert required in header


def test_decay_curves_long_format(plan, excitation):
    result = analyze_recording(
        plan, excitation, read_wav(os.path.join(FIXTURES, "recording.wav")), label="base"
    )
    import tempfile
    from sweeproom.report import write_decay_curves_csv

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "curves.csv")
        write_decay_curves_csv(result, path)
        with open(path, encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
            first = fh.readline().strip().split(",")
    assert header == [
        "band_lo_hz",
        "band_hi_hz",
        "time_s_relative_to_direct",
        "level_db",
        "gated_raw_energy",
    ]
    assert len(first) == 5


def test_analysis_does_not_read_generator_truth(plan, excitation):
    """The analyzer API takes only plan + sweep + recording arrays."""
    import inspect

    sig = inspect.signature(analyze_recording)
    assert set(sig.parameters) == {"plan", "excitation", "recording", "params", "label"}
