"""WAV I/O and diagnostic checks for the swept-sine measurement tool.

The tool only accepts the case stated in the task: 16 kHz, mono, PCM16.
Anything else is rejected with a clear message rather than silently coerced.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np


class AudioFormatError(Exception):
    pass


@dataclass
class Audio:
    samples: np.ndarray  # float64 in nominal [-1, 1] sample units
    sample_rate: int
    path: str

    @property
    def n_samples(self) -> int:
        return int(self.samples.size)


def read_wav(path: str, expected_sr: Optional[int] = None) -> Audio:
    """Read a mono PCM16 WAV file into float64 samples.

    Samples are divided by 32768, so the value 1.0 corresponds to digital full
    scale.  This scaling is a single global constant: it does not touch the
    relative amplitudes of the analysis bands.
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        nread = wf.readframes(nframes)

    if n_channels != 1:
        raise AudioFormatError(
            f"{path}: expected mono (1 channel), found {n_channels}"
        )
    if sampwidth != 2:
        raise AudioFormatError(
            f"{path}: expected PCM16 (2-byte samples), found {sampwidth}-byte"
        )
    if expected_sr is not None and sr != expected_sr:
        raise AudioFormatError(
            f"{path}: expected {expected_sr} Hz, found {sr} Hz"
        )
    raw = np.frombuffer(nread, dtype="<i2")
    if raw.size != nframes:
        raise AudioFormatError(
            f"{path}: header announced {nframes} frames but {raw.size} decoded"
        )
    samples = raw.astype(np.float64) / 32768.0
    return Audio(samples=samples, sample_rate=sr, path=path)


def full_scale_count(samples: np.ndarray, threshold: float) -> int:
    """Count samples at or beyond the digital full-scale threshold."""
    return int(np.count_nonzero(np.abs(samples) >= threshold))


def describe_signal(audio: Audio, full_scale_threshold: float) -> dict:
    x = audio.samples
    n = x.size
    if n == 0:
        return {"n_samples": 0, "rms": 0.0, "peak_abs": 0.0}
    return {
        "n_samples": int(n),
        "duration_s": float(n / audio.sample_rate),
        "rms": float(np.sqrt(np.mean(x * x))),
        "mean": float(np.mean(x)),
        "peak_abs": float(np.max(np.abs(x))),
        "peak_time_s": float(int(np.argmax(np.abs(x))) / audio.sample_rate),
        "dc_offset": float(np.mean(x)),
        "full_scale_samples": full_scale_count(x, full_scale_threshold),
        "full_scale_threshold": float(full_scale_threshold),
        "min_sample": float(np.min(x)),
        "max_sample": float(np.max(x)),
    }
