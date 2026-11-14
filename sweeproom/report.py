"""Reporting: CSV band-decay table, JSON result and matplotlib figures.

The figures deliberately show the raw evidence (recording, deconvolution,
decay curve with the noise floor and the fitted interval) rather than a single
smoothed line that hides credibility.
"""

from __future__ import annotations

import csv
import os
from typing import Optional

import numpy as np

from .analysis import AnalysisResult
from .config import Params


def write_band_csv(result: AnalysisResult, path: str) -> None:
    """Write one row per band with the verdict and all fit evidence."""
    fields = [
        "band_lo_hz",
        "band_hi_hz",
        "status",
        "status_reason",
        "noise_floor_rms",
        "dynamic_range_db",
        "analysis_window_start_s",
        "analysis_window_end_s",
        "integration_cut_time_s",
        "gate_db",
        "clearance_at_minus_5_db",
        "clearance_at_minus_15_db",
        "clearance_at_minus_25_db",
        "t5_s",
        "t15_s",
        "t25_s",
        "t20_s",
        "t20_fit_start_s",
        "t20_fit_end_s",
        "t20_fit_points",
        "t20_residual_db",
        "t20_slope_db_per_s",
        "t10_observed_s",
        "t10_fit_start_s",
        "t10_fit_end_s",
        "t10_residual_db",
        "t10_slope_db_per_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for b in result.bands:
            writer.writerow(
                {
                    "band_lo_hz": b.band[0],
                    "band_hi_hz": b.band[1],
                    "status": b.status,
                    "status_reason": b.status_reason,
                    "noise_floor_rms": _fmt(b.noise_floor_rms),
                    "dynamic_range_db": _fmt(b.dynamic_range_db),
                    "analysis_window_start_s": _fmt(b.analysis_window_start_s),
                    "analysis_window_end_s": _fmt(b.analysis_window_end_s),
                    "integration_cut_time_s": _fmt(b.integration_cut_time_s),
                    "gate_db": _fmt(b.gate_db),
                    "clearance_at_minus_5_db": _fmt(b.noise_clearance_db_at_minus_5),
                    "clearance_at_minus_15_db": _fmt(b.noise_clearance_db_at_minus_15),
                    "clearance_at_minus_25_db": _fmt(b.noise_clearance_db_at_minus_25),
                    "t5_s": _fmt(b.t5_s),
                    "t15_s": _fmt(b.t15_s),
                    "t25_s": _fmt(b.t25_s),
                    "t20_s": _fmt(b.t20_s),
                    "t20_fit_start_s": _fmt(b.t20_fit_start_s),
                    "t20_fit_end_s": _fmt(b.t20_fit_end_s),
                    "t20_fit_points": b.t20_fit_points,
                    "t20_residual_db": _fmt(b.t20_residual_db),
                    "t20_slope_db_per_s": _fmt(b.t20_slope_db_per_s),
                    "t10_observed_s": _fmt(b.t10_s),
                    "t10_fit_start_s": _fmt(b.t10_fit_start_s),
                    "t10_fit_end_s": _fmt(b.t10_fit_end_s),
                    "t10_residual_db": _fmt(b.t10_residual_db),
                    "t10_slope_db_per_s": _fmt(b.t10_slope_db_per_s),
                }
            )


def write_decay_curves_csv(result: AnalysisResult, path: str) -> None:
    """Write decay curves in long format: band, time, level, raw energy.

    Each band is gated at its own cut time and therefore has its own time
    vector; long format keeps them on their true sampling grids instead of
    forcing a shared (wrong) axis.
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "band_lo_hz",
                "band_hi_hz",
                "time_s_relative_to_direct",
                "level_db",
                "gated_raw_energy",
            ]
        )
        for b in result.bands:
            idx = b.curve.decimation_indices()
            t = b.curve.times_s[idx]
            level = b.curve.level_db[idx]
            raw = b.curve.raw_energy[idx]
            for k in range(t.size):
                writer.writerow(
                    [f"{b.band[0]:g}", f"{b.band[1]:g}", f"{t[k]:.6f}", f"{level[k]:.6f}", f"{raw[k]:.9e}"]
                )


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return f"{value:.6f}" if abs(value) < 1e6 else f"{value:.6e}"


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_report(result: AnalysisResult, path: str, params: Params) -> None:
    """Produce the combined report figure: recording, deconvolution, decays."""
    plt = _setup_matplotlib()
    sr = result.plan.sample_rate
    fig = plt.figure(figsize=(13, 12))

    # (a) raw recording
    ax = fig.add_subplot(4, 1, 1)
    ax.set_title("Recording (raw PCM16, sample units)")
    rec = result.recording_samples
    t_rec = np.arange(rec.size) / sr
    ax.plot(t_rec, rec, linewidth=0.4, color="#444444")
    ax.axvline(result.localisation.direct_time_s, color="tab:red", linestyle="--", linewidth=0.8,
               label=f"direct lag {result.localisation.direct_time_s:.4f} s")
    for h in result.localisation.harmonic_images:
        if h.in_causal_range and h.predicted_lag >= 0:
            ax.axvspan(
                h.window_start / sr,
                h.window_end / sr,
                color="tab:orange",
                alpha=0.25,
                label=f"harmonic k={h.order} image",
            )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("sample units")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.25)

    # (b) deconvolution
    ax = fig.add_subplot(4, 1, 2)
    ir, scale = result.linear_ir(normalise_peak=True)
    t = np.arange(ir.size) / sr
    ax.plot(t, ir, linewidth=0.4, color="#1f4e79")
    ax.axvline(result.localisation.direct_time_s, color="tab:red", linestyle="--", linewidth=0.8,
               label=f"direct {result.localisation.direct_time_s:.4f} s")
    ax.axvline(result.localisation.valid_time_s, color="tab:gray", linestyle=":", linewidth=0.8,
               label=f"valid end {result.localisation.valid_time_s:.3f} s")
    ax.set_xlim(0, result.localisation.valid_time_s)
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("normalised amplitude")
    ax.set_title("Deconvolution (linear lag-domain response, peak-normalised for display only)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.25)

    # (c) early response zoom
    ax = fig.add_subplot(4, 1, 3)
    zoom_end = min(ir.size, result.localisation.direct_lag + int(0.05 * sr))
    zoom_start = max(0, result.localisation.direct_lag - int(0.02 * sr))
    ax.plot(np.arange(zoom_start, zoom_end) / sr, ir[zoom_start:zoom_end], linewidth=0.8,
            color="#1f4e79")
    ax.axvline(result.localisation.direct_time_s, color="tab:red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("normalised amplitude")
    ax.set_title("Direct arrival zoom")
    ax.grid(alpha=0.25)

    # (d) decay curve for the first band; the rest go in a companion figure
    ax = fig.add_subplot(4, 1, 4)
    _plot_decay(ax, result.bands[0], params)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

    # companion figure with every band's decay curve side by side
    fig2, axes = plt.subplots(len(result.bands), 1, figsize=(11, 4 * max(1, len(result.bands))), squeeze=False)
    for ax, band in zip(axes[:, 0], result.bands):
        _plot_decay(ax, band, params)
    fig2.tight_layout()
    stem, ext = os.path.splitext(path)
    fig2.savefig(f"{stem}_decay{ext}", dpi=130)
    plt.close(fig2)


def plot_comparison(base: AnalysisResult, noisy: AnalysisResult, path: str, params: Params) -> None:
    """Side-by-side base vs high-noise decay curves and credibility bands."""
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, len(base.bands), figsize=(5.2 * len(base.bands), 4.6), squeeze=False)
    for col, (b_base, b_noisy) in enumerate(zip(base.bands, noisy.bands)):
        ax = axes[0, col]
        for band, style, label in (
            (b_noisy, dict(color="tab:red", linestyle="-"), "high noise"),
            (b_base, dict(color="tab:blue", linestyle="-"), "base"),
        ):
            idx = band.curve.decimation_indices()
            t = band.curve.times_s[idx]
            level = band.curve.level_db[idx]
            if t.size > 1:
                ax.plot(t, level, linewidth=1.0, label=label, **style)
                if band.t20_fit_start_s is not None and band.t20_fit_end_s is not None:
                    ax.axvspan(band.t20_fit_start_s, band.t20_fit_end_s,
                               color=style["color"], alpha=0.10)
                ax.axvline(band.integration_cut_time_s, color=style["color"],
                           linestyle=":", linewidth=0.7)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(-45, 5)
        ax.set_title(f"{b_base.band[0]:g}-{b_base.band[1]:g} Hz")
        ax.set_xlabel("time from direct arrival (s)")
        ax.set_ylabel("level (dB)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        # annotate verdicts
        text = (
            f"base: {b_base.status}\nnoisy: {b_noisy.status}\n"
            f"clr@-25: {b_base.noise_clearance_db_at_minus_25:.1f} / "
            f"{b_noisy.noise_clearance_db_at_minus_25:.1f} dB"
        )
        ax.text(0.98, 0.03, text, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_decay(ax, band, params: Params) -> None:
    curve = band.curve
    if curve.times_s.size > 1:
        idx = curve.decimation_indices()
        t = curve.times_s[idx]
        ax.plot(t, curve.level_db[idx], color="#1f4e79", linewidth=1.0,
                label="noise-compensated reverse integral (fitted)")
        # Uncompensated reverse integral: this is the noise plateau that a naive
        # whole-tail integration would produce.  It is shown so the reader can
        # see why the integration is cut at the gate.
        raw = curve.raw_energy
        raw_reverse = np.cumsum(raw[::-1])[::-1]
        if raw_reverse[0] > 0:
            raw_level = 10.0 * np.log10(np.maximum(raw_reverse, raw_reverse[0] * 1e-14) / raw_reverse[0])
            ax.plot(t, raw_level[idx], color="tab:gray", linestyle=":", linewidth=0.7,
                    label="uncompensated integral (noise plateau)")
        if band.integration_cut_time_s == band.integration_cut_time_s:
            ax.axvline(band.integration_cut_time_s, color="tab:green", linestyle="--", linewidth=0.8,
                       label="integration cut (gate)")
        if band.t20_fit_start_s is not None and band.t20_fit_end_s is not None:
            ax.axvspan(band.t20_fit_start_s, band.t20_fit_end_s, color="tab:blue", alpha=0.15,
                       label="T20 fit interval")
        if band.t10_fit_start_s is not None and band.t10_fit_end_s is not None:
            ax.axvspan(band.t10_fit_start_s, band.t10_fit_end_s, color="tab:purple", alpha=0.12,
                       label="-5..-15 dB observed span")
    ax.set_xlim(0, max(0.5, band.integration_cut_time_s * 1.2))
    ax.set_ylim(-60, 5)
    ax.set_xlabel("time from direct arrival (s)")
    ax.set_ylabel("level (dB)")
    status = band.status
    value = ""
    if band.t20_s is not None:
        value = f"T20={band.t20_s:.3f}s"
    elif band.t10_s is not None:
        value = f"observed T10={band.t10_s:.3f}s"
    ax.set_title(f"{band.band[0]:g}-{band.band[1]:g} Hz  [{status}] {value}")
    if band.noise_sensitivity_t20_range_s is not None:
        lo, hi = band.noise_sensitivity_t20_range_s
        ax.text(
            0.98,
            0.03,
            f"floor sensitivity: {lo:.3f}-{hi:.3f} s",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, loc="lower left")
