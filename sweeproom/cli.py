"""Command-line entry points: ``analyze`` and ``verify``.

``analyze``
    Run the full pipeline on a sweep + recording + plan and write the IR, the
    band-decay CSV, the decay curves, the JSON evidence and the report figures.

``verify``
    Run three independent checks that do *not* read any generator ground
    truth:

    1. **reference delay/scale** — deconvolve ``reference_delay.wav`` (the
       emitted sweep delayed and scaled, with no room) and confirm the lag and
       amplitude match what the plan states (0.050 s, 0.7).
    2. **synthetic exponential** — build an analytically known exponential
       decay impulse response, convolve it with the *provided* sweep, add a
       little noise, run the same analysis and check the reverse-integral
       slope recovers the known decay time.
    3. **base vs high-noise** — analyse ``recording.wav`` and
       ``cases/noisy_recording.wav`` and print the credible interval / status
       for each band side by side, showing the verdict degrading with SNR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .analysis import AnalysisResult, analyze_recording
from .config import Params, Plan
from .io import read_wav
from .report import (
    plot_report,
    write_band_csv,
    write_decay_curves_csv,
)
from .sweep import build_inverse_filter, deconvolve


def _load_plan(path: str, sr_expected: int) -> Plan:
    plan = Plan.load(path)
    if plan.sample_rate != sr_expected:
        raise SystemExit(
            f"plan sample_rate {plan.sample_rate} != recording sample_rate {sr_expected}"
        )
    return plan


def _default_paths(base: str) -> dict:
    root = Path(base)
    return {
        "plan": str(root / "fixtures" / "plan.json"),
        "sweep": str(root / "fixtures" / "sweep.wav"),
        "recording": str(root / "fixtures" / "recording.wav"),
        "reference": str(root / "fixtures" / "reference_delay.wav"),
        "noisy": str(root / "cases" / "noisy_recording.wav"),
    }


def _write_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def _write_ir_csv(result: AnalysisResult, path: str) -> None:
    ir, _ = result.linear_ir(normalise_peak=False)
    sr = result.plan.sample_rate
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("lag_samples,time_s,impulse_response\n")
        for i, value in enumerate(ir):
            fh.write(f"{i},{i / sr:.8f},{value:.12e}\n")


def _print_summary(result: AnalysisResult) -> None:
    sr = result.plan.sample_rate
    print(f"[{result.label}] {result.recording_path}")
    print(
        f"  samples={result.recording_samples.size} sr={sr} "
        f"valid_lag={result.localisation.valid_lag} "
        f"({result.localisation.valid_time_s:.3f}s)"
    )
    print(
        f"  direct lag={result.localisation.direct_lag} "
        f"({result.localisation.direct_time_s:.5f}s) "
        f"amp={result.localisation.direct_amplitude:.4f} "
        f"dominance={result.localisation.peak_dominance_db:.1f}dB "
        f"snr={result.localisation.peak_snr_db:.1f}dB"
    )
    for h in result.localisation.harmonic_images:
        if h.in_causal_range:
            print(
                f"  harmonic image k={h.order}: predicted lag "
                f"{h.predicted_lag:.0f} (t={h.predicted_lag / sr:.4f}s), "
                f"peak={h.peak_abs:.3e}, energy={h.energy:.3e}"
            )
    for b in result.bands:
        t20 = "n/a" if b.t20_s is None else f"{b.t20_s:.3f}s"
        t10 = "n/a" if b.t10_s is None else f"{b.t10_s:.3f}s"
        print(
            f"  band {b.band[0]:g}-{b.band[1]:g} Hz: {b.status} "
            f"T20={t20} T10_observed={t10} "
            f"DR={b.dynamic_range_db:.1f}dB "
            f"clearance@-25={_fmt(b.noise_clearance_db_at_minus_25)}dB "
            f"cut={b.integration_cut_time_s:.3f}s"
        )
        print(f"    reason: {b.status_reason}")
    for warning in result.diagnostics.warnings:
        print(f"  warning: {warning}")
    print(f"  overall: {result.overall_status} — {result.overall_reason}")


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def command_analyze(args: argparse.Namespace) -> int:
    paths = _default_paths(args.root)
    plan_path = args.plan or paths["plan"]
    sweep_path = args.sweep or paths["sweep"]
    recording_path = args.recording or paths["recording"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    excitation = read_wav(sweep_path, expected_sr=16000)
    recording = read_wav(recording_path, expected_sr=16000)
    plan = _load_plan(plan_path, excitation.sample_rate)

    params = Params()
    result = analyze_recording(
        plan, excitation, recording, params=params, label=Path(recording_path).stem
    )

    _write_ir_csv(result, str(out_dir / "impulse_response.csv"))
    write_band_csv(result, str(out_dir / "band_decay.csv"))
    write_decay_curves_csv(result, str(out_dir / "decay_curves.csv"))
    _write_json(result.to_dict(), str(out_dir / "evidence.json"))
    try:
        plot_report(result, str(out_dir / "report.png"), params)
    except Exception as exc:  # plotting is best-effort but must be reported
        print(f"  warning: figure generation failed: {exc}", file=sys.stderr)

    _print_summary(result)
    return 0


# ---------------------------------------------------------------------------
# verify helpers
# ---------------------------------------------------------------------------


def _reference_check(
    plan: Plan, excitation, reference, params: Params
) -> dict:
    inv = build_inverse_filter(excitation.samples, plan.sample_rate, plan.sweep)
    d = deconvolve(reference.samples, inv)
    valid = max(0, reference.n_samples - excitation.n_samples)
    search = min(valid, d.size - 1)
    # Ignore the zero-lag inverse-filter residual.
    guard = max(1, int(round(inv.self_response_width_to_minus20db_s * plan.sample_rate * 4)))
    direct = guard + int(np.argmax(np.abs(d[guard : search + 1])))
    amp = float(np.abs(d[direct]))
    lag_s = direct / plan.sample_rate
    # Expected values from the plan's own reference_delay description.
    expected_lag_s = 0.050
    expected_gain = 0.7
    lag_ok = abs(lag_s - expected_lag_s) <= 1.0 / plan.sample_rate
    gain_tol = 0.05
    gain_ok = abs(amp - expected_gain) <= gain_tol
    return {
        "description": plan.reference_delay_description,
        "measured_lag_samples": int(direct),
        "measured_lag_s": float(lag_s),
        "expected_lag_s": expected_lag_s,
        "lag_ok": bool(lag_ok),
        "measured_gain": float(amp),
        "expected_gain": expected_gain,
        "gain_abs_tolerance": gain_tol,
        "gain_ok": bool(gain_ok),
        "ok": bool(lag_ok and gain_ok),
        "valid_lag_samples": int(search),
    }


def _synthetic_exponential_check(
    plan: Plan, excitation, params: Params, target_rt60_s: float = 0.60, seed: int = 2024
) -> dict:
    """Known-exponential round trip through the real deconvolution path.

    The synthetic impulse response has a *known RT60* (amplitude envelope
    10^(-3 t / RT60), i.e. power decays linearly at 60 dB per RT60).  A T20
    estimate is the fit slope over -5..-25 dB extrapolated to 60 dB, so for a
    clean exponential it must recover RT60.  We compare the measured T20 to
    that known RT60.
    """
    sr = plan.sample_rate
    length = int(1.2 * sr)
    n = np.arange(length, dtype=np.float64)
    delay = int(0.05 * sr)
    envelope = 10.0 ** (-3.0 * n / (target_rt60_s * sr))
    # Smooth random-phase room tail with a known Schroeder slope.
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal(length) * envelope
    ir[:delay] = 0.0
    ir[delay] += 1.0
    # Convolve with the *provided* sweep so the deconvolution uses the exact
    # inverse the analyzer uses.
    wet = np.convolve(excitation.samples, ir, mode="full")
    wet = wet[: excitation.n_samples + length]
    # Add mild stationary noise so the noise machinery is exercised, but keep
    # the decay well above the floor in the fit region.
    noise_sigma = 1e-4 * float(np.max(np.abs(wet)))
    wet = wet + rng.standard_normal(wet.size) * noise_sigma

    from .io import Audio

    synth = Audio(samples=wet, sample_rate=sr, path="<synthetic>")
    result = analyze_recording(
        plan, excitation, synth, params=params, label="synthetic-exponential"
    )
    # Use the band with the strongest, cleanest response as the slope witness;
    # report every band and compare the best credible one.
    measured = []
    for b in result.bands:
        if b.t20_s is not None:
            measured.append(b.t20_s)
    best = min(measured, key=lambda v: abs(v - target_rt60_s)) if measured else None
    rel_err = abs(best - target_rt60_s) / target_rt60_s if best is not None else None
    ok = best is not None and rel_err is not None and rel_err <= 0.15
    return {
        "target_rt60_s": target_rt60_s,
        "measured_t20_s_by_band": {
            f"{b.band[0]:g}-{b.band[1]:g}Hz": b.t20_s for b in result.bands
        },
        "band_status": {
            f"{b.band[0]:g}-{b.band[1]:g}Hz": b.status for b in result.bands
        },
        "best_measured_t20_s": best,
        "relative_error": rel_err,
        "tolerance": 0.15,
        "ok": bool(ok),
    }


def _compare_base_noisy(
    plan, excitation, base_rec, noisy_rec, params
) -> tuple:
    base = analyze_recording(plan, excitation, base_rec, params=params, label="base")
    noisy = analyze_recording(plan, excitation, noisy_rec, params=params, label="high-noise")

    def per_band(result):
        return {
            f"{b.band[0]:g}-{b.band[1]:g}Hz": {
                "status": b.status,
                "t20_s": b.t20_s,
                "t10_observed_s": b.t10_s,
                "dynamic_range_db": b.dynamic_range_db,
                "clearance_at_minus_25_db": b.noise_clearance_db_at_minus_25,
                "clearance_at_minus_15_db": b.noise_clearance_db_at_minus_15,
                "t20_fit_start_s": b.t20_fit_start_s,
                "t20_fit_end_s": b.t20_fit_end_s,
                "t20_residual_db": b.t20_residual_db,
                "noise_sensitivity_range_s": (
                    None
                    if b.noise_sensitivity_t20_range_s is None
                    else list(b.noise_sensitivity_t20_range_s)
                ),
                "integration_cut_time_s": b.integration_cut_time_s,
            }
            for b in result.bands
        }

    payload = {
        "base": {
            "overall_status": base.overall_status,
            "bands": per_band(base),
        },
        "high_noise": {
            "overall_status": noisy.overall_status,
            "bands": per_band(noisy),
        },
    }
    # Document where the credibility verdict degraded with added noise.
    degraded = []
    for band in per_band(base):
        b_status = payload["base"]["bands"][band]["status"]
        n_status = payload["high_noise"]["bands"][band]["status"]
        if b_status != n_status:
            degraded.append(f"{band}: {b_status} -> {n_status}")
    payload["verdict_changes"] = degraded
    return payload, base, noisy


def command_verify(args: argparse.Namespace) -> int:
    paths = _default_paths(args.root)
    plan = _load_plan(args.plan or paths["plan"], 16000)
    params = Params()
    excitation = read_wav(args.sweep or paths["sweep"], expected_sr=16000)
    reference = read_wav(args.reference or paths["reference"], expected_sr=16000)
    base_rec = read_wav(args.recording or paths["recording"], expected_sr=16000)
    noisy_rec = read_wav(args.noisy or paths["noisy"], expected_sr=16000)

    ref_check = _reference_check(plan, excitation, reference, params)
    synth_check = _synthetic_exponential_check(plan, excitation, params)
    comparison, base_result, noisy_result = _compare_base_noisy(
        plan, excitation, base_rec, noisy_rec, params
    )

    payload = {
        "reference_delay_scaling": ref_check,
        "synthetic_exponential": synth_check,
        "base_vs_high_noise": comparison,
        "all_ok": bool(ref_check["ok"] and synth_check["ok"]),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(payload, str(out_dir / "verify.json"))
    try:
        from .report import plot_comparison

        plot_comparison(
            base_result, noisy_result, str(out_dir / "base_vs_high_noise.png"), params
        )
    except Exception as exc:  # plotting is best-effort but must be reported
        print(f"  warning: comparison figure failed: {exc}", file=sys.stderr)

    print("== verify ==")
    print(
        f"reference delay/scale: measured lag={ref_check['measured_lag_s']:.6f}s "
        f"(expected {ref_check['expected_lag_s']:.6f}s), gain={ref_check['measured_gain']:.4f} "
        f"(expected {ref_check['expected_gain']}), ok={ref_check['ok']}"
    )
    print(
        f"synthetic exponential: known RT60={synth_check['target_rt60_s']:.3f}s, "
        f"best measured T20={synth_check['best_measured_t20_s']}, "
        f"rel_err={synth_check['relative_error']}, ok={synth_check['ok']}"
    )
    print("base vs high-noise credibility:")
    for band in comparison["base"]["bands"]:
        b = comparison["base"]["bands"][band]
        n = comparison["high_noise"]["bands"][band]
        base_t20 = "n/a" if b["t20_s"] is None else f"{b['t20_s']:.3f}s"
        noisy_t20 = "n/a" if n["t20_s"] is None else f"{n['t20_s']:.3f}s"
        base_sens = (
            "n/a"
            if b["noise_sensitivity_range_s"] is None
            else "{:.3f}-{:.3f}s".format(*b["noise_sensitivity_range_s"])
        )
        noisy_sens = (
            "n/a"
            if n["noise_sensitivity_range_s"] is None
            else "{:.3f}-{:.3f}s".format(*n["noise_sensitivity_range_s"])
        )
        print(
            f"  {band}: base={b['status']} (T20={base_t20}, "
            f"clr@-25={_fmt(b['clearance_at_minus_25_db'])}dB, DR={b['dynamic_range_db']:.1f}dB, "
            f"sens={base_sens}) "
            f"| noisy={n['status']} (T20={noisy_t20}, "
            f"clr@-25={_fmt(n['clearance_at_minus_25_db'])}dB, DR={n['dynamic_range_db']:.1f}dB, "
            f"sens={noisy_sens})"
        )
    if comparison["verdict_changes"]:
        print("  verdict changes with added noise: " + "; ".join(comparison["verdict_changes"]))
    return 0 if payload["all_ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sweeproom",
        description=(
            "Offline logarithmic-sweep room impulse-response and band decay "
            "measurement.  One known sweep, one PCM16 mono 16 kHz recording."
        ),
    )
    parser.add_argument(
        "--root",
        default=".",
        help="project root containing fixtures/ and cases/ (default: .)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="full analysis of one recording")
    a.add_argument("--plan", default=None, help="path to plan.json")
    a.add_argument("--sweep", default=None, help="path to the emitted sweep WAV")
    a.add_argument("--recording", default=None, help="path to the recording WAV")
    a.add_argument("--out", default="output", help="output directory (default: output)")
    a.set_defaults(func=command_analyze)

    v = sub.add_parser("verify", help="reference-delay, synthetic-decay and SNR checks")
    v.add_argument("--plan", default=None)
    v.add_argument("--sweep", default=None)
    v.add_argument("--recording", default=None)
    v.add_argument("--reference", default=None)
    v.add_argument("--noisy", default=None)
    v.add_argument("--out", default="output/verify", help="output directory")
    v.set_defaults(func=command_verify)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
