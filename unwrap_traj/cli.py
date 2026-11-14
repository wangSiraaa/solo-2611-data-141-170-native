"""Command-line entry point for ``unwrap``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .core import AnalysisError, Cell, analyze_dataset
from .validate import InputError, load_inputs

ARBITRARY_GAUGE_NOTE = (
    "The absolute lattice image of the first frame in each continuous segment "
    "is an arbitrary gauge choice: the folded input does not fix it. Only "
    "within-segment relative displacements, shapes and bond lengths are "
    "gauge-invariant. Adding one common lattice vector to every frame of a "
    "segment leaves molecular shape and within-segment displacements unchanged."
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        fv = float(value)
        return fv if np.isfinite(fv) else None
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="unwrap",
        description=(
            "Reconstruct finite molecules and their continuous centre-of-mass "
            "trajectories from periodically folded atomic coordinates."
        ),
    )
    p.add_argument("--trajectory", required=True, help="NPZ with positions_nm, time_ps, cell_columns_nm")
    p.add_argument("--topology", required=True, help="topology JSON (atoms, masses, bonds)")
    p.add_argument("--plan", required=True, help="plan JSON (bounds, lag times)")
    p.add_argument("--output", required=True, help="output JSON path")
    p.add_argument("--npz-output", help="optional output NPZ with reconstructed arrays")
    p.add_argument(
        "--allow-missing-masses",
        action="store_true",
        help="diagnostic mode: do not require mass on every atom",
    )
    p.add_argument(
        "--max-bond-nm",
        type=float,
        default=None,
        help=(
            "override the bond bound (default: topology max_bond_nm if present, "
            "else plan max_bond_nm)"
        ),
    )
    p.add_argument("--quiet", action="store_true", help="suppress progress summary on stderr")
    return p


def run(args: argparse.Namespace) -> int:
    loaded = load_inputs(
        args.trajectory,
        args.topology,
        args.plan,
        allow_missing_masses=args.allow_missing_masses,
        max_bond_nm_override=args.max_bond_nm,
    )
    result = analyze_dataset(
        loaded.positions_nm,
        loaded.time_ps,
        loaded.cell,
        loaded.topology,
        loaded.options,
    )

    arrays = result["arrays"]
    report = {
        "command": "unwrap",
        "version": "1.0.0",
        "conventions": {
            "positions_shape": "frame x atom x xyz",
            "length_unit": "nm",
            "time_unit": "ps",
            "cell_layout": "cell_columns_nm columns are lattice vectors",
            "bond_indices": "topology bonds_zero_based, zero-based global atom indices",
            "lattice_image": (
                "unwrapped = folded + sum_k m_k a_k, where m_k are integer "
                "lattice-offset components and a_k are the cell columns"
            ),
            "mass_unit": "relative atomic mass",
            "origin_gauge_note": ARBITRARY_GAUGE_NOTE,
            "continuous_segment": (
                "A contiguous run of frames joined by unique feasible COM "
                "lattice translations. Breaks never carry accumulated "
                "displacement across them."
            ),
        },
        "provenance": loaded.provenance,
        "checks": loaded.checks,
        "summary": result["summary"],
        "molecules": result["molecules"],
        "boundary_crossings": result["boundary_crossings"],
        "msd_lag_policy": (
            "MSD is computed only from pairs inside one continuous segment and "
            "only for lags with at least one such pair. Uncovered lags are "
            "reported with covered=false and msd_nm2=null; they are never 0."
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(report), fh, indent=2, allow_nan=False)
        fh.write("\n")

    if args.npz_output:
        npz_path = Path(args.npz_output)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            frame_index=arrays["frame_index"],
            atom_index=arrays["atom_index"],
            atom_ids=arrays["atom_ids"],
            atom_molecule_ids=arrays["atom_molecule_ids"],
            molecule_ids=arrays["molecule_ids"],
            time_ps=arrays["time_ps"],
            positions_folded_nm=arrays["positions_folded_nm"],
            positions_unwrapped_nm=arrays["positions_unwrapped_nm"],
            atom_unwrapped_valid=arrays["atom_unwrapped_valid"],
            atom_lattice_offsets_total=arrays["atom_lattice_offsets_total"],
            com_unwrapped_nm=arrays["com_unwrapped_nm"],
            com_valid=arrays["com_valid"],
            segment_id=arrays["segment_id"],
            cumulative_lattice_translation=arrays["cumulative_lattice_translation"],
            cumulative_translation_valid=arrays["cumulative_translation_valid"],
            step_lattice_translation=arrays["step_lattice_translation"],
            step_translation_valid=arrays["step_translation_valid"],
            cell_columns_nm=arrays["cell_columns_nm"],
        )

    if not args.quiet:
        s = result["summary"]
        print(
            f"unwrap: {s['n_frames']} frames, {s['n_atoms']} atoms, "
            f"{s['n_molecules']} molecules; all_frames_reconstructed="
            f"{s['all_frames_reconstructed']}; molecules_with_breaks="
            f"{s['molecules_with_breaks']}",
            file=sys.stderr,
        )
        for lag_key, item in s["pooled_msd"].items():
            if item["covered"]:
                print(
                    f"  lag {lag_key} ps: MSD={item['msd_nm2']:.6g} nm^2 "
                    f"over {item['pair_count']} pairs",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  lag {lag_key} ps: uncovered (no within-segment pair)",
                    file=sys.stderr,
                )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (InputError, AnalysisError) as exc:
        print(f"unwrap: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
