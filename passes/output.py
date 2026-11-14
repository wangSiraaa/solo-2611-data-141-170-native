"""Serialisation of the solver artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .solver import SolveOutput


def _json_ready(value: Any) -> Any:
    """Recursively convert NumPy scalars/arrays and validate finiteness."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_ready(value), indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


_OBSERVATION_COLUMNS = [
    "utc",
    "jd",
    "seconds_from_start_s",
    "pass_id",
    "window_id",
    "row_role",
    "azimuth_deg",
    "elevation_deg",
    "horizon_threshold_deg",
    "gap_deg",
    "visible",
    "gmst_deg",
    "range_km",
    "teme_x_km",
    "teme_y_km",
    "teme_z_km",
    "ecef_x_km",
    "ecef_y_km",
    "ecef_z_km",
    "enu_east_km",
    "enu_north_km",
    "enu_up_km",
]


def write_observations_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_OBSERVATION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key)) for key in _OBSERVATION_COLUMNS})


def _format_cell(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, (np.floating,)):
        return format(float(value), ".12g")
    return value


def write_artifacts(out_dir: Path, result: SolveOutput) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "passes": out_dir / "passes.json",
        "boundaries": out_dir / "boundary_evidence.json",
        "observations_csv": out_dir / "observations.csv",
        "verification": out_dir / "verification.json",
    }
    write_json(paths["passes"], result.payload)
    write_json(paths["boundaries"], result.boundary_evidence)
    write_json(paths["verification"], result.verification)
    write_observations_csv(paths["observations_csv"], result.observations)
    return paths


def render_summary(result: SolveOutput) -> str:
    payload = result.payload
    summary = payload["summary"]
    lines: list[str] = []
    lines.append(f"status: {payload['status']}")
    lines.append(
        f"plan: {payload['plan']['start_utc']} .. {payload['plan']['end_utc']} "
        f"(half-open, {payload['plan']['duration_s']:.0f} s)"
    )
    lines.append(
        f"passes: {summary['pass_count']}  visible windows: {summary['window_count']}  "
        f"split passes: {summary['split_pass_count']}  "
        f"visible total: {summary['visible_seconds_total']:.3f} s"
    )
    lines.append("")
    lines.append(
        f"{'pass':>4} {'win':>4} {'start UTC':>26} {'end UTC':>26} {'dur s':>9} "
        f"{'peak el':>9} {'entry az':>9} {'exit az':>9}"
    )
    for window in payload["windows"]:
        lines.append(
            f"{window['pass_id']:>4} {window['window_id'].split('_')[-1]:>4} "
            f"{window['start_utc']:>26} {window['end_utc']:>26} "
            f"{window['duration_s']:>9.3f} {window['peak']['elevation_deg']:>9.3f} "
            f"{window['entry']['azimuth_deg']:>9.3f} {window['exit']['azimuth_deg']:>9.3f}"
        )
    lines.append("")
    lines.append("split gaps confirmed by fine audit:")
    for item in result.verification.get("gap_fine_audit", []):
        lines.append(
            f"  {item['before_end_utc']} .. {item['after_start_utc']}  "
            f"{item['duration_s']:.3f} s  max gap {item['max_gap_deg']:.3f} deg  "
            f"confirmed={item['confirmed_full_occlusion']}"
        )
    lines.append("")
    lines.append(
        f"verification: ok={result.verification['ok']} "
        f"(ENU={result.verification['enu_axis_checks_ok']}, "
        f"boundaries={result.verification['boundary_checks_ok']}, "
        f"scalar/vector={result.verification['scalar_vector_comparison']['ok']}, "
        f"gap audit={result.verification['gap_fine_audit_ok']}, "
        f"neighbourhoods={result.verification['boundary_neighbourhood_audit_ok']})"
    )
    lines.append(
        f"audit roots: {result.verification['audit']['roots']} "
        f"(solver {payload['search']['merged_roots']}), "
        f"unmatched audit={summary['unmatched_audit_roots']}, "
        f"unmatched solver={summary['unmatched_solver_roots']}"
    )
    return "\n".join(lines)
