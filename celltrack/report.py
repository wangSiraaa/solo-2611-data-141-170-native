"""Output tables: objects, events, lineage, mother-daughter relations.

All tables are written as CSV (machine-friendly) and a JSON summary.  Every row
carries enough provenance to return to the original label pixels:
``frame_id`` + ``label`` (and optionally the TIFF file name / bounding box).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .lineage import TrackingResult


def _fmt(x: Any, ndigits: int = 4) -> Any:
    if isinstance(x, float):
        return round(x, ndigits)
    return x


def write_objects(result: TrackingResult, out_dir: Path) -> Path:
    path = out_dir / "objects.csv"
    fieldnames = [
        "obs_id", "frame_id", "frame_index", "time_minutes", "file", "label",
        "area_px", "area_um2", "centroid_x_px", "centroid_y_px",
        "centroid_x_um", "centroid_y_um", "bbox_min_x", "bbox_min_y",
        "bbox_max_x", "bbox_max_y", "equiv_diameter_um", "major_axis_um",
        "minor_axis_um", "eccentricity", "circularity", "solidity",
        "orientation_deg", "perimeter_um", "track_id", "mother_track_id",
        "warnings",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for o in sorted(result.objects, key=lambda r: (r.frame_index, r.label)):
            w.writerow({
                "obs_id": o.obs_id, "frame_id": o.frame_id,
                "frame_index": o.frame_index, "time_minutes": o.time_minutes,
                "file": o.file, "label": o.label,
                "area_px": o.area_px, "area_um2": _fmt(o.area_um2),
                "centroid_x_px": _fmt(o.centroid_x_px, 3),
                "centroid_y_px": _fmt(o.centroid_y_px, 3),
                "centroid_x_um": _fmt(o.centroid_x_um),
                "centroid_y_um": _fmt(o.centroid_y_um),
                "bbox_min_x": o.bbox_min_x, "bbox_min_y": o.bbox_min_y,
                "bbox_max_x": o.bbox_max_x, "bbox_max_y": o.bbox_max_y,
                "equiv_diameter_um": _fmt(o.equiv_diameter_um),
                "major_axis_um": _fmt(o.major_axis_um),
                "minor_axis_um": _fmt(o.minor_axis_um),
                "eccentricity": _fmt(o.eccentricity),
                "circularity": _fmt(o.circularity),
                "solidity": _fmt(o.solidity),
                "orientation_deg": _fmt(o.orientation_deg, 2),
                "perimeter_um": _fmt(o.perimeter_um),
                "track_id": result.object_track.get(o.obs_id, ""),
                "mother_track_id": (
                    "" if result.object_mother.get(o.obs_id) is None
                    else result.object_mother[o.obs_id]
                ),
                "warnings": ";".join(o.warnings),
            })
    return path


def _event_row(ln, idx: int, pair_dt: float | None, labels: dict) -> dict:
    prev_lbl = labels.get(ln.prev_id, ("", "")) if ln.prev_id else ("", "")
    next_lbls = [labels.get(b, ("", "")) for b in ln.next_ids]
    return {
        "event_id": f"e{idx:03d}",
        "kind": ln.kind,
        "prev_frame": "" if ln.prev_frame is None else ln.prev_frame,
        "next_frame": "" if ln.next_frame is None else ln.next_frame,
        "prev_frame_id": prev_lbl[0],
        "next_frame_ids": ";".join(lbl[0] for lbl in next_lbls),
        "prev_label": prev_lbl[1],
        "next_labels": ";".join(str(lbl[1]) for lbl in next_lbls),
        "dt_minutes": "" if pair_dt is None else pair_dt,
        "prev_obs_id": ln.prev_id or "",
        "next_obs_ids": ";".join(ln.next_ids),
        "cost": _fmt(ln.cost, 6),
        "undecided": str(ln.undecided),
        "chosen_is_representative": str(ln.chosen_is_representative),
        "evidence_json": json.dumps(ln.evidence, sort_keys=True),
    }


def write_events(result: TrackingResult, out_dir: Path) -> Path:
    path = out_dir / "events.csv"
    fieldnames = ["event_id", "kind", "prev_frame", "next_frame", "prev_frame_id",
                  "next_frame_ids", "prev_label", "next_labels", "dt_minutes",
                  "prev_obs_id", "next_obs_ids", "cost", "undecided",
                  "chosen_is_representative", "evidence_json"]
    # attach dt per pair
    dts: dict[tuple[int, int], float] = {}
    for ps in result.frame_pairs:
        dts[(ps.pair.frame_index_prev, ps.pair.frame_index_next)] = ps.pair.dt_minutes
    labels = {o.obs_id: (o.frame_id, o.label) for o in result.objects}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for i, ln in enumerate(result.links):
            dt = None
            if ln.prev_frame is not None and ln.next_frame is not None:
                dt = dts.get((ln.prev_frame, ln.next_frame))
            w.writerow(_event_row(ln, i, dt, labels))
    return path


def write_tracks(result: TrackingResult, out_dir: Path) -> Path:
    path = out_dir / "tracks.csv"
    fieldnames = ["track_id", "start_frame", "end_frame", "n_observations",
                  "obs_ids", "frame_ids", "parent_track_id", "mother_track_id",
                  "children_track_ids", "lineage_fate", "boundary_start",
                  "undecided_any"]
    nobs = {o.obs_id: o for o in result.objects}
    # undecided events map to obs ids
    undecided_obs = set()
    for ln in result.links:
        if ln.undecided:
            if ln.prev_id:
                undecided_obs.add(ln.prev_id)
            undecided_obs.update(ln.next_ids)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for tid, tr in result.tracks.items():
            fids = ",".join(
                o.frame_id for o in sorted((nobs[o] for o in tr.obs_ids),
                                           key=lambda r: r.time_minutes)
            )
            w.writerow({
                "track_id": tid, "start_frame": tr.start_frame, "end_frame": tr.end_frame,
                "n_observations": len(tr.obs_ids), "obs_ids": ";".join(tr.obs_ids),
                "frame_ids": fids,
                "parent_track_id": "" if tr.parent_track_id is None else tr.parent_track_id,
                "mother_track_id": "" if tr.mother_track_id is None else tr.mother_track_id,
                "children_track_ids": ";".join(str(c) for c in tr.children_track_ids),
                "lineage_fate": tr.lineage_fate,
                "boundary_start": str(tr.boundary_start),
                "undecided_any": str(any(o in undecided_obs for o in tr.obs_ids)),
            })
    return path


def write_mother_daughters(result: TrackingResult, out_dir: Path) -> Path:
    path = out_dir / "mother_daughters.csv"
    fieldnames = ["mother_track_id", "mother_obs_id", "division_frame_prev",
                  "division_frame_next", "daughter1_track_id", "daughter1_obs_id",
                  "daughter2_track_id", "daughter2_obs_id", "cost", "undecided",
                  "evidence_json"]
    div_links = [ln for ln in result.links if ln.kind == "division"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for ln in div_links:
            mtid = result.object_track.get(ln.prev_id)
            d1, d2 = ln.next_ids[0], ln.next_ids[1]
            w.writerow({
                "mother_track_id": mtid,
                "mother_obs_id": ln.prev_id,
                "division_frame_prev": ln.prev_frame,
                "division_frame_next": ln.next_frame,
                "daughter1_track_id": result.object_track.get(d1),
                "daughter1_obs_id": d1,
                "daughter2_track_id": result.object_track.get(d2),
                "daughter2_obs_id": d2,
                "cost": _fmt(ln.cost, 6),
                "undecided": str(ln.undecided),
                "evidence_json": json.dumps(ln.evidence, sort_keys=True),
            })
    return path


def write_candidates(result: TrackingResult, out_dir: Path) -> Path:
    """Full candidate audit: every admissible candidate, selected or not."""
    path = out_dir / "candidates.csv"
    fieldnames = ["pair_prev_frame", "pair_next_frame", "kind", "prev_obs_id",
                  "next_obs_ids", "cost", "selected", "undecided_pair",
                  "evidence_json"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for ps in result.frame_pairs:
            selected_keys = ps.chosen.event_keys if ps.chosen else frozenset()
            for cand in ps.pair.candidates:
                w.writerow({
                    "pair_prev_frame": ps.pair.frame_index_prev,
                    "pair_next_frame": ps.pair.frame_index_next,
                    "kind": cand.kind,
                    "prev_obs_id": cand.a_id or "",
                    "next_obs_ids": ";".join(cand.b_ids),
                    "cost": _fmt(cand.cost, 6),
                    "selected": str(cand.key in selected_keys),
                    "undecided_pair": str(ps.undecided),
                    "evidence_json": json.dumps(cand.evidence, sort_keys=True),
                })
    return path


def write_alternatives(result: TrackingResult, out_dir: Path) -> Path:
    """Every near-equivalent event set for every flagged frame pair.

    This is the audit trail behind the ``undecided`` flag: the objective is
    reported together with *all* combinations within the ambiguity margin, not
    just the representative that continuity chose.
    """
    path = out_dir / "alternatives.csv"
    fieldnames = ["pair_prev_frame", "pair_next_frame", "alternative_index",
                  "is_chosen", "total_cost", "events", "evaluated_event_sets",
                  "undecided_event_keys", "ambiguity_reason"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for ps in result.frame_pairs:
            if not ps.undecided:
                continue
            chosen_keys = ps.chosen.event_keys if ps.chosen else frozenset()
            contested = ";".join(sorted(repr(k) for k in ps.undecided_event_keys))
            for i, sol in enumerate(ps.alternatives):
                evs = "; ".join(
                    f"{e.kind}({e.a_id or '-'}->{','.join(e.b_ids) or '-'})"
                    for e in sorted(sol.events, key=lambda e: (e.kind, e.a_id or "", e.b_ids))
                )
                w.writerow({
                    "pair_prev_frame": ps.pair.frame_index_prev,
                    "pair_next_frame": ps.pair.frame_index_next,
                    "alternative_index": i,
                    "is_chosen": str(sol.event_keys == chosen_keys),
                    "total_cost": _fmt(sol.cost, 6),
                    "events": evs,
                    "evaluated_event_sets": len(ps.alternatives),
                    "undecided_event_keys": contested,
                    "ambiguity_reason": ps.ambiguity_reason,
                })
    return path


def write_summary(result: TrackingResult, out_dir: Path, series_anomalies: list) -> Path:
    path = out_dir / "summary.json"
    pair_summaries = []
    for ps in result.frame_pairs:
        pair_summaries.append({
            "prev_frame": ps.pair.frame_index_prev,
            "next_frame": ps.pair.frame_index_next,
            "dt_minutes": ps.pair.dt_minutes,
            "n_candidates": len(ps.pair.candidates),
            "n_prev_objects": len(ps.pair.prev_objects),
            "n_next_objects": len(ps.pair.next_objects),
            "optimal_cost": None if ps.chosen is None else round(ps.chosen.cost, 6),
            "undecided": ps.undecided,
            "chosen_is_representative": ps.chosen_is_representative,
            "ambiguity_reason": ps.ambiguity_reason,
            "n_alternatives": len(ps.alternatives),
            "alternatives": [
                {
                    "cost": round(s.cost, 6),
                    "events": [
                        {"kind": e.kind, "prev": e.a_id, "next": list(e.b_ids)}
                        for e in s.events
                    ],
                }
                for s in ps.alternatives
            ],
            "rejected_candidates": ps.pair.rejected,
            "chosen_events": [] if ps.chosen is None else [
                {"kind": e.kind, "prev": e.a_id, "next": list(e.b_ids),
                 "cost": round(e.cost, 6), "evidence": e.evidence}
                for e in ps.chosen.events
            ],
        })
    payload = {
        "validation": result.validation,
        "n_objects": len(result.objects),
        "n_tracks": len(result.tracks),
        "n_links": len(result.links),
        "events_by_kind": {},
        "anomalies": [{"kind": a.kind, "message": a.message,
                       "frame_id": a.frame_id, "severity": a.severity}
                      for a in series_anomalies],
        "pairs": pair_summaries,
    }
    counts: dict[str, int] = {}
    for ln in result.links:
        counts[ln.kind] = counts.get(ln.kind, 0) + 1
    payload["events_by_kind"] = counts
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    return path


def write_all(result: TrackingResult, out_dir: str | Path, series_anomalies: list) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "objects": write_objects(result, out_dir),
        "events": write_events(result, out_dir),
        "tracks": write_tracks(result, out_dir),
        "mother_daughters": write_mother_daughters(result, out_dir),
        "candidates": write_candidates(result, out_dir),
        "alternatives": write_alternatives(result, out_dir),
        "summary": write_summary(result, out_dir, series_anomalies),
    }
