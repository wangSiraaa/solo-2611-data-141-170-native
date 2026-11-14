"""``track`` command-line entry point.

Usage
-----
    python -m celltrack track --series fixtures/series.json --out-dir out

Options allow overriding the candidate gates and cost weights from
:mod:`celltrack.config`; defaults come from series.json plus documented
modelling assumptions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import config_from_series, TrackConfig
from .features import extract_features
from .lineage import run_tracking
from .report import write_all
from .series import parse_series


def _add_overrides(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-speed-um-per-min", type=float, default=None,
                   help="override series max_centroid_speed_um_per_min")
    p.add_argument("--divide-ratio-low", type=float, default=None)
    p.add_argument("--divide-ratio-high", type=float, default=None)
    p.add_argument("--division-max-daughter-sep-um", type=float, default=None)
    p.add_argument("--division-max-mother-daughter-dist-um", type=float, default=None)
    p.add_argument("--ambiguity-margin", type=float, default=None)
    p.add_argument("--max-alternatives", type=int, default=None)
    p.add_argument("--birth-cost", type=float, default=None)
    p.add_argument("--disappearance-cost", type=float, default=None)
    p.add_argument("--no-velocity", action="store_true",
                   help="disable constant-velocity prediction (nearest-centroid cost)")


def _cfg_from_args(series, args) -> TrackConfig:
    overrides = {}
    if args.birth_cost is not None:
        overrides["birth_cost"] = args.birth_cost
    if args.disappearance_cost is not None:
        overrides["disappearance_cost"] = args.disappearance_cost
    if args.division_max_daughter_sep_um is not None:
        overrides["division_max_daughter_sep_um"] = args.division_max_daughter_sep_um
    if args.division_max_mother_daughter_dist_um is not None:
        overrides["division_max_mother_daughter_dist_um"] = args.division_max_mother_daughter_dist_um
    if args.ambiguity_margin is not None:
        overrides["ambiguity_cost_margin"] = args.ambiguity_margin
    if args.max_alternatives is not None:
        overrides["max_alternatives"] = args.max_alternatives
    if getattr(args, "no_velocity", False):
        overrides["use_velocity_prediction"] = False
    base = config_from_series(series.raw)
    if args.max_speed_um_per_min is not None:
        overrides["max_centroid_speed_um_per_min"] = args.max_speed_um_per_min
    if args.divide_ratio_low is not None or args.divide_ratio_high is not None:
        lo, hi = base.division_area_ratio_range
        overrides["division_area_ratio_range"] = (
            args.divide_ratio_low if args.divide_ratio_low is not None else lo,
            args.divide_ratio_high if args.divide_ratio_high is not None else hi,
        )
    if overrides:
        cfg = TrackConfig(**{**base.__dict__, **overrides})
    else:
        cfg = base
    return cfg


def cmd_track(args: argparse.Namespace) -> int:
    try:
        series = parse_series(args.series, args.base_dir)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    cfg = _cfg_from_args(series, args)
    features = extract_features(series, cfg.min_reliable_area_um2)
    result = run_tracking(series, cfg, features=features)
    outputs = write_all(result, args.out_dir, series.anomalies)

    # ---- console report -------------------------------------------------
    print(f"series        : {args.series}")
    print(f"frames        : {len(series.frames)}  objects: {len(features)}")
    print(f"tracks        : {len(result.tracks)}  links: {len(result.links)}")
    print(f"validation    : {'OK' if result.validation['ok'] else 'FAILED'}")
    if not result.validation["ok"]:
        for err in result.validation["errors"]:
            print(f"  - {err}", file=sys.stderr)
    counts: dict[str, int] = {}
    for ln in result.links:
        counts[ln.kind] = counts.get(ln.kind, 0) + 1
    print(f"events        : {counts}")
    undecided_pairs = [ps for ps in result.frame_pairs if ps.undecided]
    if undecided_pairs:
        print(f"undecided     : {len(undecided_pairs)} frame pair(s) flagged")
        for ps in undecided_pairs:
            print(f"  pair {ps.pair.frame_index_prev}->{ps.pair.frame_index_next}: "
                  f"{len(ps.alternatives)} near-equivalent sets; representative chosen")
    else:
        print("undecided     : none")
    if series.anomalies:
        print("input anomalies:")
        for a in series.anomalies:
            print(f"  [{a.severity}] {a.kind}: {a.message}")
    if args.quiet:
        pass
    else:
        print("\nmother / daughters:")
        for ln in result.links:
            if ln.kind == "division":
                print(f"  {ln.prev_id} -> {ln.next_ids[0]} + {ln.next_ids[1]}  "
                      f"(cost {ln.cost:.3f})")
        print("\ntrack fates:")
        for tid, tr in result.tracks.items():
            fate = tr.lineage_fate or ("daughter" if tr.mother_track_id else
                                       "birth" if tr.boundary_start else "present_at_first_frame")
            print(f"  T{tid:02d} frames {tr.start_frame}-{tr.end_frame} "
                  f"({len(tr.obs_ids)} obs)  {fate}")
    print("\noutputs:")
    for name, path in outputs.items():
        print(f"  {name:16s} {path}")
    return 0 if result.validation["ok"] else 2


def cmd_validate(args: argparse.Namespace) -> int:
    """Run only the integrity checks on a produced output directory."""
    out = Path(args.out_dir)
    summary_path = out / "summary.json"
    if not summary_path.exists():
        print(f"missing {summary_path}", file=sys.stderr)
        return 2
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ok = summary.get("validation", {}).get("ok", False)
    print(json.dumps(summary.get("validation", {}), indent=2))
    return 0 if ok else 2


def _explain_obs(obs_id: str, result, labels: dict) -> None:
    """Print, in order, every event that touches obs_id and why."""
    feats = {o.obs_id: o for o in result.objects}
    if obs_id not in feats:
        # allow a suffix match such as "l141" or ":l141"
        matches = [oid for oid in feats if oid.endswith(obs_id)]
        if len(matches) == 1:
            obs_id = matches[0]
        else:
            print(f"unknown observation {obs_id!r}. Known ids:", file=sys.stderr)
            for o in sorted(result.objects, key=lambda r: (r.frame_index, r.label)):
                print(f"  {o.obs_id}  ({o.frame_id}, label {o.label})", file=sys.stderr)
            return
    o = feats[obs_id]
    print(f"observation {o.obs_id}: frame {o.frame_id} ({o.time_minutes:g} min), "
          f"label {o.label}, file {o.file}")
    print(f"  area {o.area_um2:.3f} um^2, centroid ({o.centroid_x_um:.3f}, "
          f"{o.centroid_y_um:.3f}) um / pixel ({o.centroid_x_px:.2f}, {o.centroid_y_px:.2f})")
    print(f"  track {result.object_track.get(obs_id)}, mother track "
          f"{result.object_mother.get(obs_id)}")
    incoming = [ln for ln in result.links if obs_id in ln.next_ids]
    outgoing = [ln for ln in result.links if ln.prev_id == obs_id]
    print("\n  incoming (how it came to be):")
    if not incoming:
        print("    born at series start / no incoming event (first frame)")
    for ln in incoming:
        print(f"    {ln.kind}: {ln.prev_id or '-'} -> {','.join(ln.next_ids)}  "
              f"cost {ln.cost:.4f}  undecided={ln.undecided}")
        _print_evidence(ln.evidence)
    print("\n  outgoing (what happened next):")
    if not outgoing:
        print("    still present at the last frame / no outgoing event")
    for ln in outgoing:
        print(f"    {ln.kind}: {ln.prev_id or '-'} -> {','.join(ln.next_ids) or '-'}  "
              f"cost {ln.cost:.4f}  undecided={ln.undecided}")
        _print_evidence(ln.evidence)


def _print_evidence(ev: dict) -> None:
    for key, val in ev.items():
        if key == "cost_terms":
            terms = ", ".join(f"{k}={v}" for k, v in val.items())
            print(f"      cost terms: {terms}")
        else:
            print(f"      {key}: {val}")


def cmd_explain(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    if not any((out / f).exists() for f in ("objects.csv", "events.csv")):
        # re-run in memory from the series so explain does not need side files
        pass
    series = parse_series(args.series, args.base_dir)
    cfg = _cfg_from_args(series, args)
    features = extract_features(series, cfg.min_reliable_area_um2)
    result = run_tracking(series, cfg, features=features)
    labels = {o.obs_id: (o.frame_id, o.label) for o in result.objects}
    _explain_obs(args.obs_id, result, labels)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="celltrack",
                                description="Adjacent-frame cell lineage tracking")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("track", help="track a series of label TIFFs")
    t.add_argument("--series", required=True, help="path to series.json")
    t.add_argument("--base-dir", default=None,
                   help="directory containing TIFF files (default: series.json dir)")
    t.add_argument("--out-dir", default="out", help="output directory")
    t.add_argument("--quiet", action="store_true")
    _add_overrides(t)
    t.set_defaults(func=cmd_track)

    v = sub.add_parser("validate", help="check integrity of a produced output dir")
    v.add_argument("--out-dir", default="out")
    v.set_defaults(func=cmd_validate)

    e = sub.add_parser("explain", help="explain events for one observation")
    e.add_argument("--series", required=True, help="path to series.json")
    e.add_argument("--base-dir", default=None)
    e.add_argument("--out-dir", default="out")
    e.add_argument("--obs-id", required=True, help="obs id like f2:l141 (label optional suffix)")
    _add_overrides(e)
    e.set_defaults(func=cmd_explain)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
