"""Global track identity construction and lineage validation.

Adjacent pairs are solved in time order.  The chosen events of pair (i, i+1)
turn into links:

* continuation   keeps the prior track id on the next observation
* division       closes the mother's track and starts two new daughter tracks
* birth          starts a new track
* disappearance  closes a track

Two kinds of cross-pair coupling are handled here:

1. **Velocity prediction.**  Once pair (i-1, i) is solved, each continuation
   gives a velocity estimate for an object at frame i.  That prediction is fed
   into pair (i, i+1)'s continuation cost.  This is what keeps cell identities
   straight through a crossing even when the next-frame nearest centroid is the
   *other* cell.  (Feasibility still uses the raw displacement vs. speed gate.)
2. **Divisions.**  A daughter track that begins at frame i+1 is a brand-new
   track id and is *not* available to continue an unrelated prior object that
   was not its mother -- the per-pair constraints already enforce that.

Validation is independent of the solver and runs on the final event set:
  * observation uniqueness: every object belongs to exactly one event
  * parent uniqueness: every object has <= 1 parent
  * acyclicity: tracks form a forest rooted at births/boundaries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import TrackConfig
from .features import ObjectRecord, group_by_frame
from .series import Series
from .candidates import build_pair_candidates
from .solve import PairSolution, solve_pair


@dataclass
class Link:
    kind: str                     # continuation | division | birth | disappearance
    prev_id: str | None           # obs_id in prior frame
    next_ids: tuple[str, ...]     # obs_id(s) in next frame
    prev_frame: int | None
    next_frame: int | None
    cost: float
    evidence: dict[str, Any]
    undecided: bool = False
    chosen_is_representative: bool = False


@dataclass
class Track:
    track_id: int
    start_frame: int
    end_frame: int
    obs_ids: list[str] = field(default_factory=list)
    parent_track_id: int | None = None
    mother_track_id: int | None = None
    children_track_ids: list[int] = field(default_factory=list)
    lineage_fate: str = ""        # ended_by_disappearance | mother_of_division
    boundary_start: bool = False


@dataclass
class TrackingResult:
    objects: list[ObjectRecord]
    frame_pairs: list[PairSolution]
    links: list[Link]
    tracks: dict[int, Track]
    object_track: dict[str, int]          # obs_id -> track_id
    object_mother: dict[str, int | None]  # obs_id -> mother obs_id's track? -> see below
    validation: dict[str, Any]
    anomalies: list[str] = field(default_factory=list)


def _predictions_from_links(
    prev_objs: dict[str, ObjectRecord],
    cur_objs: dict[str, ObjectRecord],
    links: list[Link],
    prev_dt: float,
    cur_dt: float,
    cfg_use_velocity: bool = True,
) -> dict[str, np.ndarray | None]:
    """For each object at the current frame, predict its position one step on.

    Only continuation links from the previous pair yield a velocity.  The
    velocity is converted to um/minute using the previous frame interval and
    re-scaled by the *current* interval, so variable frame spacing is handled
    correctly.  Division daughters and births have no reliable single-parent
    velocity, so they fall back to position-only prediction (predicted = current
    position).
    """
    pred: dict[str, np.ndarray | None] = {oid: None for oid in cur_objs}
    if not cfg_use_velocity or prev_dt <= 0 or cur_dt <= 0:
        return pred
    for ln in links:
        if ln.kind != "continuation":
            continue
        a = prev_objs.get(ln.prev_id) if ln.prev_id else None
        b = cur_objs.get(ln.next_ids[0]) if ln.next_ids else None
        if a is None or b is None:
            continue
        velocity_per_min = (b.position_um - a.position_um) / prev_dt
        pred[b.obs_id] = b.position_um + velocity_per_min * cur_dt
    return pred


def run_tracking(series: Series, cfg: TrackConfig, features: list[ObjectRecord] | None = None) -> TrackingResult:
    if features is None:
        from .features import extract_features
        features = extract_features(series, cfg.min_reliable_area_um2)

    by_frame = group_by_frame(features)
    obs_by_id = {o.obs_id: o for o in features}
    links: list[Link] = []
    pair_solutions: list[PairSolution] = []

    # Previous pair's continuation links, used to seed velocity predictions.
    prev_links: list[Link] = []
    prev_obs_by_id: dict[str, ObjectRecord] = {}

    for i in range(len(series.frames) - 1):
        f_prev = series.frames[i]
        f_next = series.frames[i + 1]
        dt = f_next.time_minutes - f_prev.time_minutes
        prev_dt = f_prev.time_minutes - series.frames[i - 1].time_minutes if i > 0 else dt
        prev_objects = by_frame.get(f_prev.index, [])
        next_objects = by_frame.get(f_next.index, [])

        preds = _predictions_from_links(
            prev_obs_by_id, {o.obs_id: o for o in prev_objects}, prev_links,
            prev_dt, dt,
            cfg_use_velocity=cfg.use_velocity_prediction,
        )

        pair = build_pair_candidates(prev_objects, next_objects, dt, cfg, preds)
        sol = solve_pair(pair, cfg)
        pair_solutions.append(sol)

        new_links: list[Link] = []
        if sol.chosen is not None:
            for ev in sol.chosen.events:
                prev_obj = obs_by_id.get(ev.a_id) if ev.a_id else None
                next_objs = [obs_by_id[b] for b in ev.b_ids]
                link = Link(
                    kind=ev.kind,
                    prev_id=ev.a_id,
                    next_ids=ev.b_ids,
                    prev_frame=prev_obj.frame_index if prev_obj else None,
                    next_frame=next_objs[0].frame_index if next_objs else None,
                    cost=ev.cost,
                    evidence=ev.evidence,
                    undecided=ev.key in sol.undecided_event_keys,
                    chosen_is_representative=sol.chosen_is_representative,
                )
                new_links.append(link)
                links.append(link)

        prev_links = [ln for ln in new_links if ln.kind == "continuation"]
        prev_obs_by_id = {o.obs_id: o for o in prev_objects}

    # ------------------------------------------------------------------
    # Global track assembly
    # ------------------------------------------------------------------
    obs_ids_by_frame = {idx: [o.obs_id for o in by_frame.get(idx, [])] for idx in by_frame}
    object_track: dict[str, int] = {}
    tracks: dict[int, Track] = {}
    next_track_id = 1

    # map: next object -> its incoming link
    incoming: dict[str, Link] = {}
    for ln in links:
        for b in ln.next_ids:
            incoming[b] = ln

    frame_indices_all = sorted({o.frame_index for o in features})
    first_frame_index = frame_indices_all[0] if frame_indices_all else None

    def new_track(start_frame: int, obs_id: str) -> int:
        nonlocal next_track_id
        tid = next_track_id
        next_track_id += 1
        tracks[tid] = Track(track_id=tid, start_frame=start_frame, end_frame=start_frame,
                            obs_ids=[obs_id])
        object_track[obs_id] = tid
        return tid

    for fi in sorted(obs_ids_by_frame):
        for oid in obs_ids_by_frame[fi]:
            if oid in object_track:
                continue
            ln_in = incoming.get(oid)
            if ln_in is None:
                # No incoming event.  For a first-frame object this is simply
                # "present at series start"; for a later object this should not
                # happen because every next-frame object has a birth candidate.
                tid = new_track(fi, oid)
                if fi != first_frame_index:
                    tracks[tid].boundary_start = True
                continue
            if ln_in.kind == "continuation":
                parent_oid = ln_in.prev_id
                if parent_oid in object_track:
                    tid = object_track[parent_oid]
                    tracks[tid].obs_ids.append(oid)
                    tracks[tid].end_frame = fi
                    object_track[oid] = tid
                else:
                    # parent should have been assigned in an earlier frame
                    raise RuntimeError(f"continuation parent {parent_oid} has no track when assigning {oid}")
            elif ln_in.kind == "division":
                mother_oid = ln_in.prev_id
                mother_tid = object_track.get(mother_oid)
                tid = new_track(fi, oid)
                tracks[tid].parent_track_id = mother_tid
                tracks[tid].mother_track_id = mother_tid
                if mother_tid is not None:
                    tracks[mother_tid].children_track_ids.append(tid)
            elif ln_in.kind == "birth":
                tid = new_track(fi, oid)
                tracks[tid].boundary_start = True
            else:  # pragma: no cover - incoming cannot be disappearance
                raise RuntimeError(f"unexpected incoming event {ln_in.kind}")

    # lineage fate annotations
    for ln in links:
        if ln.kind == "disappearance" and ln.prev_id in object_track:
            tracks[object_track[ln.prev_id]].lineage_fate = "ended_by_disappearance"
        if ln.kind == "division" and ln.prev_id in object_track:
            tracks[object_track[ln.prev_id]].lineage_fate = "mother_of_division"

    tracks = dict(sorted(tracks.items(), key=lambda kv: min(
        (obs_by_id[o].frame_index, obs_by_id[o].centroid_y_um, obs_by_id[o].centroid_x_um)
        for o in kv[1].obs_ids)))

    # ------------------------------------------------------------------
    # Canonical track numbering.  Track *grouping* is label-invariant; the
    # numeric ids are assigned here from physical trajectory geometry alone
    # (start frame, then the sequence of centroid positions), so they do not
    # depend on label numbers, per-frame label ordering, or which track object
    # happened to be created first.
    # ------------------------------------------------------------------
    def track_geom_key(tid: int) -> tuple:
        tr = tracks[tid]
        pts = []
        for oid in tr.obs_ids:
            o = obs_by_id[oid]
            pts.append((o.frame_index, round(o.centroid_x_um, 6), round(o.centroid_y_um, 6)))
        # obs_ids are stored in temporal order already
        return (tr.start_frame, tuple(pts))

    old_order = sorted(tracks.keys(), key=track_geom_key)
    remap = {old: new for new, old in enumerate(old_order, start=1)}
    new_tracks: dict[int, Track] = {}
    for old in old_order:
        tr = tracks[old]
        new_tracks[remap[old]] = Track(
            track_id=remap[old],
            start_frame=tr.start_frame,
            end_frame=tr.end_frame,
            obs_ids=list(tr.obs_ids),
            parent_track_id=None if tr.parent_track_id is None else remap[tr.parent_track_id],
            mother_track_id=None if tr.mother_track_id is None else remap[tr.mother_track_id],
            children_track_ids=sorted(remap[c] for c in tr.children_track_ids),
            lineage_fate=tr.lineage_fate,
            boundary_start=tr.boundary_start,
        )
    tracks = new_tracks
    object_track = {oid: remap[tid] for oid, tid in object_track.items()}

    # Mother lineage is a property of the *track* (not just the birth
    # observation): every observation of a daughter track reports the same
    # mother track, so the lineage is consistent along the whole track.
    object_mother: dict[str, int | None] = {}
    for o in features:
        tid = object_track.get(o.obs_id)
        if tid is None:
            object_mother[o.obs_id] = None
        else:
            object_mother[o.obs_id] = tracks[tid].mother_track_id

    validation = validate_result(
        features=features,
        links=links,
        tracks=tracks,
        object_track=object_track,
        tracks_by_frame=obs_ids_by_frame,
    )

    return TrackingResult(
        objects=features,
        frame_pairs=pair_solutions,
        links=links,
        tracks=tracks,
        object_track=object_track,
        object_mother=object_mother,
        validation=validation,
    )


def validate_result(
    features: list[ObjectRecord],
    links: list[Link],
    tracks: dict[int, Track],
    object_track: dict[str, int],
    tracks_by_frame: dict[int, list[str]],
) -> dict[str, Any]:
    """Solver-independent integrity checks on the final event set."""
    all_obs = {o.obs_id for o in features}
    errors: list[str] = []

    # 1. observation uniqueness: each obs appears as exactly one endpoint event
    #    (continuation consumes 1 prev + 1 next; division 1 prev + 2 next).
    #    Objects in the first frame need no incoming event and objects in the
    #    last frame need no outgoing event -- that is the series boundary, not a
    #    tracking failure.
    frame_indices = sorted({o.frame_index for o in features})
    first_frame = frame_indices[0] if frame_indices else None
    last_frame = frame_indices[-1] if frame_indices else None
    by_id = {o.obs_id: o for o in features}

    prev_count: dict[str, int] = {o: 0 for o in all_obs}
    next_count: dict[str, int] = {o: 0 for o in all_obs}
    for ln in links:
        if ln.prev_id is not None:
            prev_count[ln.prev_id] += 1
        for b in ln.next_ids:
            next_count[b] += 1

    unclaimed_prev = [
        o for o, n in prev_count.items()
        if n != 1 and by_id[o].frame_index != last_frame
    ]
    unclaimed_next = [
        o for o, n in next_count.items()
        if n != 1 and by_id[o].frame_index != first_frame
    ]
    # objects on the last frame may still spuriously carry outgoing events?
    extra_out = [o for o, n in prev_count.items()
                 if n > 0 and by_id[o].frame_index == last_frame]
    extra_in = [o for o, n in next_count.items()
                if n > 0 and by_id[o].frame_index == first_frame]
    if unclaimed_prev:
        errors.append(f"prior objects without exactly one outgoing event: {sorted(unclaimed_prev)}")
    if unclaimed_next:
        errors.append(f"next objects without exactly one incoming event: {sorted(unclaimed_next)}")
    if extra_out:
        errors.append(f"last-frame objects with an outgoing event: {sorted(extra_out)}")
    if extra_in:
        errors.append(f"first-frame objects with an incoming event: {sorted(extra_in)}")

    # 2. parent uniqueness: each object has at most one incoming parent link
    parent_count: dict[str, int] = {o: 0 for o in all_obs}
    for ln in links:
        if ln.kind in ("continuation", "division"):
            for b in ln.next_ids:
                parent_count[b] += 1
    multi_parent = [o for o, n in parent_count.items() if n > 1]
    if multi_parent:
        errors.append(f"objects with more than one parent: {sorted(multi_parent)}")

    # 3. a mother cannot both divide and continue (one outgoing event each)
    mothers_who_also_continued: list[str] = []
    outgoing_by_prev: dict[str, list[str]] = {}
    for ln in links:
        if ln.prev_id is not None:
            outgoing_by_prev.setdefault(ln.prev_id, []).append(ln.kind)
    for oid, kinds in outgoing_by_prev.items():
        if len(kinds) > 1:
            mothers_who_also_continued.append(oid)
    if mothers_who_also_continued:
        errors.append(f"objects with multiple outgoing events: {sorted(mothers_who_also_continued)}")

    # 4. acyclicity / consistency of the track forest
    #    Every track either starts at a birth, a division daughter, or frame 0.
    cycles: list[int] = []
    for tid, tr in tracks.items():
        parent = tr.parent_track_id
        seen = {tid}
        while parent is not None:
            if parent in seen:
                cycles.append(tid)
                break
            seen.add(parent)
            parent = tracks.get(parent, Track(-1, -1, -1)).parent_track_id
    if cycles:
        errors.append(f"lineage cycles detected in tracks: {sorted(set(cycles))}")

    # 5. daughter tracks must have a mother; mother must have exactly 2 daughters
    for tid, tr in tracks.items():
        if tr.mother_track_id is not None:
            mom = tracks.get(tr.mother_track_id)
            if mom is None:
                errors.append(f"track {tid} references missing mother {tr.mother_track_id}")
            elif len(mom.children_track_ids) != 2:
                errors.append(
                    f"mother track {mom.track_id} has {len(mom.children_track_ids)} daughters, expected 2"
                )

    # 6. contiguity: each track's obs frames are consecutive
    for tid, tr in tracks.items():
        # obs_id is f<idx>:l<label>; frame parts must be contiguous
        idxs = sorted({int(o.split(':')[0][1:]) for o in tr.obs_ids})
        if idxs and idxs != list(range(idxs[0], idxs[-1] + 1)):
            errors.append(f"track {tid} has non-contiguous frames {idxs}")

    # 7. every object assigned to exactly one track and that track contains it
    for o in features:
        tid = object_track.get(o.obs_id)
        if tid is None or o.obs_id not in tracks[tid].obs_ids:
            errors.append(f"object {o.obs_id} not consistently assigned to a track")

    return {
        "ok": not errors,
        "errors": errors,
        "n_objects": len(all_obs),
        "n_links": len(links),
        "n_tracks": len(tracks),
    }
