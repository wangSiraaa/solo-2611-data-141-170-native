"""Candidate events between adjacent frames, with support gates and costs.

Event vocabulary (per adjacent frame pair)
------------------------------------------
* continuation     one prior object -> one next object
* division         one prior mother  -> two next daughters
* birth            one next object with no prior parent
* disappearance    one prior object with no next successor

Each candidate carries:
  * a hard *support gate* (why the event is geometrically/physically admissible)
  * a continuous, explainable *cost* used by the joint solver
  * an ``evidence`` dictionary that is written verbatim to the event table, so
    every selected link can be justified.

Design choices (also documented in README):
  * Continuation feasibility uses the raw centroid displacement and the series'
    maximum speed (``max_centroid_speed_um_per_min * dt``).  The *cost* uses
    displacement from a constant-velocity prediction when a reliable velocity
    exists (derived from the already-solved previous adjacent pair).  This is
    what prevents a nearest-centroid swap when two cells cross: each cell is
    scored against where it was heading, not merely where it is.
  * Division is admissible only when (a) each daughter lies near the mother,
    (b) the daughters are close enough to have come from one mother, and (c) the
    combined daughter area is consistent with the mother's area within the
    series' ``division_area_ratio_range``.  The cost additionally penalises area
    mismatch, daughter imbalance, daughter separation, midpoint offset and a
    mother that is smaller than a daughter.  Merely "two objects appeared
    nearby" is therefore not enough.
  * Birth and disappearance are first-class events with flat costs.  They are
    legitimate outcomes, not errors, and a link is only chosen when its cost is
    genuinely competitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import TrackConfig
from .features import ObjectRecord


@dataclass(frozen=True)
class Candidate:
    kind: str                       # continuation | division | birth | disappearance
    a_id: str | None                # prior object (mother for division)
    b_ids: tuple[str, ...] = ()     # next object(s) (canonical daughter order)
    cost: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    # label-numbering-independent geometric signature, used only for
    # deterministic representative selection among near-equivalent solutions
    geom_signature: tuple = ()

    @property
    def key(self) -> tuple:
        return (self.kind, self.a_id, self.b_ids)


def _obj_geom(o: ObjectRecord) -> tuple:
    """Physical, label-free geometry key for an object."""
    return (
        round(o.centroid_x_um, 6),
        round(o.centroid_y_um, 6),
        round(o.area_um2, 6),
    )


def _log_ratio(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return float("inf")
    return abs(float(np.log(a / b)))


def continuation_candidate(
    a: ObjectRecord,
    b: ObjectRecord,
    dt: float,
    cfg: TrackConfig,
    predicted_xy_um: np.ndarray | None,
    velocity_used: bool,
) -> Candidate | None:
    """Return a continuation candidate or None if outside the support gate."""
    speed_limit = cfg.max_centroid_speed_um_per_min * dt
    disp = float(np.hypot(b.centroid_x_um - a.centroid_x_um, b.centroid_y_um - a.centroid_y_um))
    if disp > speed_limit + cfg.gate_tolerance_um:
        return None  # faster than physically allowed -> not a continuation candidate

    pred = a.position_um if predicted_xy_um is None else np.asarray(predicted_xy_um, float)
    pred_disp = float(np.hypot(b.centroid_x_um - pred[0], b.centroid_y_um - pred[1]))
    move_norm = min(pred_disp / speed_limit, 3.0) if speed_limit > 0 else 0.0
    area_delta = _log_ratio(b.area_um2, a.area_um2)
    if not np.isfinite(area_delta):
        return None
    shape_delta = abs(b.eccentricity - a.eccentricity) + abs(b.circularity - a.circularity)
    move_term = cfg.move_cost_weight * move_norm
    area_term_w = cfg.area_cost_weight * area_delta
    shape_term_w = cfg.shape_cost_weight * shape_delta
    cost = move_term + area_term_w + shape_term_w
    return Candidate(
        kind="continuation",
        a_id=a.obs_id,
        b_ids=(b.obs_id,),
        cost=float(cost),
        evidence={
            "displacement_um": round(disp, 4),
            "speed_limit_um": round(speed_limit, 4),
            "predicted_position_um": [round(float(pred[0]), 4), round(float(pred[1]), 4)],
            "predicted_displacement_um": round(pred_disp, 4),
            "velocity_used": bool(velocity_used),
            "area_ratio_b_over_a": round(float(b.area_um2 / a.area_um2), 5),
            "log_area_change": round(area_delta, 5),
            "shape_delta": round(shape_delta, 5),
            "centroid_prev_px": [round(a.centroid_x_px, 3), round(a.centroid_y_px, 3)],
            "centroid_next_px": [round(b.centroid_x_px, 3), round(b.centroid_y_px, 3)],
            "cost_terms": {
                "move": round(move_term, 6),
                "area": round(area_term_w, 6),
                "shape": round(shape_term_w, 6),
            },
        },
        geom_signature=("continuation", _obj_geom(a), _obj_geom(b)),
    )


def division_candidate(
    m: ObjectRecord,
    d1: ObjectRecord,
    d2: ObjectRecord,
    dt: float,
    cfg: TrackConfig,
) -> Candidate | None:
    """Return a division candidate or None if any support gate fails."""
    # canonical daughter order independent of label numbering
    if _canon_key(d1) > _canon_key(d2):
        d1, d2 = d2, d1

    dist1 = float(np.hypot(d1.centroid_x_um - m.centroid_x_um, d1.centroid_y_um - m.centroid_y_um))
    dist2 = float(np.hypot(d2.centroid_x_um - m.centroid_x_um, d2.centroid_y_um - m.centroid_y_um))
    if max(dist1, dist2) > cfg.division_max_mother_daughter_dist_um + cfg.gate_tolerance_um:
        return None
    sep = float(np.hypot(d2.centroid_x_um - d1.centroid_x_um, d2.centroid_y_um - d1.centroid_y_um))
    if sep > cfg.division_max_daughter_sep_um + cfg.gate_tolerance_um:
        return None

    combined = d1.area_um2 + d2.area_um2
    conservation = combined / m.area_um2 if m.area_um2 > 0 else float("inf")
    lo, hi = cfg.division_area_ratio_range
    if not (lo - cfg.gate_tolerance_um <= conservation <= hi + cfg.gate_tolerance_um):
        return None  # area not conserved -> not admissible as a division

    area_term = abs(float(np.log(max(conservation, 1e-12)))
                    ) if conservation > 0 else float("inf")
    balance_term = _log_ratio(d1.area_um2, d2.area_um2)
    if not np.isfinite(balance_term):
        return None
    sep_term = sep / cfg.division_max_daughter_sep_um
    mid = np.array([(d1.centroid_x_um + d2.centroid_x_um) / 2.0,
                    (d1.centroid_y_um + d2.centroid_y_um) / 2.0])
    place = float(np.hypot(mid[0] - m.centroid_x_um, mid[1] - m.centroid_y_um))
    place_term = place / cfg.division_max_mother_daughter_dist_um
    # Mother evidence: a daughter larger than the mother is implausible.
    mother_penalty = max(0.0, float(np.log(max(d1.area_um2, d2.area_um2) / m.area_um2)))
    # Elongation evidence: dividing mothers tend to be at least as elongated as
    # daughters; record (small) bonus-penalty without dominating geometry.
    elong_penalty = max(0.0, float(d1.eccentricity - m.eccentricity)) + \
        max(0.0, float(d2.eccentricity - m.eccentricity))

    cost = (
        cfg.division_base_cost
        + cfg.division_area_weight * area_term
        + cfg.division_balance_weight * balance_term
        + cfg.division_separation_weight * sep_term
        + cfg.division_placement_weight * place_term
        + cfg.division_mother_weight * mother_penalty
        + cfg.division_shape_weight * elong_penalty
    )
    return Candidate(
        kind="division",
        a_id=m.obs_id,
        b_ids=(d1.obs_id, d2.obs_id),
        cost=float(cost),
        evidence={
            "mother_area_um2": round(m.area_um2, 4),
            "daughter_areas_um2": [round(d1.area_um2, 4), round(d2.area_um2, 4)],
            "combined_daughter_area_um2": round(combined, 4),
            "area_conservation_ratio": round(float(conservation), 5),
            "allowed_area_ratio_range": [lo, hi],
            "log_area_mismatch": round(area_term, 5),
            "daughter_balance_log_ratio": round(balance_term, 5),
            "daughter_separation_um": round(sep, 4),
            "max_daughter_separation_um": cfg.division_max_daughter_sep_um,
            "mother_daughter_distances_um": [round(dist1, 4), round(dist2, 4)],
            "max_mother_daughter_dist_um": cfg.division_max_mother_daughter_dist_um,
            "midpoint_offset_um": round(place, 4),
            "mother_eccentricity": round(m.eccentricity, 5),
            "daughter_eccentricities": [round(d1.eccentricity, 5), round(d2.eccentricity, 5)],
            "mother_larger_than_daughters": bool(m.area_um2 >= max(d1.area_um2, d2.area_um2)),
            "dt_minutes": dt,
            "cost_terms": {
                "base": round(cfg.division_base_cost, 6),
                "area": round(cfg.division_area_weight * area_term, 6),
                "balance": round(cfg.division_balance_weight * balance_term, 6),
                "separation": round(cfg.division_separation_weight * sep_term, 6),
                "placement": round(cfg.division_placement_weight * place_term, 6),
                "mother": round(cfg.division_mother_weight * mother_penalty, 6),
                "shape": round(cfg.division_shape_weight * elong_penalty, 6),
            },
        },
        geom_signature=("division", _obj_geom(m), _obj_geom(d1), _obj_geom(d2)),
    )


def birth_candidate(b: ObjectRecord, nearest_prev_um: float | None, cfg: TrackConfig) -> Candidate:
    return Candidate(
        kind="birth",
        a_id=None,
        b_ids=(b.obs_id,),
        cost=cfg.birth_cost,
        evidence={
            "reason": "no prior-frame object satisfies the continuation gate",
            "area_um2": round(b.area_um2, 4),
            "nearest_prior_centroid_distance_um": None if nearest_prev_um is None else round(nearest_prev_um, 4),
            "cost_terms": {"flat": round(cfg.birth_cost, 6)},
        },
        geom_signature=("birth", None, _obj_geom(b)),
    )


def disappearance_candidate(a: ObjectRecord, nearest_next_um: float | None, cfg: TrackConfig) -> Candidate:
    return Candidate(
        kind="disappearance",
        a_id=a.obs_id,
        b_ids=(),
        cost=cfg.disappearance_cost,
        evidence={
            "reason": "no next-frame object satisfies the continuation gate",
            "area_um2": round(a.area_um2, 4),
            "nearest_next_centroid_distance_um": None if nearest_next_um is None else round(nearest_next_um, 4),
            "cost_terms": {"flat": round(cfg.disappearance_cost, 6)},
        },
        geom_signature=("disappearance", _obj_geom(a), None),
    )


def _canon_key(o: ObjectRecord) -> tuple:
    """Geometric, label-numbering-independent ordering key."""
    return (
        round(o.centroid_x_um, 6),
        round(o.centroid_y_um, 6),
        round(o.area_um2, 6),
        o.bbox_min_x,
        o.bbox_min_y,
    )


@dataclass
class PairCandidates:
    frame_index_prev: int
    frame_index_next: int
    dt_minutes: float
    prev_objects: list[ObjectRecord]
    next_objects: list[ObjectRecord]
    candidates: list[Candidate]
    # hard-gate diagnostics for events that were *considered* but rejected
    rejected: list[dict[str, Any]] = field(default_factory=list)


def build_pair_candidates(
    prev_objects: list[ObjectRecord],
    next_objects: list[ObjectRecord],
    dt: float,
    cfg: TrackConfig,
    predictions: dict[str, np.ndarray | None],
) -> PairCandidates:
    """Enumerate all admissible candidates for one adjacent frame pair."""
    # Sort canonically so that enumeration order (and thus deterministic
    # tie-breaking) never depends on label numbers or file/listing order.
    prev_sorted = sorted(prev_objects, key=_canon_key)
    next_sorted = sorted(next_objects, key=_canon_key)

    def nearest_next(a: ObjectRecord) -> float | None:
        if not next_sorted:
            return None
        return min(float(np.hypot(b.centroid_x_um - a.centroid_x_um,
                                  b.centroid_y_um - a.centroid_y_um)) for b in next_sorted)

    def nearest_prev(b: ObjectRecord) -> float | None:
        if not prev_sorted:
            return None
        return min(float(np.hypot(b.centroid_x_um - a.centroid_x_um,
                                  b.centroid_y_um - a.centroid_y_um)) for a in prev_sorted)

    candidates: list[Candidate] = []
    rejected: list[dict[str, Any]] = []

    for a in prev_sorted:
        pred = predictions.get(a.obs_id)
        used_velocity = pred is not None
        for b in next_sorted:
            cand = continuation_candidate(a, b, dt, cfg, pred, used_velocity)
            if cand is not None:
                candidates.append(cand)
            else:
                disp = float(np.hypot(b.centroid_x_um - a.centroid_x_um,
                                      b.centroid_y_um - a.centroid_y_um))
                speed_limit = cfg.max_centroid_speed_um_per_min * dt
                if disp <= speed_limit * 3.0:
                    rejected.append({
                        "kind": "continuation_rejected",
                        "a_id": a.obs_id, "b_id": b.obs_id,
                        "reason": "displacement exceeds speed gate",
                        "displacement_um": round(disp, 4),
                        "speed_limit_um": round(speed_limit, 4),
                    })
        # divisions: candidate pair of next objects
        for i in range(len(next_sorted)):
            for j in range(i + 1, len(next_sorted)):
                d1, d2 = next_sorted[i], next_sorted[j]
                cand = division_candidate(a, d1, d2, dt, cfg)
                if cand is not None:
                    candidates.append(cand)
                else:
                    # record only when it looked geographically plausible but
                    # failed an evidence gate -- useful audit trail
                    near = max(
                        float(np.hypot(d1.centroid_x_um - a.centroid_x_um, d1.centroid_y_um - a.centroid_y_um)),
                        float(np.hypot(d2.centroid_x_um - a.centroid_x_um, d2.centroid_y_um - a.centroid_y_um)),
                    )
                    if near <= cfg.division_max_mother_daughter_dist_um * 2.0:
                        combined = d1.area_um2 + d2.area_um2
                        conservation = combined / a.area_um2 if a.area_um2 else float("inf")
                        lo, hi = cfg.division_area_ratio_range
                        reason = []
                        if not (lo <= conservation <= hi):
                            reason.append("combined daughter/mother area outside series range")
                        if near > cfg.division_max_mother_daughter_dist_um:
                            reason.append("daughter farther than mother-daughter gate")
                        sep = float(np.hypot(d2.centroid_x_um - d1.centroid_x_um,
                                             d2.centroid_y_um - d1.centroid_y_um))
                        if sep > cfg.division_max_daughter_sep_um:
                            reason.append("daughter separation exceeds gate")
                        rejected.append({
                            "kind": "division_rejected",
                            "a_id": a.obs_id, "b_ids": [d1.obs_id, d2.obs_id],
                            "reason": "; ".join(reason) or "failed support gate",
                            "area_conservation_ratio": round(float(conservation), 5),
                            "daughter_separation_um": round(sep, 4),
                        })
        candidates.append(disappearance_candidate(a, nearest_next(a), cfg))

    for b in next_sorted:
        candidates.append(birth_candidate(b, nearest_prev(b), cfg))

    return PairCandidates(
        frame_index_prev=prev_sorted[0].frame_index if prev_sorted else -1,
        frame_index_next=next_sorted[0].frame_index if next_sorted else -1,
        dt_minutes=dt,
        prev_objects=prev_sorted,
        next_objects=next_sorted,
        candidates=candidates,
        rejected=rejected,
    )
