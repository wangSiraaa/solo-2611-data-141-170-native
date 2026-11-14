"""Configuration: physical units, candidate-support gates, and cost weights.

All distances are in micrometres (um) and times in minutes.  The defaults are
calibrated from ``series.json`` (pixel size, frame interval, stated maximum
centroid speed, division area-ratio range) and the documented evidence model.
Every number here is either taken from the fixture metadata or is an explicit,
overridable modelling assumption -- none is a hidden heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackConfig:
    # ---- physical metadata (filled from series.json) -------------------
    pixel_size_um: tuple[float, float] = (1.0, 1.0)
    max_centroid_speed_um_per_min: float = 2.5
    division_area_ratio_range: tuple[float, float] = (0.65, 1.6)

    # ---- candidate support gates (hard feasibility) --------------------
    # A continuation is a candidate only if the centroid displacement does not
    # exceed max_speed * dt.  A small tolerance lets objects sitting exactly on
    # the boundary remain feasible despite floating-point noise.
    gate_tolerance_um: float = 1e-6

    # A division additionally requires the two daughters' *combined* area to be
    # consistent with the mother's area (area conservation) and the daughters to
    # lie within a plausible splitting radius of the mother.  Each daughter's
    # area ratio to the mother must also fall inside the configured range.
    division_max_daughter_sep_um: float = 20.0
    division_max_mother_daughter_dist_um: float = 16.0

    # Minimum per-object area in um^2 below which a segmentation is reported as
    # suspicious (sliver) but still tracked.
    min_reliable_area_um2: float = 1.0

    # ---- cost weights --------------------------------------------------
    # Continuation cost = w_move * normalised displacement
    #                   + w_area * |log area ratio|
    #                   + w_shape * normalised shape change
    move_cost_weight: float = 1.0
    area_cost_weight: float = 0.35
    shape_cost_weight: float = 0.15

    # Division cost = w_div_area * log-area-mismatch penalty
    #               + w_div_sep  * daughter-separation penalty
    #               + w_div_place* placement penalty
    #               + w_div_mother * mother-evidence penalty
    # A division is intrinsically more expensive than a clean continuation so
    # that, absent positive evidence, a 1-to-1 link wins when both are feasible.
    division_base_cost: float = 0.60
    division_area_weight: float = 1.0
    division_balance_weight: float = 0.4
    division_separation_weight: float = 0.5
    division_placement_weight: float = 0.5
    division_mother_weight: float = 0.5
    division_shape_weight: float = 0.15

    # ---- birth / disappearance flat costs ------------------------------
    # Kept equal and moderate: an unmatched object is a birth or disappearance,
    # not an error, but it should not be preferred over a genuinely good link.
    birth_cost: float = 1.20
    disappearance_cost: float = 1.20

    # ---- near-equivalence / ambiguity reporting ------------------------
    # Two event sets whose total costs differ by <= this are "near equivalent".
    ambiguity_cost_margin: float = 0.05
    # Maximum number of distinct near-optimal alternatives to enumerate.
    max_alternatives: int = 8
    # Hard cap on branch-and-bound nodes while enumerating alternatives.
    max_enumeration_nodes: int = 200_000

    # ---- determinism ---------------------------------------------------
    # Cost comparisons are rounded to this many decimals before ranking so that
    # mathematically-equal event sets are not separated by float noise.
    cost_decimals: int = 9

    # ---- motion model --------------------------------------------------
    # When True, continuation cost uses a constant-velocity prediction derived
    # from the already-solved previous frame pair.  Disabling it falls back to
    # nearest-centroid scoring (still gated by max speed).  Exposed mainly to
    # demonstrate that crossing identities rely on the prediction and that,
    # without it, near-equivalent swaps are reported as undecided rather than
    # silently resolved by label order.
    use_velocity_prediction: bool = True

    def centroid_threshold_um(self, dt_minutes: float) -> float:
        return self.max_centroid_speed_um_per_min * dt_minutes + self.gate_tolerance_um


def config_from_series(series: dict, **overrides) -> TrackConfig:
    """Build a TrackConfig from parsed series.json content."""
    px = series.get("pixel_size_um", [1.0, 1.0])
    if not (isinstance(px, (list, tuple)) and len(px) == 2):
        raise ValueError("series.pixel_size_um must be a 2-element [x, y] list")
    ratio = series.get("division_area_ratio_range", [0.65, 1.6])
    if not (isinstance(ratio, (list, tuple)) and len(ratio) == 2):
        raise ValueError("series.division_area_ratio_range must be a 2-element list")
    speed = series.get("max_centroid_speed_um_per_min", 2.5)
    cfg = TrackConfig(
        pixel_size_um=(float(px[0]), float(px[1])),
        max_centroid_speed_um_per_min=float(speed),
        division_area_ratio_range=(float(ratio[0]), float(ratio[1])),
    )
    if overrides:
        cfg = TrackConfig(**{**cfg.__dict__, **overrides})
    return cfg
