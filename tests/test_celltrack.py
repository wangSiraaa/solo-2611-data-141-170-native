"""End-to-end and unit tests for celltrack.

Runs with:  .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from celltrack.config import config_from_series
from celltrack.features import extract_features, group_by_frame
from celltrack.lineage import run_tracking
from celltrack.report import write_all
from celltrack.series import parse_series

from tests.synth import disc, ellipse, write_series

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run(series_path, out_dir, **overrides):
    series = parse_series(series_path)
    cfg = config_from_series(series.raw, **overrides)
    feats = extract_features(series)
    result = run_tracking(series, cfg, features=feats)
    return series, feats, result, cfg


def event_tuples(result):
    return [(ln.kind, ln.prev_id, ln.next_ids) for ln in result.links]


def read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


# ---------------------------------------------------------------------------
# Fixture series: crossing, division, boundary disappearance, birth
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("fixture_out")
    series, feats, result, cfg = run(FIXTURES / "series.json", out)
    write_all(result, out, series.anomalies)
    return series, feats, result, cfg, out


def test_validation_ok(fixture_run):
    *_, result, cfg, out = fixture_run
    assert result.validation["ok"], result.validation["errors"]


def test_crossing_identities_are_geometrically_consistent(fixture_run):
    """The crossing pair f2->f3 must keep each cell on its own trajectory.

    Cell A travels +9px x between f2 and f3 (continues +9 each interval); cell B
    travels -9px x.  Nearest-centroid would swap them because at the crossing
    instant the distances tie; velocity prediction disambiguates.
    """
    series, feats, result, cfg, out = fixture_run
    by_id = {o.obs_id: o for o in feats}
    # frames 2 and 3 objects
    f2 = group_by_frame(feats)[2]
    # identify A as the object at y=28 (moving +x) and B at y=37 (moving -x)
    a2 = min(f2, key=lambda o: abs(o.centroid_y_um - 14.0))   # y px 28 * 0.5
    b2 = min(f2, key=lambda o: abs(o.centroid_y_um - 18.5))   # y px 37 * 0.5
    # continuations from these
    cont = {ln.prev_id: ln.next_ids[0] for ln in result.links if ln.kind == "continuation"}
    a3 = by_id[cont[a2.obs_id]]
    b3 = by_id[cont[b2.obs_id]]
    # A must keep moving in +x, B in -x
    assert a3.centroid_x_um > a2.centroid_x_um
    assert b3.centroid_x_um < b2.centroid_x_um
    # and they must not be swapped: A's y stays near 28, B near 37
    assert abs(a3.centroid_y_px - 28) < 1
    assert abs(b3.centroid_y_px - 37) < 1


def test_division_detected_with_mother_evidence(fixture_run):
    series, feats, result, cfg, out = fixture_run
    divs = [ln for ln in result.links if ln.kind == "division"]
    assert len(divs) == 1
    d = divs[0]
    mother = next(o for o in feats if o.obs_id == d.prev_id)
    daughters = [o for o in feats if o.obs_id in d.next_ids]
    assert len(daughters) == 2
    # mother area ~ sum of daughters (area-conserving division)
    assert 0.75 <= sum(x.area_um2 for x in daughters) / mother.area_um2 <= 1.35
    # mother is larger than each daughter
    assert mother.area_um2 >= max(x.area_um2 for x in daughters)
    # evidence includes explicit terms
    for key in ("area_conservation_ratio", "daughter_separation_um",
                "mother_daughter_distances_um", "mother_larger_than_daughters"):
        assert key in d.evidence


def test_boundary_disappearance_is_event_not_error(fixture_run):
    series, feats, result, cfg, out = fixture_run
    dis = [ln for ln in result.links if ln.kind == "disappearance"]
    assert len(dis) == 1
    # the disappearing object is at the right image boundary (x px ~125 of 128)
    gone = next(o for o in feats if o.obs_id == dis[0].prev_id)
    assert gone.bbox_max_x >= series.image_shape[1] - 3


def test_birth_is_event_not_forced_link(fixture_run):
    series, feats, result, cfg, out = fixture_run
    births = [ln for ln in result.links if ln.kind == "birth"]
    assert len(births) == 1


def test_mother_not_also_continuing(fixture_run):
    series, feats, result, cfg, out = fixture_run
    outgoing = {}
    for ln in result.links:
        if ln.prev_id:
            outgoing.setdefault(ln.prev_id, []).append(ln.kind)
    for oid, kinds in outgoing.items():
        assert len(kinds) == 1, f"{oid} has multiple outgoing events {kinds}"
    # specifically the mother does not continue
    div = next(ln for ln in result.links if ln.kind == "division")
    assert outgoing[div.prev_id] == ["division"]


def test_each_observation_single_assignment(fixture_run):
    series, feats, result, cfg, out = fixture_run
    first = series.frames[0].index
    last = series.frames[-1].index
    incoming = {o.obs_id: 0 for o in feats}
    outgoing = {o.obs_id: 0 for o in feats}
    for ln in result.links:
        if ln.prev_id:
            outgoing[ln.prev_id] += 1
        for b in ln.next_ids:
            incoming[b] += 1
    for o in feats:
        if o.frame_index != first:
            assert incoming[o.obs_id] == 1, o.obs_id
        else:
            assert incoming[o.obs_id] == 0
        if o.frame_index != last:
            assert outgoing[o.obs_id] == 1, o.obs_id
        else:
            assert outgoing[o.obs_id] == 0


def test_lineage_acyclic_and_daughters_have_mother(fixture_run):
    series, feats, result, cfg, out = fixture_run
    # no cycles
    for tid, tr in result.tracks.items():
        seen = set()
        cur = tr
        while cur is not None:
            assert cur.track_id not in seen, "cycle"
            seen.add(cur.track_id)
            cur = result.tracks.get(cur.mother_track_id) if cur.mother_track_id else None
    # every division mother has exactly two daughters and they point back
    for tid, tr in result.tracks.items():
        if tr.children_track_ids:
            assert len(tr.children_track_ids) == 2
            for c in tr.children_track_ids:
                assert result.tracks[c].mother_track_id == tid


def test_provenance_back_to_pixels(fixture_run):
    """Every object's pixels, read back from the TIFF, match its label mask."""
    import tifffile
    series, feats, result, cfg, out = fixture_run
    cache = {}
    for o in feats:
        if o.file not in cache:
            cache[o.file] = tifffile.imread(FIXTURES / o.file)
        img = cache[o.file]
        r, c = o.pixel_lookup().T
        assert np.all(img[r, c] == o.label)
        # centroid recomputed from those pixels
        assert abs(c.mean() - o.centroid_x_px) < 1e-9
        assert abs(r.mean() - o.centroid_y_px) < 1e-9


# ---------------------------------------------------------------------------
# Label-renumbering and file-order invariance
# ---------------------------------------------------------------------------
def _permute_labels(series_path, tmp_path, permute_seed=1234):
    """Rewrite a series with every frame's labels randomly permuted."""
    import tifffile
    rng = np.random.default_rng(permute_seed)
    series = json.loads(Path(series_path).read_text())
    out = Path(tmp_path)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(series_path).parent
    entries = []
    for i, e in enumerate(series["frames"]):
        arr = tifffile.imread(base / e["file"])
        labs = np.unique(arr)
        labs = labs[labs != 0]
        new_labs = rng.permutation(labs)
        remap = {int(a): int(b) for a, b in zip(labs, new_labs)}
        out_arr = np.zeros_like(arr)
        for old, new in remap.items():
            out_arr[arr == old] = new
        fn = f"perm_{i:02d}.tif"
        tifffile.imwrite(out / fn, out_arr)
        entries.append({**e, "file": fn})
    series["frames"] = entries
    (out / "series.json").write_text(json.dumps(series))
    return out / "series.json"


def _shuffle_file_order(series_path, tmp_path):
    series = json.loads(Path(series_path).read_text())
    out = Path(tmp_path)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(series_path).parent
    import shutil
    entries = list(series["frames"])[::-1]  # reverse listing order
    for e in entries:
        shutil.copy(base / e["file"], out / e["file"])
    series["frames"] = entries
    (out / "series.json").write_text(json.dumps(series))
    return out / "series.json"


def _normalized_events(result):
    """Label-free event description: kind + physical centroids, no label ids."""
    feats = {o.obs_id: o for o in result.objects}

    def pos(oid):
        o = feats[oid]
        return (round(o.centroid_x_um, 3), round(o.centroid_y_um, 3))

    out = []
    for ln in result.links:
        out.append((
            ln.kind,
            pos(ln.prev_id) if ln.prev_id else None,
            tuple(sorted(pos(b) for b in ln.next_ids)),
            ln.undecided,
        ))
    return sorted(out, key=repr)


def _normalized_tracks(result):
    """Label-free track description, including canonical track ids."""
    feats = {o.obs_id: o for o in result.objects}
    out = []
    for tid, tr in result.tracks.items():
        pts = tuple(
            (feats[o].frame_index, round(feats[o].centroid_x_um, 3),
             round(feats[o].centroid_y_um, 3))
            for o in tr.obs_ids
        )
        out.append((tid, pts, tr.mother_track_id, tuple(tr.children_track_ids),
                    tr.lineage_fate))
    return sorted(out, key=repr)


def test_label_renumbering_invariance(fixture_run, tmp_path):
    series, feats, result_ref, cfg, out = fixture_run
    permuted = _permute_labels(FIXTURES / "series.json", tmp_path / "perm")
    s2, f2, r2, c2 = run(permuted, tmp_path / "perm_out")
    assert r2.validation["ok"], r2.validation["errors"]
    assert _normalized_events(r2) == _normalized_events(result_ref)
    # track ids, grouping, mothers, daughters and fates are also invariant
    assert _normalized_tracks(r2) == _normalized_tracks(result_ref)


def test_file_listing_order_invariance(fixture_run, tmp_path):
    series, feats, result_ref, cfg, out = fixture_run
    shuffled = _shuffle_file_order(FIXTURES / "series.json", tmp_path / "shuf")
    s2, f2, r2, c2 = run(shuffled, tmp_path / "shuf_out")
    assert r2.validation["ok"], r2.validation["errors"]
    assert _normalized_events(r2) == _normalized_events(result_ref)
    assert _normalized_tracks(r2) == _normalized_tracks(result_ref)


# ---------------------------------------------------------------------------
# Ambiguity: without velocity prediction, the crossing is near-equivalent
# ---------------------------------------------------------------------------
def test_crossing_ambiguous_without_velocity(fixture_run, tmp_path):
    series, feats, result_ref, cfg, out = fixture_run
    s2, f2, r2, c2 = run(FIXTURES / "series.json", tmp_path, use_velocity_prediction=False)
    ps23 = next(p for p in r2.frame_pairs if p.pair.frame_index_prev == 2)
    assert ps23.undecided
    assert len(ps23.alternatives) == 2
    # the division must be stable across both alternatives -> not contested
    div_key = ("division", "f2:l141", ("f3:l158", "f3:l161"))
    assert div_key not in ps23.undecided_event_keys
    # the two crossing continuations are contested: all four possible
    # prior->next links across the two cells appear in some alternative
    contested = {k for k in ps23.undecided_event_keys if k[0] == "continuation"}
    assert len(contested) == 4
    # each prior cell has exactly two contested outgoing links (one per target)
    priors = {k[1] for k in contested}
    assert len(priors) == 2


def test_crossing_variable_frame_interval(tmp_path):
    """Velocity prediction must scale with the actual frame interval.  Two cells
    cross with a non-uniform dt sequence; identities still follow trajectories."""
    shape = (60, 100)
    frames = []
    for t in (0, 3, 9, 12):           # intervals 3, 6, 3 minutes
        xa = 20 + 4 * t               # 4 px/min = 2 um/min < speed gate
        xb = 70 - 4 * t
        frames.append({10: disc(shape, xa, 20, 4.0),
                       20: disc(shape, xb, 24, 4.0)})
    sp = write_series(tmp_path, frames, [0, 3, 9, 12], shape=shape,
                      pixel_size=(0.5, 0.5), speed=2.5)
    series = parse_series(sp)
    cfg = config_from_series(series.raw)  # velocity prediction on
    feats = extract_features(series)
    result = run_tracking(series, cfg, features=feats)
    assert result.validation["ok"], result.validation["errors"]
    by_id = {o.obs_id: o for o in feats}

    def xseq(tid):
        return [by_id[o].centroid_x_px for o in result.tracks[tid].obs_ids]

    left = min(result.tracks, key=lambda t: xseq(t)[0])
    right = max(result.tracks, key=lambda t: xseq(t)[0])
    assert all(b > a for a, b in zip(xseq(left), xseq(left)[1:]))
    assert all(b < a for a, b in zip(xseq(right), xseq(right)[1:]))
    assert len(result.tracks) == 2


# ---------------------------------------------------------------------------
# Minimal independent enumeration check of the joint objective
# ---------------------------------------------------------------------------
def test_minimal_enumeration_matches_milp(tmp_path):
    """Build a tiny two-frame problem, enumerate *all* event combinations, and
    verify the MILP optimum equals the brute-force optimum and that constraints
    are respected.  This is an independent oracle for the objective."""
    shape = (40, 60)
    # frame 0: two objects; frame 1: two objects.  A clean 1:1 swap would be
    # ambiguous; a slight area change breaks the tie.
    m0 = {10: disc(shape, 12, 20, 4.5), 20: disc(shape, 30, 20, 4.5)}
    m1 = {30: disc(shape, 21, 20, 4.5), 40: ellipse(shape, 40, 20, 5.5, 4.0)}
    sp = write_series(tmp_path, [m0, m1], [0, 3], pixel_size=(0.5, 0.5),
                      speed=2.5, ratio=(0.65, 1.6), shape=shape)
    series = parse_series(sp)
    cfg = config_from_series(series.raw, use_velocity_prediction=False)
    feats = extract_features(series)
    from celltrack.candidates import build_pair_candidates
    from celltrack.solve import solve_pair
    byf = group_by_frame(feats)
    pair = build_pair_candidates(byf[0], byf[1], 3.0, cfg, {})
    sol = solve_pair(pair, cfg)
    assert sol.chosen is not None

    # brute-force: choose, for each prior object, one outgoing candidate; and
    # for each next object exactly one incoming candidate.  Enumerate all
    # consistent event sets by backtracking over prior objects' outgoing events.
    cands = pair.candidates
    prev_ids = [o.obs_id for o in pair.prev_objects]
    next_ids = [o.obs_id for o in pair.next_objects]

    from collections import defaultdict
    out_by_prev = defaultdict(list)
    for c in cands:
        if c.a_id in prev_ids:
            out_by_prev[c.a_id].append(c)
    best = [float("inf")]
    feasible_sets = []

    def rec(i, used_next, chosen, cost):
        if i == len(prev_ids):
            if used_next == set(next_ids):
                feasible_sets.append((cost, tuple(chosen)))
                best[0] = min(best[0], cost)
            return
        a = prev_ids[i]
        for c in out_by_prev[a]:
            kids = set(c.b_ids)
            if kids & used_next:
                continue
            rec(i + 1, used_next | kids, chosen + [c.key], cost + c.cost)

    rec(0, set(), [], 0.0)
    assert feasible_sets, "brute force found no feasible event set"
    brute_best = min(fs[0] for fs in feasible_sets)
    assert abs(brute_best - sol.chosen.cost) < 1e-6

    # MILP chosen set must itself be feasible under the brute-force rules
    chosen_keys = sol.chosen.event_keys
    assert any(abs(fs[0] - sol.chosen.cost) < 1e-6 and set(fs[1]) == set(chosen_keys)
               for fs in feasible_sets)
    # and it must be observation-consistent
    used_next = []
    used_prev = []
    for c in sol.chosen.events:
        if c.a_id:
            used_prev.append(c.a_id)
        used_next.extend(c.b_ids)
    assert sorted(used_prev) == sorted(prev_ids)
    assert sorted(used_next) == sorted(next_ids)


# ---------------------------------------------------------------------------
# Split must not fire merely because two objects are nearby
# ---------------------------------------------------------------------------
def test_no_false_split_from_two_nearby_objects(tmp_path):
    shape = (60, 80)
    # One big object continues; a *separate* small object appears nearby but
    # their combined area is way out of the division ratio range.
    m0 = {10: disc(shape, 20, 30, 4.5)}
    m1 = {20: disc(shape, 21, 30, 4.5), 30: disc(shape, 30, 30, 2.0)}
    sp = write_series(tmp_path, [m0, m1], [0, 3], shape=shape,
                      pixel_size=(0.5, 0.5), ratio=(0.65, 1.6))
    series = parse_series(sp)
    cfg = config_from_series(series.raw, use_velocity_prediction=False)
    feats = extract_features(series)
    result = run_tracking(series, cfg, features=feats)
    assert result.validation["ok"], result.validation["errors"]
    assert not any(ln.kind == "division" for ln in result.links)
    # exactly one of the two next objects is a continuation of the prior
    assert sum(ln.kind == "continuation" for ln in result.links) == 1
    assert sum(ln.kind == "birth" for ln in result.links) == 1


def test_nearby_newcomer_not_forced_into_division(tmp_path):
    """A cell that clearly continues, plus a small newcomer beside it, is *not*
    a division even when the combined area is inside the allowed range.  The
    balanced-but-imperfect split costs more than a clean continuation + birth,
    and the joint solver chooses the globally cheaper event set."""
    shape = (60, 80)
    m0 = {10: disc(shape, 40, 30, 4.5)}                  # 63 px
    m1 = {20: disc(shape, 40, 30, 4.5),                  # continues unchanged
          30: disc(shape, 46, 30, 2.5)}                  # small newborn nearby
    sp = write_series(tmp_path, [m0, m1], [0, 3], shape=shape,
                      pixel_size=(0.5, 0.5), ratio=(0.65, 1.6))
    series = parse_series(sp)
    cfg = config_from_series(series.raw)
    feats = extract_features(series)
    result = run_tracking(series, cfg, features=feats)
    assert result.validation["ok"], result.validation["errors"]
    kinds = sorted(ln.kind for ln in result.links)
    assert kinds == ["birth", "continuation"]
    # the continuation is the unchanged cell, the birth is the newcomer
    cont = next(ln for ln in result.links if ln.kind == "continuation")
    born = next(ln for ln in result.links if ln.kind == "birth")
    by_id = {o.obs_id: o for o in feats}
    assert by_id[cont.next_ids[0]].area_um2 > by_id[born.next_ids[0]].area_um2


def test_division_small_daughters_cannot_continue_mother(tmp_path):
    """Mother area ~ 2x each daughter: each daughter is too far in area from the
    mother to be a plausible continuation, and the pair conserves area; the
    solver must pick the division, not a continuation + birth."""
    shape = (60, 80)
    m0 = {10: disc(shape, 30, 30, 4.5)}  # area ~63 px
    m1 = {20: disc(shape, 25, 30, 3.2),  # area ~30 px
          30: disc(shape, 35, 30, 3.2)}
    sp = write_series(tmp_path, [m0, m1], [0, 3], shape=shape,
                      pixel_size=(0.5, 0.5), ratio=(0.65, 1.6))
    series = parse_series(sp)
    cfg = config_from_series(series.raw, use_velocity_prediction=False)
    feats = extract_features(series)
    result = run_tracking(series, cfg, features=feats)
    assert any(ln.kind == "division" for ln in result.links)


# ---------------------------------------------------------------------------
# Anomaly reporting
# ---------------------------------------------------------------------------
def test_duplicate_frame_id_reported(tmp_path):
    shape = (40, 50)
    sp = write_series(tmp_path, [{5: disc(shape, 10, 10, 3)}, {6: disc(shape, 12, 10, 3)}],
                      [0, 3], shape=shape)
    data = json.loads(sp.read_text())
    data["frames"][1]["frame_id"] = data["frames"][0]["frame_id"]
    sp.write_text(json.dumps(data))
    series = parse_series(sp)
    kinds = {a.kind for a in series.anomalies}
    assert "duplicate_frame_id" in kinds
    # only one frame survives (the duplicate is dropped, reported)
    assert len(series.frames) == 1


def test_empty_frame_reported(tmp_path):
    shape = (40, 50)
    sp = write_series(tmp_path, [dict(), {6: disc(shape, 12, 10, 3)}], [0, 3], shape=shape)
    series = parse_series(sp)
    kinds = {a.kind for a in series.anomalies}
    assert "empty_frame" in kinds
    # empty frame is still processed (just no objects)
    assert len(series.frames) == 2


def test_illegal_dimensions_reported(tmp_path):
    import tifffile
    shape = (40, 50)
    sp = write_series(tmp_path, [{5: disc(shape, 10, 10, 3)}, {6: disc(shape, 12, 10, 3)}],
                      [0, 3], shape=shape)
    # rewrite second TIFF with a different shape
    bad = np.zeros((30, 50), dtype=np.uint16)
    bad[5:8, 5:8] = 6
    tifffile.imwrite(tmp_path / "frame_01.tif", bad)
    series = parse_series(sp)
    kinds = {a.kind for a in series.anomalies}
    assert "illegal_dimensions" in kinds
    assert len(series.frames) == 1


def test_illegal_ndim_reported(tmp_path):
    import tifffile
    shape = (40, 50)
    sp = write_series(tmp_path, [{5: disc(shape, 10, 10, 3)}, {6: disc(shape, 12, 10, 3)}],
                      [0, 3], shape=shape)
    tifffile.imwrite(tmp_path / "frame_01.tif", np.zeros((3, 30, 50), dtype=np.uint16))
    series = parse_series(sp)
    assert "illegal_ndim" in {a.kind for a in series.anomalies}


def test_missing_file_reported(tmp_path):
    shape = (40, 50)
    sp = write_series(tmp_path, [{5: disc(shape, 10, 10, 3)}, {6: disc(shape, 12, 10, 3)}],
                      [0, 3], shape=shape)
    (tmp_path / "frame_01.tif").unlink()
    series = parse_series(sp)
    assert "missing_file" in {a.kind for a in series.anomalies}


# ---------------------------------------------------------------------------
# No forced links across the speed gate
# ---------------------------------------------------------------------------
def test_distant_object_becomes_birth_and_disappearance(tmp_path):
    shape = (80, 100)
    m0 = {10: disc(shape, 10, 40, 4.5)}
    m1 = {20: disc(shape, 80, 40, 4.5)}  # way beyond 2.5 um/min * 3 min * 2 px/um = 15 px
    sp = write_series(tmp_path, [m0, m1], [0, 3], shape=shape,
                      pixel_size=(0.5, 0.5), speed=2.5)
    series = parse_series(sp)
    cfg = config_from_series(series.raw)
    feats = extract_features(series)
    result = run_tracking(series, cfg, features=feats)
    kinds = sorted(ln.kind for ln in result.links)
    assert kinds == ["birth", "disappearance"]


# ---------------------------------------------------------------------------
# Enumeration caps must be reported truthfully
# ---------------------------------------------------------------------------
def test_alternative_cap_still_reports_undecided(tmp_path):
    """With max_alternatives=1 the second equivalent set is not stored, but the
    pair must still be flagged undecided."""
    sp = FIXTURES / "series.json"
    series = parse_series(sp)
    cfg = config_from_series(series.raw, use_velocity_prediction=False,
                             max_alternatives=1)
    feats = extract_features(series)
    result = run_tracking(series, cfg, features=feats)
    ps23 = next(p for p in result.frame_pairs if p.pair.frame_index_prev == 2)
    assert ps23.undecided
    assert len(ps23.alternatives) == 1
    assert ps23.enumeration_truncated


# ---------------------------------------------------------------------------
# Report / CLI integration
# ---------------------------------------------------------------------------
def test_report_files_and_alternatives(tmp_path):
    from celltrack.report import write_all
    sp = FIXTURES / "series.json"
    series = parse_series(sp)
    cfg = config_from_series(series.raw, use_velocity_prediction=False)
    result = run_tracking(series, cfg, features=extract_features(series))
    paths = write_all(result, tmp_path, series.anomalies)
    for name in ("objects", "events", "tracks", "mother_daughters",
                 "candidates", "alternatives", "summary"):
        assert paths[name].exists(), name
    alt_rows = read_csv(paths["alternatives"])
    # the crossing pair is undecided, so alternatives were written
    assert len(alt_rows) == 2
    assert sum(r["is_chosen"] == "True" for r in alt_rows) == 1
    summary = json.loads(paths["summary"].read_text())
    assert summary["validation"]["ok"] is True


def test_cli_track_exit_code_ok(tmp_path):
    from celltrack.cli import main
    rc = main(["track", "--series", str(FIXTURES / "series.json"),
               "--out-dir", str(tmp_path), "--quiet"])
    assert rc == 0
    assert (tmp_path / "summary.json").exists()


def test_cli_explain_runs(capsys):
    from celltrack.cli import main
    rc = main(["explain", "--series", str(FIXTURES / "series.json"),
               "--obs-id", "f2:l141"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "division" in out
    assert "area_conservation_ratio" in out
