"""Tests for the periodic-trajectory unwrap implementation.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import itertools
import json
import unittest
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unwrap_traj.core import (  # noqa: E402
    AnalysisOptions,
    Cell,
    analyze_dataset,
    compute_msd,
    lattice_candidates,
    reconstruct_molecule_frame,
)
from unwrap_traj.validate import InputError, load_inputs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures"


def triclinic_cell() -> np.ndarray:
    # columns: a1, a2, a3
    return np.array(
        [
            [10.0, 6.0, 0.0],
            [0.0, 8.0, 2.0],
            [0.0, 0.0, 9.0],
        ]
    )


def fold_into_cell(frac: np.ndarray, cell: Cell) -> tuple[np.ndarray, np.ndarray]:
    """Return (folded fractional, folded cartesian) with frac in [0, 1)."""
    frac_folded = frac - np.floor(frac)
    cart = frac_folded @ cell.matrix.T
    return frac_folded, cart


def topology_for(masses, molecule_ids, bonds):
    atoms = [
        {"atom_id": f"a{i}", "molecule_id": molecule_ids[i], "mass": float(masses[i])}
        for i in range(len(masses))
    ]
    return {"atoms": atoms, "bonds_zero_based": bonds}


def options(**kw):
    base = dict(
        max_bond_nm=1.2,
        max_com_speed_nm_per_ps=2.8,
        max_gap_ps=3.0,
        lag_times_ps=(1.0, 2.0, 3.0),
        lag_tolerance_ps=1e-8,
    )
    base.update(kw)
    return AnalysisOptions(**base)


# ---------------------------------------------------------------------------
# Candidate search
# ---------------------------------------------------------------------------


class TestCandidateSearch(unittest.TestCase):
    def setUp(self):
        self.cell = Cell(triclinic_cell())

    def brute(self, delta, radius, span=4):
        out = []
        for m in itertools.product(range(-span, span + 1), repeat=3):
            if (
                np.linalg.norm(delta + self.cell.matrix @ np.asarray(m, dtype=float))
                <= radius + 1e-9
            ):
                out.append(tuple(m))
        return sorted(out)

    def test_matches_brute_force(self):
        rng = np.random.default_rng(1234)
        for _ in range(60):
            delta = rng.uniform(-9, 9, size=3)
            radius = rng.uniform(0.3, 7.0)
            got = lattice_candidates(delta, self.cell, radius)
            want = self.brute(delta, radius)
            self.assertEqual(got, want, msg=f"delta={delta} radius={radius}")

    def test_triclinic_nearest_not_component_rounding(self):
        # A displacement whose nearest lattice image is NOT obtained by
        # rounding each fractional coordinate independently (fractional
        # component rounding is only valid for orthogonal cells).
        delta = np.array([5.53324223, 3.43813711, 0.25487704])
        frac = self.cell.cart_to_frac(delta)
        naive = tuple(int(x) for x in -np.rint(frac))
        nearest = min(
            itertools.product(range(-3, 4), repeat=3),
            key=lambda m: np.linalg.norm(
                delta + self.cell.matrix @ np.asarray(m, float)
            ),
        )
        self.assertNotEqual(tuple(naive), tuple(nearest))
        got = lattice_candidates(delta, self.cell, 4.7)
        self.assertEqual(got, [tuple(nearest)])
        self.assertNotIn(tuple(naive), got)

    def test_no_candidates_reports_empty(self):
        self.assertEqual(lattice_candidates(np.array([5.0, 0.0, 0.0]), self.cell, 2.0), [])

    def test_meta_box_bounds_every_candidate(self):
        delta = np.array([4.9, 3.9, 3.9])
        cands, meta = lattice_candidates(delta, self.cell, 2.0, return_meta=True)
        lo, hi = meta["box_lo"], meta["box_hi"]
        for m in cands:
            for k in range(3):
                self.assertGreaterEqual(m[k], lo[k])
                self.assertLessEqual(m[k], hi[k])


# ---------------------------------------------------------------------------
# Within-frame reconstruction
# ---------------------------------------------------------------------------


class TestWithinFrame(unittest.TestCase):
    def setUp(self):
        self.cell = Cell(triclinic_cell())

    def test_known_bond_lengths_analytic(self):
        # Triangle with exactly known bond lengths, all below max_bond_nm.
        local = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [-0.2, 0.7, 0.1],
            ]
        )
        bonds = [(0, 1), (0, 2), (1, 2)]
        true_lengths = [
            float(np.linalg.norm(local[j] - local[i])) for i, j in bonds
        ]
        for length in true_lengths:
            self.assertLess(length, 1.2)
        # Push the triangle across a cell face so folding splits it.
        frac = self.cell.cart_to_frac(local + np.array([9.0, 7.5, 8.9]))
        _, raw = fold_into_cell(frac, self.cell)
        recon = reconstruct_molecule_frame(raw, bonds, self.cell, 1.2)
        self.assertEqual(recon.status, "ok")
        np.testing.assert_allclose(recon.bond_lengths_nm, true_lengths, atol=1e-12)

    def test_reconstruction_is_gauge_invariant(self):
        local = np.array(
            [[0.0, 0.0, 0.0], [0.8, 0.2, 0.0], [-0.2, 0.7, 0.1]]
        )
        bonds = [(0, 1), (0, 2), (1, 2)]
        frac = self.cell.cart_to_frac(local + np.array([9.0, 7.5, 8.9]))
        _, raw = fold_into_cell(frac, self.cell)
        r1 = reconstruct_molecule_frame(raw, bonds, self.cell, 1.2)
        # Shift every atom by one lattice vector: a different folded input for
        # the same physical molecule.
        shifted_raw = raw + self.cell.shift_cart(np.array([2, -1, 3], dtype=float))
        r3 = reconstruct_molecule_frame(shifted_raw, bonds, self.cell, 1.2)
        self.assertEqual(r1.status, "ok")
        self.assertEqual(r3.status, "ok")
        np.testing.assert_allclose(r1.bond_lengths_nm, r3.bond_lengths_nm, atol=1e-12)

    def test_cycle_closure_conflict_is_detected(self):
        raw = np.array([[0.0, 0.0, 0.0], [3.3, 0.0, 0.0], [6.6, 0.0, 0.0]])
        bonds = [(0, 1), (1, 2), (2, 0)]
        recon = reconstruct_molecule_frame(raw, bonds, self.cell, 3.6)
        self.assertEqual(recon.status, "no_consistent_lattice_image")
        # Every individual bond does have a feasible image.
        for edge, cands in recon.edge_candidates.items():
            self.assertTrue(cands, msg=f"edge {edge} had no local candidate")
        self.assertIsNotNone(recon.conflict)
        self.assertEqual(recon.conflict["kind"], "cycle_closure_conflict")

    def test_open_chain_same_positions_is_consistent(self):
        # Same coordinates but a tree (no closing bond) is realisable: this
        # shows the failure above is specifically cycle closure, not geometry.
        raw = np.array([[0.0, 0.0, 0.0], [3.3, 0.0, 0.0], [6.6, 0.0, 0.0]])
        bonds = [(0, 1), (1, 2)]
        recon = reconstruct_molecule_frame(raw, bonds, self.cell, 3.6)
        self.assertEqual(recon.status, "ok")
        # All three atoms land on one straight, finite chain.
        np.testing.assert_allclose(recon.bond_lengths_nm, [3.3, 3.3], atol=1e-12)


# ---------------------------------------------------------------------------
# Synthetic analytic trajectory
# ---------------------------------------------------------------------------


class TestAnalyticTrajectory(unittest.TestCase):
    def setUp(self):
        self.cell_mat = triclinic_cell()
        self.cell = Cell(self.cell_mat)
        self.local = np.array(
            [[0.0, 0.0, 0.0], [0.8, 0.2, 0.0], [-0.2, 0.7, 0.1]]
        )
        self.masses = np.array([12.0, 1.0, 1.0])
        self.local_com = self.masses @ self.local / self.masses.sum()
        self.bonds = [(0, 1), (0, 2), (1, 2)]
        self.com0 = np.array([1.5, 2.0, 2.5])
        # modest velocity and small rotation, well inside the bounds
        self.vel = np.array([0.55, 0.35, 0.22])
        self.times = np.array([0.0, 1.0, 2.0, 4.0, 5.0])

    def _build(self, extra_shift=None):
        fracs = []
        coms = []
        for t in self.times:
            com = self.com0 + self.vel * t
            xyz = self.local + com
            physical = xyz
            if extra_shift is not None:
                physical = xyz + self.cell.shift_cart(np.asarray(extra_shift, float))
            frac = self.cell.cart_to_frac(physical)
            fracs.append(frac)
            coms.append(self.local_com + com)
        frac_folded = np.array(fracs) - np.floor(np.array(fracs))
        cart = frac_folded @ self.cell.matrix.T
        return cart, np.array(coms)

    def test_bond_lengths_and_com_gauge_invariance(self):
        raw, analytic_com = self._build()
        topo = topology_for(self.masses, ["m"] * 3, self.bonds)
        res = analyze_dataset(raw, self.times, self.cell, topo, options())
        mol = res["molecules"]["m"]
        self.assertTrue(mol["feasible_all_frames_and_steps"])
        # Analytic bond lengths recovered exactly.
        true_lengths = [
            float(np.linalg.norm(self.local[j] - self.local[i])) for i, j in self.bonds
        ]
        for f in mol["frames"]:
            np.testing.assert_allclose(f["bond_lengths_nm"], true_lengths, atol=1e-12)
        # One continuous segment.
        self.assertEqual(len(mol["segments"]), 1)
        seg = mol["segments"][0]
        self.assertEqual(seg["frame_indices"], [0, 1, 2, 3, 4])
        # Absolute COM differs from analytic by one constant lattice vector.
        unwrapped = np.asarray([fr["com_unwrapped_nm"] for fr in mol["frames"]])
        frac_gauge = self.cell.cart_to_frac(unwrapped[0] - analytic_com[0])
        for k in range(len(self.times)):
            diff_frac = self.cell.cart_to_frac(unwrapped[k] - analytic_com[k])
            np.testing.assert_allclose(diff_frac, frac_gauge, atol=1e-9)
            np.testing.assert_allclose(
                diff_frac, np.rint(diff_frac), atol=1e-7,
                err_msg="absolute offset must be an integer lattice vector",
            )
        # Gauge-invariant displacement equals the analytic velocity * dt.
        for k in range(1, len(self.times)):
            d = unwrapped[k] - unwrapped[k - 1]
            np.testing.assert_allclose(
                d, self.vel * (self.times[k] - self.times[k - 1]), atol=1e-9
            )
        # Fold-back traceability.
        self.assertTrue(mol["fold_back_check"]["ok"])
        self.assertLess(mol["fold_back_check"]["max_residual_nm"], 1e-12)

    def test_global_lattice_shift_leaves_shape_and_displacement(self):
        topo = topology_for(self.masses, ["m"] * 3, self.bonds)
        raw_a, _ = self._build()
        res_a = analyze_dataset(raw_a, self.times, self.cell, topo, options())

        # A physical translation by an integer lattice vector is removed by
        # folding, so it must be completely invisible to the analysis.
        raw_b, _ = self._build(extra_shift=[3, -2, 1])
        np.testing.assert_allclose(raw_a, raw_b, atol=1e-12)
        res_b = analyze_dataset(raw_b, self.times, self.cell, topo, options())

        mol_a = res_a["molecules"]["m"]
        mol_b = res_b["molecules"]["m"]
        for fa, fb in zip(mol_a["frames"], mol_b["frames"]):
            np.testing.assert_allclose(
                fa["bond_lengths_nm"], fb["bond_lengths_nm"], atol=1e-12
            )
            np.testing.assert_allclose(
                fa["unwrapped_atoms_nm"], fb["unwrapped_atoms_nm"], atol=1e-12
            )
        np.testing.assert_allclose(
            mol_a["segments"][0]["net_com_displacement_nm"],
            mol_b["segments"][0]["net_com_displacement_nm"],
            atol=1e-12,
        )

    def test_output_gauge_invariance(self):
        # Shape and within-segment displacements are invariant under adding an
        # arbitrary common lattice vector to a segment's absolute image.
        topo = topology_for(self.masses, ["m"] * 3, self.bonds)
        raw, _ = self._build()
        res = analyze_dataset(raw, self.times, self.cell, topo, options())
        mol = res["molecules"]["m"]
        seg = mol["segments"][0]
        gauge = self.cell.shift_cart(np.array([2, 1, -1], dtype=float))
        first = seg["frame_indices"][0]
        last = seg["frame_indices"][-1]
        shift_first = np.asarray(mol["frames"][first]["com_unwrapped_nm"]) + gauge
        shift_last = np.asarray(mol["frames"][last]["com_unwrapped_nm"]) + gauge
        displaced = shift_last - shift_first
        np.testing.assert_allclose(
            displaced, np.asarray(seg["net_com_displacement_nm"]), atol=1e-12
        )
        for entry in mol["frames"]:
            shifted = np.asarray(entry["unwrapped_atoms_nm"]) + gauge
            lengths = [
                float(np.linalg.norm(shifted[j] - shifted[i])) for i, j in self.bonds
            ]
            np.testing.assert_allclose(lengths, entry["bond_lengths_nm"], atol=1e-12)

    def test_multi_interval_flag(self):
        raw, _ = self._build()
        topo = topology_for(self.masses, ["m"] * 3, self.bonds)
        res = analyze_dataset(raw, self.times, self.cell, topo, options())
        steps = res["molecules"]["m"]["steps"]
        multi = [s for s in steps if s["from_frame"] == 2]  # dt = 2
        self.assertEqual(len(multi), 1)
        self.assertTrue(multi[0]["is_multi_interval"])
        self.assertEqual(multi[0]["nominal_interval_multiple"], 2.0)


# ---------------------------------------------------------------------------
# Continuity decisions: ambiguous / none / gap / unique
# ---------------------------------------------------------------------------


class TestContinuityDecisions(unittest.TestCase):
    def setUp(self):
        self.cell = Cell(np.diag([10.0, 10.0, 10.0]))

    def _run(self, positions, times, max_speed, max_gap):
        positions = np.asarray(positions, dtype=float)
        n = positions.shape[1]
        topo = topology_for([1.0] * n, ["m"] * n, [])
        return analyze_dataset(
            np.asarray(positions, float),
            np.asarray(times, float),
            self.cell,
            topo,
            options(
                max_com_speed_nm_per_ps=max_speed,
                max_gap_ps=max_gap,
            ),
        )

    def test_unique_translation_continues(self):
        res = self._run([[[1.0, 1.0, 1.0]], [[3.0, 1.0, 1.0]]], [0.0, 1.0], 2.8, 3.0)
        step = res["molecules"]["m"]["steps"][0]
        self.assertEqual(step["decision"], "continue")
        self.assertEqual(step["candidate_count"], 1)
        self.assertEqual(step["chosen_lattice_translation"], [0, 0, 0])
        self.assertEqual(len(res["molecules"]["m"]["segments"]), 1)

    def test_ambiguous_translation_breaks_without_picking_nearest(self):
        # 0.0 -> 5.0 nm in a 10 nm cell: images at -5 and +5 are equally near.
        res = self._run([[[0.0, 0.0, 0.0]], [[5.0, 0.0, 0.0]]], [0.0, 1.0], 6.0, 3.0)
        step = res["molecules"]["m"]["steps"][0]
        self.assertEqual(step["decision"], "break")
        self.assertEqual(step["reason"], "ambiguous_lattice_translation")
        self.assertEqual(step["candidate_count"], 2)
        self.assertIsNone(step["chosen_lattice_translation"])
        mol = res["molecules"]["m"]
        self.assertEqual(len(mol["segments"]), 2)
        self.assertEqual(mol["segments"][0]["frame_indices"], [0])
        self.assertEqual(mol["segments"][1]["frame_indices"], [1])
        # No displacement is accumulated across the break.
        self.assertEqual(mol["segments"][1]["net_com_displacement_nm"], [0.0, 0.0, 0.0])

    def test_no_feasible_translation_breaks(self):
        res = self._run([[[0.0, 0.0, 0.0]], [[5.0, 0.0, 0.0]]], [0.0, 1.0], 2.0, 3.0)
        step = res["molecules"]["m"]["steps"][0]
        self.assertEqual(step["decision"], "break")
        self.assertEqual(step["reason"], "no_feasible_lattice_translation")
        self.assertEqual(step["candidate_count"], 0)
        self.assertGreater(step["detail"]["nearest_translation_distance_nm"], 2.0)

    def test_gap_breaks_even_when_translation_unique(self):
        # dt = 4 > max_gap = 3; displacement tiny and unique.
        res = self._run([[[0.0, 0.0, 0.0]], [[0.2, 0.0, 0.0]]], [0.0, 4.0], 2.8, 3.0)
        step = res["molecules"]["m"]["steps"][0]
        self.assertEqual(step["decision"], "break")
        self.assertEqual(step["reason"], "gap_exceeds_max_allowed")
        self.assertEqual(len(res["molecules"]["m"]["segments"]), 2)

    def test_no_cross_break_displacement_accumulation(self):
        # Three frames: unique, then gap.  Segment 2 starts at its own gauge.
        res = self._run(
            [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.5, 0.0, 0.0]]],
            [0.0, 1.0, 10.0],
            2.8,
            3.0,
        )
        mol = res["molecules"]["m"]
        self.assertEqual([s["frame_indices"] for s in mol["segments"]], [[0, 1], [2]])
        # Cumulative translation resets for the new segment.
        self.assertEqual(mol["frames"][2]["cumulative_lattice_translation"], [0, 0, 0])
        self.assertIsNone(mol["frames"][2]["step_lattice_translation"])
        # Segment net displacement only reflects its own frames.
        np.testing.assert_allclose(
            mol["segments"][1]["net_com_displacement_nm"], [0.0, 0.0, 0.0]
        )


# ---------------------------------------------------------------------------
# MSD
# ---------------------------------------------------------------------------


class TestMSD(unittest.TestCase):
    def test_pair_counts_and_no_zero_fill(self):
        times = np.array([0.0, 1.0, 2.0, 3.0])
        # straight line motion, all one segment
        com = np.column_stack(
            [np.arange(4) * 1.0, np.zeros(4), np.zeros(4)]
        )
        seg = np.zeros(4, dtype=np.int64)
        msd = compute_msd(times, com, seg, (1.0, 2.0, 5.0), 1e-8)
        self.assertEqual(msd["1"]["pair_count"], 3)
        self.assertEqual(msd["1"]["pair_indices"], [[0, 1], [1, 2], [2, 3]])
        self.assertEqual(msd["2"]["pair_count"], 2)
        self.assertEqual(msd["2"]["pair_indices"], [[0, 2], [1, 3]])
        self.assertAlmostEqual(msd["2"]["msd_nm2"], 4.0)
        self.assertFalse(msd["5"]["covered"])
        self.assertIsNone(msd["5"]["msd_nm2"])
        self.assertEqual(msd["5"]["pair_count"], 0)

    def test_across_segment_pairs_excluded(self):
        times = np.array([0.0, 1.0, 2.0])
        com = np.array([[0.0, 0, 0], [0.0, 0, 0], [100.0, 0, 0]])
        seg = np.array([0, 0, 1])
        msd = compute_msd(times, com, seg, (1.0, 2.0), 1e-8)
        # frame 1 -> 2 is dt 1 but crosses a segment: excluded.
        self.assertEqual(msd["1"]["pair_indices"], [[0, 1]])
        # frame 0 -> 2 also crosses segments: excluded -> uncovered.
        self.assertFalse(msd["2"]["covered"])
        self.assertIsNone(msd["2"]["msd_nm2"])


# ---------------------------------------------------------------------------
# Fixture-level checks
# ---------------------------------------------------------------------------


class TestFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = load_inputs(
            FIX / "trajectory.npz", FIX / "topology.json", FIX / "plan.json"
        )
        cls.result = analyze_dataset(
            cls.loaded.positions_nm,
            cls.loaded.time_ps,
            cls.loaded.cell,
            cls.loaded.topology,
            cls.loaded.options,
        )

    def test_input_checks(self):
        checks = self.loaded.checks
        self.assertTrue(checks["mass_positive_and_finite"])
        self.assertTrue(checks["bonds_within_range"])
        self.assertTrue(checks["bonds_within_molecule"])
        self.assertTrue(checks["molecules_connected"])
        self.assertTrue(checks["time_strictly_increasing"])
        self.assertEqual(checks["time_diffs_ps"], [1.0, 1.0, 2.0, 1.0, 4.0, 1.0])
        self.assertAlmostEqual(checks["cell_determinant"], 720.0)
        self.assertTrue(checks["cell_is_fixed"])

    def test_all_frames_reconstructed(self):
        self.assertTrue(self.result["summary"]["all_frames_reconstructed"])
        for mol in self.result["molecules"].values():
            for f in mol["frames"]:
                self.assertTrue(f["ok"])
                self.assertEqual(f["within_frame_status"], "ok")

    def test_expected_breaks(self):
        m0 = self.result["molecules"]["m0"]
        reasons = {(b["after_frame"], b["before_frame"]): b["reason"] for b in m0["breaks"]}
        # dt=2 step has two feasible images for the fast molecule.
        self.assertEqual(reasons[(2, 3)], "ambiguous_lattice_translation")
        # dt=4 step exceeds max_gap=3.
        self.assertEqual(reasons[(4, 5)], "gap_exceeds_max_allowed")
        self.assertEqual(
            [s["frame_indices"] for s in m0["segments"]], [[0, 1, 2], [3, 4], [5, 6]]
        )
        for mid in ("m1", "m2"):
            reasons = {
                (b["after_frame"], b["before_frame"]): b["reason"]
                for b in self.result["molecules"][mid]["breaks"]
            }
            self.assertEqual(reasons, {(4, 5): "gap_exceeds_max_allowed"})

    def test_lattice_offsets_match_unwrapped(self):
        cell = self.loaded.cell
        raw = self.loaded.positions_nm
        arrays = self.result["arrays"]
        for mid, mol in self.result["molecules"].items():
            for f, entry in enumerate(mol["frames"]):
                offsets = np.asarray(entry["atom_lattice_offsets_total"], dtype=float)
                idx = mol["atom_indices"]
                expected = raw[f, idx] + cell.shift_cart(offsets)
                np.testing.assert_allclose(
                    expected, np.asarray(entry["unwrapped_atoms_nm"]), atol=1e-12
                )
                self.assertTrue(arrays["atom_unwrapped_valid"][f, idx].all())

    def test_fold_back_exact(self):
        for mol in self.result["molecules"].values():
            self.assertTrue(mol["fold_back_check"]["ok"])
            self.assertLess(mol["fold_back_check"]["max_residual_nm"], 1e-12)

    def test_boundary_crossings_detected(self):
        crossings = self.result["boundary_crossings"]
        self.assertEqual(len(crossings), 3)
        got = {(c["atom_id"], c["from_frame"], c["to_frame"]) for c in crossings}
        self.assertEqual(
            got,
            {("m0a0", 0, 1), ("m0a0", 3, 4), ("m0a1", 3, 4)},
        )
        for c in crossings:
            self.assertTrue(c["within_same_continuous_segment"])

    def test_msd_coverage_and_counts(self):
        pooled = self.result["summary"]["pooled_msd"]
        self.assertEqual(pooled["1"]["pair_count"], 12)
        self.assertEqual(pooled["2"]["pair_count"], 5)
        self.assertEqual(pooled["3"]["pair_count"], 4)
        self.assertTrue(all(v["covered"] for v in pooled.values()))
        # m0 cannot contribute to lag 3 (its longest segment spans 2 ps).
        self.assertEqual(pooled["3"]["pair_counts_by_molecule"]["m0"], 0)
        for mid, mol in self.result["molecules"].items():
            for key, item in mol["msd"].items():
                if not item["covered"]:
                    self.assertIsNone(item["msd_nm2"])
                else:
                    self.assertEqual(item["pair_count"], len(item["pair_indices"]))

    def test_multi_interval_step_reported(self):
        for mid in ("m0", "m1", "m2"):
            steps = self.result["molecules"][mid]["steps"]
            dt2 = [s for s in steps if s["to_frame"] == 3]
            self.assertTrue(dt2[0]["is_multi_interval"])
            self.assertEqual(dt2[0]["nominal_interval_multiple"], 2.0)


class TestCycleFixture(unittest.TestCase):
    def test_cycle_closure_conflict(self):
        loaded = load_inputs(
            FIX / "cycle-case.npz",
            FIX / "cycle-topology.json",
            FIX / "plan.json",
        )
        self.assertEqual(loaded.checks["plan_meta"]["bond_bound_source"], "topology")
        self.assertAlmostEqual(loaded.options.max_bond_nm, 3.6)
        res = analyze_dataset(
            loaded.positions_nm,
            loaded.time_ps,
            loaded.cell,
            loaded.topology,
            loaded.options,
        )
        mol = res["molecules"]["cycle"]
        self.assertFalse(res["summary"]["all_frames_reconstructed"])
        entry = mol["frames"][0]
        self.assertEqual(entry["within_frame_status"], "no_consistent_lattice_image")
        self.assertEqual(entry["conflict"]["kind"], "cycle_closure_conflict")
        # Locally every bond has a feasible image (the loop is the problem).
        for edge, cands in entry["edge_candidates"].items():
            self.assertTrue(cands, msg=edge)
        for key in ("1", "2", "3"):
            self.assertFalse(mol["msd"][key]["covered"])
            self.assertIsNone(mol["msd"][key]["msd_nm2"])


class TestValidationErrors(unittest.TestCase):
    def test_time_order_rejected(self):
        traj = FIX / "trajectory.npz"
        with self.assertRaises(InputError):
            # Build a temp npz with non-monotonic time via direct loader checks.
            import tempfile

            data = np.load(traj)
            bad = {
                "positions_nm": data["positions_nm"],
                "time_ps": np.array([0.0, 1.0, 2.0, 1.5, 5.0, 9.0, 10.0]),
                "cell_columns_nm": data["cell_columns_nm"],
            }
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "bad.npz"
                np.savez(p, **bad)
                load_inputs(p, FIX / "topology.json", FIX / "plan.json")

    def test_bad_bond_index_rejected(self):
        topo = {
            "atoms": [
                {"atom_id": "a0", "molecule_id": "m", "mass": 1},
                {"atom_id": "a1", "molecule_id": "m", "mass": 1},
            ],
            "bonds_zero_based": [[0, 5]],
        }
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "topo.json"
            tp.write_text(json.dumps(topo))
            with self.assertRaises(InputError):
                load_inputs(FIX / "cycle-case.npz", tp, FIX / "plan.json")

    def test_plan_missing_key_rejected(self):
        import tempfile

        plan = json.loads((FIX / "plan.json").read_text())
        del plan["max_gap_ps"]
        with tempfile.TemporaryDirectory() as td:
            pp = Path(td) / "plan.json"
            pp.write_text(json.dumps(plan))
            with self.assertRaises(InputError):
                load_inputs(
                    FIX / "cycle-case.npz",
                    FIX / "cycle-topology.json",
                    pp,
                )


if __name__ == "__main__":
    unittest.main()
