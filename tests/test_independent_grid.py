"""Independent brute-force cross-checks for the support region and risk range.

These tests avoid the package's pattern aggregation.  They enumerate the full
primitive configuration table, evaluate the circuit with separately written
recursive logic, and build a per-cycle weight vector over the four
(surge, heat) groups.  That per-cycle summary is computed once; the dense 2-D
grid is then evaluated directly with those weights.  The package results must
agree with this second implementation.
"""

from __future__ import annotations

import itertools
import unittest
from pathlib import Path

import numpy as np

from reliability.assess import assess
from reliability.fit import support_contains


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class BruteForce:
    """Independent enumerator; no helper from ``reliability.likelihood``."""

    def __init__(self, model):
        self.model = model
        self.primitives = list(model.primitives)
        self.fixed = {
            e.event_id: e.fixed_probability
            for e in model.events
            if e.fixed_probability is not None
        }
        self.param_of = {
            e.event_id: e.unknown_parameter
            for e in model.events
            if e.unknown_parameter is not None
        }
        self.groups = list(itertools.product([False, True], repeat=2))
        self.configs = list(
            itertools.product([False, True], repeat=len(self.primitives))
        )
        # fixed probability and unknown-code for each config
        self.fixed_weight = []
        self.group_code = []
        for bits in self.configs:
            assignment = dict(zip(self.primitives, bits))
            weight = 1.0
            code = []
            for name in self.primitives:
                if name in self.fixed:
                    prob = self.fixed[name]
                    weight *= prob if assignment[name] else (1 - prob)
                else:
                    code.append(assignment[name])
            assert len(code) == 2
            self.fixed_weight.append(weight)
            self.group_code.append(tuple(code))
        self.fixed_weight = np.array(self.fixed_weight)

    def node_value(self, node_id, assignment, repaired=False):
        if node_id not in self.model.nodes:
            return assignment[node_id]
        if repaired and node_id == self.model.plan.repair_node:
            return False
        node = self.model.nodes[node_id]
        vals = [self.node_value(i, assignment, repaired) for i in node.inputs]
        if node.op == "or":
            return any(vals)
        if node.op == "and":
            return all(vals)
        return sum(vals) >= int(node.k)

    def pattern_weights(self, observations, repaired=False):
        """(4,) weights over (surge,heat) groups for one cycle."""

        weights = np.zeros((2, 2))
        for idx, bits in enumerate(self.configs):
            assignment = dict(zip(self.primitives, bits))
            ok = True
            for name, observed in observations.items():
                if observed is None:
                    continue
                if self.node_value(name, assignment, repaired) != bool(observed):
                    ok = False
                    break
            if ok:
                codes = self.group_code[idx]
                weights[int(codes[0]), int(codes[1])] += self.fixed_weight[idx]
        return weights.reshape(-1)

    def risk_weights(self, repaired=False):
        """Weights for the event 'system fails', one cycle, (4,)."""

        weights = np.zeros((2, 2))
        for idx, bits in enumerate(self.configs):
            assignment = dict(zip(self.primitives, bits))
            if self.node_value(self.model.root, assignment, repaired):
                codes = self.group_code[idx]
                weights[int(codes[0]), int(codes[1])] += self.fixed_weight[idx]
        return weights.reshape(-1)


def basis4(p_surge, p_heat):
    return np.array(
        [
            (1 - p_surge) * (1 - p_heat),
            (1 - p_surge) * p_heat,
            p_surge * (1 - p_heat),
            p_surge * p_heat,
        ]
    )


class IndependentGridCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = assess(
            FIXTURES / "system.json",
            FIXTURES / "events.csv",
            FIXTURES / "plan.json",
            FIXTURES / "snapshots.csv",
            grid_size=201,
        )
        cls.weak = assess(
            FIXTURES / "system.json",
            FIXTURES / "events.csv",
            FIXTURES / "plan.json",
            FIXTURES / "weak_snapshots.csv",
            grid_size=201,
        )
        cls.brute = {
            "base": BruteForce(cls.base.model),
            "weak": BruteForce(cls.weak.model),
        }

    def _cycle_weights(self, key, assessment):
        if not hasattr(assessment, "_bf_cycle_weights"):
            bf = self.brute[key]
            assessment._bf_cycle_weights = [
                bf.pattern_weights(obs) for obs in assessment.observations
            ]
        return assessment._bf_cycle_weights

    def _brute_loglik_grid(self, key, assessment, axis0, axis1):
        weights = np.array(self._cycle_weights(key, assessment))  # (C, 4)
        flat0, flat1 = np.meshgrid(axis0, axis1, indexing="ij")
        flat0 = flat0.reshape(-1)
        flat1 = flat1.reshape(-1)
        probs = weights @ np.stack(
            [
                (1 - flat0) * (1 - flat1),
                (1 - flat0) * flat1,
                flat0 * (1 - flat1),
                flat0 * flat1,
            ]
        )  # (C, N)
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.where(probs > 0, np.log(probs), -np.inf)
        return logs.sum(axis=0).reshape(len(axis0), len(axis1))

    def test_likelihood_matches_brute_force(self):
        for key, assessment in (("base", self.base), ("weak", self.weak)):
            bf = self.brute[key]
            for theta in ([0.05, 0.12], [0.2, 0.03], [0.01, 0.25], [0.15, 0.15]):
                fast = float(
                    np.sum(
                        assessment.summary.counts
                        * np.log(
                            assessment.summary.weights
                            @ assessment.model.basis(theta)
                        )
                    )
                )
                brute = float(
                    sum(
                        np.log(bf.pattern_weights(obs) @ basis4(*theta))
                        for obs in assessment.observations
                    )
                )
                self.assertAlmostEqual(
                    fast, brute, places=9,
                    msg=f"{assessment.snapshot_path} theta={theta}",
                )

    def test_mle_grid_matches_brute_force(self):
        assessment = self.base
        axis = np.linspace(0.0, 0.3, 301)
        grid = self._brute_loglik_grid("base", assessment, axis, axis)
        best = float(np.nanmax(grid))
        self.assertLessEqual(
            abs(best - assessment.fit.loglik_mle), 2e-3,
            "independent brute-force grid maximum should match reported MLE",
        )

    def test_support_region_and_risk_extremes_match_fine_grid(self):
        for key, assessment in (("base", self.base), ("weak", self.weak)):
            fit = assessment.fit
            bf = self.brute[key]
            axis = np.linspace(0.0, 0.3, 181)
            ll = self._brute_loglik_grid(key, assessment, axis, axis)
            support = np.isfinite(ll) & (ll >= fit.support_threshold - 1e-9)
            self.assertTrue(np.any(support))

            base_w = bf.risk_weights(False)
            rep_w = bf.risk_weights(True)
            ps, ph = np.meshgrid(axis, axis, indexing="ij")
            b = base_w @ np.stack(
                [(1 - ps) * (1 - ph), (1 - ps) * ph, ps * (1 - ph), ps * ph]
            ).reshape(4, -1)
            r = rep_w @ np.stack(
                [(1 - ps) * (1 - ph), (1 - ps) * ph, ps * (1 - ph), ps * ph]
            ).reshape(4, -1)
            b = b.reshape(support.shape)
            r = r.reshape(support.shape)
            benefit = b - r
            baseline = assessment.risk["baseline"]
            repaired = assessment.risk["repaired"]
            benefit_report = assessment.risk["absolute_benefit"]
            tol = 1e-2  # 181x181 -> spacing 0.00167; SLSQP may exceed grid slightly
            self.assertLessEqual(
                abs(baseline["min"]["value"] - b[support].min()), tol)
            self.assertLessEqual(
                abs(baseline["max"]["value"] - b[support].max()), tol)
            self.assertLessEqual(
                abs(repaired["min"]["value"] - r[support].min()), tol)
            self.assertLessEqual(
                abs(repaired["max"]["value"] - r[support].max()), tol)
            self.assertLessEqual(
                abs(benefit_report["min"]["value"] - benefit[support].min()), tol)
            self.assertLessEqual(
                abs(benefit_report["max"]["value"] - benefit[support].max()), tol)
            # SLSQP-refined extrema must not be more extreme than the true
            # continuous-support values by more than optimization tolerance;
            # they can be slightly outside the coarse grid, which is expected.
            self.assertLessEqual(
                baseline["min"]["value"], b[support].min() + tol)

    def test_witnesses_reproduce_brute_force(self):
        for key, assessment in (("base", self.base), ("weak", self.weak)):
            bf = self.brute[key]
            base_w = bf.risk_weights(False)
            rep_w = bf.risk_weights(True)
            for witness in assessment.risk["witnesses"]:
                theta = witness["theta"]
                basis = basis4(*theta)
                b = float(base_w @ basis)
                r = float(rep_w @ basis)
                self.assertAlmostEqual(b, witness["baseline_risk"], places=9)
                self.assertAlmostEqual(r, witness["repaired_risk"], places=9)
                self.assertAlmostEqual(b - r, witness["absolute_benefit"], places=9)
                self.assertTrue(
                    support_contains(assessment.fit, assessment.summary, theta)
                )

    def test_identifiability_objects_and_weak_boundary(self):
        weak_fit = self.weak.fit
        base_fit = self.base.fit
        self.assertEqual(weak_fit.identifiability["rank"], 1)
        self.assertFalse(weak_fit.identifiability["identified_separately"])
        touched = {h["edge"] for h in weak_fit.boundary_hits if h["covered"]}
        self.assertTrue({"axis0_low", "axis1_low"} <= touched)
        self.assertEqual(base_fit.identifiability["rank"], 2)
        self.assertTrue(base_fit.identifiability["identified_separately"])
        self.assertGreater(weak_fit.grid_mask.mean(), base_fit.grid_mask.mean())

    def test_weak_flat_direction_is_nearly_equally_likely(self):
        probes = self.weak.fit.identifiability["weak_direction_probes"]
        self.assertLess(max(abs(p["loglik_change"]) for p in probes), 0.5)
        # And the parameter actually moves along that direction.
        self.assertGreater(max(p["theta_change"] for p in probes), 0.01)


if __name__ == "__main__":
    unittest.main()
