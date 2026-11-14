"""Hand-computable shared-shock example and graph validation tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reliability.likelihood import (
    pattern_probabilities,
    summarise_patterns,
)
from reliability.model import (
    ValidationError,
    build_model,
    load_snapshots,
)


def _write_fixture(root: Path, system: dict, events: str, plan: dict) -> tuple[Path, Path, Path]:
    system_path = root / "system.json"
    events_path = root / "events.csv"
    plan_path = root / "plan.json"
    system_path.write_text(json.dumps(system), encoding="utf-8")
    events_path.write_text(events, encoding="utf-8")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return system_path, events_path, plan_path


class SharedShockTest(unittest.TestCase):
    """Two components sharing one shock, plus an independent heat branch.

    a = local_a OR surge,  b = local_b OR surge,  c = local_c OR heat,
    sys = a OR b OR c

    With P(local_a)=1/5, P(local_b)=1/4, P(local_c)=1/2 and unknown
    (p_surge, p_heat), the hand-computed probabilities are exercised below.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        system = {
            "root": "sys",
            "observables": ["a", "b", "c", "sys"],
            "nodes": [
                {"id": "a", "op": "or", "inputs": ["local_a", "surge"]},
                {"id": "b", "op": "or", "inputs": ["local_b", "surge"]},
                {"id": "c", "op": "or", "inputs": ["local_c", "heat"]},
                {"id": "sys", "op": "or", "inputs": ["a", "b", "c"]},
            ],
        }
        events = (
            "event_id,fixed_probability,unknown_parameter\n"
            "local_a,1/5,\n"
            "local_b,1/4,\n"
            "local_c,1/2,\n"
            "surge,,p_surge\n"
            "heat,,p_heat\n"
        )
        plan = {
            "parameters": {"p_surge": [0.0, 1.0], "p_heat": [0.0, 1.0]},
            "support_loglik_drop": 1.0,
            "one_repair": {"node": "a", "forced_failure_state": False},
        }
        self.paths = _write_fixture(root, system, events, plan)
        self.model = build_model(*self.paths)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _prob(self, observations, theta) -> float:
        summary = summarise_patterns(self.model, observations)
        probs = pattern_probabilities(summary, self.model.basis(theta))
        return float(np.prod(probs))

    def test_shared_shock_two_components_health(self):
        # P(a=0, b=0) = (1-l_a)(1-l_b)(1-p_surge), independent of heat.
        for ps, ph in {(0.0, 0.0), (0.1, 0.4), (0.7, 0.2), (1.0, 1.0)}:
            expected = (1 - 0.2) * (1 - 0.25) * (1 - ps)
            got = self._prob([{"a": 0, "b": 0, "c": None, "sys": None}], [ps, ph])
            self.assertAlmostEqual(got, expected, places=12)

    def test_shared_shock_joint_and_marginals_sum_to_total(self):
        theta = [0.13, 0.37]
        joint = self._prob([{"a": 0, "b": 0, "c": None, "sys": None}], theta)
        a_only = self._prob([{"a": 0, "b": None, "c": None, "sys": None}], theta)
        b_only = self._prob([{"a": None, "b": 0, "c": None, "sys": None}], theta)
        # a_only = 1 - [l_a + p_s - l_a p_s]; b_only analogous.
        la, lb = 0.2, 0.25
        ps = theta[0]
        self.assertAlmostEqual(a_only, 1 - (la + ps - la * ps), places=12)
        self.assertAlmostEqual(b_only, 1 - (lb + ps - lb * ps), places=12)
        # The joint health probability must not equal the product of marginals
        # because surge induces dependence.
        self.assertGreater(joint, a_only * b_only - 1e-12)
        self.assertNotAlmostEqual(joint, a_only * b_only, places=6)

    def test_full_observation_all_healthy(self):
        theta = [0.2, 0.05]
        expected = (1 - 0.2) * (1 - 0.25) * (1 - 0.5) * (1 - 0.2) * (1 - 0.05)
        got = self._prob([{"a": 0, "b": 0, "c": 0, "sys": 0}], theta)
        self.assertAlmostEqual(got, expected, places=12)

    def test_system_failure_probability(self):
        theta = [0.1, 0.3]
        # P(sys=1) = 1 - P(all healthy).
        expected = 1 - (1 - 0.2) * (1 - 0.25) * (1 - 0.5) * (1 - 0.1) * (1 - 0.3)
        got = self._prob([{"a": None, "b": None, "c": None, "sys": 1}], theta)
        self.assertAlmostEqual(got, expected, places=12)

    def test_hand_checked_risk_coefficients(self):
        baseline = self.model.risk_coefficients(repaired=False)
        repaired = self.model.risk_coefficients(repaired=True)
        # Baseline: system fails unless a,b,c are all healthy.  Enumerating
        # the 4 (surge,heat) groups gives the same coefficient vector the
        # closed-form risk uses.
        theta = [0.1, 0.3]
        expected_fail = 1 - (1 - 0.2) * (1 - 0.25) * (1 - 0.5) * (1 - 0.1) * (1 - 0.3)
        self.assertAlmostEqual(float(baseline @ self.model.basis(theta)), expected_fail, places=12)
        # Repair clamps a healthy: failure then needs (local_b or surge) or
        # (local_c or heat), with surge still active on b.
        repaired_fail = 1 - (1 - 0.25) * (1 - 0.5) * (1 - 0.1) * (1 - 0.3)
        self.assertAlmostEqual(
            float(repaired @ self.model.basis(theta)), repaired_fail, places=12
        )

    def _event_probability(self, event_value: np.ndarray, theta, clamped=()) -> float:
        """P(event_value holds) in the (optionally repaired) model."""

        vals = self.model.evaluate_nodes(clamped_healthy=clamped)
        target = vals[event_value] if isinstance(event_value, str) else event_value
        weights = np.bincount(
            self.model.group_codes.astype(np.int64),
            weights=self.model.fixed_weights * target,
            minlength=4,
        )
        return float(weights @ self.model.basis(theta))

    def test_repair_is_not_conditioning_on_observed_health(self):
        # Clamping a healthy removes a's own failure but leaves surge active on
        # b.  Conditioning on a=0 instead implies surge=0, which would also
        # shield b.  The two must differ.
        theta = [0.15, 0.2]
        prob_b = self._event_probability("b", theta, clamped=("a",))
        repaired_b_fail = 1 - (1 - 0.25) * (1 - 0.15)
        self.assertAlmostEqual(prob_b, repaired_b_fail, places=12)
        # Conditioning on a=0: surge must be 0, so P(b=1 | a=0) = P(local_b=1).
        cond = self._prob(
            [{"a": 0, "b": 1, "c": None, "sys": None}], theta
        ) / self._prob([{"a": 0, "b": None, "c": None, "sys": None}], theta)
        self.assertAlmostEqual(cond, 0.25, places=12)
        self.assertGreater(prob_b, cond)

    def test_heat_survives_on_other_branch_after_repair(self):
        # Repairing a (surge-fed) leaves heat acting on c.
        theta = [0.1, 0.2]
        prob_c = self._event_probability("c", theta, clamped=("a",))
        self.assertAlmostEqual(prob_c, 1 - (1 - 0.5) * (1 - 0.2), places=12)

    def test_NA_never_becomes_healthy(self):
        summary = summarise_patterns(
            self.model, [{"a": 0, "b": None, "c": 0, "sys": None}]
        )
        pattern = summary.patterns[0]
        self.assertEqual(pattern[1], -1)  # b is unobserved, encoded -1
        theta = [0.1, 0.2]
        # a and c are each observed healthy, so a=0 forces (local_a,surge)=(0,0)
        # and c=0 forces (local_c,heat)=(0,0); b and sys are marginalised out.
        expected = (1 - 0.2) * (1 - 0.1) * (1 - 0.5) * (1 - 0.2)
        got = float(pattern_probabilities(summary, self.model.basis(theta))[0])
        self.assertAlmostEqual(got, expected, places=12)


class ValidationTest(unittest.TestCase):
    def _base(self):
        return (
            {
                "root": "sys",
                "observables": ["sys"],
                "nodes": [{"id": "sys", "op": "or", "inputs": ["e1"]}],
            },
            "event_id,fixed_probability,unknown_parameter\n"
            "e1,1/3,\n"
            "surge,,p_surge\n"
            "heat,,p_heat\n",
            {
                "parameters": {"p_surge": [0, 1], "p_heat": [0, 1]},
                "support_loglik_drop": 1.0,
                "one_repair": {"node": "sys", "forced_failure_state": False},
            },
        )

    def _build(self, system, events, plan):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_fixture(Path(tmp), system, events, plan)
            return build_model(*paths)

    def test_duplicate_input_same_gate_rejected(self):
        system, events, plan = self._base()
        system["nodes"][0]["inputs"] = ["e1", "e1"]
        with self.assertRaises(ValidationError):
            self._build(system, events, plan)

    def test_cross_gate_shared_fact_allowed(self):
        system = {
            "root": "sys",
            "observables": ["a", "b", "sys"],
            "nodes": [
                {"id": "a", "op": "or", "inputs": ["e1", "surge"]},
                {"id": "b", "op": "or", "inputs": ["e2", "surge"]},
                {"id": "sys", "op": "and", "inputs": ["a", "b"]},
            ],
        }
        events = (
            "event_id,fixed_probability,unknown_parameter\n"
            "e1,1/3,\ne2,1/4,\nsurge,,p_surge\nheat,,p_heat\n"
        )
        plan = {
            "parameters": {"p_surge": [0, 1], "p_heat": [0, 1]},
            "support_loglik_drop": 1.0,
            "one_repair": {"node": "a", "forced_failure_state": False},
        }
        model = self._build(system, events, plan)
        self.assertEqual(model.validate_graph()[0][:12], "graph is a D")

    def test_cycle_rejected(self):
        system, events, plan = self._base()
        system["nodes"] = [
            {"id": "sys", "op": "or", "inputs": ["x"]},
            {"id": "x", "op": "or", "inputs": ["sys", "e1"]},
        ]
        with self.assertRaises(ValidationError):
            self._build(system, events, plan)

    def test_unknown_input_rejected(self):
        system, events, plan = self._base()
        system["nodes"][0]["inputs"] = ["nope"]
        with self.assertRaises(ValidationError):
            self._build(system, events, plan)

    def test_kofn_k_out_of_range_rejected(self):
        system, events, plan = self._base()
        system["nodes"] = [
            {"id": "sys", "op": "kofn", "k": 3, "inputs": ["e1", "e2"]},
        ]
        events = (
            "event_id,fixed_probability,unknown_parameter\n"
            "e1,1/3,\ne2,1/4,\nsurge,,p_surge\nheat,,p_heat\n"
        )
        with self.assertRaises(ValidationError):
            self._build(system, events, plan)

    def test_zero_or_one_fixed_probability_parse(self):
        system, events, plan = self._base()
        events = (
            "event_id,fixed_probability,unknown_parameter\n"
            "e1,1,\nsurge,,p_surge\nheat,,p_heat\n"
        )
        model = self._build(system, events, plan)
        self.assertAlmostEqual(model._event_by_id["e1"].fixed_probability, 1.0)


class SnapshotContradictionTest(unittest.TestCase):
    def test_contradictory_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            system = {
                "root": "sys",
                "observables": ["a", "b", "sys"],
                "nodes": [
                    {"id": "a", "op": "or", "inputs": ["e1"]},
                    {"id": "b", "op": "or", "inputs": ["e2"]},
                    {"id": "sys", "op": "and", "inputs": ["a", "b"]},
                ],
            }
            events = (
                "event_id,fixed_probability,unknown_parameter\n"
                "e1,0,\ne2,0,\nsurge,,p_surge\nheat,,p_heat\n"
            )
            plan = {
                "parameters": {"p_surge": [0, 1], "p_heat": [0, 1]},
                "support_loglik_drop": 1.0,
                "one_repair": {"node": "a", "forced_failure_state": False},
            }
            paths = _write_fixture(root, system, events, plan)
            model = build_model(*paths)
            # e1=e2=surge=heat=0 forces all healthy; sys=1 is impossible.
            obs, _ = load_snapshots(
                self._csv(root, "cycle_id,a,b,sys\nc1,0,0,1\n"),
                system["observables"],
            )
            from reliability.likelihood import contradiction_report

            summary = summarise_patterns(model, obs)
            report = contradiction_report(model, summary)
            self.assertFalse(report["consistent"])
            self.assertEqual(len(report["inconsistent_patterns"]), 1)

    @staticmethod
    def _csv(root: Path, text: str) -> Path:
        path = root / "snap.csv"
        path.write_text(text, encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
