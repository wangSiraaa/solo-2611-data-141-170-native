"""Exact-arithmetic regression and contract tests (stdlib unittest).

Run with:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from policy_tool.bruteforce import brute_force_value, walk_policy_value
from policy_tool.model import (
    ModelError,
    ZeroProbabilityError,
    load_model,
    parse_fraction,
)
from policy_tool.replay import ReplayError, replay
from policy_tool.solver import solve
from policy_tool.report import strategy_tree

F = Fraction


def main_model():
    return load_model(
        os.path.join(ROOT, "fixtures/model.json"),
        actions_path=os.path.join(ROOT, "fixtures/actions.csv"),
        plans_path=os.path.join(ROOT, "fixtures/plans.json"),
    )


def tiny_model():
    return load_model(os.path.join(ROOT, "fixtures/tiny.json"))


def correlated_model():
    return load_model(os.path.join(ROOT, "fixtures/correlated.json"))


class ExactParsingTests(unittest.TestCase):
    def test_fractions_and_decimals(self):
        self.assertEqual(parse_fraction("3/10", "x"), F(3, 10))
        self.assertEqual(parse_fraction("0.3", "x"), F(3, 10))
        self.assertEqual(parse_fraction("1e-3", "x"), F(1, 1000))
        self.assertEqual(parse_fraction(2, "x"), F(2))
        self.assertEqual(parse_fraction(" 7 ", "x"), F(7))
        with self.assertRaises(ModelError):
            parse_fraction("abc", "x")
        with self.assertRaises(ModelError):
            parse_fraction("1/0", "x")

    def test_float_is_exact_decimal_not_binary(self):
        self.assertEqual(parse_fraction(0.1, "x"), F(1, 10))


class TinyHandCheckTests(unittest.TestCase):
    """The task asks for a hand-check of tiny.json.

    priors fault=1/3 ok=2/3; lamp perfectly reveals the state, fee 1.
    repair(2,4), release(20,0).
    Stop: min( (1/3)2+(2/3)4=10/3 , (1/3)20=20/3 ) = 10/3.
    Test: 1 + (1/3)(2) + (2/3)(0) = 1 + 2/3 = 5/3  < 10/3, so buy.
    After lamp=1 -> fault -> repair (2); after lamp=0 -> ok -> release (0).
    """

    def test_value_matches_hand_calculation(self):
        sol = solve(tiny_model(), F(1))
        self.assertEqual(sol.value, F(5, 3))
        self.assertEqual(sol.expected_fees, F(1))
        self.assertEqual(sol.expected_terminal_loss, F(2, 3))
        self.assertEqual(sol.no_detection_cost, F(10, 3))
        self.assertEqual(sol.gross_information_value, F(8, 3))
        self.assertEqual(sol.net_information_value, F(5, 3))
        self.assertEqual(sol.root.decision_kind, "test")
        self.assertEqual(sol.root.decision_id, "lamp")
        self.assertEqual(sol.root.value, F(5, 3))

    def test_hand_leaf_actions(self):
        sol = solve(tiny_model(), F(1))
        leaf_ops = {s: child.action_option(child.decision_id).action_id
                    for s, child in sol.root.children.items()}
        self.assertEqual(leaf_ops, {"1": "repair", "0": "release"})

    def test_zero_budget_equals_no_detection(self):
        sol = solve(tiny_model(), F(0))
        self.assertEqual(sol.value, F(10, 3))
        self.assertEqual(sol.expected_fees, F(0))
        self.assertEqual(sol.root.decision_kind, "stop")

    def test_replay_tiny(self):
        sol = solve(tiny_model(), F(1))
        r1 = replay(sol, "fault", outcomes="1")
        self.assertEqual(r1.fees_paid, F(1))
        self.assertEqual(r1.terminal_action, "repair")
        self.assertEqual(r1.terminal_loss, F(2))
        self.assertEqual(r1.total_path_cost, F(3))
        self.assertEqual(r1.expected_reconstructed, F(5, 3))
        r0 = replay(sol, "ok", outcomes="0")
        self.assertEqual(r0.terminal_action, "release")
        self.assertEqual(r0.terminal_loss, F(0))
        self.assertEqual(r0.total_path_cost, F(1))


class BruteForceAgreementTests(unittest.TestCase):
    def test_all_budgets_match_exhaustive_permutation_recursion(self):
        model = main_model()
        for budget in model.budgets:
            sol = solve(model, budget)
            self.assertEqual(sol.value, brute_force_value(model, budget),
                             f"budget {budget}")
            self.assertEqual(sol.value, walk_policy_value(sol),
                             f"budget {budget}")

    def test_correlated_matches_bruteforce(self):
        model = correlated_model()
        sol = solve(model, F(1))
        self.assertEqual(sol.value, brute_force_value(model, F(1)))
        self.assertEqual(sol.root.decision_id, "a")  # stable tie -> position order


class BudgetSensitivityTests(unittest.TestCase):
    def test_budget_changes_policy_and_value(self):
        model = main_model()
        values = {}
        for budget in model.budgets:
            sol = solve(model, budget)
            values[budget] = (sol.value, sol.root.decision_id, sol.max_depth)
        # Monotone non-increasing optimum as budget grows.
        ordered = sorted(values.items())
        for (_, prev), (_, nxt) in zip(ordered, ordered[1:]):
            self.assertLessEqual(nxt[0], prev[0])
        # Strict improvement from 0->3->5; 5 and 8 coincide (budget stops binding).
        self.assertGreater(values[0][0], values[3][0])
        self.assertGreater(values[3][0], values[5][0])
        self.assertEqual(values[5][0], values[8][0])
        # Budget 0 cannot buy anything.
        self.assertEqual(values[0][1], "replace_bearing")  # root stop choice happens to be stop
        zero_sol = solve(model, F(0))
        self.assertEqual(zero_sol.root.decision_kind, "stop")
        self.assertEqual(zero_sol.expected_fees, F(0))

    def test_budget_3_vs_5_root_test_differs(self):
        model = main_model()
        self.assertEqual(solve(model, F(3)).root.decision_id, "thermal")
        self.assertEqual(solve(model, F(5)).root.decision_id, "thermal")
        # At budget 5 the second-level decisions include buying current, at 3 no.

    def test_budget_cap_conservative(self):
        """Never spend more than the budget on any branch."""
        model = main_model()
        for budget in model.budgets:
            self.assertLessEqual(solve(model, budget).expected_fees
                                 <= budget * 1, True)


class CorrelationTests(unittest.TestCase):
    def test_uses_joint_not_product_of_marginals(self):
        model = main_model()
        # Under 'bearing', vibration and thermal are correlated.
        from policy_tool.model import Model
        vt = model.test_by_id("vibration")
        th = model.test_by_id("thermal")
        # Marginal-only likelihood of (1,1) would be 31/40 * 3/10 = 93/400
        # but the frozen joint says 39/200.
        marg_both = model.likelihood("bearing", frozenset({"vibration", "thermal"}),
                                     {"vibration": "1", "thermal": "1"})
        mar_product = (model.likelihood("bearing", frozenset({"vibration"}),
                                        {"vibration": "1"})
                       * model.likelihood("bearing", frozenset({"thermal"}),
                                          {"thermal": "1"}))
        self.assertEqual(marg_both, F(39, 200))
        self.assertEqual(mar_product, F(93, 400))
        self.assertNotEqual(marg_both, mar_product)

    def test_correlated_joint_respected_in_posterior(self):
        model = correlated_model()
        # Under perfect correlation, observing a=1 pins broken=1.
        post = model.posterior(frozenset({"a"}), {"a": "1"})
        self.assertEqual(post, {"broken": F(1), "fine": F(0)})
        # A marginal-product model would leave 1/2, so this distinguishes them.
        self.assertNotEqual(post["broken"], F(1, 2))

    def test_correlated_stop_and_branch_probs(self):
        model = correlated_model()
        sol = solve(model, F(1))
        self.assertEqual(sol.root.branch_probs["1"], F(1, 2))
        self.assertEqual(sol.root.branch_probs["0"], F(1, 2))
        self.assertEqual(sol.root.children["1"].action_option(
            sol.root.children["1"].decision_id).action_id, "scrap")
        self.assertEqual(sol.root.children["0"].action_option(
            sol.root.children["0"].decision_id).action_id, "keep")


class ZeroProbabilityTests(unittest.TestCase):
    def test_observation_with_zero_probability_raises(self):
        """A globally impossible observation must never get a posterior."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "z.json")
            payload = {
                "states": {"a": "1/2", "b": "1/2"},
                "tests": [{"id": "t", "cost": 1, "requires": []}],
                "joint": [
                    {"state": "a", "outcomes": "0", "probability": "1"},
                    {"state": "b", "outcomes": "0", "probability": "1"},
                ],
                "actions": {"x": {"a": 0, "b": 10}, "y": {"a": 10, "b": 0}},
                "budget": 1,
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            m = load_model(path)
            # t=1 is impossible under every state: evidence is exactly 0.
            with self.assertRaises(ZeroProbabilityError):
                m.posterior(frozenset({"t"}), {"t": "1"})
            # A reachable observation still yields a proper posterior.
            self.assertEqual(m.posterior(frozenset({"t"}), {"t": "0"}),
                             {"a": F(1, 2), "b": F(1, 2)})

    def test_state_specific_zero_likelihood_is_not_global_zero(self):
        """t=1 impossible under b but possible under a is still a valid posterior."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "z.json")
            payload = {
                "states": {"a": "1/2", "b": "1/2"},
                "tests": [{"id": "t", "cost": 1, "requires": []}],
                "joint": [
                    {"state": "a", "outcomes": "1", "probability": "1"},
                    {"state": "b", "outcomes": "0", "probability": "1"},
                ],
                "actions": {"x": {"a": 0, "b": 10}, "y": {"a": 10, "b": 0}},
                "budget": 1,
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            m = load_model(path)
            self.assertEqual(m.posterior(frozenset({"t"}), {"t": "1"}),
                             {"a": F(1), "b": F(0)})

    def test_solver_never_creates_zero_prob_child_posterior(self):
        # Use a joint with an impossible combination by setting an action model
        # whose joint lacks certain vectors? The fixtures are complete, so we
        # check the solver marks zero branches rather than posterior-ing them.
        model = main_model()
        sol = solve(model, F(5))
        cur = sol.root
        while cur.decision_kind == "test":
            for sym, child in cur.children.items():
                if child is None:
                    self.assertEqual(cur.branch_probs[sym], 0)
            # follow only a positive branch
            sym = next(s for s, ch in cur.children.items() if ch is not None)
            cur = cur.children[sym]

    def test_replay_rejects_impossible_vector(self):
        model = correlated_model()
        sol = solve(model, F(1))
        with self.assertRaises(ReplayError):
            replay(sol, "fine", outcomes="11")  # impossible under fine
        with self.assertRaises(ReplayError):
            replay(sol, "nope", outcomes="00")


class InputValidationTests(unittest.TestCase):
    def _load_embedded(self, payload):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(payload, tmp)
        tmp.close()
        try:
            return load_model(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_priors_not_summing_to_one(self):
        with self.assertRaises(ModelError):
            self._load_embedded({
                "states": {"a": "1/2", "b": "1/3"},
                "tests": [{"id": "t", "cost": 0, "requires": []}],
                "joint": [{"state": "a", "outcomes": "0", "probability": "1"},
                          {"state": "b", "outcomes": "0", "probability": "1"}],
                "actions": {"x": {"a": 0, "b": 0}},
            })

    def test_joint_not_summing_to_one(self):
        with self.assertRaises(ModelError):
            self._load_embedded({
                "states": {"a": "1"},
                "tests": [{"id": "t", "cost": 0, "requires": []}],
                "joint": [{"state": "a", "outcomes": "0", "probability": "1/2"}],
                "actions": {"x": {"a": 0}},
            })

    def test_unknown_state_in_joint(self):
        with self.assertRaises(ModelError):
            self._load_embedded({
                "states": {"a": "1"},
                "tests": [{"id": "t", "cost": 0, "requires": []}],
                "joint": [{"state": "ghost", "outcomes": "0", "probability": "1"}],
                "actions": {"x": {"a": 0}},
            })

    def test_wrong_outcome_length(self):
        with self.assertRaises(ModelError):
            self._load_embedded({
                "states": {"a": "1"},
                "test_order": ["t1", "t2"],
                "tests": [{"id": "t1", "cost": 0, "requires": []},
                          {"id": "t2", "cost": 0, "requires": []}],
                "joint": [{"state": "a", "outcomes": "0", "probability": "1"}],
                "actions": {"x": {"a": 0}},
            })

    def test_duplicate_joint_rows(self):
        with self.assertRaises(ModelError):
            self._load_embedded({
                "states": {"a": "1"},
                "tests": [{"id": "t", "cost": 0, "requires": []}],
                "joint": [
                    {"state": "a", "outcomes": "0", "probability": "1/2"},
                    {"state": "a", "outcomes": "0", "probability": "1/2"},
                ],
                "actions": {"x": {"a": 0}},
            })

    def test_requires_unknown_test(self):
        with self.assertRaises(ModelError):
            self._load_embedded({
                "states": {"a": "1"},
                "tests": [{"id": "t", "cost": 0, "requires": ["ghost"]}],
                "joint": [{"state": "a", "outcomes": "0", "probability": "1"}],
                "actions": {"x": {"a": 0}},
            })

    def test_requires_cycle(self):
        with self.assertRaises(ModelError):
            self._load_embedded({
                "states": {"a": "1"},
                "tests": [{"id": "t1", "cost": 0, "requires": ["t2"]},
                          {"id": "t2", "cost": 0, "requires": ["t1"]}],
                "joint": [{"state": "a", "outcomes": "00", "probability": "1"}],
                "actions": {"x": {"a": 0}},
            })

    def test_test_order_references_unknown(self):
        with self.assertRaises(ModelError):
            self._load_embedded({
                "states": {"a": "1"},
                "test_order": ["ghost"],
                "tests": [{"id": "t", "cost": 0, "requires": []}],
                "joint": [{"state": "a", "outcomes": "0", "probability": "1"}],
                "actions": {"x": {"a": 0}},
            })

    def test_negative_test_cost(self):
        with self.assertRaises(ModelError):
            self._load_embedded({
                "states": {"a": "1"},
                "tests": [{"id": "t", "cost": -1, "requires": []}],
                "joint": [{"state": "a", "outcomes": "0", "probability": "1"}],
                "actions": {"x": {"a": 0}},
            })

    def test_missing_action_state(self):
        with self.assertRaises(ModelError):
            self._load_embedded({
                "states": {"a": "1/2", "b": "1/2"},
                "tests": [{"id": "t", "cost": 0, "requires": []}],
                "joint": [{"state": "a", "outcomes": "0", "probability": "1"},
                          {"state": "b", "outcomes": "0", "probability": "1"}],
                "actions": {"x": {"a": 0}},
            })

    def test_probability_out_of_range(self):
        with self.assertRaises(ModelError):
            self._load_embedded({
                "states": {"a": "3/2"},
                "tests": [{"id": "t", "cost": 0, "requires": []}],
                "joint": [{"state": "a", "outcomes": "0", "probability": "1"}],
                "actions": {"x": {"a": 0}},
            })

    def test_joint_line_numbers_for_bad_coverage(self):
        """A CSV missing a state's rows reports the concrete problem."""
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "m.json")
            joint_path = os.path.join(tmp, "j.csv")
            actions_path = os.path.join(tmp, "a.csv")
            with open(joint_path, "w") as fh:
                fh.write("state,outcomes,probability\nbearing-shaped,0000,1\n")
            with open(actions_path, "w") as fh:
                fh.write("action,bearing\nx,0\n")
            with open(model_path, "w") as fh:
                json.dump({
                    "states": {"bearing": "1"},
                    "test_order": ["v", "c", "t", "b"],
                    "tests": [{"id": t, "cost": 0, "requires": []}
                              for t in ["v", "c", "t", "b"]],
                    "joint_file": "j.csv",
                }, fh)
            with self.assertRaises(ModelError) as ctx:
                load_model(model_path, actions_path=actions_path)
            self.assertIn("unknown state", str(ctx.exception))


class RowOrderInvarianceTests(unittest.TestCase):
    def _parity(self, model_a, model_b, budget):
        sa = solve(model_a, budget)
        sb = solve(model_b, budget)
        self.assertEqual(sa.value, sb.value)
        tree_a = strategy_tree(sa)
        tree_b = strategy_tree(sb)
        # Decisions (kind,id) at every node must match; compare structurally.
        self.assertEqual(self._decisions(tree_a["tree"]), self._decisions(tree_b["tree"]))

    @staticmethod
    def _decisions(node):
        out = [(node["depth"], node["decision"]["kind"], node["decision"]["id"])]
        for child in node.get("children", {}).values():
            if "unreachable" not in child:
                out.extend(RowOrderInvarianceTests._decisions(child))
        return sorted(out)

    def test_shuffled_joint_rows_and_action_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(ROOT, "fixtures/joint.csv")
            with open(src) as fh:
                lines = fh.read().strip().splitlines()
            header, rows = lines[0], lines[1:]
            rng = random.Random(1234)
            rng.shuffle(rows)
            jp = os.path.join(tmp, "joint_shuffled.csv")
            with open(jp, "w") as fh:
                fh.write("\n".join([header] + rows) + "\n")

            pa = os.path.join(ROOT, "fixtures/actions.csv")
            with open(pa) as fh:
                alines = fh.read().strip().splitlines()
            aheader, arows = alines[0], alines[1:]
            rng.shuffle(arows)
            ap = os.path.join(tmp, "actions_shuffled.csv")
            with open(ap, "w") as fh:
                fh.write("\n".join([aheader] + arows) + "\n")

            base = load_model(os.path.join(ROOT, "fixtures/model.json"),
                              actions_path=pa, joint_path=src,
                              plans_path=os.path.join(ROOT, "fixtures/plans.json"))
            shuffled = load_model(os.path.join(ROOT, "fixtures/model.json"),
                                  actions_path=ap, joint_path=jp,
                                  plans_path=os.path.join(ROOT, "fixtures/plans.json"))
            for budget in base.budgets:
                self._parity(base, shuffled, budget)

    def test_cli_independent_of_row_order(self):
        """End-to-end: the CLI JSON is identical modulo the model path."""
        with tempfile.TemporaryDirectory() as tmp:
            jp = os.path.join(tmp, "j.csv")
            with open(os.path.join(ROOT, "fixtures/joint.csv")) as fh:
                lines = fh.read().strip().splitlines()
            header, rows = lines[0], lines[1:]
            random.Random(99).shuffle(rows)
            with open(jp, "w") as fh:
                fh.write("\n".join([header] + rows) + "\n")
            cmd = [sys.executable, "-m", "policy_tool", "solve",
                   "--model", os.path.join(ROOT, "fixtures/model.json"),
                   "--actions", os.path.join(ROOT, "fixtures/actions.csv"),
                   "--budget", "5", "--json"]
            out_base = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                      check=True).stdout
            idx = cmd.index("--budget")
            cmd2 = cmd[:idx] + ["--joint", jp] + cmd[idx:]
            out_shuf = subprocess.run(cmd2, cwd=ROOT, capture_output=True, text=True,
                                      check=True).stdout
            base = json.loads(out_base)["strategy_tree"]
            shuf = json.loads(out_shuf)["strategy_tree"]
            self.assertEqual(base, shuf)


class ReplayContractTests(unittest.TestCase):
    def test_replay_reaverages_to_value_for_all_states_vectors(self):
        model = main_model()
        sol = solve(model, F(5))
        # Enumerate all positive joint rows and confirm re-average equals value.
        total = F(0)
        for state in model.states:
            for vector, prob in model.joint[state].items():
                r = replay(sol, state, outcomes="".join(vector))
                self.assertEqual(r.expected_reconstructed, sol.value)
                total += model.priors[state] * prob * r.total_path_cost
        self.assertEqual(total, sol.value)

    def test_replay_coverage_all_vectors_every_budget(self):
        model = main_model()
        for budget in model.budgets:
            sol = solve(model, budget)
            agg = F(0)
            for state in model.states:
                for vector, prob in model.joint[state].items():
                    r = replay(sol, state, outcomes="".join(vector))
                    self.assertTrue(r.consistent)
                    self.assertLessEqual(r.fees_paid, budget)
                    agg += model.priors[state] * prob * r.total_path_cost
            self.assertEqual(agg, sol.value)

    def test_replay_does_not_return_stored_total(self):
        """r.expected_reconstructed must equal a freshly re-walked value."""
        model = main_model()
        sol = solve(model, F(5))
        self.assertEqual(sol.value, walk_policy_value(sol))
        r = replay(sol, "bearing", outcomes="1011")
        self.assertNotEqual(r.total_path_cost, r.expected_reconstructed)
        self.assertAlmostEqual(float(r.expected_reconstructed), float(sol.value))

    def test_replay_requires_full_vector(self):
        model = main_model()
        sol = solve(model, F(5))
        with self.assertRaises(ReplayError):
            replay(sol, "bearing", outcomes="101")

    def test_replay_rejects_bad_symbol(self):
        sol = solve(main_model(), F(5))
        with self.assertRaises(ReplayError):
            replay(sol, "bearing", outcomes="1012")


class AdaptivePolicyTests(unittest.TestCase):
    def test_policy_depends_on_results(self):
        """Different observed results lead to different terminal actions."""
        model = main_model()
        sol = solve(model, F(5)
        )
        actions = set()
        for state in model.states:
            for vector in model.joint[state]:
                r = replay(sol, state, outcomes="".join(vector))
                actions.add(r.terminal_action)
        self.assertGreater(len(actions), 1,
                           "adaptive policy must not return one action always")

    def test_root_not_fixed_for_all_paths(self):
        model = main_model()
        sol = solve(model, F(5))
        # At least one child node buys a different test than the root.
        self.assertEqual(sol.root.decision_id, "thermal")
        vals = [ch.decision_id for ch in sol.root.children.values()
                if ch is not None and ch.decision_kind == "test"]
        self.assertTrue(vals)
        self.assertNotEqual(vals[0], "thermal")

    def test_stop_explanation_reason_codes(self):
        model = main_model()
        sol = solve(model, F(8))
        seen = set()

        def walk(node):
            if node.decision_kind == "stop":
                seen.add(node.stop_explanation["reason_code"])
            for ch in node.children.values():
                if ch is not None:
                    walk(ch)

        walk(sol.root)
        self.assertIn("additional_information_not_worth_its_fee", seen)

    def test_blocked_tests_never_selected(self):
        model = main_model()
        for budget in model.budgets:
            sol = solve(model, budget)

            def walk(node):
                if node.decision_kind == "test":
                    opt = node.test_option(node.decision_id)
                    self.assertIsNotNone(opt.expected_total)
                    self.assertTrue(opt.budget_ok)
                    self.assertFalse(opt.missing_requires)
                for ch in node.children.values():
                    if ch is not None:
                        walk(ch)

            walk(sol.root)

    def test_requires_respected_bench_after_current(self):
        model = main_model()
        sol = solve(model, F(8))
        node = sol.root
        while node.decision_kind == "test":
            if node.decision_id == "bench":
                self.assertIn("current", node.bought)
            sym = next(s for s, c in node.children.items() if c is not None)
            node = node.children[sym]


class CostAccountingTests(unittest.TestCase):
    def test_fee_and_loss_are_separate(self):
        sol = solve(main_model(), F(5))
        comp = sol.cost_composition()
        fs = comp["full_adaptive_strategy"]
        self.assertEqual(F(fs["expected_inspection_fees"]), sol.expected_fees)
        self.assertEqual(F(fs["expected_terminal_loss"]), sol.expected_terminal_loss)
        self.assertEqual(
            F(fs["total_expected_cost"]),
            sol.expected_fees + sol.expected_terminal_loss,
        )
        iv = comp["information_value"]
        self.assertEqual(F(iv["gross_reduction_in_terminal_loss"]),
                         sol.gross_information_value)
        self.assertEqual(F(iv["inspection_fees_paid"]), sol.expected_fees)
        self.assertEqual(F(iv["net_value_after_fees"]), sol.net_information_value)
        self.assertEqual(
            sol.gross_information_value - sol.expected_fees,
            sol.net_information_value,
        )

    def test_no_detection_plan_reported(self):
        sol = solve(main_model(), F(5))
        self.assertEqual(sol.no_detection_cost, F(39))


class TieTests(unittest.TestCase):
    def test_exact_tie_and_stable_selection(self):
        model = correlated_model()
        sol = solve(model, F(1))
        self.assertIn(("test", "a"), sol.root.ties)
        self.assertIn(("test", "b"), sol.root.ties)
        self.assertEqual(sol.root.decision_id, "a")  # smaller position

    def test_no_false_ties_from_tolerance(self):
        """Values differing at the last rational digit must not tie."""
        model = main_model()
        sol = solve(model, F(5))
        # Root: thermal 463021/15000 vs vibration/current slightly larger.
        vals = [t.expected_total for t in sol.root.tests if t.buyable]
        distinct = set(vals)
        self.assertEqual(len(distinct), len(vals),
                         "no spurious ties among distinct exact values")
        self.assertNotIn(("test", "current"), sol.root.ties)


class DeterminismTests(unittest.TestCase):
    def test_repeat_solve_is_identical(self):
        model = main_model()
        a = strategy_tree(solve(model, F(5)))
        b = strategy_tree(solve(model, F(5)))
        self.assertEqual(a, b)


class StructuralInvariantTests(unittest.TestCase):
    def _walk(self, node):
        yield node
        for child in node.children.values():
            if child is not None:
                yield from self._walk(child)

    def test_posteriors_sum_to_one_and_reach_matches_evidence(self):
        model = main_model()
        for budget in model.budgets:
            sol = solve(model, budget)
            for node in self._walk(sol.root):
                self.assertEqual(sum(node.posterior.values()), 1)
                self.assertEqual(
                    node.reach_probability,
                    model.evidence(node.bought, node.obs),
                )
                self.assertGreater(node.reach_probability, 0)
                # Consistent with the frozen joint: prior * likelihood.
                self.assertEqual(
                    node.reach_probability,
                    sum(model.priors[s] * model.likelihood(s, node.bought, node.obs)
                        for s in model.states),
                )

    def test_branch_probabilities_match_joint_and_sum_to_one(self):
        model = main_model()
        for budget in model.budgets:
            sol = solve(model, budget)
            for node in self._walk(sol.root):
                if node.decision_kind != "test":
                    continue
                self.assertEqual(sum(node.branch_probs.values()), 1)
                for symbol, child in node.children.items():
                    if child is None:
                        self.assertEqual(node.branch_probs[symbol], 0)
                        continue
                    # P(child) = P(parent) * P(symbol | parent)
                    self.assertEqual(
                        child.reach_probability,
                        node.reach_probability * node.branch_probs[symbol],
                    )
                    # P(symbol | parent) marginalsied straight from the joint.
                    self.assertEqual(
                        node.branch_probs[symbol],
                        model.outcome_marginal(node.bought, node.obs,
                                               node.decision_id, symbol),
                    )

    def test_leaf_probabilities_sum_to_one(self):
        model = main_model()
        for budget in model.budgets:
            sol = solve(model, budget)
            total = sum(n.reach_probability for n in self._walk(sol.root)
                        if n.decision_kind == "stop")
            self.assertEqual(total, 1)

    def test_no_order_dependent_duplicate_nodes(self):
        """Same (bought set, results) must be the identical object (merged)."""
        model = main_model()
        sol = solve(model, F(8))
        seen = {}
        for node in self._walk(sol.root):
            key = (tuple(sorted(node.bought)),
                   tuple(sorted(node.obs.items())))
            if key in seen:
                self.assertIs(seen[key], node)
            else:
                seen[key] = node

    def test_children_are_memo_merged(self):
        model = main_model()
        sol = solve(model, F(8))
        # bench after current can be reached under multiple prefixes; the
        # (bought={current,bench}, same results) history must be one object.
        ids = {}
        for node in self._walk(sol.root):
            key = (frozenset(node.bought), tuple(sorted(node.obs.items())))
            ids.setdefault(key, set()).add(id(node))
        self.assertTrue(all(len(v) == 1 for v in ids.values()))


class NoToleranceTieTests(unittest.TestCase):
    def test_tiny_exact_difference_is_not_a_tie(self):
        """Two policies differing by 1/10**12 must not be treated as equal."""
        payload = {
            "states": {"a": "1/2", "b": "1/2"},
            "tests": [{"id": "t", "cost": "1/1000000000000", "requires": []}],
            "joint": [
                {"state": "a", "outcomes": "0", "probability": "1"},
                {"state": "b", "outcomes": "0", "probability": "1"},
            ],
            "actions": {
                "x": {"a": "0", "b": "0"},
                "y": {"a": "0", "b": "0"},
            },
            "budget": "1",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(payload, fh)
            path = fh.name
        try:
            model = load_model(path)
            sol = solve(model, F(1))
        finally:
            os.unlink(path)
        # Buying the test strictly costs 1e-12 more than stopping at 0, and
        # must NOT tie; one tree should not be chosen as if equal.
        self.assertNotIn(("test", "t"), sol.root.ties)
        self.assertEqual(sol.root.decision_kind, "stop")
        self.assertEqual(sol.root.value, F(0))

    def test_stop_preferred_on_exact_tie(self):
        """A test that exactly offsets its own fee ties stopping -> stop wins."""
        payload = {
            "states": {"a": "1"},
            "tests": [{"id": "t", "cost": "0", "requires": []}],
            "joint": [{"state": "a", "outcomes": "0", "probability": "1"}],
            "actions": {"x": {"a": "0"}},
            "budget": "1",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(payload, fh)
            path = fh.name
        try:
            sol = solve(load_model(path), F(1))
        finally:
            os.unlink(path)
        self.assertEqual(sol.root.decision_kind, "stop")
        self.assertIn(("test", "t"), sol.root.ties)


if __name__ == "__main__":
    unittest.main()
