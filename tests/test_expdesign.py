"""pytest 测试套件：覆盖约束、可估计性、三类失败区分、枚举核对与编码不变性。

运行：``PYTHONPATH=src .venv/bin/python -m pytest tests/ -q``
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from expdesign.constraints import (  # noqa: E402
    block_replicate_satisfied, load_locked_runs,
)
from expdesign.criteria import (  # noqa: E402
    block_projector, conditional_information, target_indices,
)
from expdesign.design import (  # noqa: E402
    diagnostics_payload, load_bundle, randomized_order, run_design,
    run_table_rows, rebuild_and_verify,
)
from expdesign.enumerate import count_multisets, enumerate_optimal  # noqa: E402
from expdesign.errors import (  # noqa: E402
    ConfigError, InfeasibleError, SearchFailure,
)
from expdesign.factors import FactorSet, build_candidate_set  # noqa: E402
from expdesign.invariance import check_coding_invariance  # noqa: E402
from expdesign.model import INTERCEPT_NAME, Model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
FX = os.path.join(ROOT, "fixtures")


def _paths(plan="plan.json", factors=os.path.join(FX, "factors.json"),
           model=os.path.join(FX, "model.json"),
           constraints=os.path.join(FX, "constraints.json")):
    return {
        "factors_path": factors,
        "model_path": model,
        "constraints_path": constraints,
        "plan_path": os.path.join(FX, plan),
    }


# --------------------------------------------------------------------------- #
# 1. 因素与候选集
# --------------------------------------------------------------------------- #
class TestFactors:
    def setup_method(self):
        self.fs = FactorSet.from_json_file(os.path.join(FX, "factors.json"))

    def test_numeric_coding(self):
        f = self.fs.get("temperature")
        assert f.coded_value(20) == -1.0
        assert f.coded_value(40) == 0.0
        assert f.coded_value(60) == 1.0

    def test_categorical_reference_and_dummies(self):
        f = self.fs.get("catalyst")
        assert f.reference == "A"
        assert f.dummy_columns() == [("catalyst[B]", "B")]

    def test_cartesian_product_size(self):
        assert len(self.fs.all_combinations()) == 3 * 3 * 2

    def test_forbidden_removed_never_reintroduced(self):
        rules = [{"temperature": 60, "catalyst": "B"}]
        cs = build_candidate_set(self.fs, rules)
        # 18 - 被禁的 3 个 concentration 水平 = 15
        assert cs.n_candidates == 15
        assert len(cs.forbidden) == 3
        for combo in cs.combos:
            assert not (combo[self.fs.index_of("temperature")] == 60
                        and combo[self.fs.index_of("catalyst")] == "B")

    def test_unknown_factor_in_forbidden_rejected(self):
        with pytest.raises(ConfigError):
            build_candidate_set(self.fs, [{"nope": 1}])

    def test_duplicate_levels_rejected(self):
        with pytest.raises(ConfigError):
            FactorSet.from_dict({"factors": [
                {"name": "a", "type": "numeric", "levels": [1, 1],
                 "center": 1, "scale": 1}]})

    def test_zero_scale_rejected(self):
        with pytest.raises(ConfigError):
            FactorSet.from_dict({"factors": [
                {"name": "a", "type": "numeric", "levels": [1, 2],
                 "center": 1, "scale": 0}]})

    def test_center_may_lie_between_levels(self):
        # 两水平因素的中心可以是中点（aliased 案例的情形）。
        fs = FactorSet.from_json_file(
            os.path.join(FX, "cases", "aliased-factors.json"))
        assert fs.get("temperature").center == 40


# --------------------------------------------------------------------------- #
# 2. 模型 schema 与列身份
# --------------------------------------------------------------------------- #
class TestModel:
    def setup_method(self):
        self.fs = FactorSet.from_json_file(os.path.join(FX, "factors.json"))

    def test_declared_column_identity(self):
        model = Model.from_json_file(self.fs, os.path.join(FX, "model.json"))
        assert model.treatment_names == [
            "Intercept", "temperature", "concentration", "catalyst[B]",
            "temperature^2", "temperature*concentration",
        ]

    def test_unknown_term_kind_rejected(self):
        payload = {"terms": [
            {"kind": "intercept"},
            {"factor": "temperature", "cubic": True}]}
        with pytest.raises(ConfigError):
            Model.from_dict(self.fs, payload)

    def test_unknown_factor_rejected(self):
        with pytest.raises(ConfigError):
            Model.from_dict(self.fs, {"terms": [
                {"kind": "intercept"}, {"factor": "ghost"}]})

    def test_missing_intercept_rejected(self):
        with pytest.raises(ConfigError):
            Model.from_dict(self.fs, {"terms": [{"factor": "temperature"}]})

    def test_categorical_power_rejected(self):
        with pytest.raises(ConfigError):
            Model.from_dict(self.fs, {"terms": [
                {"kind": "intercept"}, {"factor": "catalyst", "power": 2}]})

    def test_interaction_wrong_arity_rejected(self):
        with pytest.raises(ConfigError):
            Model.from_dict(self.fs, {"terms": [
                {"kind": "intercept"},
                {"interaction": ["temperature"]}]})

    def test_duplicate_intercept_rejected(self):
        with pytest.raises(ConfigError):
            Model.from_dict(self.fs, {"terms": [
                {"kind": "intercept"}, {"kind": "intercept"}]})

    def test_design_matrix_values(self):
        model = Model.from_json_file(self.fs, os.path.join(FX, "model.json"))
        row = model.treatment_row((20, 0.2, "B"))
        # t=-1, c=-1, cat=1, t^2=1, t*c=1
        assert np.allclose(row, [1, -1, -1, 1, 1, 1])


# --------------------------------------------------------------------------- #
# 3. 锁定运行与重复
# --------------------------------------------------------------------------- #
class TestConstraints:
    def test_locked_forbidden_combo_rejected(self, tmp_path):
        fs = FactorSet.from_json_file(os.path.join(FX, "factors.json"))
        cs = build_candidate_set(fs, [{"temperature": 60, "catalyst": "B"}])
        bad = tmp_path / "bad.csv"
        bad.write_text(
            "run_id,block,temperature,concentration,catalyst\n"
            "X,day1,60,0.2,B\n")
        with pytest.raises(ConfigError, match="禁用"):
            load_locked_runs(str(bad), fs, cs, block_ids={"day1"})

    def test_locked_unknown_block_rejected(self, tmp_path):
        fs = FactorSet.from_json_file(os.path.join(FX, "factors.json"))
        cs = build_candidate_set(fs, [])
        bad = tmp_path / "bad.csv"
        bad.write_text(
            "run_id,block,temperature,concentration,catalyst\n"
            "X,ghost,20,0.2,A\n")
        with pytest.raises(ConfigError, match="未声明的块"):
            load_locked_runs(str(bad), fs, cs, block_ids={"day1"})

    def test_replicate_predicate(self):
        assert block_replicate_satisfied([0, 0, 1])
        assert not block_replicate_satisfied([0, 1, 2])
        assert not block_replicate_satisfied([0])


# --------------------------------------------------------------------------- #
# 4. 端到端主案例
# --------------------------------------------------------------------------- #
class TestMainDesign:
    def setup_method(self):
        self.bundle = run_design(load_bundle(**_paths()),
                                 n_random_starts=4, max_evaluations=8000)

    def test_full_pipeline_verification(self):
        v = rebuild_and_verify(self.bundle)
        assert v["all_passed"]
        assert v["block_quotas_ok"]
        assert v["all_candidates_legal"]
        assert v["treatment_estimable"]
        assert v["full_matrix_full_rank"]
        assert v["matrix_matches_search_object"]

    def test_quotas_and_replicates(self):
        design = self.bundle.result.design
        for block in self.bundle.plan.blocks:
            bd = design[block.block_id]
            assert len(bd.candidate_ids) == block.runs
            assert block_replicate_satisfied(bd.candidate_ids)

    def test_locked_runs_immutable(self):
        rows = run_table_rows(self.bundle)
        l1 = next(r for r in rows if r["run_id"] == "L1")
        l2 = next(r for r in rows if r["run_id"] == "L2")
        assert (l1["block"], l1["temperature"], l1["concentration"],
                l1["catalyst"]) == ("day1", 20, 0.2, "A")
        assert (l2["block"], l2["temperature"], l2["concentration"],
                l2["catalyst"]) == ("day2", 20, 0.8, "B")
        # 锁定位置不参与交换
        for bid in ("day1", "day2"):
            assert self.bundle.result.design[bid].locked_positions == [0]

    def test_forbidden_never_used(self):
        for rows in (run_table_rows(self.bundle), randomized_order(self.bundle)):
            for r in rows:
                assert not (r["temperature"] == 60 and r["catalyst"] == "B")

    def test_random_order_preserves_multiset_and_estimability(self):
        canonical = run_table_rows(self.bundle)
        randomised = randomized_order(self.bundle)
        # 块内组合多重集不变（只是位置置换）
        for bid in self.bundle.plan.block_ids:
            a = sorted((r["temperature"], r["concentration"], r["catalyst"])
                       for r in canonical if r["block"] == bid)
            b = sorted((r["temperature"], r["concentration"], r["catalyst"])
                       for r in randomised if r["block"] == bid)
            assert a == b
        # 执行顺序是块内 1..k 的一个置换
        for bid in self.bundle.plan.block_ids:
            orders = sorted(r["execution_order_in_block"]
                            for r in randomised if r["block"] == bid)
            assert orders == list(range(1, len(orders) + 1))
        # 可估计性结论不变
        assert rebuild_and_verify(self.bundle)["all_passed"]

    def test_repetitions_consume_slots(self):
        v = rebuild_and_verify(self.bundle)
        # 每块 6 个名额、5 个不同组合 => 1 个重复名额
        assert v["repeat_slots"] == {"day1": 1, "day2": 1}
        assert v["distinct_combinations"] == {"day1": 5, "day2": 5}

    def test_baseline_feasible_and_search_not_worse(self):
        base = self.bundle.baseline_quality
        best = self.bundle.result.quality
        assert base.feasible_rank
        assert best.feasible_rank
        assert best.log_det >= base.log_det - 1e-9

    def test_coding_invariance(self):
        combos = [
            self.bundle.candidates.combos[cid]
            for bid in self.bundle.plan.block_ids
            for cid in self.bundle.result.design[bid].candidate_ids
        ]
        blocks = [
            bid for bid in self.bundle.plan.block_ids
            for _ in self.bundle.result.design[bid].candidate_ids
        ]
        inv = check_coding_invariance(
            self.bundle.factor_set, self.bundle.model, self.bundle.candidates,
            combos, blocks, self.bundle.plan.block_ids,
        )
        assert inv.estimability_consistent
        assert inv.same_column_space
        assert inv.max_subspace_distance < 1e-10
        assert inv.declared_treatment_rank == inv.alternative_treatment_rank

    def test_search_report_auditable(self):
        rep = self.bundle.result.report
        assert rep.n_objective_evaluations > 0
        assert rep.n_distinct_designs_checked > 0
        assert rep.stop_reason in {
            "all_starts_exhausted", "max_evaluations_reached",
            "time_budget_reached"}
        assert len(rep.start_points) >= 4
        assert rep.found_feasible

    def test_diagnostics_contains_required_sections(self):
        d = diagnostics_payload(self.bundle)
        for key in ("criterion", "selected_design", "baseline_round_robin",
                    "improvement_over_baseline", "search_report",
                    "structural_evidence", "enumeration",
                    "independent_verification", "coding_identity"):
            assert key in d

    def test_seed_reproducible(self):
        b1 = run_design(load_bundle(**_paths()), n_random_starts=4,
                        max_evaluations=4000)
        b2 = run_design(load_bundle(**_paths()), n_random_starts=4,
                        max_evaluations=4000)
        ids1 = {bid: tuple(d.candidate_ids)
                for bid, d in b1.result.design.items()}
        ids2 = {bid: tuple(d.candidate_ids)
                for bid, d in b2.result.design.items()}
        assert ids1 == ids2


# --------------------------------------------------------------------------- #
# 5. 微型候选集枚举核对
# --------------------------------------------------------------------------- #
MICRO = os.path.join(FX, "micro")


class TestEnumeration:
    def setup_method(self):
        self.bundle = run_design(load_bundle(
            factors_path=os.path.join(MICRO, "factors.json"),
            model_path=os.path.join(MICRO, "model.json"),
            constraints_path=os.path.join(MICRO, "constraints.json"),
            plan_path=os.path.join(MICRO, "plan.json"),
        ), n_random_starts=4, max_evaluations=4000)

    def test_exhaustive_enumeration_matches_search(self):
        enum = self.bundle.enumeration
        assert enum is not None and enum.exhaustive
        assert enum.n_designs_evaluated > 0
        assert abs(enum.best_objective
                   - self.bundle.result.quality.log_det) <= 1e-8

    def test_enumeration_respects_replicates_and_locked(self):
        # 枚举与搜索的最优都必须通过独立重建。
        assert rebuild_and_verify(self.bundle)["all_passed"]

    def test_multiset_count_formula(self):
        assert count_multisets(4, 3) == 20  # C(6,3)
        assert count_multisets(4, 0) == 1

    def test_enumeration_cap_returns_none_for_large(self):
        out = enumerate_optimal(
            self.bundle.plan, self.bundle.model, self.bundle.candidates,
            self.bundle.constraints, self.bundle.locked,
            max_multisets_per_block=1, max_total_designs=1,
        )
        assert out is None  # 超阈值必须明确放弃，不得抽样冒充


# --------------------------------------------------------------------------- #
# 6. 三类失败严格区分
# --------------------------------------------------------------------------- #
class TestFailureModes:
    def test_budget_insufficient_distinct_from_search(self):
        with pytest.raises(InfeasibleError) as exc:
            load_bundle(**_paths(plan=os.path.join("cases", "too-small-plan.json")))
        assert exc.value.reason == InfeasibleError.REASON_BUDGET
        assert exc.value.evidence["total_runs"] == 4
        assert not isinstance(exc.value, SearchFailure)

    def test_structural_alias_with_null_space_evidence(self):
        with pytest.raises(InfeasibleError) as exc:
            load_bundle(
                factors_path=os.path.join(FX, "cases", "aliased-factors.json"),
                model_path=os.path.join(FX, "model.json"),
                constraints_path=os.path.join(FX, "constraints.json"),
                plan_path=os.path.join(FX, "plan.json"),
            )
        assert exc.value.reason == InfeasibleError.REASON_STRUCTURAL_ALIAS
        nulls = exc.value.evidence["null_space"]
        assert nulls and nulls[0]["dominant_columns"]
        cols = {item["column"] for item in nulls[0]["dominant_columns"]}
        assert {"Intercept", "temperature^2"} <= cols

    def test_config_error_unknown_model_term(self, tmp_path):
        bad = tmp_path / "model.json"
        bad.write_text(json.dumps({"terms": [
            {"kind": "intercept"}, {"spline": "x"}]}))
        with pytest.raises(ConfigError):
            load_bundle(
                factors_path=os.path.join(FX, "factors.json"),
                model_path=str(bad),
                constraints_path=os.path.join(FX, "constraints.json"),
                plan_path=os.path.join(FX, "plan.json"),
            )

    def test_search_failure_is_not_infeasibility(self, monkeypatch):
        # 用极小评估预算迫使搜索找不到满秩方案；结构检查已通过，
        # 因此必须报 SearchFailure 而不是 InfeasibleError。
        bundle = load_bundle(**_paths())
        with pytest.raises(SearchFailure):
            run_design(bundle, n_random_starts=0, max_evaluations=1,
                       time_budget_seconds=0.001)


# --------------------------------------------------------------------------- #
# 7. 准则数学性质
# --------------------------------------------------------------------------- #
class TestCriteriaMath:
    def test_block_projector_centers_within_blocks(self):
        z = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
        q = block_projector(z)
        x = np.array([[1.0], [3.0], [10.0], [20.0]])
        centered = q @ x
        # 块内均值被扣除：第一块均值 2，第二块均值 15
        assert np.allclose(centered.ravel(), [-1, 1, -5, 5])
        # Q 对称幂等
        assert np.allclose(q, q.T)
        assert np.allclose(q @ q, q)

    def test_target_indices_exclude_intercept_with_blocks(self):
        names = [INTERCEPT_NAME, "a", "b"]
        assert target_indices(names, True) == [1, 2]
        assert target_indices(names, False) == [0, 1, 2]

    def test_conditional_information_symmetric(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(8, 3))
        q = np.eye(8)
        m = conditional_information(x, q)
        assert np.allclose(m, m.T)

    def test_projector_identity_without_block_columns(self):
        z_empty = np.zeros((5, 0))
        q = block_projector(z_empty)
        assert np.allclose(q, np.eye(5))


# --------------------------------------------------------------------------- #
# 8. 无块效应模型（Q=I，截距正常可估计）
# --------------------------------------------------------------------------- #
class TestNoBlockModel:
    NB = os.path.join(FX, "noblock")

    def setup_method(self):
        self.bundle = run_design(load_bundle(
            factors_path=os.path.join(self.NB, "factors.json"),
            model_path=os.path.join(self.NB, "model.json"),
            constraints_path=os.path.join(self.NB, "constraints.json"),
            plan_path=os.path.join(self.NB, "plan.json"),
        ), n_random_starts=3, max_evaluations=3000)

    def test_intercept_estimable_and_full_rank(self):
        v = rebuild_and_verify(self.bundle)
        assert v["all_passed"]
        # 无块效应：目标列含截距，共 4 列，Q=I 不消除任何列。
        assert v["n_target_columns"] == 4
        assert v["intercept_block_alias_expected"] is False
        assert v["rebuilt_target_rank_adjusted"] == 4

    def test_replicate_still_present(self):
        bd = self.bundle.result.design["only"]
        assert block_replicate_satisfied(bd.candidate_ids)
        assert len(bd.candidate_ids) == 6


# --------------------------------------------------------------------------- #
# 9. 重复约束导致的真实不可行（与搜索失败区分）
# --------------------------------------------------------------------------- #
class TestReplicateInfeasible:
    def test_distinct_capacity_below_columns(self, tmp_path):
        # 2 个二水平数值因素只有 4 个候选；3 个名额且强制重复 => 最多 2 个
        # 不同组合，而含截距+两主效应+交互的模型需要 4 列，可证明不可行。
        factors = {"factors": [
            {"name": "x1", "type": "numeric", "levels": [-1, 1],
             "center": 0, "scale": 1},
            {"name": "x2", "type": "numeric", "levels": [-1, 1],
             "center": 0, "scale": 1}]}
        model = {"terms": [
            {"kind": "intercept"}, {"factor": "x1"}, {"factor": "x2"},
            {"interaction": ["x1", "x2"]}],
            "include_block_effect": False}
        constraints = {
            "forbidden_all": [],
            "replicates": "at least one identical factor combination "
                          "repeated within each block; repetitions consume run slots"}
        plan = {"blocks": [{"id": "solo", "runs": 3}], "seed": 11}
        fp = tmp_path / "f.json"; fp.write_text(json.dumps(factors))
        mp = tmp_path / "m.json"; mp.write_text(json.dumps(model))
        cp = tmp_path / "c.json"; cp.write_text(json.dumps(constraints))
        pp = tmp_path / "p.json"; pp.write_text(json.dumps(plan))
        with pytest.raises(InfeasibleError) as exc:
            load_bundle(str(fp), str(mp), str(cp), str(pp))
        # 预算/重复容量不足属于有证据的不可行，而非 SearchFailure。
        assert isinstance(exc.value, InfeasibleError)
        assert not isinstance(exc.value, SearchFailure)
