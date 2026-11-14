"""判读规则测试：状态优先级、阈值生效、两类问题模式的区分。"""

from mcmc_diag.interpret import (
    STATUS_BLOCKED,
    STATUS_CHAINS_DISAGREE,
    STATUS_CONSTANT,
    STATUS_CONSTANT_DIFFERS,
    STATUS_INSUFFICIENT_DATA,
    STATUS_INSUFFICIENT_ESS,
    STATUS_NONFINITE,
    STATUS_OK,
    classify,
)


def _base_summary(**over):
    s = {
        "n_chains": 4,
        "n_draws_per_chain": 500,
        "nonfinite": False,
        "all_constant": False,
        "chain_constant_differs": False,
        "rhat_max": 1.0,
        "ess_bulk": 1000.0,
        "ess_tail": 1000.0,
    }
    s.update(over)
    return s


def test_ok_when_all_pass():
    v = classify(_base_summary())
    assert v["status"] == STATUS_OK
    assert v["failed_checks"] == []
    assert "不证明" in v["explanation"]  # 通过不等于证明收敛/模型正确


def test_chains_disagree_when_rhat_fails():
    v = classify(_base_summary(rhat_max=1.7))
    assert v["status"] == STATUS_CHAINS_DISAGREE
    assert v["failed_checks"] == ["rhat"]


def test_insufficient_ess_when_only_ess_fails():
    """链间均值看似接近（R-hat 达标）但自相关强 → INSUFFICIENT_ESS。"""
    v = classify(_base_summary(ess_bulk=50.0, ess_tail=900.0))
    assert v["status"] == STATUS_INSUFFICIENT_ESS
    assert v["failed_checks"] == ["bulk_ess"]
    v2 = classify(_base_summary(ess_tail=100.0))
    assert v2["status"] == STATUS_INSUFFICIENT_ESS
    assert v2["failed_checks"] == ["tail_ess"]


def test_both_failures_reported_together():
    """R-hat 与 ESS 同时失败：状态为链间不一致，但 ESS 失败不丢失。"""
    v = classify(_base_summary(rhat_max=1.2, ess_bulk=30.0, ess_tail=40.0))
    assert v["status"] == STATUS_CHAINS_DISAGREE
    assert set(v["failed_checks"]) == {"rhat", "bulk_ess", "tail_ess"}
    assert "ESS" in v["explanation"]


def test_degenerate_states_have_explicit_status():
    assert classify(_base_summary(nonfinite=True))["status"] == STATUS_NONFINITE
    assert classify(_base_summary(all_constant=True))["status"] == STATUS_CONSTANT
    assert classify(_base_summary(chain_constant_differs=True))["status"] == STATUS_CONSTANT_DIFFERS
    assert classify(_base_summary(n_chains=1))["status"] == STATUS_BLOCKED
    assert classify(_base_summary(n_draws_per_chain=3))["status"] == STATUS_INSUFFICIENT_DATA


def test_nan_metrics_never_pass():
    """NaN 指标必须落入失败分支，而不是被当成通过。"""
    v = classify(_base_summary(rhat_max=float("nan")))
    assert v["status"] != STATUS_OK
    assert "rhat" in v["failed_checks"]
    v2 = classify(_base_summary(ess_bulk=float("nan")))
    assert v2["status"] == STATUS_INSUFFICIENT_ESS


def test_limits_come_from_manifest():
    """阈值由 decision_limits 驱动，不是写死的。"""
    strict = classify(_base_summary(ess_bulk=500.0), limits={"bulk_ess": 1000.0})
    assert strict["status"] == STATUS_INSUFFICIENT_ESS
    loose = classify(_base_summary(rhat_max=1.05), limits={"rhat": 1.1})
    assert loose["status"] == STATUS_OK
