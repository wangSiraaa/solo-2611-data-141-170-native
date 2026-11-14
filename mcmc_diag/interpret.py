"""判读规则：把多链一致性（R-hat）与单链有效信息量（ESS/MCSE）合成为参数级结论。

设计原则：
- 两类问题分开报告，不混为一谈：
  * CHAINS_DISAGREE：链间不一致（rhat_max 超限）。典型表现是各链各自平稳
    却卡在不同模式，或链未平稳。此时 ESS 再高也不能说明可合并估计。
  * INSUFFICIENT_ESS：链间看起来一致（R-hat 达标）但有效样本量不足
    （bulk/tail ESS 低于复核阈值），典型表现是均值看似接近但自相关强，
    均值与尾部估计的 Monte Carlo 误差仍不可控。
  * 两者同时失败时状态取 CHAINS_DISAGREE（链间不一致时合并估计首先不可信），
    但 failed_checks 与解释中明确 ESS 同样不足——强自相关本身也会抬高 R-hat，
    与“卡在不同模式”是不同的问题模式，解释中分别注明。
- 退化输入有显式状态，绝不把 NaN 当“通过”：
  NONFINITE_VALUES / CONSTANT_ALL_CHAINS / CONSTANT_DIFFERS_ACROSS_CHAINS /
  INSUFFICIENT_DATA / DIAGNOSTICS_BLOCKED。
- 阈值来自 manifest 的 decision_limits，只是复核辅助：全部通过仅说明
  “这些有限诊断未发现问题”，不能证明收敛或后验模型正确。
- 单参数出问题不影响其他参数出现在报告中（逐参数独立判定）。
"""

from __future__ import annotations

import numpy as np

from .diagnostics import MIN_CHAIN_LENGTH, MIN_CHAINS

STATUS_OK = "OK_NO_ISSUE_DETECTED"
STATUS_CHAINS_DISAGREE = "CHAINS_DISAGREE"
STATUS_INSUFFICIENT_ESS = "INSUFFICIENT_ESS"
STATUS_CONSTANT = "CONSTANT_ALL_CHAINS"
STATUS_CONSTANT_DIFFERS = "CONSTANT_DIFFERS_ACROSS_CHAINS"
STATUS_NONFINITE = "NONFINITE_VALUES"
STATUS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
STATUS_BLOCKED = "DIAGNOSTICS_BLOCKED"

DEFAULT_LIMITS = {"rhat": 1.01, "bulk_ess": 400.0, "tail_ess": 400.0}

_FIXED_EXPLANATIONS = {
    STATUS_CONSTANT: "参数在全部链中为同一常量：R-hat/ESS 无定义（非“通过”），均值即常量、MCSE 为 0。",
    STATUS_CONSTANT_DIFFERS: "各链各自为常量但链间取值不同：R-hat=inf，链间严重不一致。",
    STATUS_NONFINITE: "保留样本中存在 NaN/Inf，诊断指标无定义，需先排查数据。",
    STATUS_INSUFFICIENT_DATA: "有效链数或共同保留长度不足，无法计算多链诊断。",
    STATUS_BLOCKED: "有效链不足两条（链恢复阶段被阻断），多链诊断不可用。",
}


def classify(summary: dict, limits: dict | None = None) -> dict:
    """根据 summarize_parameter 的指标字典给出状态、未通过检查项与解释。"""
    lim = dict(DEFAULT_LIMITS)
    if limits:
        lim.update({k: float(v) for k, v in limits.items()})

    failed: list[str] = []
    pattern = "degenerate"
    if summary["nonfinite"]:
        status = STATUS_NONFINITE
    elif summary["n_chains"] < MIN_CHAINS:
        status = STATUS_BLOCKED
    elif summary["n_draws_per_chain"] < MIN_CHAIN_LENGTH:
        status = STATUS_INSUFFICIENT_DATA
    elif summary["all_constant"]:
        status = STATUS_CONSTANT
    elif summary["chain_constant_differs"]:
        status = STATUS_CONSTANT_DIFFERS
        failed.append("rhat")
    else:
        rhat_bad = not (summary["rhat_max"] <= lim["rhat"])  # NaN/inf 亦落入
        bulk_bad = not (summary["ess_bulk"] >= lim["bulk_ess"])
        tail_bad = not (summary["ess_tail"] >= lim["tail_ess"])
        if rhat_bad:
            failed.append("rhat")
        if bulk_bad:
            failed.append("bulk_ess")
        if tail_bad:
            failed.append("tail_ess")
        if rhat_bad:
            status = STATUS_CHAINS_DISAGREE
            pattern = "chains_disagree_and_low_ess" if (bulk_bad or tail_bad) else "chains_disagree"
        elif bulk_bad or tail_bad:
            status = STATUS_INSUFFICIENT_ESS
            pattern = "low_ess"
        else:
            status = STATUS_OK
            pattern = "ok"

    if status == STATUS_OK:
        explanation = (
            "R-hat 与 bulk/tail ESS 均通过复核阈值：这些有限诊断未发现问题"
            "（不证明收敛或后验模型正确）。"
        )
    elif status == STATUS_CHAINS_DISAGREE:
        parts = ["链间不一致：rhat_max 超过阈值，各链可能卡在不同模式或尚未平稳，链间合并估计尚不可信。"]
        rr = summary.get("rhat_rank", float("nan"))
        rf = summary.get("rhat_folded", float("nan"))
        if np.isfinite(rr) and np.isfinite(rf) and rr - rf > 0.2 and rf <= lim["rhat"]:
            parts.append(
                "rank R-hat 明显高于 folded R-hat，呈链间位置/模式漂移特征"
                "（各链各自平稳却卡在不同位置，而非尺度差异）。"
            )
        if pattern == "chains_disagree_and_low_ess":
            parts.append(
                "同时 bulk/tail ESS 低于阈值：有效信息量不足；强自相关也会抬高 R-hat，"
                "需延长抽样后复查才能区分“混合慢”与“多模式”。"
            )
        explanation = "".join(parts)
    elif status == STATUS_INSUFFICIENT_ESS:
        explanation = (
            "链间均值看似一致（R-hat 达标）但有效样本量不足：自相关强或尾部探索不够，"
            "均值/尾部估计的 Monte Carlo 误差仍偏大，尚不支持精确估计。"
        )
    else:
        explanation = _FIXED_EXPLANATIONS[status]

    return {
        "status": status,
        "pattern": pattern,
        "failed_checks": failed,
        "explanation": explanation,
        "limits": lim,
    }
