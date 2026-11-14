"""报告生成：summary.json、summary.md 与静态图（matplotlib，Agg 后端）。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 图中含中文标注，优先使用可用的 CJK 字体，找不到时回退默认字体（仅影响图内文字）
for _candidate in ("Noto Serif CJK SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
    try:
        from matplotlib import font_manager

        font_manager.findfont(_candidate, fallback_to_default=False)
        plt.rcParams["font.family"] = [_candidate, "DejaVu Sans"]
        break
    except Exception:
        continue

from .interpret import STATUS_OK
from .recovery import CHAIN_BLOCKED, RecoveredData

STATUS_LABELS_ZH = {
    "OK_NO_ISSUE_DETECTED": "未发现问题",
    "CHAINS_DISAGREE": "链间不一致",
    "INSUFFICIENT_ESS": "有效样本量不足",
    "CONSTANT_ALL_CHAINS": "全链同一常量",
    "CONSTANT_DIFFERS_ACROSS_CHAINS": "链间常量不同",
    "NONFINITE_VALUES": "含非有限值",
    "INSUFFICIENT_DATA": "样本不足",
    "DIAGNOSTICS_BLOCKED": "诊断被阻断",
}


def _fmt(v: float, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NaN"
    if isinstance(v, float) and np.isinf(v):
        return "inf" if v > 0 else "-inf"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if digits == 1:
        return f"{v:.1f}"
    return f"{v:.{digits}g}"


def run_diagnostics(recovered: RecoveredData) -> dict:
    """对每个参数计算指标并判定状态，返回可 JSON 序列化的结果字典。"""
    from .diagnostics import summarize_parameter
    from .interpret import classify

    results: dict[str, dict] = {}
    for p in recovered.parameters:
        if recovered.common_length is None or len(recovered.valid_chain_ids) == 0:
            summary = {
                "n_chains": 0,
                "n_draws_per_chain": 0,
                "nonfinite": False,
                "all_constant": False,
                "chain_constant_differs": False,
            }
        else:
            summary = summarize_parameter(recovered.used_matrix(p))
        verdict = classify(summary, recovered.decision_limits)
        results[p] = {"metrics": summary, "verdict": verdict}
    return results


def write_summary_json(recovered: RecoveredData, results: dict, out_dir: Path) -> Path:
    payload = {
        "chain_recovery": recovered.to_record(),
        "parameters": results,
        "disclaimer": "decision_limits 仅为复核辅助；全部通过只说明这些有限诊断未发现问题，不能证明收敛或后验模型正确。",
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    return path


def write_summary_md(recovered: RecoveredData, results: dict, out_dir: Path) -> Path:
    lines: list[str] = []
    lines.append("# MCMC 分片样本诊断报告\n")
    lines.append("## 链恢复记录\n")
    lines.append("| 链 | 状态 | 片段数 | 读入行 | 去重 | 去warmup | 保留 | 共同长度L | 丢弃尾部 | 问题 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in recovered.chains:
        lines.append(
            "| {id} | {st} | {nf} | {nr} | {nd} | {nw} | {nk} | {L} | {dt} | {pb} |".format(
                id=c.chain_id,
                st=c.status,
                nf=len(c.fragments),
                nr=c.n_rows_loaded,
                nd=c.n_duplicates_removed,
                nw=c.n_warmup_removed,
                nk=c.n_retained,
                L=c.common_length if c.common_length is not None else "-",
                dt=c.n_discarded_tail if c.n_discarded_tail is not None else "-",
                pb="; ".join(c.problems) if c.problems else "-",
            )
        )
    lines.append("")
    lines.append(f"- 有效链: {recovered.valid_chain_ids}；共同保留长度 L = {recovered.common_length}")
    for note in recovered.notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## 参数级结论\n")
    lines.append("| 参数 | 状态 | 未通过项 | mean | q05 | q50 | q95 | MCSE(mean) | R-hat(rank) | R-hat(folded) | ESS bulk | ESS tail |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p, r in results.items():
        m, v = r["metrics"], r["verdict"]
        lines.append(
            "| {p} | {st} | {fc} | {mean} | {q05} | {q50} | {q95} | {mcse} | {rr} | {rf} | {eb} | {et} |".format(
                p=p,
                st=v["status"],
                fc=",".join(v["failed_checks"]) or "-",
                mean=_fmt(m.get("mean")),
                q05=_fmt(m.get("q05")),
                q50=_fmt(m.get("q50")),
                q95=_fmt(m.get("q95")),
                mcse=_fmt(m.get("mcse_mean")),
                rr=_fmt(m.get("rhat_rank")),
                rf=_fmt(m.get("rhat_folded")),
                eb=_fmt(m.get("ess_bulk"), 1),
                et=_fmt(m.get("ess_tail"), 1),
            )
        )
    lines.append("")
    lines.append("### 逐项说明\n")
    for p, r in results.items():
        v = r["verdict"]
        lines.append(f"- **{p}** [{v['status']} / {STATUS_LABELS_ZH.get(v['status'], v['status'])}] {v['explanation']}")
    lines.append("")
    lines.append("## 口径与免责声明\n")
    lines.append("- 秩变换：平均秩 + Blom 变换 Φ⁻¹((r−3/8)/(S+1/4))；split 奇数长度丢弃末样本。")
    lines.append("- ESS 截断：Geyer 初始正序列（成对和首个负值处截断）+ 单调化；ESS 上限 m·n·log10(m·n)。")
    lines.append("- tail ESS：5%/95% 经验分位指示序列 ESS 的较小者；MCSE(mean)=sd/√ESS(原始值)。")
    lines.append("- 阈值来自 manifest decision_limits，仅为复核辅助；全部通过只说明这些有限诊断未发现问题，不能证明收敛或后验模型正确。")
    lines.append("- 图文件：traces_<参数>.png（原链轨迹，灰底为已去除 warmup，虚线为被丢弃的尾部样本），overview.png。")
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_figures(recovered: RecoveredData, results: dict, out_dir: Path) -> list[str]:
    """每个参数一张证据图（原链轨迹 + 链间直方图 + 指标），另加一张总览图。"""
    written: list[str] = []
    valid = recovered.valid_chain_ids
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for p in recovered.parameters:
        r = results[p]
        m, v = r["metrics"], r["verdict"]
        fig, (ax_tr, ax_hi) = plt.subplots(
            1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2.2, 1.0]}
        )
        for i, c in enumerate(recovered.chains):
            color = colors[i % len(colors)]
            if c.status == CHAIN_BLOCKED or c.iterations is None:
                ax_tr.plot([], [], color=color, label=f"{c.chain_id} (blocked)")
                continue
            w_end = c.warmup_end or 0
            # 已去除的 warmup 区间以灰底标出
            ax_tr.axvspan(0.5, w_end + 0.5, color="0.92", zorder=0)
            # 共同长度内使用的样本（实线）
            L = c.common_length or 0
            used_it = c.iterations[:L]
            used_val = c.values[p][:L]
            ax_tr.plot(used_it, used_val, color=color, lw=0.7, label=c.chain_id)
            # 被丢弃的尾部样本（虚线，公开披露）
            if c.n_discarded_tail:
                tail_it = c.iterations[L:]
                tail_val = c.values[p][L:]
                ax_tr.plot(tail_it, tail_val, color=color, lw=0.7, ls=":", alpha=0.6)
            ax_hi.hist(
                c.values[p][:L], bins=40, histtype="step", density=True, color=color, lw=1.2
            )
        ax_tr.set_xlabel("original sampler iteration")
        ax_tr.set_ylabel(p)
        ax_tr.set_title(f"{p}: 原链轨迹（灰底=已去warmup，虚线=丢弃尾部）")
        ax_tr.legend(fontsize=8, loc="best")
        ax_hi.set_title("链间密度对比（使用样本）")
        ax_hi.set_yticks([])
        txt = (
            f"status: {v['status']}\n"
            f"mean={_fmt(m.get('mean'))}  MCSE={_fmt(m.get('mcse_mean'))}\n"
            f"q05={_fmt(m.get('q05'))}  q50={_fmt(m.get('q50'))}  q95={_fmt(m.get('q95'))}\n"
            f"R-hat rank={_fmt(m.get('rhat_rank'))}  folded={_fmt(m.get('rhat_folded'))}\n"
            f"ESS bulk={_fmt(m.get('ess_bulk'), 1)}  tail={_fmt(m.get('ess_tail'), 1)}"
        )
        ax_tr.text(
            0.01, 0.98, txt, transform=ax_tr.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
        )
        fig.tight_layout()
        fname = f"traces_{p}.png"
        fig.savefig(out_dir / fname, dpi=110)
        plt.close(fig)
        written.append(fname)

    # 总览：R-hat 与 ESS 对阈值（NaN 为无定义，画 0 并标注 N/A，不伪装成数值）
    params = [p for p in recovered.parameters]
    rhat = [results[p]["metrics"].get("rhat_max", float("nan")) for p in params]
    eb = [results[p]["metrics"].get("ess_bulk", float("nan")) for p in params]
    et = [results[p]["metrics"].get("ess_tail", float("nan")) for p in params]
    lim = results[params[0]]["verdict"]["limits"] if params else {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(params))

    def _display(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.0, "N/A"
        if np.isinf(v):
            return 3.0, "inf"
        return min(v, 3.0), None

    def _bar_color(p):
        st = results[p]["verdict"]["status"]
        if st == STATUS_OK:
            return "seagreen"
        if st in ("CHAINS_DISAGREE", "INSUFFICIENT_ESS"):
            return "indianred"
        return "0.6"  # 退化状态（常量/非有限/样本不足）用灰色，不算“失败”

    heights, tags = zip(*[_display(v) for v in rhat])
    ax1.bar(x, heights, color=[_bar_color(p) for p in params])
    for xi, tag in zip(x, tags):
        if tag:
            ax1.text(xi, 0.05, tag, ha="center", va="bottom", fontsize=9, rotation=90)
    if "rhat" in lim:
        ax1.axhline(lim["rhat"], color="black", ls="--", lw=1, label=f"阈值 {lim['rhat']}")
        ax1.legend(fontsize=8)
    ax1.set_xticks(x, params, rotation=20, ha="right")
    ax1.set_ylabel("R-hat max（>3 截断显示）")
    ax1.set_title("多链一致性：rank/folded R-hat 较大者")
    width = 0.38
    ax2.bar(x - width / 2, [0.0 if v is None or np.isnan(v) else v for v in eb], width, label="bulk ESS")
    ax2.bar(x + width / 2, [0.0 if v is None or np.isnan(v) else v for v in et], width, label="tail ESS")
    for xi, v in zip(x, eb):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            ax2.text(xi, 5, "N/A", ha="center", va="bottom", fontsize=9)
    if "bulk_ess" in lim:
        ax2.axhline(lim["bulk_ess"], color="black", ls="--", lw=1, label=f"阈值 {lim['bulk_ess']:.0f}")
    ax2.set_xticks(x, params, rotation=20, ha="right")
    ax2.set_title("单链有效信息量：ESS 对阈值")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=110)
    plt.close(fig)
    written.append("overview.png")
    return written
