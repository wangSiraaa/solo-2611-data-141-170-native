"""交付物写出：候选 JSON、采样窗依据、逐候选相位/拟合/残差 CSV、静态 Markdown 报告。

报告中的每个数值均来自同一批拟合对象与写出的 CSV/JSON，不另做口径转换。
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

from .model import bic
from .periodogram import period_uncertainty


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"不可序列化对象: {type(o)!r}")


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_json_default)


def write_phase_csv(path: str, lc, candidate) -> None:
    fit = candidate.fits[candidate.preferred_harmonics]
    phase = ((fit.freq * (lc.time - fit.t0)) % 1.0)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["observation_id", "night_id", "time_day", "brightness", "sigma",
                    "phase_0_to_1", "predicted_brightness", "residual",
                    "weighted_residual"])
        for i in range(lc.n_points):
            w.writerow([
                lc.observation_id[i], lc.night_id[i],
                f"{lc.time[i]:.10f}", f"{lc.brightness[i]:.10f}", f"{lc.sigma[i]:.10f}",
                f"{phase[i]:.10f}", f"{fit.predicted[i]:.10f}",
                f"{fit.residual[i]:.10f}",
                f"{fit.residual[i] / lc.sigma[i]:.10f}",
            ])


def write_periodogram_csv(path: str, grid: np.ndarray, chi_by_k: dict[int, np.ndarray]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["freq_per_day", "period_day"] + [f"chi2_K{k}" for k in sorted(chi_by_k)])
        for i, f in enumerate(grid):
            w.writerow([f"{f:.10f}", f"{1.0 / f:.10f}"] +
                       [f"{chi_by_k[k][i]:.10f}" for k in sorted(chi_by_k)])


def write_window_csv(path: str, nu: np.ndarray, power: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["offset_freq_per_day", "normalized_window_power"])
        for f, p in zip(nu, power):
            w.writerow([f"{f:.10f}", f"{p:.10f}"])


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def build_report(case_name: str, lc, payload: dict) -> str:
    win = payload["sampling_window"]
    meta = payload["grid_meta"]
    cands = payload["candidates"]
    lines: list[str] = []
    lines.append(f"# 周期分析报告 — {case_name}")
    lines.append("")
    lines.append("## 1. 数据与质量校验")
    lines.append("")
    lines.append(f"- 输入文件：`{os.path.basename(lc.source_csv)}`，"
                 f"观测点 {lc.n_points} 个，观测夜 {win['sampling_summary']['n_nights']} 个，"
                 f"时间跨度 T = {meta['time_span_day']:.4f} 天。")
    if lc.issues:
        lines.append(f"- ⚠️ 发现 {len(lc.issues)} 条数据问题（详见 candidates.json 的 data_issues），"
                     "本报告只在干净数据上拟合。")
    else:
        lines.append("- 重复 observation_id、非有限数、非正 sigma 检查：均未发现。")
    lines.append("")
    lines.append("## 2. 频率分辨率与网格设计")
    lines.append("")
    lines.append(f"- Rayleigh 分辨率 1/T = **{meta['rayleigh_resolution_per_day']:.5f} /天**"
                 f"（对应周期方向 {meta['rayleigh_resolution_per_day'] / cands[0]['freq_per_day']**2:.4f} 天量级）。"
                 "时间跨度决定两个峰至少相差约 1/T 才可能被采样独立分辨；")
    lines.append(f"- 粗网格步长 1/(10T) = {meta['coarse_grid_step_per_day']:.6f} /天，"
                 f"共 {meta['n_grid']} 个频率点；每个局部极小再做 3 轮局部细化"
                 "（每轮 41 点、区间缩至 1/5），候选 JSON 给出各候选最终网格步长；")
    lines.append("- 全部计算只在真实观测时刻构建设计矩阵做加权最小二乘，"
                 "没有把长空白插补成规则序列，也没有使用 FFT。")
    lines.append("")
    lines.append("## 3. 采样窗审计")
    lines.append("")
    ss = win["sampling_summary"]
    lines.append(f"- 夜间隔：{ss['night_gap_day']['min']:.3f}–{ss['night_gap_day']['max']:.3f} 天，"
                 f"其中 {ss['night_gap_day']['near_one_day_fraction']*100:.0f}% 落在 1±0.15 天；")
    lines.append(f"- 每夜平均本地时刻 (t mod 1)：{ss['nightly_local_time_mod1']['min']:.4f}"
                 f"–{ss['nightly_local_time_mod1']['max']:.4f}，环形标准差 "
                 f"{ss['nightly_local_time_mod1']['circular_std_day']:.5f} 天。"
                 "该散布越小，每日采样越接近精确，f 与 f±k/天 的别名越难区分；")
    lines.append("- 谱窗主要旁瓣（只由观测时刻决定，完整曲线见 window_function.csv）：")
    for pk in win["window_peaks"][:6]:
        tag = f"（≈ {pk['harmonic_of_daily']} /天 的每日谐波）" if pk["harmonic_of_daily"] else ""
        lines.append(f"  - ν = {pk['freq_per_day']:+.4f} /天，归一化功率 {pk['power']:.4f} {tag}")
    lines.append("")
    lines.append("## 4. 候选周期")
    lines.append("")
    lines.append("模型为常数加最多 2 个谐波，权重 w=1/σ²；谐波阶数按 BIC 选择。"
                 "`explained_score = 1 − χ²/χ²_常数` 是拟合优度的描述性分数，"
                 "**不是后验概率**，候选间分数差也不能解释为概率差。")
    lines.append("")
    lines.append("| # | 周期 (天) | 频率 (/天) | K | χ² | 约化χ² | BIC | explained | 留出夜约化χ² | 标注 |")
    lines.append("|---|-----------|-----------|---|----|-------|-----|-----------|-------------|------|")
    for i, c in enumerate(cands):
        labels = ", ".join(c["labels"]) or "—"
        hr = c["holdout"]["holdout_reduced_chi2"] if c["holdout"] else None
        lines.append(
            f"| {i} | {c['period_day']:.4f} | {c['freq_per_day']:.5f} | "
            f"{c['preferred_harmonics']} | {c['chi2']:.2f} | {c['reduced_chi2']:.3f} | "
            f"{c['bic']:.2f} | {c['explained_score']:.4f} | {_fmt(hr, 3)} | {labels} |"
        )
    lines.append("")
    lines.append("### 4.1 首选候选的谐波参数与误差口径")
    lines.append("")
    c0 = cands[0]
    lines.append(f"候选 0：f = {c0['freq_per_day']:.5f} ± {c0['freq_1sigma_halfwidth']:.6f} /天，"
                 f"P = {c0['period_day']:.4f} 天（周期 1σ ≈ "
                 f"{period_uncertainty(c0['freq_per_day'], c0['freq_1sigma_halfwidth']):.5f} 天，"
                 "由 χ² = χ²_min+1 的频率半宽一阶传播）。")
    lines.append(f"常数项 c = {c0['constant']:.5f}。各谐波写作 A_k·cos(2πkf(t−t0)+φ_k)，"
                 "振幅/相位标准误由参数协方差矩阵 (X'WX)⁻¹ 线性传播：")
    lines.append("")
    lines.append("| 谐波 k | A_k | σ(A_k) | φ_k (rad) | σ(φ_k) (rad) |")
    lines.append("|--------|-----|--------|-----------|--------------|")
    for h in c0["harmonics"]:
        lines.append(f"| {h['order']} | {h['amplitude']:.5f} | {h['amplitude_err']:.5f} | "
                     f"{h['phase_rad']:.5f} | {h['phase_err_rad']:.5f} |")
    lines.append("")
    lines.append("### 4.2 每日采样别名族")
    lines.append("")
    fam = win["alias_families"]
    if fam["links"]:
        lines.append(f"按 |f_i−f_j−k/天| < 1 Rayleigh（{fam['rayleigh_tolerance_per_day']:.5f} /天）连族，"
                     f"共 {fam['n_families']} 族。族内 χ² 最小者为主成员，其余为每日别名：")
        for link in fam["links"]:
            lines.append(f"- 候选 {link['candidate_i']} (f={link['freq_i']:.4f}) 与 "
                         f"候选 {link['candidate_j']} (f={link['freq_j']:.4f})，"
                         f"频率差 {link['offset_per_day']:+.4f} /天 ≈ {link['daily_order']:+d} /天")
    else:
        lines.append("候选之间未发现间距在 1 Rayleigh 内的整数/天关系。")
    lines.append("")
    lines.append("### 4.3 P 与 2P 的近等价解释")
    lines.append("")
    if payload["half_period_pairs"]:
        for p in payload["half_period_pairs"]:
            verdict = "近等价，保留 P 与 2P 两种解释" if p["near_equivalent"] else "结构配对但证据可区分"
            lines.append(f"- 候选 {p['low_freq_candidate_id']}（2P = {p['period_long_day']:.4f} 天）"
                         f"对候选 {p['high_freq_candidate_id']}（P = {p['period_short_day']:.4f} 天）："
                         f"χ² = {p['chi2_long_period']:.2f} vs {p['chi2_short_period']:.2f}"
                         f"（相对差 {p['rel_chi2_diff']*100:.1f}%），"
                         f"BIC = {p['bic_long_period']:.1f} vs {p['bic_short_period']:.1f}，"
                         f"留出 χ² = {_fmt(p['holdout_chi2_long'],1)} vs {_fmt(p['holdout_chi2_short'],1)}，"
                         f"长周期模型 A1/A2 = {_fmt(p['fundamental_to_second_amplitude_ratio_long_model'],3)}，"
                         f"判定：**{verdict}**。")
    else:
        lines.append("未发现 f_hi ≈ 2 f_lo 的候选对。")
    lines.append("")
    lines.append("## 5. 结论与可分辨性说明")
    lines.append("")
    lines.append(
        build_conclusion(cands, win, payload["half_period_pairs"])
    )
    lines.append("")
    lines.append("整夜留出（每折仅用其余 11 夜拟合、预测整条留出夜）与相位折叠 CSV 是支持性证据；"
                 "有限且带窗的采样不保证恢复唯一真周期。采样或信号形状不足以区分时，"
                 "本报告保留候选集并给出原因，而不是强制只报一个最高峰。")
    lines.append("")
    lines.append("## 附：复算入口")
    lines.append("")
    lines.append("- `candidates.json`：全部候选、逐 K 拟合系数、数据问题与网格元数据；")
    lines.append("- `candidate_N_phase.csv`：逐点相位、模型预测、残差、加权残差；")
    lines.append("- `periodogram.csv`：粗网格上 K=1/K=2 的逐频率 χ²；")
    lines.append("- `window_function.csv`、`sampling_window.json`：谱窗与别名依据；")
    lines.append("- 设计矩阵定义见 `period_analysis/model.py::design_matrix`，"
                 "可在任意单个频率独立构建并核对加权残差。")
    lines.append("")
    return "\n".join(lines)


def build_conclusion(cands, win, pairs) -> str:
    notes = []
    best = cands[0]
    notes.append(f"全数据 χ² 最优候选为 P = {best['period_day']:.4f} 天"
                 f"（f = {best['freq_per_day']:.5f} /天，K = {best['preferred_harmonics']}）。")
    alias_members = [c for c in cands if "daily_alias" in c["labels"]]
    if alias_members:
        notes.append("存在与每日采样有关的别名候选（标注 daily_alias）：谱窗在 ±1/天等处有强旁瓣，"
                     "这些频率与主频率在当前采样下近线性相关，χ² 接近不代表多个真实周期，"
                     "需夜内时刻更分散或更长时间跨度才能拆开。")
    if any(p["near_equivalent"] for p in pairs):
        notes.append("存在 P 与 2P 近等价的候选对：信号近似半周期对称（长周期模型一次谐波很弱），"
                     "现有采样在两种解释之间无法给出有意义的偏好，两个周期同时保留。")
    hold = [(i, c["holdout"]["holdout_reduced_chi2"]) for i, c in enumerate(cands) if c["holdout"]]
    if hold:
        i_best_ho, v = min(hold, key=lambda z: z[1])
        if i_best_ho != 0:
            notes.append(f"注意整夜留出表现最好的是候选 {i_best_ho} 而非全数据 χ² 最优候选，"
                         "说明最优点对单夜信息较敏感，结论应按候选集表述。")
        else:
            notes.append("整夜留出中最优候选同样排名第一，与全数据拟合一致。")
    return " ".join(notes)
