"""整夜留出（leave-one-night-out）评价与 P/2P 近等价识别。

每折只用其余夜的数据拟合，再预测整条留出夜；绝不随机拆散同一夜内的密集点。
留出 chi2 是预测口径（留出点不参与该折参数估计），与全数据拟合 chi2 口径不同，
不可直接相减当作似然比。
"""

from __future__ import annotations

import numpy as np

from .model import weighted_fit

HALF_PERIOD_TOL_FACTOR = 1.5   # |f_hi - 2 f_lo| < 1.5 Rayleigh 视为 P/2P 配对


def leave_one_night_out(lc, candidate) -> dict:
    groups = lc.night_groups()
    t, y, s = lc.time, lc.brightness, lc.sigma
    k = candidate.preferred_harmonics
    t0 = float(np.mean(t))
    per_night = []
    total_chi2 = 0.0
    total_points = 0
    for nid in sorted(groups):
        idx_hold = groups[nid]
        mask = np.ones(t.size, dtype=bool)
        mask[idx_hold] = False
        if mask.sum() <= 1 + 2 * k:
            per_night.append({"night_id": nid, "n_points": int(idx_hold.size),
                              "skipped": "训练点不足"})
            continue
        fold_fit = weighted_fit(t[mask], y[mask], s[mask], candidate.freq, k, t0)
        pred = fold_fit.predict(t[idx_hold])
        chi2 = float(np.sum(((y[idx_hold] - pred) / s[idx_hold]) ** 2))
        total_chi2 += chi2
        total_points += idx_hold.size
        per_night.append({
            "night_id": nid,
            "n_points": int(idx_hold.size),
            "n_train_points": int(mask.sum()),
            "holdout_chi2": chi2,
            "holdout_reduced_chi2": chi2 / idx_hold.size,
            "mean_weighted_residual": float(np.sum((y[idx_hold] - pred) / s[idx_hold]**2)
                                            / np.sum(1.0 / s[idx_hold]**2)),
        })
    return {
        "scheme": "leave-one-whole-night-out",
        "n_folds": len(groups),
        "harmonics_used": k,
        "total_holdout_chi2": total_chi2,
        "total_holdout_points": total_points,
        "holdout_reduced_chi2": total_chi2 / total_points if total_points else None,
        "per_night": per_night,
    }


def constant_holdout(lc) -> dict:
    """常数模型的整夜留出基线，供候选比较。"""
    groups = lc.night_groups()
    t, y, s = lc.time, lc.brightness, lc.sigma
    total_chi2, total_points = 0.0, 0
    for nid in sorted(groups):
        idx = groups[nid]
        mask = np.ones(t.size, dtype=bool)
        mask[idx] = False
        w = 1.0 / s[mask] ** 2
        c = np.sum(w * y[mask]) / np.sum(w)
        total_chi2 += float(np.sum(((y[idx] - c) / s[idx]) ** 2))
        total_points += idx.size
    return {"scheme": "leave-one-whole-night-out", "model": "constant",
            "total_holdout_chi2": total_chi2,
            "holdout_reduced_chi2": total_chi2 / total_points}


def mark_half_period_pairs(candidates, rayleigh: float) -> list[dict]:
    """识别 f_hi ≈ 2 f_lo 的候选对（即 P_lo = 2 P_hi）。

    结构性配对条件（同时满足，避免把差候选凑成噪声对）：
      a) |f_hi - 2 f_lo| < 1.5 Rayleigh；
      b) 长周期模型必须含二次谐波（K=2），否则无法表达对称结构；
      c) 两个模型都不能明显失配（约化 chi2 <= 5）。
    近等价判据综合四类可复算证据：全数据 chi2 相对差 <= 10%、
    |BIC 差| <= 10、整夜留出 chi2 相对差 <= 20%，且长周期模型
    一次/二次谐波振幅比 < 0.25（信号近似半周期对称）。
    只有近等价对才在候选上打 half_period_* 标注。
    """
    pairs = []
    for i, lo in enumerate(candidates):
        if lo.preferred_harmonics < 2:
            continue
        for j, hi in enumerate(candidates):
            if i == j:
                continue
            if abs(hi.freq - 2.0 * lo.freq) >= HALF_PERIOD_TOL_FACTOR * rayleigh:
                continue
            if lo.reduced_chi2 > 5.0 or hi.reduced_chi2 > 5.0:
                continue
            fit_lo = lo.fits[lo.preferred_harmonics]
            terms = {h.order: h for h in fit_lo.harmonic_terms()}
            a1, a2 = terms[1].amplitude, terms[2].amplitude
            ratio = a1 / a2 if a2 > 0 else np.inf
            h_lo = lo.holdout or {}
            h_hi = hi.holdout or {}
            rel_chi2 = abs(lo.chi2 - hi.chi2) / max(lo.chi2, hi.chi2)
            rel_hold = None
            if h_lo.get("total_holdout_chi2") and h_hi.get("total_holdout_chi2"):
                rel_hold = (abs(h_lo["total_holdout_chi2"] - h_hi["total_holdout_chi2"])
                            / max(h_lo["total_holdout_chi2"], h_hi["total_holdout_chi2"]))
            near = (
                rel_chi2 <= 0.10
                and abs(lo.bic - hi.bic) <= 10.0
                and (rel_hold is None or rel_hold <= 0.20)
                and ratio < 0.25
            )
            pairs.append({
                "low_freq_candidate_id": i,   # 周期 2P 解释
                "high_freq_candidate_id": j,  # 周期 P 解释
                "period_long_day": lo.period,
                "period_short_day": hi.period,
                "freq_mismatch_per_day": hi.freq - 2.0 * lo.freq,
                "chi2_long_period": lo.chi2,
                "chi2_short_period": hi.chi2,
                "rel_chi2_diff": rel_chi2,
                "bic_long_period": lo.bic,
                "bic_short_period": hi.bic,
                "holdout_chi2_long": h_lo.get("total_holdout_chi2"),
                "holdout_chi2_short": h_hi.get("total_holdout_chi2"),
                "fundamental_to_second_amplitude_ratio_long_model": ratio,
                "near_equivalent": bool(near),
                "criteria": {
                    "rel_chi2_diff_le_0.10": rel_chi2 <= 0.10,
                    "abs_bic_diff_le_10": abs(lo.bic - hi.bic) <= 10.0,
                    "rel_holdout_diff_le_0.20": None if rel_hold is None else rel_hold <= 0.20,
                    "amplitude_ratio_lt_0.25": ratio < 0.25,
                },
            })
            if near:
                if "half_period_long_2P" not in lo.labels:
                    lo.labels.append("half_period_long_2P")
                if "half_period_short_P" not in hi.labels:
                    hi.labels.append("half_period_short_P")
                lo.half_period_pair = f"candidate_{j}"
                hi.half_period_pair = f"candidate_{i}"
    return pairs
