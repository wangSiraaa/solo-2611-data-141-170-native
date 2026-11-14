"""采样窗审计：真实观测时刻本身决定哪些频率不可分辨。

两部分证据：
1. 时间结构统计（夜间隔、夜内本地时刻 t mod 1 的散布、相邻点间隔）；
2. 谱窗 W(nu) = |sum_i exp(2*pi*i*nu*t_i)|^2 / N^2 —— 只由采样时刻决定，
   与亮度无关。每日采样会在 nu = ±1/day 及其谐波处产生强窗峰；真实频率 f0
   与别名 f0 ± k/day 在这样的采样下近线性相关，周期图会同时出峰。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WINDOW_NU_RANGE = 3.0          # 扫描 |nu| <= 3 /day，覆盖 1、2、3/day
WINDOW_OVERSAMPLE = 20         # 窗函数网格步长 1/(20 T)
WINDOW_PEAK_MIN = 0.05         # 只记录归一化功率 >= 5% 的窗峰
DAILY_FREQ = 1.0               # 相对日，每日采样频率即 1 /day


@dataclass
class WindowPeak:
    freq_per_day: float
    power: float
    harmonic_of_daily: int | None

    def to_dict(self) -> dict:
        return {
            "freq_per_day": self.freq_per_day,
            "power": self.power,
            "harmonic_of_daily": self.harmonic_of_daily,
        }


def spectral_window(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """归一化谱窗 W(nu)，在真实时刻上直接求和，不做 FFT/插补。"""
    T = float(t.max() - t.min())
    step = 1.0 / (WINDOW_OVERSAMPLE * T)
    nu = np.arange(-WINDOW_NU_RANGE, WINDOW_NU_RANGE + step, step)
    # 分块求和避免大外占内存
    power = np.empty(nu.size)
    block = 4096
    n = t.size
    for s in range(0, nu.size, block):
        nu_b = nu[s : s + block]
        z = np.exp(2j * np.pi * np.outer(nu_b, t - t.mean())).sum(axis=1)
        power[s : s + block] = (z.real**2 + z.imag**2) / n**2
    return nu, power


def window_peaks(nu: np.ndarray, power: np.ndarray) -> list[WindowPeak]:
    from scipy.signal import argrelextrema

    idx = argrelextrema(power, np.greater)[0]
    idx = idx[power[idx] >= WINDOW_PEAK_MIN]
    peaks: list[WindowPeak] = []
    for i in idx:
        f = float(nu[i])
        if abs(f) < 1e-9:
            continue
        harm = int(round(abs(f) / DAILY_FREQ))
        is_harm = abs(abs(f) - harm * DAILY_FREQ) < 0.05 and harm >= 1
        peaks.append(WindowPeak(f, float(power[i]), harm if is_harm else None))
    peaks.sort(key=lambda p: -p.power)
    return peaks


def sampling_summary(t: np.ndarray, night_id: np.ndarray) -> dict:
    order = np.argsort(t)
    gaps = np.diff(t[order])
    groups: dict[str, list[float]] = {}
    for tt, nid in zip(t, night_id):
        groups.setdefault(str(nid), []).append(float(tt))
    starts = np.array([min(v) for v in groups.values()])
    night_gaps = np.diff(np.sort(starts))
    local = np.array([np.mean(v) % 1.0 for v in groups.values()])
    # 环形标准差（本地时刻在 [0,1) 上）
    ang = 2 * np.pi * local
    circ_std = float(np.sqrt(-2.0 * np.log(max(abs(np.mean(np.exp(1j * ang))), 1e-12))) / (2 * np.pi))
    return {
        "n_observations": int(t.size),
        "n_nights": len(groups),
        "time_span_day": float(t.max() - t.min()),
        "night_gap_day": {
            "min": float(night_gaps.min()),
            "max": float(night_gaps.max()),
            "mean": float(night_gaps.mean()),
            "near_one_day_fraction": float(np.mean(np.abs(night_gaps - 1.0) < 0.15)),
        },
        "nightly_local_time_mod1": {
            "mean": float(local.mean()),
            "min": float(local.min()),
            "max": float(local.max()),
            "circular_std_day": circ_std,
        },
        "adjacent_sample_gap_day": {
            "min": float(gaps.min()),
            "median": float(np.median(gaps)),
            "max": float(gaps.max()),
        },
    }


def alias_table(candidate_freqs: list[float], nu: np.ndarray, power: np.ndarray,
                rayleigh: float) -> list[dict]:
    """对每个候选频率列出 f ± k/day 的谱窗强度与最近候选，作为别名判断依据。"""
    def power_at(f: float) -> float:
        if f < nu[0] or f > nu[-1]:
            return 0.0
        return float(power[np.argmin(np.abs(nu - f))])

    table = []
    for f in candidate_freqs:
        entries = []
        for k in (-3, -2, -1, 1, 2, 3):
            fa = f + k * DAILY_FREQ
            nearest = min(candidate_freqs, key=lambda g: abs(g - fa))
            entries.append({
                "k_per_day": k,
                "alias_freq_per_day": fa,
                "window_power_at_offset_k": power_at(k * DAILY_FREQ),
                "nearest_candidate_freq_per_day": nearest,
                "nearest_candidate_match": bool(abs(nearest - fa) < rayleigh),
            })
        table.append({"freq_per_day": f, "daily_aliases": entries})
    return table


def assign_alias_families(candidates, rayleigh: float) -> dict:
    """按 |f_i - f_j - k/day| < 1 Rayleigh 把候选连成别名族。

    每族内 chi2 最小者标 primary_daily_family，其余标 daily_alias；
    无法与任何候选构成每日别名关系者标 standalone。
    """
    n = len(candidates)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    links = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = candidates[i].freq - candidates[j].freq
            k = int(round(diff / DAILY_FREQ))
            if k != 0 and abs(abs(diff) - abs(k) * DAILY_FREQ) < rayleigh:
                union(i, j)
                links.append({
                    "candidate_i": i, "candidate_j": j,
                    "freq_i": candidates[i].freq, "freq_j": candidates[j].freq,
                    "offset_per_day": diff, "daily_order": k,
                })

    families: dict[int, list[int]] = {}
    for i in range(n):
        families.setdefault(find(i), []).append(i)
    fam_id_by_root = {root: idx + 1 for idx, root in enumerate(sorted(families, key=lambda r: min(families[r])))}
    for root, members in families.items():
        members_sorted = sorted(members, key=lambda i: candidates[i].chi2)
        primary = members_sorted[0]

        def tag(c, label):
            if label not in c.labels:
                c.labels.append(label)

        for i in members_sorted:
            c = candidates[i]
            c.family_id = fam_id_by_root[root]
            if len(members) == 1:
                tag(c, "standalone")
            elif i == primary:
                tag(c, "primary_daily_family")
            else:
                tag(c, "daily_alias")
    return {"links": links, "n_families": len(families),
            "rayleigh_tolerance_per_day": rayleigh}
