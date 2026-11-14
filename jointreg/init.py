"""可解释的初始化：锚站为世界基准，沿支撑最强的生成树逐站加权 Kabsch 挂载。

为什么不是"每站独立对 A 求变换后平均"：

* 有些站（如数据中的 D）与 A 没有直接共有标靶，只能经其他站间接挂到世界系；
* 闭环边在初始化阶段不参与姿态求解，只留给联合调整阶段做闭合，
  从而能显式比较初始化闭环误差与联合调整后的闭环一致性；
* 共线站的 Kabsch 结果不唯一，初始化只给出一个代表姿态并明确标注，
  真正的可识别性结论由后续分析给出。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import weighted_kabsch
from .graph import GraphReport
from .io import Dataset


@dataclass
class InitRecord:
    station: str
    parent: str | None
    shared_targets: list[str]
    kabsch_rms_m: float | None
    reflection_corrected: bool
    rotational_rank: int | None
    note: str


@dataclass
class Initialization:
    R: dict[str, np.ndarray]
    t: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    target_weight: dict[str, float]          # 累积的 1/sigma^2 权重
    records: list[InitRecord]
    spanning_tree: list[tuple[str, str]]


def _edge_strength(graph: GraphReport, a: str, b: str) -> tuple:
    """BFS 出边排序键：共有标靶多、支撑秩高者优先。"""
    shared = set(graph.supports[a].targets) & set(graph.supports[b].targets)
    rank_b = graph.supports[b].report["exact_rank"]
    return (len(shared), rank_b)


def initialize(ds: Dataset, graph: GraphReport) -> Initialization:
    R: dict[str, np.ndarray] = {}
    t: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    tw: dict[str, float] = {}
    records: list[InitRecord] = []
    tree: list[tuple[str, str]] = []

    # ---- 锚站固定为世界基准：R=I, t=0，锚站观测直接给出首批世界标靶 ----
    anchor = ds.anchor_station
    R[anchor] = np.eye(3)
    t[anchor] = np.zeros(3)
    records.append(
        InitRecord(anchor, None, [], 0.0, False, 3, "锚站，固定 R=I, t=0 作为世界基准")
    )

    def absorb_station_obs(station: str) -> None:
        """把该站全部拟合观测按当前 R,t 投影为世界标靶估计（加权累积）。"""
        for o in ds.by_station(station):
            if o.holdout:
                continue
            wpt = R[station] @ o.local + t[station]
            w = 1.0 / o.sigma**2
            if o.target in targets:
                acc = targets[o.target] * tw[o.target] + wpt * w
                tw[o.target] += w
                targets[o.target] = acc / tw[o.target]
            else:
                targets[o.target] = wpt.copy()
                tw[o.target] = w

    absorb_station_obs(anchor)

    # ---- BFS 生成树：每次从已挂载站集合中选支撑最强的跨集边扩展 ----
    placed = {anchor}
    remaining = set(graph.anchor_component) - placed
    while remaining:
        candidates = []
        for a in sorted(placed):
            for b in graph.adjacency[a]:
                if b in remaining:
                    candidates.append((_edge_strength(graph, a, b), a, b))
        if not candidates:
            break  # 理论上不会发生（连通分量内部必有边）
        candidates.sort(reverse=True)
        (_, _), parent, child = candidates[0]

        shared = sorted(
            set(graph.supports[parent].targets)
            & set(graph.supports[child].targets)
        )
        local_pts, world_pts, sigs = [], [], []
        obs_by_target = {o.target: o for o in ds.by_station(child)}
        for tg in shared:
            local_pts.append(obs_by_target[tg].local)
            world_pts.append(targets[tg])
            sigs.append(obs_by_target[tg].sigma)
        kab = weighted_kabsch(
            np.array(local_pts), np.array(world_pts), np.array(sigs)
        )
        R[child], t[child] = kab.R, kab.t
        tree.append((parent, child))
        rank_note = {
            3: "完整 3D 支撑，Kabsch 姿态唯一",
            2: "共面支撑，平面内姿态唯一（法向由右手定则确定）",
            1: "共线支撑：绕直线轴旋转不可识别，此处仅取 SVD 代表姿态",
            0: "单点支撑：旋转完全不可识别，仅平移有效",
        }[kab.rotational_rank]
        refl_note = "；无约束最优为反射，已强制翻回 det=+1" if kab.reflection_corrected else ""
        records.append(
            InitRecord(
                child, parent, shared, kab.rms_m,
                kab.reflection_corrected, kab.rotational_rank,
                f"经 {parent} 挂载，共有标靶 {len(shared)} 个；{rank_note}{refl_note}",
            )
        )
        absorb_station_obs(child)
        placed.add(child)
        remaining.discard(child)

    # ---- 与锚站不连通的分量：取分量内排序最小的站作局部锚（R=I,t=0），
    #      同样走生成树；其世界系与 A 世界系之间的 6 个刚体自由度不可识别 ----
    for dc in graph.disconnected_stations:
        if dc in R:
            continue
        component = {dc}
        frontier = [dc]
        while frontier:
            cur = frontier.pop()
            for nxt in graph.adjacency[cur]:
                if nxt not in component and nxt not in placed:
                    component.add(nxt)
                    frontier.append(nxt)
        local_anchor = sorted(component)[0]
        R[local_anchor] = np.eye(3)
        t[local_anchor] = np.zeros(3)
        records.append(
            InitRecord(
                local_anchor, None, [], None, False, None,
                f"与锚站 {anchor} 不连通，分量内临时锚（局部坐标），"
                "与 A 世界系的 6 个刚体自由度不可识别",
            )
        )
        absorb_station_obs(local_anchor)
        sub_placed = {local_anchor}
        sub_remaining = component - sub_placed
        while sub_remaining:
            cands = []
            for a in sorted(sub_placed):
                for b in graph.adjacency[a]:
                    if b in sub_remaining:
                        cands.append((_edge_strength(graph, a, b), a, b))
            cands.sort(reverse=True)
            (_, _), parent, child = cands[0]
            shared = sorted(
                set(graph.supports[parent].targets)
                & set(graph.supports[child].targets)
            )
            obs_by_target = {o.target: o for o in ds.by_station(child)}
            local_pts = np.array([obs_by_target[tg].local for tg in shared])
            world_pts = np.array([targets[tg] for tg in shared])
            sigs = np.array([obs_by_target[tg].sigma for tg in shared])
            kab = weighted_kabsch(local_pts, world_pts, sigs)
            R[child], t[child] = kab.R, kab.t
            tree.append((parent, child))
            records.append(
                InitRecord(
                    child, parent, shared, kab.rms_m,
                    kab.reflection_corrected, kab.rotational_rank,
                    f"断开分量内经 {parent} 挂载，共有标靶 {len(shared)} 个",
                )
            )
            absorb_station_obs(child)
            sub_placed.add(child)
            sub_remaining.discard(child)
        placed |= component

    # 健全性检查：所有站都应有姿态
    for s in ds.stations:
        assert s in R and s in t, f"站 {s} 初始化失败"
    return Initialization(R, t, targets, tw, records, tree)
