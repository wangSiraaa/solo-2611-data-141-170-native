"""连接图与几何支撑分析。

两类问题严格分开报告：

1. 图连接（graph connectivity）：站与站之间是否通过共有标靶相连、能否回到锚站 A；
2. 几何支撑（geometric support）：即使连通，点集是完整 3D、共面还是共线，
   决定姿态的哪些自由度真正可识别。

"未连到 A" 与 "共线/近退化" 是不同性质的问题，输出中分别标注，
不允许用一个看起来很小的残差掩盖其中任何一种。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import support_report
from .io import Dataset, Observation


@dataclass
class StationSupport:
    station: str
    n_fit_targets: int
    n_holdout: int
    targets: list[str]
    report: dict
    connected_to_anchor: bool
    distance_hops: int | None          # 到锚站的站间跳数，不连通为 None


@dataclass
class Edge:
    station_i: str
    station_j: str
    shared_fit_targets: list[str]
    n_shared: int


@dataclass
class GraphReport:
    anchor: str
    supports: dict[str, StationSupport]
    edges: list[Edge]
    anchor_component: list[str]
    disconnected_stations: list[str]
    adjacency: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "anchor_station": self.anchor,
            "anchor_connected_component": self.anchor_component,
            "disconnected_stations": self.disconnected_stations,
            "stations": {
                s: {
                    "n_fit_targets": sup.n_fit_targets,
                    "n_holdout": sup.n_holdout,
                    "fit_targets": sup.targets,
                    "geometric_support": sup.report,
                    "connected_to_anchor": sup.connected_to_anchor,
                    "hops_to_anchor": sup.distance_hops,
                }
                for s, sup in sorted(self.supports.items())
            },
            "station_edges": [
                {
                    "station_i": e.station_i,
                    "station_j": e.station_j,
                    "shared_fit_targets": e.shared_fit_targets,
                    "n_shared": e.n_shared,
                }
                for e in self.edges
            ],
        }


def analyze_graph(ds: Dataset) -> GraphReport:
    fit = ds.fit_observations()

    # 站 -> 参与拟合的标靶集合
    station_targets: dict[str, set[str]] = {}
    station_obs: dict[str, list[Observation]] = {}
    for o in fit:
        station_targets.setdefault(o.station, set()).add(o.target)
        station_obs.setdefault(o.station, []).append(o)

    # 标靶 -> 观测到它的站（仅拟合观测）
    target_stations: dict[str, set[str]] = {}
    for o in fit:
        target_stations.setdefault(o.target, set()).add(o.station)

    stations = ds.stations
    adjacency: dict[str, set[str]] = {s: set() for s in stations}
    edges: list[Edge] = []
    for tg, st_set in sorted(target_stations.items()):
        ss = sorted(st_set)
        for a in range(len(ss)):
            for b in range(a + 1, len(ss)):
                if ss[b] not in adjacency[ss[a]]:
                    shared = sorted(
                        station_targets[ss[a]] & station_targets[ss[b]]
                    )
                    edges.append(Edge(ss[a], ss[b], shared, len(shared)))
                adjacency[ss[a]].add(ss[b])
                adjacency[ss[b]].add(ss[a])

    # BFS：锚站连通分量与跳数
    hops = {ds.anchor_station: 0}
    queue = [ds.anchor_station]
    while queue:
        cur = queue.pop(0)
        for nxt in sorted(adjacency[cur]):
            if nxt not in hops:
                hops[nxt] = hops[cur] + 1
                queue.append(nxt)
    anchor_component = sorted(hops)
    disconnected = sorted(s for s in stations if s not in hops)

    supports: dict[str, StationSupport] = {}
    holdout_count = {s: 0 for s in stations}
    for o in ds.holdout_observations():
        holdout_count[o.station] = holdout_count.get(o.station, 0) + 1

    for s in stations:
        obs_list = station_obs.get(s, [])
        pts = np.array([o.local for o in obs_list]) if obs_list else np.empty((0, 3))
        sig = np.array([o.sigma for o in obs_list]) if obs_list else None
        rep = support_report(pts, sig).as_dict()
        supports[s] = StationSupport(
            station=s,
            n_fit_targets=len(obs_list),
            n_holdout=holdout_count.get(s, 0),
            targets=sorted(o.target for o in obs_list),
            report=rep,
            connected_to_anchor=s in hops,
            distance_hops=hops.get(s),
        )

    edges.sort(key=lambda e: (e.station_i, e.station_j))
    return GraphReport(
        anchor=ds.anchor_station,
        supports=supports,
        edges=edges,
        anchor_component=anchor_component,
        disconnected_stations=disconnected,
        adjacency={s: sorted(v) for s, v in adjacency.items()},
    )
