"""上游累计单元数与平方米汇水面积。

每个有效可解决单元贡献一个单位（含自身）；未解决封闭组件独立标注，
不参与可解决图的累计。面积 = 累计单元数 * cellsize^2（固定米制方格）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AccumResult:
    count: np.ndarray   # 上游累计单元数（含自身），无效/未解决处为 nodata
    area: np.ndarray    # 平方米汇水面积
    order: np.ndarray   # 后序拓扑编号（先上游后下游），供核验回查
    exit_cells: np.ndarray  # 出流单元线性编号


def accumulate(flow_result, fill_result, header) -> AccumResult:
    receiver = flow_result.receiver
    resolved = fill_result.resolved
    nrows, ncols = resolved.shape
    n = nrows * ncols
    recv_flat = receiver.reshape(-1)
    res_flat = resolved.reshape(-1)
    exit_flat = flow_result.is_exit.reshape(-1)

    count = np.full(n, header.nodata, dtype=np.float64)
    order = np.full(n, -1, dtype=np.int32)
    count[res_flat] = 1.0  # 自身贡献

    # 流向图是一片以“边界出流单元”为根的有向森林（每个单元至多一个
    # 接收格；平坦 BFS 父指针结构上无环）。先建反向邻接表，再从所有
    # 出流根做迭代 DFS 求后序：节点退出时把自身计数加给接收格，
    # 共享下游会被所有上游分支完整访问，不会漏加或重复。
    parents: list[list[int]] = [[] for _ in range(n)]
    for p in np.flatnonzero(res_flat):
        q = int(recv_flat[p])
        if q >= 0:
            if not res_flat[q]:
                raise RuntimeError(f"单元 {p} 的接收格 {q} 不是可解决单元")
            parents[q].append(int(p))

    roots = [int(p) for p in np.flatnonzero(res_flat) if recv_flat[p] < 0]
    if not np.any(res_flat):
        roots = []
    post_index = 0
    visited = np.zeros(n, dtype=bool)
    for root in roots:
        # (节点, 已展开) 栈；树无环，visited 仅用于防御性断言
        stack: list[tuple[int, bool]] = [(root, False)]
        while stack:
            p, expanded = stack.pop()
            if not expanded:
                if visited[p]:
                    raise RuntimeError(f"流向图存在环或重复访问：{p}")
                stack.append((p, True))
                for child in parents[p]:
                    stack.append((child, False))
            else:
                visited[p] = True
                order[p] = post_index
                post_index += 1
                q = int(recv_flat[p])
                if q >= 0:  # 出流根的计数不再加入任何栅格内单元
                    count[q] += count[p]

    if np.any(~visited[res_flat]):
        raise RuntimeError("存在未完成遍历的可解决单元（流向图可能有环）")

    area = np.full(n, header.nodata, dtype=np.float64)
    area[res_flat] = count[res_flat] * float(header.cellsize) ** 2

    return AccumResult(
        count=count.reshape(nrows, ncols),
        area=area.reshape(nrows, ncols),
        order=order.reshape(nrows, ncols),
        exit_cells=np.flatnonzero(exit_flat & res_flat),
    )
