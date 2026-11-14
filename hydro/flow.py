"""单一 D8 流向与无环平坦区路由。

方向编码（栅格北朝上，固定公开）：
    1=N  2=NE  3=E  4=SE  5=S  6=SW  7=W  8=NW
相邻中心距离：正交 = cellsize，对角 = sqrt(2) * cellsize。

严格下坡规则
------------
对每个有效可解决单元，在 8 个有效（非 nodata、非未解决）邻居中找
严格更低者，坡度 = (z_self - z_nbr) / 中心距离，取最大坡度；
坡度并列时取最小方向码（上面的 1..8 顺序），结果完全确定、可复现。

平坦区规则（填洼后出现的同高程台面，含真实原生平地）
----------------------------------------------------
platter 单元 = 没有任何严格更低有效邻居的可解决单元（局部最低点台面）。
为每个 platter 单元计算两个 BFS 辅助距离（8 邻接、以格为跳数，仅在
同高程 platter 邻居间扩散）：
  dist_lower  到“泄水点”的跳数。泄水点两类：
              (a) 与同高程、自身有严格下坡邻居的非 platter 单元（spill）
                  相邻的 platter 单元；
              (b) 位于栅格外边界、可直接出流的 platter 单元。
  dist_higher 到“高地边缘”的跳数：与严格更高有效邻居相邻的 platter
              单元为 0 距离源；全域高台没有更高邻居时保持无穷大。

路由采用带父指针的多源 BFS（不是给同高邻居固定方向，也不微加高程）：
  * 种子（dist_lower=0）：若邻接 spill，直接指向 spill（并列取小编号）；
    否则作为外边界单元向图外出流。
  * 其余单元：接收格 = 它在 dist_lower BFS 中首次被发现时的父单元。
    父单元的 dist_lower 严格小 1，因此每条链距离严格递减、必终止于
    种子，结构上无环（不依赖运气或局部比较）。
  * 同层多个父候选时的公开并列裁决：dist_higher 较大者优先（远离高地
    边缘，水流不贴着台地坎边走），再并列取线性编号较小者。
dist_lower、dist_higher、父单元全部落盘，可逐格回查。

边界退出规则
------------
图外没有来水；有效外边界单元在没有栅格内接收格时允许出流，方向取
指向图外的最小方向码（角点在两个朝外方向中取小码），逐格记录在案。
nodata 与未解决单元永远不是接收格，也不允许“穿过”它们出流。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# (方向码 1..8, 行偏移, 列偏移, 相对中心距离系数)
DIR_TABLE = (
    (1, -1, 0, 1.0),
    (2, -1, 1, math.sqrt(2.0)),
    (3, 0, 1, 1.0),
    (4, 1, 1, math.sqrt(2.0)),
    (5, 1, 0, 1.0),
    (6, 1, -1, math.sqrt(2.0)),
    (7, 0, -1, 1.0),
    (8, -1, -1, math.sqrt(2.0)),
)
EXIT_DIR_OFFSET = 10          # 出流接收格编码 = -(EXIT_DIR_OFFSET + 方向码)
NO_RECEIVER = -1              # 无效/未解决单元的接收格编码
INF_DIST = np.iinfo(np.int32).max


@dataclass
class FlowResult:
    direction: np.ndarray       # int8：1..8 内部；0 出流；无接收处写 0
    receiver: np.ndarray        # int32：线性编号；出流为负码；无接收为 -1
    is_exit: np.ndarray         # bool：该单元向图外出流
    exit_direction: np.ndarray  # int8：出流方向码，否则 0
    platter: np.ndarray         # bool：平坦台面单元（无严格更低邻居）
    spill: np.ndarray           # bool：台面边缘、自身有严格下坡的同高程单元
    dist_lower: np.ndarray      # int32 BFS 距离证据，非 platter 为 INF
    dist_higher: np.ndarray     # int32 BFS 距离证据，无穷大为 INF
    bfs_parent: np.ndarray      # int32 平坦路由父单元，-1 表示非父指针路由
    records: list[dict] = field(default_factory=list)


def _neighbors(r, c, nrows, ncols):
    """生成 (方向码, nr, nc, 距离系数, 是否在栅格内)。"""
    for code, dr, dc, dist in DIR_TABLE:
        rr, cc = r + dr, c + dc
        inside = 0 <= rr < nrows and 0 <= cc < ncols
        yield code, rr, cc, dist, inside


def compute_flow(fill_result) -> FlowResult:
    z = fill_result.filled
    valid = fill_result.valid
    resolved = fill_result.resolved
    header = fill_result.header
    nrows, ncols = valid.shape
    n = nrows * ncols
    cell = float(header.cellsize)

    direction = np.zeros(n, dtype=np.int8)
    receiver = np.full(n, NO_RECEIVER, dtype=np.int32)
    is_exit = np.zeros(n, dtype=bool)
    exit_direction = np.zeros(n, dtype=np.int8)
    platter = np.zeros(n, dtype=bool)
    spill = np.zeros(n, dtype=bool)

    # ---- 第一步：严格下坡 D8；没有严格下坡的可解决单元标为 platter ----
    for p in range(n):
        if not resolved.flat[p]:
            continue
        r, c = divmod(p, ncols)
        best = None  # (-slope, 方向码, 邻居编号)
        for code, rr, cc, dist_coef, inside in _neighbors(r, c, nrows, ncols):
            if not inside:
                continue
            q = rr * ncols + cc
            if not resolved.flat[q]:
                continue  # nodata / 未解决封闭区：封闭，不可穿过
            drop = float(z.flat[p]) - float(z.flat[q])
            if drop > 0.0:
                slope = drop / (dist_coef * cell)
                key = (-slope, code, q)
                if best is None or key < best:
                    best = key
        if best is not None:
            _, code, q = best
            direction[p] = code
            receiver[p] = q
        else:
            platter[p] = True

    # ---- 第二步：标记 spill、两类种子与 dist_higher 源 ----
    dist_lower = np.full(n, INF_DIST, dtype=np.int32)
    dist_higher = np.full(n, INF_DIST, dtype=np.int32)
    spill_parent = np.full(n, NO_RECEIVER, dtype=np.int32)  # 种子 -> spill
    seeds: list[int] = []
    high_seeds: list[int] = []

    for p in np.flatnonzero(platter):
        p = int(p)
        r, c = divmod(p, ncols)
        on_edge = r == 0 or r == nrows - 1 or c == 0 or c == ncols - 1
        for code, rr, cc, _, inside in _neighbors(r, c, nrows, ncols):
            if not inside:
                continue
            q = rr * ncols + cc
            if not resolved.flat[q]:
                continue
            if float(z.flat[q]) == float(z.flat[p]):
                if not platter[q]:
                    spill[q] = True  # 同高程但自身能下坡：台面泄水点
                    if spill_parent[p] == NO_RECEIVER or q < spill_parent[p]:
                        spill_parent[p] = q
            elif float(z.flat[q]) > float(z.flat[p]):
                dist_higher[p] = 0  # 紧贴高地边缘
        if dist_higher[p] == 0:
            high_seeds.append(p)
        if spill_parent[p] != NO_RECEIVER or on_edge:
            dist_lower[p] = 0  # (a) 邻接 spill；(b) 边界直接出口
            seeds.append(p)

    # ---- 第三步：两个多源 BFS，同层按线性编号排序，过程可复现可回查 ----
    def bfs(seeds_in: list[int], dist: np.ndarray) -> None:
        dq = deque(sorted(seeds_in))
        enqueued = set(dq)
        while dq:
            p = dq.popleft()
            r, c = divmod(p, ncols)
            nxt = []
            for _, rr, cc, _, inside in _neighbors(r, c, nrows, ncols):
                if not inside:
                    continue
                q = rr * ncols + cc
                if (
                    platter[q]
                    and q not in enqueued
                    and float(z.flat[q]) == float(z.flat[p])
                ):
                    nxt.append(q)
            for q in sorted(nxt):
                enqueued.add(q)
                dist[q] = dist[p] + 1
                dq.append(q)

    bfs(seeds, dist_lower)
    bfs(high_seeds, dist_higher)

    orphan = np.flatnonzero(platter & (dist_lower == INF_DIST))
    if orphan.size:
        raise RuntimeError(
            f"平坦路由内部错误：{orphan.size} 个台面单元无法到达任何泄水点"
        )

    # ---- 第四步：dist_lower BFS 父指针（逐层在全部同层父候选间裁决）----
    # 并列规则：dist_higher 较大者优先（INF 视为最远、最优先），再取小编号。
    bfs_parent = np.full(n, NO_RECEIVER, dtype=np.int32)
    current = sorted(seeds)
    finalized = set(current)
    while current:
        children: dict[int, list[int]] = {}
        for p in current:
            r, c = divmod(p, ncols)
            for _, rr, cc, _, inside in _neighbors(r, c, nrows, ncols):
                if not inside:
                    continue
                q = rr * ncols + cc
                if (
                    platter[q]
                    and q not in finalized
                    and float(z.flat[q]) == float(z.flat[p])
                ):
                    children.setdefault(q, []).append(p)
        next_level: list[int] = []
        for q in sorted(children):  # 子单元编号排序，跨邻居展开顺序确定
            parents = children[q]
            parent = min(
                parents,
                key=lambda par: (
                    -int(dist_higher[par])
                    if dist_higher[par] != INF_DIST
                    else -INF_DIST,
                    par,
                ),
            )
            bfs_parent[q] = parent
            finalized.add(q)
            next_level.append(q)
        current = next_level

    # ---- 第五步：为每个 platter 单元落定接收格 ----
    for p in np.flatnonzero(platter):
        p = int(p)
        r, c = divmod(p, ncols)
        if spill_parent[p] != NO_RECEIVER:
            q = int(spill_parent[p])
            qr, qc = divmod(q, ncols)
            code = _code_between(r, c, qr, qc)
            direction[p], receiver[p] = code, q
        elif bfs_parent[p] != NO_RECEIVER:
            q = int(bfs_parent[p])
            qr, qc = divmod(q, ncols)
            code = _code_between(r, c, qr, qc)
            direction[p], receiver[p] = code, q
        else:
            # 无栅格内接收格：只有外边界单元允许出流
            if not (r == 0 or r == nrows - 1 or c == 0 or c == ncols - 1):
                raise RuntimeError(f"单元 {p} 无接收格且不在外边界，路由失败")
            outside_codes = [
                code
                for code, rr, cc, _, inside in _neighbors(r, c, nrows, ncols)
                if not inside
            ]
            code = min(outside_codes)  # 公开并列规则：最小朝外方向码
            direction[p] = 0
            receiver[p] = -(EXIT_DIR_OFFSET + code)
            is_exit[p] = True
            exit_direction[p] = code

    # ---- 第六步：逐格可回查记录 ----
    records: list[dict] = []
    for p in range(n):
        r, c = divmod(p, ncols)
        if not valid.flat[p]:
            kind = "void"
        elif not resolved.flat[p]:
            kind = "unresolved"
        elif is_exit[p]:
            kind = "exit"
        elif platter[p]:
            kind = "flat"
        else:
            kind = "slope"
        rec = {
            "row": r,
            "col": c,
            "kind": kind,
            "z_original": float(fill_result.z.flat[p]) if valid.flat[p] else np.nan,
            "z_filled": float(z.flat[p]) if valid.flat[p] else np.nan,
            "fill_depth": float(fill_result.depth.flat[p]) if valid.flat[p] else np.nan,
            "direction": int(direction[p]) if resolved.flat[p] else np.nan,
            "receiver": int(receiver[p]) if valid.flat[p] else np.nan,
            "dist_lower": int(dist_lower[p]) if platter[p] else np.nan,
            "dist_higher": (
                int(dist_higher[p])
                if platter[p] and dist_higher[p] != INF_DIST
                else np.nan
            ),
            "bfs_parent": int(bfs_parent[p]) if platter[p] else np.nan,
            "component": int(fill_result.comp.flat[p]),
        }
        if is_exit[p]:
            rec["receiver_row"] = "outside"
            rec["receiver_col"] = "outside"
            rec["exit_direction"] = int(exit_direction[p])
        elif resolved.flat[p]:
            qr, qc = divmod(int(receiver[p]), ncols)
            rec["receiver_row"] = qr
            rec["receiver_col"] = qc
            rec["exit_direction"] = 0
        else:
            rec["receiver_row"] = ""
            rec["receiver_col"] = ""
            rec["exit_direction"] = ""
        records.append(rec)

    return FlowResult(
        direction=direction.reshape(valid.shape),
        receiver=receiver.reshape(valid.shape),
        is_exit=is_exit.reshape(valid.shape),
        exit_direction=exit_direction.reshape(valid.shape),
        platter=platter.reshape(valid.shape),
        spill=spill.reshape(valid.shape),
        dist_lower=dist_lower.reshape(valid.shape),
        dist_higher=dist_higher.reshape(valid.shape),
        bfs_parent=bfs_parent.reshape(valid.shape),
        records=records,
    )


def _code_between(r: int, c: int, qr: int, qc: int) -> int:
    """由两个格心的行列差返回方向码。"""
    dr, dc = qr - r, qc - c
    for code, ddr, ddc, _ in DIR_TABLE:
        if dr == ddr and dc == ddc:
            return code
    raise ValueError(f"非 D8 相邻：({r},{c}) -> ({qr},{qc})")
