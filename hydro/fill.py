"""连通组件标记与洼地填平（Priority Flood，Barnes et al. 2014 思路）。

规则（plan.json）：
- nodata 是封闭无效区，不能作为泄水出口，也不能被水流穿过。
- 有效单元按 8 邻接（与 D8 邻域一致）划分连通组件。
- 能连到有效外边界的组件：可解决，单元高程只升到“能够向外排水所需的
  最低高程”，即最小填洼面；原高程、填后高程、填深全部保留。
- 被 nodata 完全包围、没有任何有效外边界出口的组件：标记为未解决，
  不开洞、不凭空指定内部终点，填洼与流向均不对其声称成功。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

# 8 邻接偏移（行, 列），顺序即 flow 模块中固定的方向 1..8。
NEIGHBOR8 = (
    (-1, 0),   # 1 东? 见 flow 模块定义；此处只做连通与泛洪，方向语义无关
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


@dataclass
class FillResult:
    z: np.ndarray            # 原高程（无效位置为输入 nodata 值）
    valid: np.ndarray        # bool 掩膜
    comp: np.ndarray         # 组件编号 int32，无效为 -1
    comp_touches_boundary: np.ndarray  # 每个组件是否含有效外边界单元
    resolved: np.ndarray     # 可解决有效单元（组件接触外边界）
    unresolved: np.ndarray   # 未解决有效单元（封闭组件）
    filled: np.ndarray       # 填后高程；未解决单元保持原高程
    depth: np.ndarray        # 填深 = filled - z；未解决/无效为 0
    header: object


def label_components(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """8 邻接连通组件标记。

    返回 (组件编号数组[无效=-1], 各组件是否接触栅格外边界)。
    迭代 BFS，输入再小也不依赖 scipy。
    """
    nrows, ncols = valid.shape
    comp = np.full(valid.shape, -1, dtype=np.int32)
    touches: list[bool] = []
    next_id = 0
    for start in range(valid.size):
        if not valid.flat[start] or comp.flat[start] != -1:
            continue
        cid = next_id
        next_id += 1
        touches.append(False)
        stack = [start]
        comp.flat[start] = cid
        while stack:
            p = stack.pop()
            r, c = divmod(p, ncols)
            if r == 0 or r == nrows - 1 or c == 0 or c == ncols - 1:
                touches[cid] = True
            for dr, dc in NEIGHBOR8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < nrows and 0 <= cc < ncols:
                    q = rr * ncols + cc
                    if valid.flat[q] and comp.flat[q] == -1:
                        comp.flat[q] = cid
                        stack.append(q)
    return comp, np.array(touches, dtype=bool)


def priority_flood_fill(z, valid, resolved, header) -> tuple[np.ndarray, np.ndarray]:
    """在可解决组件上执行优先洪水，返回 (填后高程, 填深)。

    最小堆按 (当前水位, 线性编号) 排序；编号作为确定性并列裁决，
    保证同水位时处理顺序可复现。每个单元只入堆一次，首次被“漫到”
    时得到的水位就是它能排向外边界所需的最低填洼高程。
    """
    nrows, ncols = valid.shape
    n = valid.size
    filled = z.astype(np.float64, copy=True)
    pushed = np.zeros(n, dtype=bool)
    heap: list[tuple[float, int]] = []

    # 种子：所有可解决的外边界有效单元，以原高程入堆（图外无来水，
    # 外边界本身是合法出口，不需要抬高）。
    edge = np.zeros(n, dtype=bool)
    edge[:ncols] = True
    edge[-ncols:] = True
    edge[::ncols] = True
    edge[ncols - 1::ncols] = True
    seed_mask = resolved.reshape(-1) & edge
    for p in np.flatnonzero(seed_mask):
        heapq.heappush(heap, (float(z.flat[p]), int(p)))
        pushed[p] = True

    while heap:
        level, p = heapq.heappop(heap)
        r, c = divmod(p, ncols)
        for dr, dc in NEIGHBOR8:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < nrows and 0 <= cc < ncols):
                continue
            q = rr * ncols + cc
            if not resolved.flat[q] or pushed[q]:
                continue
            pushed[q] = True
            new_level = max(float(z.flat[q]), level)
            filled.flat[q] = new_level
            heapq.heappush(heap, (new_level, q))

    if not np.all(pushed[resolved.reshape(-1)]):
        missing = int(np.count_nonzero(resolved.reshape(-1) & ~pushed))
        raise RuntimeError(f"优先洪水未覆盖全部可解决单元，剩余 {missing} 个")

    depth = np.zeros_like(filled)
    depth[resolved] = filled[resolved] - z[resolved]
    # 数值上可能出现极小负零，统一截断。
    depth[depth < 0] = 0.0
    return filled, depth


def fill_depressions(z, valid, header) -> FillResult:
    """完整填洼流程：组件标记 -> 封闭组件识别 -> 可解决组件最小填平。"""
    comp, touches = label_components(valid)
    if touches.size == 0:  # 整幅都是 nodata：没有任何有效组件
        resolved = np.zeros_like(valid)
    else:
        resolved = valid & touches[comp]
    unresolved = valid & ~resolved
    filled, depth = priority_flood_fill(z, valid, resolved, header)
    return FillResult(
        z=z,
        valid=valid,
        comp=comp,
        comp_touches_boundary=touches,
        resolved=resolved,
        unresolved=unresolved,
        filled=filled,
        depth=depth,
        header=header,
    )
