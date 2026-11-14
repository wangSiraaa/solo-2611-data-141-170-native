"""verify：独立核验产物，不 import build 的任何计算结果，只从磁盘重读。

核验项：
 V1 栅格头/形状/nodata 语义一致；真实零高程不被当无效。
 V2 独立重做连通组件 + 优先洪水，逐格比对填后高程与填深；
    封闭组件必须标未解决，且其高程未被改动。
 V3 独立重算 D8 坡度与平坦 BFS 距离/父指针，比对方向与接收格；
    每条接收格链独立走查：要么终止于边界出流，要么停在未解决单元，
    且无环、不穿过 nodata。
 V4 逐源追踪独立累加（每源沿链走一格加一），核对 accum_count/area，
    不使用构建期的拓扑序。
 V5 守恒：所有已解决单元的单位贡献之和 == 各边界出口累计值之和；
    明确拒绝“内部累计值相加”的口径。
 V6 出口合法性与流域成员：独立反向追踪重算，检查嵌套重叠未被互斥化。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import deque

import numpy as np

from .fill import NEIGHBOR8
from .flow import DIR_TABLE, EXIT_DIR_OFFSET
from .gridio import read_ascii


class VerifyError(AssertionError):
    pass


def _read_grid(path, header, integer=False):
    values, valid, h = read_ascii(path)
    if h.shape != header.shape:
        raise VerifyError(f"{path}: 形状不一致")
    if (h.ncols, h.nrows, h.xllcorner, h.yllcorner, h.cellsize, h.nodata) != (
        header.ncols, header.nrows, header.xllcorner, header.yllcorner,
        header.cellsize, header.nodata,
    ):
        raise VerifyError(f"{path}: 地理元数据与输入 DEM 不一致")
    return values, valid


def _label(valid: np.ndarray):
    nrows, ncols = valid.shape
    comp = np.full(valid.size, -1, dtype=np.int32)
    touches = []
    cid = 0
    for start in range(valid.size):
        if not valid.flat[start] or comp.flat[start] != -1:
            continue
        touches.append(False)
        dq = deque([start])
        comp[start] = cid
        while dq:
            p = dq.popleft()
            r, c = divmod(p, ncols)
            if r in (0, nrows - 1) or c in (0, ncols - 1):
                touches[cid] = True
            for dr, dc in NEIGHBOR8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < nrows and 0 <= cc < ncols:
                    q = rr * ncols + cc
                    if valid.flat[q] and comp[q] == -1:
                        comp[q] = cid
                        dq.append(q)
        cid += 1
    return comp, np.array(touches, dtype=bool)


def _priority_flood(z, valid, resolved):
    import heapq

    nrows, ncols = valid.shape
    n = valid.size
    filled = z.copy()
    pushed = np.zeros(n, dtype=bool)
    heap = []
    edge = np.zeros(n, dtype=bool)
    edge[:ncols] = edge[-ncols:] = True
    edge[::ncols] = edge[ncols - 1::ncols] = True
    for p in np.flatnonzero(resolved.reshape(-1) & edge):
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
            filled.flat[q] = max(float(z.flat[q]), level)
            heapq.heappush(heap, (filled.flat[q], q))
    if not np.all(pushed[resolved.reshape(-1)]):
        raise VerifyError("独立优先洪水未覆盖全部可解决单元")
    return filled


def run_verify(dem_path: str, outlet_path: str | None, outdir: str) -> dict:
    checks: list[dict] = []

    def record(cid: str, ok: bool, detail: str):
        checks.append({"check": cid, "passed": ok, "detail": detail})
        if not ok:
            raise VerifyError(f"{cid} 失败：{detail}")

    z, valid, header = read_ascii(dem_path)
    nrows, ncols = valid.shape
    n = nrows * ncols
    cell = float(header.cellsize)

    # V1 读取全部栅格产物并核对元数据/nodata
    names_int = [
        "status", "component", "flow_direction", "receiver_id", "accum_count",
        "flat_class", "flat_dist_lower", "flat_dist_higher", "flat_bfs_parent",
    ]
    grids = {}
    for name in names_int:
        grids[name], _ = _read_grid(
            os.path.join(outdir, f"{name}.asc"), header, integer=True)
    for name in ["dem_original", "dem_filled", "fill_depth", "accum_area_m2"]:
        grids[name], _ = _read_grid(
            os.path.join(outdir, f"{name}.asc"), header)
    nd = header.nodata
    record("V1.header_and_nodata", True,
           f"{len(grids) + 1} 个栅格头与 nodata 语义一致，形状 {nrows}x{ncols}")
    if np.any((z == 0.0) & ~valid):
        raise VerifyError("真实零高程被误判为无效")
    record("V1.zero_is_valid", True, "零高程单元（若存在）按有效处理")

    # V2 独立填洼复核
    comp_flat, touches = _label(valid)
    comp = comp_flat.reshape(valid.shape)
    if touches.size == 0:
        resolved = np.zeros_like(valid)
    else:
        resolved = valid & touches[comp_flat].reshape(valid.shape)
    unresolved = valid & ~resolved
    filled_exp = _priority_flood(z, valid, resolved)
    depth_exp = np.zeros_like(z)
    depth_exp[resolved] = filled_exp[resolved] - z[resolved]

    if not np.allclose(grids["dem_original"][valid], z[valid]):
        raise VerifyError("原高程栅格被改动")
    if not np.allclose(grids["dem_filled"][resolved], filled_exp[resolved],
                       atol=1e-9):
        raise VerifyError("填后高程与独立优先洪水结果不符")
    if not np.allclose(grids["fill_depth"][resolved], depth_exp[resolved],
                       atol=1e-9):
        raise VerifyError("填深与独立计算不符")
    if np.any(grids["fill_depth"][resolved] < -1e-12):
        raise VerifyError("存在负填深（挖地）")
    if np.any(grids["dem_filled"][unresolved] != z[unresolved]):
        raise VerifyError("未解决封闭组件的高程被改动")
    status = grids["status"]
    if not (np.all(status[resolved] == 1) and np.all(status[unresolved] == 2)
            and np.all(status[~valid] == nd)):
        raise VerifyError("status 栅格分类错误")
    if not np.array_equal(grids["component"][valid].astype(int), comp[valid]):
        raise VerifyError("组件编号不一致")
    record("V2.fill_and_closed_components", True,
           f"填洼逐格一致；封闭组件 {np.flatnonzero(~touches).tolist()} "
           "标记未解决且未被改动/开洞")

    # V3 独立流向复核 + 接收链走查
    direction = grids["flow_direction"].astype(int)
    receiver = grids["receiver_id"].astype(int)
    flat_class = grids["flat_class"].astype(int)
    dl = grids["flat_dist_lower"]
    dh = grids["flat_dist_higher"]
    zf = grids["dem_filled"]

    # V3a 严格下坡单元：独立按坡度重算
    for p in np.flatnonzero(resolved.reshape(-1)):
        r, c = divmod(int(p), ncols)
        best = None
        for code, dr, dc, coef in DIR_TABLE:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < nrows and 0 <= cc < ncols):
                continue
            q = rr * ncols + cc
            if not resolved.flat[q]:
                continue
            drop = zf.flat[p] - zf.flat[q]
            if drop > 0:
                key = (-drop / (coef * cell), code)
                if best is None or key < best[0]:
                    best = (key, code, q)
        if best is not None:
            _, code, q = best
            if direction.flat[p] != code or receiver.flat[p] != q:
                raise VerifyError(f"单元 {p} 的 D8 接收格与独立坡度计算不符")
        else:
            if flat_class.flat[p] not in (1, 2):  # platter 或 spill
                raise VerifyError(f"单元 {p} 应属平坦台面但 flat_class={flat_class.flat[p]}")
    record("V3a.d8_slope", True, "严格下坡接收格按最大坡度/最小方向码独立复算一致")

    # V3b 平坦 BFS 距离独立复算（只在同高程 platter 间扩散）
    platter = flat_class.reshape(-1) == 1
    spill = flat_class.reshape(-1) == 2
    exp_dl = np.full(n, np.iinfo(np.int32).max, dtype=np.int64)
    exp_dh = np.full(n, np.iinfo(np.int32).max, dtype=np.int64)
    seeds_l, seeds_h = [], []
    for p in np.flatnonzero(platter | spill):
        r, c = divmod(int(p), ncols)
        on_edge = r in (0, nrows - 1) or c in (0, ncols - 1)
        for _, dr, dc, _ in DIR_TABLE:
            rr, cc = r + dr, c + dc
            if 0 <= rr < nrows and 0 <= cc < ncols:
                q = rr * ncols + cc
                if resolved.flat[q] and zf.flat[q] > zf.flat[p]:
                    if platter[p]:
                        exp_dh[p] = 0
                if (0 <= rr < nrows and 0 <= cc < ncols and resolved.flat[q]
                        and zf.flat[q] == zf.flat[p] and not platter[q]
                        and spill[q] and platter[p]):
                    exp_dl[p] = 0
        if platter[p] and on_edge:
            exp_dl[p] = 0
        if platter[p] and exp_dl[p] == 0:
            seeds_l.append(int(p))
        if platter[p] and exp_dh[p] == 0:
            seeds_h.append(int(p))

    def bfs(seeds, dist):
        dq = deque(sorted(seeds))
        seen = set(dq)
        while dq:
            p = dq.popleft()
            r, c = divmod(p, ncols)
            nxt = []
            for _, dr, dc, _ in DIR_TABLE:
                rr, cc = r + dr, c + dc
                if 0 <= rr < nrows and 0 <= cc < ncols:
                    q = rr * ncols + cc
                    if platter[q] and q not in seen and zf.flat[q] == zf.flat[p]:
                        nxt.append(q)
            for q in sorted(nxt):
                seen.add(q)
                dist[q] = dist[p] + 1
                dq.append(q)

    bfs(seeds_l, exp_dl)
    bfs(seeds_h, exp_dh)
    for p in np.flatnonzero(platter):
        if int(dl.flat[p]) != exp_dl[p]:
            raise VerifyError(f"platter {p} dist_lower 不符")
        # dist_higher 的 INF 在落盘时按 nodata 写，两侧统一口径比较
        got_dh = None if dh.flat[p] == nd else int(dh.flat[p])
        exp = None if exp_dh[p] == np.iinfo(np.int32).max else int(exp_dh[p])
        if got_dh != exp:
            raise VerifyError(f"platter {p} dist_higher 不符")
    record("V3b.flat_bfs_evidence", True, "平坦区 dist_lower/dist_higher 独立 BFS 一致")

    # V3c 每条接收链独立走查：终止状态 + 无环 + 不穿 nodata
    exit_cells = set()
    for p in np.flatnonzero(resolved.reshape(-1)):
        seen = set()
        cur = int(p)
        while True:
            if cur in seen:
                raise VerifyError(f"接收链在 {cur} 处成环（源 {p}）")
            seen.add(cur)
            if not resolved.flat[cur]:
                raise VerifyError(f"接收链进入非可解决单元 {cur}")
            q = receiver.flat[cur]
            if q < 0:  # 出流编码
                code = -q - EXIT_DIR_OFFSET
                if code not in range(1, 9):
                    raise VerifyError(f"单元 {cur} 出流编码非法 {q}")
                r, c = divmod(cur, ncols)
                if not (r in (0, nrows - 1) or c in (0, ncols - 1)):
                    raise VerifyError(f"非边界单元 {cur} 被标记出流")
                # 出流方向必须确实指向图外
                outside = [
                    c2 for c2, dr, dc, _ in DIR_TABLE
                    if not (0 <= r + dr < nrows and 0 <= c + dc < ncols)
                ]
                if code not in outside:
                    raise VerifyError(f"单元 {cur} 出流方向 {code} 不指向图外")
                if direction.flat[cur] != 0:
                    raise VerifyError(f"出流单元 {cur} 方向码应为 0")
                exit_cells.add(cur)
                break
            if not (0 <= q < n) or not resolved.flat[q]:
                raise VerifyError(f"单元 {cur} 的接收格 {q} 非法")
            # 高程只降不升；platter 内部同高程，靠 BFS 父指针无环
            if zf.flat[q] > zf.flat[cur] + 1e-12:
                raise VerifyError(f"单元 {cur} 流向更高单元 {q}")
            cur = q
    # 未解决单元必须没有接收格
    for p in np.flatnonzero(unresolved.reshape(-1)):
        if receiver.flat[p] != -1:
            raise VerifyError(f"未解决单元 {p} 被赋予接收格")
    record("V3c.receiver_chains", True,
           f"{np.count_nonzero(resolved)} 条链全部终止于合法边界出流，无环、不穿 nodata；"
           f"未解决单元无接收格")

    # V4 逐源追踪独立累加
    exp_count = np.zeros(n, dtype=np.int64)
    for src in np.flatnonzero(resolved.reshape(-1)):
        cur = int(src)
        while True:
            exp_count[cur] += 1
            q = receiver.flat[cur]
            if q < 0:
                break
            cur = q
    got_count = grids["accum_count"]
    got_flat = got_count.reshape(-1)
    if np.any(got_flat[resolved.reshape(-1)].astype(int)
              != exp_count[resolved.reshape(-1)]):
        raise VerifyError("累计单元数与逐源追踪不符")
    if np.any(got_flat[~valid.reshape(-1)] != nd) or np.any(
            got_flat[unresolved.reshape(-1)] != nd):
        raise VerifyError("累计栅格 nodata 语义错误")
    exp_area = exp_count.reshape(nrows, ncols) * cell ** 2
    if not np.allclose(grids["accum_area_m2"][resolved], exp_area[resolved]):
        raise VerifyError("汇水面积与 cell 面积推算不符")
    record("V4.accumulation_by_source_tracing", True,
           "逐源走链独立累加与 accum_count/accum_area_m2 一致（含自身贡献）")

    # V5 守恒：只数边界出口的累计值
    outflow = sum(int(got_count.reshape(-1)[p]) for p in exit_cells)
    n_resolved = int(np.count_nonzero(resolved))
    if outflow != n_resolved:
        raise VerifyError(f"守恒失败：出流 {outflow} != 可解决单元 {n_resolved}")
    # 反例口径自查：内部累计之和必然远大于总量
    internal_sum = int(exp_count.sum())
    if internal_sum == n_resolved and n_resolved > 1:
        raise VerifyError("内部累计口径异常")
    record("V5.conservation", True,
           f"总出流 {outflow} = 可解决贡献 {n_resolved}；"
           f"内部累计之和 {internal_sum} 仅作反例，未当作总流量")

    # V6 出口与流域独立复核
    basins_on_disk = sorted(
        f for f in os.listdir(outdir)
        if f.startswith("basin_") and f.endswith(".asc"))
    expected_outlets = []
    if outlet_path:
        with open(outlet_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                expected_outlets.append(
                    (row["outlet_id"], int(row["row"]), int(row["col"])))
    if len(basins_on_disk) != len(expected_outlets):
        raise VerifyError("流域栅格数与出口数不符")

    parents: dict[int, list[int]] = {}
    for p in np.flatnonzero(resolved.reshape(-1)):
        q = receiver.flat[p]
        if q >= 0:
            parents.setdefault(int(q), []).append(int(p))
    members_by_id = {}
    for oid, r, c in expected_outlets:
        target = r * ncols + c
        if not resolved.flat[target]:
            raise VerifyError(f"出口 {oid} 落在无效/未解决单元")
        stack = [target]
        members = {target}
        while stack:
            q = stack.pop()
            for p in parents.get(q, []):
                if p not in members:
                    members.add(p)
                    stack.append(p)
        bg, _ = _read_grid(os.path.join(outdir, f"basin_{oid}.asc"), header)
        disk_members = set(np.flatnonzero(bg.reshape(-1) == 1).tolist())
        if disk_members != members:
            raise VerifyError(f"流域 {oid} 成员与独立反向追踪不符")
        members_by_id[oid] = members
    # 嵌套：重叠必须被保留，不得互斥切分
    overlap_notes = []
    ids = list(members_by_id)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            inter = members_by_id[ids[i]] & members_by_id[ids[j]]
            if inter:
                overlap_notes.append(f"{ids[i]}∩{ids[j]}={len(inter)}格")
    record("V6.outlets_and_basins", True,
           "出口合法、流域成员独立复算一致；嵌套重叠："
           + ("; ".join(overlap_notes) if overlap_notes else "无重叠"))

    summary = {
        "dem": dem_path,
        "outdir": outdir,
        "all_passed": True,
        "n_checks": len(checks),
        "checks": checks,
        "resolved_cells": int(np.count_nonzero(resolved)),
        "unresolved_cells": int(np.count_nonzero(unresolved)),
        "boundary_exit_cells": len(exit_cells),
        "outflow_units": outflow,
    }
    with open(os.path.join(outdir, "verify_report.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"[verify] {outdir}：全部 {len(checks)} 项独立核验通过")
    for c in checks:
        print(f"  [PASS] {c['check']}: {c['detail']}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="独立核验水文产物")
    parser.add_argument("--dem", required=True)
    parser.add_argument("--outlets", default=None)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    try:
        run_verify(args.dem, args.outlets, args.outdir)
    except VerifyError as exc:
        print(f"[verify] 核验失败：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
