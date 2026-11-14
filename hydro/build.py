"""build：离线全流程编排与栅格/表格/守恒报告产物落盘。

用法：
    python -m hydro build --dem fixtures/dem.asc \
        --outlets fixtures/outlets.csv --outdir outputs/dem
    python -m hydro build --dem fixtures/tiny_flat.asc --outdir outputs/tiny
    python -m hydro build --dem cases/enclosed.asc --outdir outputs/enclosed
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from . import accumulation as accum_mod
from . import basins as basin_mod
from . import fill as fill_mod
from . import flow as flow_mod
from .gridio import GridHeader, read_ascii, write_ascii

INF_DIST = flow_mod.INF_DIST


def _int_grid(header: GridHeader, values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    out[~np.isfinite(out)] = header.nodata
    return out


def run_build(dem_path: str, outlet_path: str | None, outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    z, valid, header = read_ascii(dem_path)
    nrows, ncols = valid.shape

    # 1) 组件标记 + 最小填洼
    fr = fill_mod.fill_depressions(z, valid, header)

    # 2) D8 + 无环平坦路由
    fl = flow_mod.compute_flow(fr)

    # 3) 汇流累计
    ac = accum_mod.accumulate(fl, fr, header)

    # 4) 出口流域（可选；封闭组件场景允许不给出口）
    outlets = basin_mod.read_outlets(outlet_path) if outlet_path else []
    basin_results = [
        basin_mod.delineate_basin(o, fl.receiver, fr.resolved, header)
        for o in outlets
    ]

    # ---------------- 栅格产物（全部保留原网格与 nodata 语义）----------------
    write_ascii(os.path.join(outdir, "dem_original.asc"), z, header)
    filled_out = fr.filled.copy()
    filled_out[~valid] = header.nodata
    write_ascii(os.path.join(outdir, "dem_filled.asc"), filled_out, header)
    depth_out = fr.depth.copy()
    depth_out[~valid] = header.nodata
    write_ascii(os.path.join(outdir, "fill_depth.asc"), depth_out, header)

    status = np.full(valid.shape, header.nodata, dtype=np.float64)
    status[valid] = 1.0
    status[fr.unresolved] = 2.0
    write_ascii(os.path.join(outdir, "status.asc"), status, header, integer=True)

    comp_out = fr.comp.astype(np.float64)
    comp_out[~valid] = header.nodata
    write_ascii(os.path.join(outdir, "component.asc"), comp_out, header, integer=True)

    dir_out = fl.direction.astype(np.float64)
    dir_out[~fr.resolved] = header.nodata
    write_ascii(os.path.join(outdir, "flow_direction.asc"), dir_out, header, integer=True)

    recv_out = fl.receiver.astype(np.float64)
    recv_out[~valid] = header.nodata
    write_ascii(os.path.join(outdir, "receiver_id.asc"), recv_out, header, integer=True)

    write_ascii(os.path.join(outdir, "accum_count.asc"), ac.count, header, integer=True)
    write_ascii(os.path.join(outdir, "accum_area_m2.asc"), ac.area, header)

    flat_out = np.full(valid.shape, header.nodata, dtype=np.float64)
    flat_out[fr.resolved] = 0.0
    flat_out[fl.platter] = 1.0
    flat_out[fl.spill] = 2.0
    write_ascii(os.path.join(outdir, "flat_class.asc"), flat_out, header, integer=True)

    dl = fl.dist_lower.astype(np.float64)
    dl[~fl.platter] = header.nodata
    write_ascii(os.path.join(outdir, "flat_dist_lower.asc"), dl, header, integer=True)
    dh = fl.dist_higher.astype(np.float64)
    dh[~fl.platter] = header.nodata
    dh[fl.dist_higher == INF_DIST] = header.nodata
    write_ascii(os.path.join(outdir, "flat_dist_higher.asc"), dh, header, integer=True)
    bp = fl.bfs_parent.astype(np.float64)
    bp[~fl.platter] = header.nodata
    write_ascii(os.path.join(outdir, "flat_bfs_parent.asc"), bp, header, integer=True)

    for br in basin_results:
        b = np.full(valid.shape, header.nodata, dtype=np.float64)
        b[br.member] = 1.0
        write_ascii(os.path.join(outdir, f"basin_{br.outlet_id}.asc"), b, header,
                    integer=True)

    # ---------------- 逐格可回查记录 ----------------
    rec_fields = [
        "row", "col", "kind", "component", "z_original", "z_filled",
        "fill_depth", "direction", "receiver", "receiver_row", "receiver_col",
        "dist_lower", "dist_higher", "bfs_parent", "exit_direction",
        "x_center", "y_center", "accum_count", "accum_area_m2",
    ]
    with open(os.path.join(outdir, "flow_records.csv"), "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rec_fields)
        writer.writeheader()
        for rec in fl.records:
            p = rec["row"] * ncols + rec["col"]
            rec["x_center"] = header.x_of_col(rec["col"])
            rec["y_center"] = header.y_of_row(rec["row"])
            rec["accum_count"] = (
                int(ac.count.reshape(-1)[p])
                if fr.resolved.reshape(-1)[p]
                else ""
            )
            rec["accum_area_m2"] = (
                round(float(ac.area.reshape(-1)[p]), 4)
                if fr.resolved.reshape(-1)[p]
                else ""
            )
            writer.writerow(rec)

    # 边界退出清单（出流即流出研究区，外部无来水）
    exits = np.flatnonzero(fl.is_exit)
    with open(os.path.join(outdir, "boundary_exits.csv"), "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "row", "col", "linear_id", "exit_direction", "exit_direction_name",
            "z_filled", "accum_count", "accum_area_m2", "x_center", "y_center",
        ])
        dir_names = {1: "N", 2: "NE", 3: "E", 4: "SE", 5: "S", 6: "SW",
                     7: "W", 8: "NW"}
        for p in exits:
            r, c = divmod(int(p), ncols)
            writer.writerow([
                r, c, int(p), int(fl.exit_direction.reshape(-1)[p]),
                dir_names[int(fl.exit_direction.reshape(-1)[p])],
                round(float(fr.filled.reshape(-1)[p]), 4),
                int(ac.count.reshape(-1)[p]),
                round(float(ac.area.reshape(-1)[p]), 4),
                header.x_of_col(c), header.y_of_row(r),
            ])

    # 平坦路由辅助证据（只列 platter/spill 单元）
    with open(os.path.join(outdir, "flat_evidence.csv"), "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "row", "col", "role", "z_filled", "dist_lower", "dist_higher",
            "bfs_parent", "receiver", "receiver_row", "receiver_col",
        ])
        for p in np.flatnonzero(fl.platter | fl.spill):
            r, c = divmod(int(p), ncols)
            role = "spill" if fl.spill.reshape(-1)[p] else "platter"
            dh_v = fl.dist_higher.reshape(-1)[p]
            writer.writerow([
                r, c, role, round(float(fr.filled.reshape(-1)[p]), 4),
                int(fl.dist_lower.reshape(-1)[p]) if fl.platter.reshape(-1)[p]
                else "",
                int(dh_v) if fl.platter.reshape(-1)[p] and dh_v != INF_DIST
                else ("INF" if fl.platter.reshape(-1)[p] else ""),
                int(fl.bfs_parent.reshape(-1)[p])
                if fl.platter.reshape(-1)[p] else "",
                int(fl.receiver.reshape(-1)[p]),
                fl.records[p]["receiver_row"], fl.records[p]["receiver_col"],
            ])

    # 流域成员（嵌套/重叠允许，逐出口独立列出）
    with open(os.path.join(outdir, "basin_members.csv"), "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["outlet_id", "outlet_row", "outlet_col",
                         "member_row", "member_col", "linear_id"])
        for br in basin_results:
            for p in br.member_cells:
                r, c = divmod(int(p), ncols)
                writer.writerow([br.outlet_id, br.row, br.col, r, c, int(p)])

    # ---------------- 守恒报告 ----------------
    n_valid = int(np.count_nonzero(valid))
    n_resolved = int(np.count_nonzero(fr.resolved))
    n_unresolved = int(np.count_nonzero(fr.unresolved))
    n_void = int(np.count_nonzero(~valid))
    n_filled_cells = int(np.count_nonzero(fr.depth > 0.0))
    total_fill_volume = float(np.sum(fr.depth[fr.resolved]) * header.cellsize ** 2)
    exit_counts = [int(ac.count.reshape(-1)[p]) for p in ac.exit_cells]
    total_outflow = int(sum(exit_counts))
    conserved = total_outflow == n_resolved

    unresolved_components = [
        int(cid) for cid in range(fr.comp_touches_boundary.size)
        if not fr.comp_touches_boundary[cid]
    ]
    basin_summaries = [
        {
            "outlet_id": br.outlet_id,
            "outlet_row": br.row,
            "outlet_col": br.col,
            "cell_count": br.cell_count,
            "area_m2": br.area_m2,
        }
        for br in basin_results
    ]
    # 嵌套关系独立核对：两两集合包含
    for i, a in enumerate(basin_results):
        for b in basin_results[i + 1:]:
            inter = int(np.count_nonzero(a.member & b.member))
            if inter:
                basin_summaries[[x["outlet_id"] for x in basin_summaries]
                                .index(a.outlet_id)].setdefault(
                    "overlaps", []).append(
                    {"with": b.outlet_id, "shared_cells": inter})

    report = {
        "input": {
            "dem": os.path.abspath(dem_path),
            "outlets": os.path.abspath(outlet_path) if outlet_path else None,
            "nrows": nrows,
            "ncols": ncols,
            "cellsize_m": header.cellsize,
            "cell_area_m2": header.cellsize ** 2,
            "xllcorner": header.xllcorner,
            "yllcorner": header.yllcorner,
            "nodata_value": header.nodata,
        },
        "counts": {
            "total_cells": nrows * ncols,
            "void_cells": n_void,
            "valid_cells": n_valid,
            "resolved_cells": n_resolved,
            "unresolved_cells": n_unresolved,
            "platter_cells": int(np.count_nonzero(fl.platter)),
            "spill_cells": int(np.count_nonzero(fl.spill)),
            "depression_modified_cells": n_filled_cells,
            "boundary_exit_cells": int(ac.exit_cells.size),
        },
        "fill": {
            "max_depth_m": float(np.max(fr.depth)) if n_resolved else 0.0,
            "total_fill_volume_m3": total_fill_volume,
            "elevation_only_raised": bool(
                np.all(fr.filled[fr.resolved] >= fr.z[fr.resolved] - 1e-12)
            ),
        },
        "unresolved_components": unresolved_components,
        "unresolved_note": (
            "封闭组件：被 nodata 完全包围、无有效外边界出口，"
            "未开洞、未指定内部终点，不参与流向与累计。"
            if unresolved_components else "无"
        ),
        "conservation": {
            "unit_contribution_per_valid_resolved_cell": 1,
            "resolved_unit_total": n_resolved,
            "boundary_exit_count_values": exit_counts,
            "total_outflow_units": total_outflow,
            "conserved": conserved,
            "check": "总出流 = 各边界出口累计值之和；绝不把内部累计值相加。",
        },
        "basins": basin_summaries,
    }
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # 控制台简明摘要
    print(f"[build] {dem_path} -> {outdir}")
    print(f"  有效 {n_valid} | 可解决 {n_resolved} | 未解决 {n_unresolved} "
          f"| 无效 {n_void}")
    print(f"  填洼改动单元 {n_filled_cells}，最大填深 {report['fill']['max_depth_m']} m，"
          f"填方量 {total_fill_volume:.2f} m^3")
    print(f"  边界出口 {ac.exit_cells.size} 个，总出流 {total_outflow} 单位，"
          f"守恒：{conserved}")
    for br in basin_results:
        print(f"  流域 {br.outlet_id} @({br.row},{br.col})：{br.cell_count} 单元，"
              f"{br.area_m2:.1f} m^2")
    if unresolved_components:
        print(f"  未解决封闭组件编号：{unresolved_components}（未伪造排水）")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="小DEM填洼/D8/汇流/流域离线构建")
    parser.add_argument("--dem", required=True)
    parser.add_argument("--outlets", default=None)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    report = run_build(args.dem, args.outlets, args.outdir)
    return 0 if report["conservation"]["conserved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
