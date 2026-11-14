"""地图报告：多面板 PNG（洼地、平坦区、无效洞）+ 小栅格终端箭头图。

matplotlib 使用 Agg 后端，全程离线。tiny_flat 这类小图额外输出
ASCII 箭头表，便于手工逐格核对平坦路由。
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .gridio import read_ascii

# 方向码 -> 终端箭头（北朝上）
ARROW = {1: "↑", 2: "↗", 3: "→", 4: "↘", 5: "↓", 6: "↙", 7: "←", 8: "↖"}
DIR_VECTORS = {  # 绘图向量（东向 x, 北向 y）
    1: (0.0, 1.0), 2: (1.0, 1.0), 3: (1.0, 0.0), 4: (1.0, -1.0),
    5: (0.0, -1.0), 6: (-1.0, -1.0), 7: (-1.0, 0.0), 8: (-1.0, 1.0),
}


def _load(outdir, name):
    values, valid, header = read_ascii(os.path.join(outdir, f"{name}.asc"))
    return values, valid, header


def text_arrow_map(outdir: str) -> str:
    """生成终端箭头表：流向 + 状态符号，写入 flow_arrows.txt 并返回。"""
    z, valid, h = _load(outdir, "dem_filled")
    direction, _, _ = _load(outdir, "flow_direction")
    status, _, _ = _load(outdir, "status")
    receiver, _, _ = _load(outdir, "receiver_id")
    nrows, ncols = valid.shape
    nd = h.nodata
    lines = [
        "流向箭头手查图（北朝上；↑N ↗NE →E ↘SE ↓S ↙SW ←W ↖NW）",
        "符号：箭头=内部接收格  X=边界出流  #=nodata  U=未解决封闭组件",
        f"网格 {nrows} 行 x {ncols} 列，cellsize={h.cellsize} m",
        "",
    ]
    for r in range(nrows):
        cells = []
        for c in range(ncols):
            if not valid[r, c]:
                cells.append("#")
            elif status[r, c] == 2:
                cells.append("U")
            elif receiver[r, c] < 0:
                cells.append("X")
            else:
                cells.append(ARROW[int(direction[r, c])])
        lines.append(" ".join(f"{ch:>2}" for ch in cells))
    lines.append("")
    lines.append("各格填后高程（z_filled）：")
    for r in range(nrows):
        vals = [
            "  # " if not valid[r, c]
            else ("  U " if status[r, c] == 2 else f"{z[r, c]:4.0f}")
            for c in range(ncols)
        ]
        lines.append(" ".join(vals))
    text = "\n".join(lines) + "\n"
    with open(os.path.join(outdir, "flow_arrows.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def render_map(outdir: str, png_name: str = "map_report.png") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    z, valid, h = _load(outdir, "dem_original")
    zf, _, _ = _load(outdir, "dem_filled")
    depth, _, _ = _load(outdir, "fill_depth")
    direction, _, _ = _load(outdir, "flow_direction")
    status, _, _ = _load(outdir, "status")
    flat_class, _, _ = _load(outdir, "flat_class")
    count, _, _ = _load(outdir, "accum_count")
    nd = h.nodata
    unresolved = status == 2.0  # 未解决封闭组件：除状态面板外显式留白

    extent = [
        h.xllcorner,
        h.xllcorner + h.ncols * h.cellsize,
        h.yllcorner,
        h.yllcorner + h.nrows * h.cellsize,
    ]

    def masked(a):
        return np.ma.masked_where(a == nd, a)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"DEM hydrology report  {h.nrows}x{h.ncols}  cellsize={h.cellsize} m  "
        f"SW corner=({h.xllcorner},{h.yllcorner})",
        fontsize=13,
    )

    # 1 原高程，无效洞用灰色覆盖
    ax = axes[0, 0]
    show = z.astype(float).copy()
    show[~valid] = np.nan
    im = ax.imshow(show, extent=extent, origin="upper", cmap="terrain")
    ax.imshow(np.where(~valid, 1.0, np.nan), extent=extent, origin="upper",
              cmap=ListedColormap(["0.35"]))
    ax.set_title("Original DEM (grey = nodata void)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 2 填后高程
    ax = axes[0, 1]
    im = ax.imshow(masked(zf), extent=extent, origin="upper", cmap="terrain")
    ax.set_title("Filled DEM (least-level spill surface)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 3 填深（洼地改动）；nodata 灰覆盖、未解决留白，均不参与色阶
    ax = axes[0, 2]
    dshow = depth.astype(float).copy()
    dshow[~valid | unresolved] = np.nan
    finite = dshow[np.isfinite(dshow)]
    im = ax.imshow(dshow, extent=extent, origin="upper", cmap="Blues",
                   vmin=0.0, vmax=max(float(finite.max()), 1e-9)
                   if finite.size else 1.0)
    ax.imshow(np.where(~valid, 1.0, np.nan), extent=extent, origin="upper",
              cmap=ListedColormap(["0.35"]))
    ax.set_title("Fill depth (m) — depression modification")
    if finite.size:
        fig.colorbar(im, ax=ax, fraction=0.046)

    # 4 状态：已解决/未解决/无效
    ax = axes[1, 0]
    cmap = ListedColormap(["#2ca02c", "#d62728", "#444444"])
    sshow = np.full(status.shape, 2.0)
    sshow[valid] = status[valid] - 1  # 1->0 resolved, 2->1 unresolved
    ax.imshow(sshow, extent=extent, origin="upper", cmap=cmap,
              vmin=0, vmax=2)
    ax.set_title("Status: green=resolved, red=unresolved, dark=void")

    # 5 平坦区分类 + D8 箭头 + 出流；未解决区域留白
    ax = axes[1, 1]
    base = np.full(flat_class.shape, np.nan)
    base[valid & ~unresolved] = 0.0
    base[flat_class == 1] = 1.0
    base[flat_class == 2] = 2.0
    ax.imshow(base, extent=extent, origin="upper",
              cmap=ListedColormap(["#c7e9c0", "#fec44f", "#fe9929"]))
    ax.imshow(np.where(~valid, 1.0, np.nan), extent=extent, origin="upper",
              cmap=ListedColormap(["0.35"]))
    nrows, ncols = valid.shape
    for r in range(nrows):
        for c in range(ncols):
            if not valid[r, c] or status[r, c] == 2:
                continue
            x = h.x_of_col(c)
            y = h.y_of_row(r)
            if direction[r, c] == 0:
                ax.plot(x, y, marker="s", color="red", markersize=5)
            else:
                dx, dy = DIR_VECTORS[int(direction[r, c])]
                ax.annotate("", xy=(x + dx * h.cellsize * 0.4,
                                    y + dy * h.cellsize * 0.4),
                            xytext=(x, y),
                            arrowprops=dict(arrowstyle="->", color="black",
                                            lw=0.8))
    ax.set_title("Flats (yellow=platter, orange=spill), D8 arrows, ■=exit")

    # 6 汇流累计；nodata 灰覆盖、未解决留白后再配色
    ax = axes[1, 2]
    cshow = count.astype(float).copy()
    cshow[~valid | unresolved] = np.nan
    im = ax.imshow(np.ma.masked_invalid(cshow), extent=extent,
                   origin="upper", cmap="viridis")
    ax.imshow(np.where(~valid, 1.0, np.nan), extent=extent, origin="upper",
              cmap=ListedColormap(["0.35"]))
    ax.set_title("Flow accumulation (upstream cells incl. self)")
    if np.isfinite(cshow).any():
        fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes.flat:
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(outdir, png_name)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成地图与终端箭头报告")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    png = render_map(args.outdir)
    text = text_arrow_map(args.outdir)
    print(f"[map] PNG -> {png}")
    print(f"[map] 箭头手查图 -> {os.path.join(args.outdir, 'flow_arrows.txt')}")
    print()
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
