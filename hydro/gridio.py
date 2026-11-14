"""ESRI ASCII Grid 读写与地理元数据。

ASC 头字段：
    ncols / nrows          列数 / 行数
    xllcorner / yllcorner  西南角（栅格外侧角点）坐标，米
    cellsize               固定方格边长，米
    NODATA_value          无效值（封闭无效区语义，见 plan.json）

同时兼容 xllcenter/yllcenter 写法（读入时换算成 corner）。
真实零高程是合法高程：只按 NODATA_value 判定无效，不能把 0 当空。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridHeader:
    """栅格地理元数据。行列与地理位置全部由此头定义。"""

    ncols: int
    nrows: int
    xllcorner: float
    yllcorner: float
    cellsize: float
    nodata: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nrows, self.ncols)

    def x_of_col(self, col: int) -> float:
        """列 col 的格心东向坐标（零基列，col 0 最西）。"""
        return self.xllcorner + (col + 0.5) * self.cellsize

    def y_of_row(self, row: int) -> float:
        """行 row 的格心北向坐标（零基行，row 0 最北）。"""
        return self.yllcorner + (self.nrows - row - 0.5) * self.cellsize


def read_ascii(path: str) -> tuple[np.ndarray, np.ndarray, GridHeader]:
    """读取 ASC，返回 (高程 float64 数组, 有效布尔掩膜, 头)。

    无效判定严格使用 NODATA_value；NaN 也视为无效以增强稳健性。
    """
    header_items: dict[str, str] = {}
    with open(path, "r", encoding="ascii") as fh:
        # 头固定 6 行，但按关键字逐行解析以容忍顺序差异。
        while len(header_items) < 6:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: 文件在完整头之前结束")
            parts = line.split()
            if len(parts) >= 2:
                header_items[parts[0].lower()] = parts[1]
        ncols = int(header_items["ncols"])
        nrows = int(header_items["nrows"])
        cellsize = float(header_items["cellsize"])
        nodata = float(header_items["nodata_value"])
        if "xllcorner" in header_items:
            xll = float(header_items["xllcorner"])
            yll = float(header_items["yllcorner"])
        else:  # 兼容 center 写法：换算为西南外侧角点
            xll = float(header_items["xllcenter"]) - 0.5 * cellsize
            yll = float(header_items["yllcenter"]) - 0.5 * cellsize
        values = np.loadtxt(fh, dtype=np.float64)
    if values.shape != (nrows, ncols):
        raise ValueError(
            f"{path}: 数据形状 {values.shape} 与头声明 ({nrows},{ncols}) 不一致"
        )
    valid = np.isfinite(values) & (values != nodata)
    header = GridHeader(ncols, nrows, xll, yll, cellsize, nodata)
    return values, valid, header


def _format_values(values: np.ndarray, nodata: float) -> np.ndarray:
    """写出前把无效位置统一替换成 nodata，其余保持原样。"""
    out = values.astype(np.float64, copy=True)
    out[~np.isfinite(out)] = nodata
    return out


def write_ascii(
    path: str,
    values: np.ndarray,
    header: GridHeader,
    *,
    integer: bool = False,
) -> None:
    """按原网格与 nodata 语义写出 ASC。

    integer=True 时按整数栅格写（方向、编号、掩膜、整数汇流计数等）。
    """
    if values.shape != header.shape:
        raise ValueError(f"{path}: 输出形状 {values.shape} 与头 {header.shape} 不符")
    out = _format_values(values, header.nodata)
    with open(path, "w", encoding="ascii") as fh:
        fh.write(f"ncols {header.ncols}\n")
        fh.write(f"nrows {header.nrows}\n")
        fh.write(f"xllcorner {_num(header.xllcorner)}\n")
        fh.write(f"yllcorner {_num(header.yllcorner)}\n")
        fh.write(f"cellsize {_num(header.cellsize)}\n")
        fh.write(f"NODATA_value {_num(header.nodata)}\n")
        if integer:
            np.savetxt(fh, out, fmt="%d")
        else:
            # 高程与面积：整数原样无小数，非整数保留 4 位小数，避免无谓噪声。
            np.savetxt(fh, out, fmt="%.4f")


def _num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.6g}"
