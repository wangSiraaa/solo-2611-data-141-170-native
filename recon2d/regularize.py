"""空间正则：公开的邻格平滑（一阶最近邻差分）与无支持格先验。

平滑项对每一对共享边的相邻单元 (i,j) 加一行 (e_i - e_j)，
即二维各向同性 TV 的二次替代 / Tikhonov 邻域平滑：
    R_smooth(x) = ||D x||^2
D 的每行是无量纲单位差分（未除以格宽；grid 为等宽 1m 时二者等价，
非等宽时在文档与诊断中按实际 dx/dy 说明，本工具的 fixtures 均为 1m）。

无射线单元的先验：选择矩阵 P 在无直接支持的单元上取单位行，
    R_prior(x) = ||w_prior P (x - x_prior)||^2
默认 w_prior=0，即不凭空锚定数值——无支持格的解完全来自平滑项的
“渗漏”，并在输出中单独标记；给定权重时先验值与权重同样显式记录。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def neighbor_difference_matrix(nx: int, ny: int) -> sparse.csr_matrix:
    """构建最近邻差分矩阵 D，行数 = 横向邻接数 + 纵向邻接数。"""
    n = nx * ny
    rows, cols, vals = [], [], []

    def add_edge(r: int, i: int, j: int) -> None:
        rows.extend((r, r))
        cols.extend((i, j))
        vals.extend((1.0, -1.0))

    edge = 0
    for row in range(ny):
        for col in range(nx):
            i = row * nx + col
            if col + 1 < nx:  # 横向（x 方向）邻接
                add_edge(edge, i, i + 1)
                edge += 1
            if row + 1 < ny:  # 纵向（y 方向）邻接
                add_edge(edge, i, i + nx)
                edge += 1
    D = sparse.coo_matrix((vals, (rows, cols)), shape=(edge, n),
                          dtype=float).tocsr()
    return D


def prior_selector(support_mask: np.ndarray) -> sparse.csr_matrix:
    """无直接射线支持单元的选择矩阵 P（每个无支持格一行单位行）。"""
    unsupported = np.flatnonzero(~support_mask)
    k = unsupported.size
    if k == 0:
        return sparse.csr_matrix((0, support_mask.size), dtype=float)
    rows = np.arange(k)
    P = sparse.coo_matrix(
        (np.ones(k), (rows, unsupported)),
        shape=(k, support_mask.size), dtype=float).tocsr()
    return P


def support_from_operator(A: sparse.csr_matrix) -> np.ndarray:
    """每格射线支持：该列上非零路径长度的射线条数。"""
    if A.shape[0] == 0:
        return np.zeros(A.shape[1], dtype=int)
    return np.asarray((A != 0).sum(axis=0)).ravel().astype(int)
