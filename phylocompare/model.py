"""JC69 模型、位点模式压缩与对数域 Felsenstein 剪枝。

数值策略：全程在对数空间保存部分似然（log-partials），
组合用 ``logsumexp``，因此重复位点再多也不会下溢，
不需要对位点做额外的重标定。N（编码 4）在四个潜隐碱基下的
对数发射似然都是 0，即发射似然为 1。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .fasta import Alignment, UNKNOWN
from .tree import UnrootedTree

N_BASES = 4
_LOG_FREQ = np.log(0.25)  # JC69 根状态均匀，且为平稳频率


def jc69_log_matrix(t: float) -> np.ndarray:
    """返回枝长 t 下的 4x4 对数转移概率矩阵。

    同态: 1/4 + 3/4 exp(-4t/3)；异态: 1/4 - 1/4 exp(-4t/3)。
    t 被压到非负；t=0 时退化为单位矩阵（log 1 = 0, log 0 = -inf）。
    """
    t = max(float(t), 0.0)
    diag_prob = 0.25 + 0.75 * np.exp(-4.0 * t / 3.0)
    off_prob = 0.25 - 0.25 * np.exp(-4.0 * t / 3.0)
    mat = np.full((N_BASES, N_BASES), np.log(max(off_prob, 1e-300)))
    np.fill_diagonal(mat, np.log(diag_prob))
    return mat


@dataclass(frozen=True)
class SitePatterns:
    """压缩后的位点模式：patterns 每行一个模式（0..3 碱基，4 为 N）。"""

    patterns: np.ndarray  # shape (n_patterns, n_taxa), int8
    counts: np.ndarray  # shape (n_patterns,), int64

    @property
    def n_sites(self) -> int:
        return int(self.counts.sum())


def compress_sites(alignment: Alignment, columns: np.ndarray | None = None) -> SitePatterns:
    """把（可由 bootstrap 列索引重采样后的）位点压缩为模式与计数。

    用 5 进制编码做字典压缩，N 占第 5 个符号；同一模式只剪枝一次，
    其贡献按出现次数加权——这是处理重复位点模式的标准做法。
    """
    stacked = np.vstack([alignment.sequences[name] for name in alignment.names])
    if columns is not None:
        stacked = stacked[:, columns]
    n_taxa = stacked.shape[0]
    powers = 5 ** np.arange(n_taxa - 1, -1, -1)
    codes = (stacked.T * powers).sum(axis=1)
    unique_codes, counts = np.unique(codes, return_counts=True)
    patterns = np.empty((unique_codes.size, n_taxa), dtype=np.int8)
    remaining = unique_codes.astype(np.int64)
    for col in range(n_taxa - 1, -1, -1):
        patterns[:, col] = remaining % 5
        remaining //= 5
    return SitePatterns(patterns=patterns, counts=counts.astype(np.int64))


def _leaf_log_partials(patterns: np.ndarray) -> np.ndarray:
    """叶部分似然，形状 (n_patterns, n_taxa, 4)。

    观测碱基对应位置为 0（似然 1），其余为 -inf；
    N 在四个潜隐碱基下全部为 0（发射似然均为 1）。
    """
    n_patterns, n_taxa = patterns.shape
    partials = np.full((n_patterns, n_taxa, N_BASES), -np.inf, dtype=np.float64)
    unknown = patterns == UNKNOWN
    rows = np.broadcast_to(np.arange(n_patterns)[:, None], patterns.shape)
    cols = np.broadcast_to(np.arange(n_taxa)[None, :], patterns.shape)
    partials[rows[unknown], cols[unknown], :] = 0.0
    known = ~unknown
    partials[rows[known], cols[known], patterns[known].astype(np.int64)] = 0.0
    return partials


def log_likelihood(
    tree: UnrootedTree,
    patterns: SitePatterns,
    lengths: np.ndarray,
) -> float:
    """整序列对数似然：按模式剪枝后以模式计数加权求和。

    长度为 n_patterns 的批处理向量化；对数空间运算避免下溢。
    """
    leaf_partials = _leaf_log_partials(patterns.patterns)
    n_leaves = len(tree.leaves)

    def integrate_edge(child_partial: np.ndarray, edge_idx: int) -> np.ndarray:
        """把孩子侧的部分对数似然沿边传到父节点。

        child_partial: (n_patterns, 4)，列为孩子碱基 j；
        log_mat[j, i] = log P(j -> i | t)，对 j 做 logsumexp。
        """
        log_mat = jc69_log_matrix(float(lengths[edge_idx]))
        return logsumexp(child_partial[:, :, None] + log_mat[None, :, :], axis=1)

    def subtrial(node: int, parent: int) -> np.ndarray:
        if node < n_leaves:
            return leaf_partials[:, node, :]
        combined: np.ndarray | None = None
        for nbr, edge_idx in tree.adjacency[node]:
            if nbr == parent:
                continue
            passed = integrate_edge(subtrial(nbr, node), edge_idx)
            combined = passed if combined is None else (combined + passed)
        assert combined is not None
        return combined

    # 任选一个内部节点作剪枝根（无根树选点不影响似然）。
    root_id = min(nid for nid in tree.adjacency if nid >= n_leaves)
    total: np.ndarray | None = None
    for nbr, edge_idx in tree.adjacency[root_id]:
        passed = integrate_edge(subtrial(nbr, root_id), edge_idx)
        total = passed if total is None else (total + passed)
    assert total is not None

    per_pattern = _LOG_FREQ + logsumexp(total, axis=1)  # 均匀根状态
    if not np.all(np.isfinite(per_pattern)):
        return float("-inf")
    weighted = np.dot(patterns.counts, per_pattern)
    if not np.isfinite(weighted):
        return float("-inf")
    return float(weighted)
