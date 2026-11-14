"""共享列重采样 bootstrap。

关键约束：同一组重采样列用于该数据集的**全部**候选树，
每次重拟合都重新优化枝长，而不是复用原拟合的长度。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .fasta import Alignment
from .model import SitePatterns, compress_sites


@dataclass(frozen=True)
class BootstrapPlan:
    n_replicates: int
    seed: int
    # shape (n_replicates, n_sites)，元素为原始比对中的列下标。
    columns: np.ndarray

    def fingerprint(self) -> str:
        """对整组重采样列做指纹，方便在输出中核对各候选用的是同一组列。"""
        digest = hashlib.sha256(self.columns.astype(np.int64).tobytes())
        digest.update(str(self.seed).encode("utf-8"))
        digest.update(str(self.n_replicates).encode("utf-8"))
        return digest.hexdigest()[:16]


def make_bootstrap_plan(alignment: Alignment, n_replicates: int, seed: int) -> BootstrapPlan:
    """生成固定的重采样列矩阵（numpy legacy RandomState，结果确定）。"""
    rng = np.random.RandomState(seed)
    columns = rng.randint(
        low=0,
        high=alignment.n_sites,
        size=(n_replicates, alignment.n_sites),
        dtype=np.int64,
    )
    return BootstrapPlan(n_replicates=n_replicates, seed=seed, columns=columns)


def resampled_patterns(alignment: Alignment, columns: np.ndarray) -> SitePatterns:
    """把一组列下标映射为压缩位点模式（重复模式只剪枝一次）。"""
    return compress_sites(alignment, columns=columns)
