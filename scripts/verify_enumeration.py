#!/usr/bin/env python3
"""独立核对：对极短 3 叶 / 4 叶样例枚举内部碱基，与树上剪枝似然逐项对比。

该脚本不经过 pytest，可直接运行：

    .venv/bin/python scripts/verify_enumeration.py

检查内容：
1. 3 叶星状树、4 叶无根二叉树在含 N 模式下的剪枝对数似然，
   与 4^k 种内部碱基全枚举的参考实现一致；
2. N 处理：N 位点等价于对该叶四个观测状态边际化；
3. 叶序处理：重排叶顺序后似然不变；
4. 长序列下对数空间剪枝保持有限（验证下溢防护）。
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phylocompare.fasta import UNKNOWN  # noqa: E402
from phylocompare.model import (  # noqa: E402
    N_BASES,
    SitePatterns,
    jc69_log_matrix,
    log_likelihood,
)
from phylocompare.newick import parse_newick  # noqa: E402
from phylocompare.tree import UnrootedTree  # noqa: E402

MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def make_patterns(rows: list[str]) -> SitePatterns:
    patterns = np.array(
        [[MAPPING[ch] for ch in row] for row in rows], dtype=np.int8
    ).T
    return SitePatterns(
        patterns=patterns, counts=np.ones(patterns.shape[0], dtype=np.int64)
    )


def brute_force(tree: UnrootedTree, patterns: SitePatterns, lengths: np.ndarray) -> float:
    """完全独立的参考实现：枚举所有内部碱基组合，在普通概率空间相乘求和。"""
    n_leaves = len(tree.leaves)
    internal_ids = [nid for nid in tree.adjacency if nid >= n_leaves]
    probs = [np.exp(jc69_log_matrix(float(t))) for t in lengths]

    total = 0.0
    for pattern, count in zip(patterns.patterns, patterns.counts):
        site_like = 0.0
        for assignment in itertools.product(range(N_BASES), repeat=len(internal_ids)):
            states = dict(zip(internal_ids, assignment))
            weight = 0.25
            for edge in tree.edges:
                u, v = edge.endpoints
                if u >= n_leaves and v >= n_leaves:
                    weight *= probs[edge.index][states[u], states[v]]
                    continue
                leaf_id, internal_id = (v, u) if u >= n_leaves else (u, v)
                row = probs[edge.index][states[internal_id]]
                observed = int(pattern[leaf_id])
                weight *= float(row.sum()) if observed == UNKNOWN else float(row[observed])
            site_like += weight
        total += int(count) * np.log(site_like)
    return total


def check(label: str, got: float, expected: float, tol: float = 1e-9) -> None:
    diff = abs(got - expected)
    status = "OK " if diff < tol else "FAIL"
    print(f"[{status}] {label}: 剪枝={got:.10f} 枚举={expected:.10f} |Δ|={diff:.2e}")
    if diff >= tol:
        raise SystemExit(1)


def main() -> int:
    # --- 3 叶星状树（1 个内部节点，枚举 4 种内部状态） ------------------
    leaves3 = ("A", "B", "C")
    tree3 = UnrootedTree.from_newick(
        parse_newick("(A:0.11,B:0.27,C:0.43);", "<3taxon>")[0], leaves3
    )
    patterns3 = make_patterns(["ACGTN", "ACGTA", "TTTAN"])
    lengths3 = np.array([0.11, 0.27, 0.43])
    ll3 = log_likelihood(tree3, patterns3, lengths3)
    check("3 叶含 N 全模式", ll3, brute_force(tree3, patterns3, lengths3))

    # --- 4 叶无根二叉树（2 个内部节点，枚举 16 种组合） -----------------
    leaves4 = ("A", "B", "C", "D")
    tree4 = UnrootedTree.from_newick(
        parse_newick("(A:0.09,B:0.31,(C:0.22,D:0.05):0.17);", "<4taxon>")[0],
        leaves4,
    )
    patterns4 = make_patterns(["ACGTAC", "ACGTNT", "TTTACG", "TNTACG"])
    rng = np.random.RandomState(7)
    lengths4 = rng.uniform(0.02, 0.8, size=tree4.n_edges)
    ll4 = log_likelihood(tree4, patterns4, lengths4)
    check("4 叶含 N 全模式", ll4, brute_force(tree4, patterns4, lengths4))

    # --- N 的边际化核对 --------------------------------------------------
    unknown = make_patterns(["N", "C", "G"])
    ll_n = log_likelihood(tree3, unknown, lengths3)
    ll_marginal = np.logaddexp.reduce([
        log_likelihood(tree3, make_patterns([base, "C", "G"]), lengths3)
        for base in "ACGT"
    ])
    check("N 等于四碱基边际化", ll_n, float(ll_marginal), tol=1e-10)
    check(
        "N 的枚举实现一致",
        ll_n,
        brute_force(tree3, unknown, lengths3),
        tol=1e-10,
    )

    # --- 叶序不变性 ------------------------------------------------------
    order = ("C", "A", "B")
    reordered = make_patterns(["TTTAN", "ACGTN", "ACGTA"])  # 与 patterns3 同数据换叶序
    tree3_re = tree3.reorder_leaf_sequences(order)
    ll_re = log_likelihood(tree3_re, reordered, lengths3)
    check("叶序重排似然不变", ll3, ll_re, tol=1e-12)

    # --- 下溢防护：构造极长的恒定/近恒定序列 -----------------------------
    big = SitePatterns(
        patterns=np.array([[0, 0, 0], [0, 1, 2]], dtype=np.int8),
        counts=np.array([100_000, 100_000], dtype=np.int64),
    )
    ll_big = log_likelihood(tree3, big, lengths3)
    assert np.isfinite(ll_big), "长序列对数似然应为有限值"
    print(f"[OK ] 20 万位点长序列对数似然有限: {ll_big:.4f}")

    print("\n全部独立核对通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
