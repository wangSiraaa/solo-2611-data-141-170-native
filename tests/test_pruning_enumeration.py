"""不依赖剪枝实现的独立似然核对：枚举所有内部碱基状态。

对 3 叶（星状三叉树，3 条边）和 4 叶（含 1 个内部边，共 5 条边）
的极小树，把内部节点的 4^k 种碱基组合全部枚举一遍，
按 JC69 转移概率直接相乘求和；结果必须与 ``model.log_likelihood``
在普通概率空间（不经对数）逐项一致。
"""

from __future__ import annotations

import itertools

import numpy as np

from phylocompare.fasta import UNKNOWN
from phylocompare.model import N_BASES, SitePatterns, jc69_log_matrix, log_likelihood
from phylocompare.newick import parse_newick
from phylocompare.tree import UnrootedTree


def _tree_from(newick_text: str, leaves: tuple[str, ...]) -> UnrootedTree:
    root = parse_newick(newick_text, "<test>")[0]
    return UnrootedTree.from_newick(root, leaves)


def brute_force_loglik(
    tree: UnrootedTree, patterns: SitePatterns, lengths: np.ndarray
) -> float:
    """枚举内部碱基状态的参考实现，完全独立于剪枝代码路径。"""
    n_leaves = len(tree.leaves)
    internal_ids = [nid for nid in tree.adjacency if nid >= n_leaves]
    probs = [np.exp(jc69_log_matrix(float(t))) for t in lengths]

    total = 0.0
    for pattern, count in zip(patterns.patterns, patterns.counts):
        site_like = 0.0
        for assignment in itertools.product(range(N_BASES), repeat=len(internal_ids)):
            internal_state = dict(zip(internal_ids, assignment))
            weight = 0.25  # 均匀根先验（JC69 平稳频率），挂在任一内部节点上等价
            for edge in tree.edges:
                u, v = edge.endpoints
                if u >= n_leaves and v >= n_leaves:
                    weight *= probs[edge.index][internal_state[u], internal_state[v]]
                else:
                    leaf_id, internal_id = (v, u) if u >= n_leaves else (u, v)
                    observed = int(pattern[leaf_id])
                    trans = probs[edge.index][internal_state[internal_id]]
                    # 已知碱基取对应转移概率；N 对四个碱基边际化（发射似然均为 1）。
                    if observed == UNKNOWN:
                        weight *= float(trans.sum())
                    else:
                        weight *= float(trans[observed])
            site_like += weight
        total += int(count) * np.log(site_like)
    return float(total)


def _make_patterns(rows: list[str]) -> SitePatterns:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    patterns = np.array(
        [[mapping[ch] for ch in row] for row in rows], dtype=np.int8
    ).T
    return SitePatterns(patterns=patterns, counts=np.ones(patterns.shape[0], dtype=np.int64))


def test_three_leaf_star_matches_enumeration() -> None:
    leaves = ("A", "B", "C")
    tree = _tree_from("(A:0.11,B:0.27,C:0.43);", leaves)
    assert tree.n_edges == 3
    patterns = _make_patterns([
        "ACGTN",   # A 叶
        "ACGTA",   # B 叶
        "TTTAN",   # C 叶
    ])
    lengths = np.array([0.11, 0.27, 0.43])
    got = log_likelihood(tree, patterns, lengths)
    expected = brute_force_loglik(tree, patterns, lengths)
    assert abs(got - expected) < 1e-10, (got, expected)


def test_four_leaf_tree_matches_enumeration() -> None:
    leaves = ("A", "B", "C", "D")
    # 4 叶无根二叉树：2 个内部节点、5 条边；三叉根带两个叶和一个二叉内部节点。
    tree = _tree_from("(A:0.09,B:0.31,(C:0.22,D:0.05):0.17);", leaves)
    assert tree.n_edges == 5
    patterns = _make_patterns([
        "ACGTAC",
        "ACGTNT",
        "TTTACG",
        "TNTACG",
    ])
    # 边按规范 key 编号，长度按同一顺序给出即可，这里直接用一组固定值。
    rng = np.random.RandomState(7)
    lengths = rng.uniform(0.02, 0.8, size=tree.n_edges)
    got = log_likelihood(tree, patterns, lengths)
    expected = brute_force_loglik(tree, patterns, lengths)
    assert abs(got - expected) < 1e-9, (got, expected)


def test_n_unknown_sums_over_all_bases() -> None:
    """N 位点的似然必须等于该叶四个具体碱基观测的似然之和（发射似然均为 1）。"""
    leaves = ("A", "B", "C")
    tree = _tree_from("(A:0.15,B:0.25,C:0.35);", leaves)
    base_rows = ["A", "C", "G"]  # 三个叶各一个已知碱基
    known = _make_patterns(base_rows)
    lengths = np.array([0.15, 0.25, 0.35])
    ll_known = log_likelihood(tree, known, lengths)

    # 把 A 叶换成 N，枚举核对：N 等价于对该叶四个观测状态边际化。
    unknown_patterns = _make_patterns(["N", "C", "G"])
    ll_n = log_likelihood(tree, unknown_patterns, lengths)
    marginal = [
        log_likelihood(tree, _make_patterns([base, "C", "G"]), lengths)
        for base in "ACGT"
    ]
    # 转移概率行和为 1，均匀根先验已含在每个似然内，因此直接相加、不除 4。
    ll_marginal = np.logaddexp.reduce(marginal)
    assert abs(ll_n - ll_marginal) < 1e-10
    assert abs(brute_force_loglik(tree, unknown_patterns, lengths) - ll_n) < 1e-10
    assert ll_n > ll_known  # 边际化掉一个观测，似然密度通常变高


def test_leaf_order_does_not_change_likelihood() -> None:
    """重排 FASTA 叶顺序与 Newick 书写顺序，剪枝似然与拟合结果保持不变。"""
    from phylocompare.optim import fit_branch_lengths

    leaves = ("A", "B", "C", "D", "E")
    newick_text = (
        "((A:0.04,B:0.04):0.15,(C:0.04,D:0.04):0.12,E:0.06);"
    )
    tree = _tree_from(newick_text, leaves)

    rng = np.random.RandomState(11)
    cols = rng.randint(0, 4, size=(5, 30)).astype(np.int8)
    cols[cols == 4] = 0

    def alignment_like(names: tuple[str, ...], order: list[int]) -> SitePatterns:
        reordered = cols[order, :]
        return SitePatterns(
            patterns=reordered.T.copy(),
            counts=np.ones(30, dtype=np.int64),
        )

    order_a = [0, 1, 2, 3, 4]
    order_b = [4, 0, 3, 1, 2]
    patterns_a = alignment_like(leaves, order_a)
    names_b = tuple(leaves[i] for i in order_b)
    patterns_b = alignment_like(names_b, order_b)

    lengths = np.full(tree.n_edges, 0.2)
    ll_a = log_likelihood(tree, patterns_a, lengths)

    tree_b = tree.reorder_leaf_sequences(names_b)
    # 邻接关系不变，长度按规范边 key 对应，结果应完全相同。
    ll_b = log_likelihood(tree_b, patterns_b, lengths)
    assert abs(ll_a - ll_b) < 1e-12

    fit_a = fit_branch_lengths(tree, patterns_a)
    fit_b = fit_branch_lengths(tree_b, patterns_b)
    assert fit_a.ok and fit_b.ok
    assert abs(fit_a.loglik - fit_b.loglik) < 1e-8
    assert np.allclose(fit_a.lengths, fit_b.lengths, atol=1e-7)
