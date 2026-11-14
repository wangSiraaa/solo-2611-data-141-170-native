"""排列不变性：叶名顺序与候选顺序变化不应改变有意义结果。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phylocompare.compare import run_comparison
from phylocompare.newick import parse_newick
from phylocompare.tree import UnrootedTree

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_candidate_permutation_does_not_change_results(tmp_path: Path) -> None:
    nwk = (FIXTURES / "candidates.nwk").read_text(encoding="utf-8").splitlines()
    permuted = tmp_path / "permuted.nwk"
    # 逆序排列三个候选，并交换每棵树 Newick 根孩子的书写顺序。
    permuted.write_text(
        "\n".join(
            _swap_root_children(nwk[i]) for i in (2, 1, 0)
        )
        + "\n",
        encoding="utf-8",
    )

    base = run_comparison(
        FIXTURES / "weak.fasta", FIXTURES / "candidates.nwk", n_bootstrap=8, seed=4242
    )
    swapped = run_comparison(
        FIXTURES / "weak.fasta", permuted, n_bootstrap=8, seed=4242
    )

    def by_topology(report: dict) -> dict:
        return {
            tuple(tuple(edge) for edge in cand["topology"]): cand
            for cand in report["candidates"]
        }

    base_map, swapped_map = by_topology(base), by_topology(swapped)
    assert set(base_map) == set(swapped_map)
    for key, cand_a in base_map.items():
        cand_b = swapped_map[key]
        assert abs(cand_a["log_likelihood"] - cand_b["log_likelihood"]) < 1e-8
        assert np.allclose(
            list(cand_a["fitted_branch_lengths"].values()),
            list(cand_b["fitted_branch_lengths"].values()),
            atol=1e-7,
        )

    # 同一 bootstrap 种子 → 同一组重采样列；逐次最优拓扑集合应一致。
    for rep_a, rep_b in zip(base["bootstrap_replicates"], swapped["bootstrap_replicates"]):
        assert (
            rep_a["resample_columns_sha256_prefix"]
            == rep_b["resample_columns_sha256_prefix"]
        )


def _swap_root_children(line: str) -> str:
    root = parse_newick(line, "<perm>")[0]
    # 只交换根孩子的书写顺序（连同各自枝长），拓扑与边不变。
    root.children = [root.children[2], root.children[1], root.children[0]]

    def render(node) -> str:
        if node.is_leaf:
            return f"{node.name}:{node.length}"
        inner = ",".join(render(child) for child in node.children)
        if node.length is None:
            return f"({inner})"
        return f"({inner}):{node.length}"

    return render(root) + ";"


def test_newick_root_child_order_gives_same_tree() -> None:
    leaves = tuple("ABCDE")
    text_a = "((A:0.04,B:0.04):0.15,(C:0.04,D:0.04):0.12,E:0.06);"
    text_b = "(E:0.06,(C:0.04,D:0.04):0.12,(B:0.04,A:0.04):0.15);"
    tree_a = UnrootedTree.from_newick(parse_newick(text_a)[0], leaves)
    tree_b = UnrootedTree.from_newick(parse_newick(text_b)[0], leaves)
    assert tree_a.topology_key == tree_b.topology_key
    assert tree_a.initial_lengths == tree_b.initial_lengths
