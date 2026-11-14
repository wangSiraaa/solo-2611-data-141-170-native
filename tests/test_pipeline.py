"""输入校验、bootstrap 共享列与优化失败留证的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phylocompare.bootstrap import make_bootstrap_plan
from phylocompare.compare import load_inputs, run_comparison
from phylocompare.errors import CompareError
from phylocompare.fasta import read_fasta
from phylocompare.model import compress_sites
from phylocompare.newick import parse_newick
from phylocompare.optim import DEFAULT_BOUNDS, fit_branch_lengths
from phylocompare.tree import UnrootedTree

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _write_fasta(tmp_path: Path, rows: dict[str, str]) -> Path:
    text = "".join(f">{name}\n{seq}\n" for name, seq in rows.items())
    path = tmp_path / "x.fasta"
    path.write_text(text, encoding="utf-8")
    return path


def test_fasta_rejects_unequal_lengths(tmp_path: Path) -> None:
    path = _write_fasta(tmp_path, {"A": "ACG", "B": "AC"})
    with pytest.raises(CompareError, match="长度不一致"):
        read_fasta(path)


def test_fasta_rejects_bad_alphabet_and_duplicates(tmp_path: Path) -> None:
    path = _write_fasta(tmp_path, {"A": "ACGX", "B": "ACGT", "C": "ACGT", "D": "ACGT", "E": "ACGT"})
    with pytest.raises(CompareError, match="ACGTN"):
        read_fasta(path)
    dup = tmp_path / "d.fasta"
    dup.write_text(">A\nACGT\n>A\nACGT\n", encoding="utf-8")
    with pytest.raises(CompareError, match="叶名重复"):
        read_fasta(dup)


def test_tree_leaf_set_must_match_alignment(tmp_path: Path) -> None:
    fasta = _write_fasta(
        tmp_path,
        {name: "ACGTACGT" for name in "ABCDE"},
    )
    nwk = tmp_path / "c.nwk"
    nwk.write_text(
        "((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2,X:0.1);\n",
        encoding="utf-8",
    )
    with pytest.raises(CompareError, match="叶集合"):
        load_inputs(fasta, nwk)


def test_tree_must_be_trifurcating_root(tmp_path: Path) -> None:
    tree_leaves = tuple("ABCDE")
    # 合法：根有三个孩子，内部节点均为二叉，每条边都带初始枝长。
    root = parse_newick("((A:0.1,B:0.1):0.2,C:0.2,(D:0.1,E:0.1):0.3);", "<t>")[0]
    good = UnrootedTree.from_newick(root, tree_leaves)
    assert good.n_edges == 7
    # 非法：根只有两个孩子（被当成有根二叉树序列化）。
    bad = parse_newick("((A:0.1,B:0.1):0.2,((C:0.1,D:0.1):0.2,E:0.1):0.3);", "<t>")[0]
    with pytest.raises(CompareError, match="三叉根"):
        UnrootedTree.from_newick(bad, tree_leaves)


def test_shared_bootstrap_columns_are_identical_for_all_candidates() -> None:
    alignment, trees = load_inputs(
        FIXTURES / "resolved.fasta", FIXTURES / "candidates.nwk"
    )
    plan = make_bootstrap_plan(alignment, n_replicates=24, seed=271828)
    assert plan.columns.shape == (24, alignment.n_sites)
    # 指纹只依赖列矩阵本身；同种子同数据必须逐元素复现。
    again = make_bootstrap_plan(alignment, 24, 271828)
    assert np.array_equal(plan.columns, again.columns)
    assert plan.fingerprint() == again.fingerprint()

    report = run_comparison(
        FIXTURES / "resolved.fasta",
        FIXTURES / "candidates.nwk",
        n_bootstrap=24,
        seed=271828,
    )
    fingerprints = {
        rep["resample_columns_sha256_prefix"] for rep in report["bootstrap_replicates"]
    }
    # 每次重采样自身的列指纹在各候选条目间隐式共享（同一 rep 只存一次），
    # 这里核对每个 rep 都对三个候选做了独立的重新拟合。
    for rep in report["bootstrap_replicates"]:
        assert len(rep["candidates"]) == len(trees)
        assert all(c["optimization_ok"] for c in rep["candidates"].values())
        lengths = [
            tuple(c["fitted_branch_lengths"].values())
            for c in rep["candidates"].values()
        ]
        # 重新拟合确实发生：不同拓扑的最优枝长向量不应逐边相同。
        assert lengths[0] != lengths[1]
    assert len(fingerprints) == 24


def test_refit_uses_new_lengths_not_original() -> None:
    alignment, trees = load_inputs(
        FIXTURES / "weak.fasta", FIXTURES / "candidates.nwk"
    )
    patterns = compress_sites(alignment)
    result = fit_branch_lengths(trees[0], patterns, DEFAULT_BOUNDS)
    assert result.ok
    initial = trees[0].initial_lengths
    assert result.lengths != pytest.approx(initial, abs=1e-12)
    assert all(DEFAULT_BOUNDS[0] <= v <= DEFAULT_BOUNDS[1] for v in result.lengths)


def test_optimization_failure_is_recorded_not_filled(tmp_path: Path, monkeypatch) -> None:
    """拟合失败必须保留失败证据，不能用最差数冒充完整结果。"""
    import phylocompare.optim as optim_mod

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic numerical failure")

    monkeypatch.setattr(optim_mod, "minimize", boom)
    alignment, trees = load_inputs(
        FIXTURES / "weak.fasta", FIXTURES / "candidates.nwk"
    )
    patterns = compress_sites(alignment)
    result = fit_branch_lengths(trees[0], patterns, DEFAULT_BOUNDS)
    assert not result.ok
    assert result.failure_reason == "全部优化起点均未收敛"
    assert len(result.starts) == 4
    assert all(not s.success and s.final_loglik is None for s in result.starts)


def test_end_to_end_report_contract(tmp_path: Path) -> None:
    report = run_comparison(
        FIXTURES / "weak.fasta",
        FIXTURES / "candidates.nwk",
        n_bootstrap=6,
        seed=271828,
    )
    # 支持频率字段完整，且显式声明不是后验概率。
    note = report["run_parameters"]["support_frequency_note"]
    assert "不是贝叶斯后验概率" in note
    for cid, item in report["bootstrap_support_frequencies"].items():
        assert 0.0 <= item["strict_win_frequency"] <= 1.0
        assert 0.0 <= item["best_or_near_tied_frequency"] <= 1.0
    # JSON 可序列化且不含 NaN/Infinity。
    out = tmp_path / "r.json"
    from phylocompare.compare import write_report

    write_report(report, out)
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["bootstrap_replicates"][0]["replicate"] == 0
