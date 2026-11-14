"""模式压缩与 CLI 行为的补充测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phylocompare.cli import main
from phylocompare.fasta import read_fasta
from phylocompare.model import compress_sites

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_pattern_compression_counts_and_n_handling() -> None:
    alignment = read_fasta(FIXTURES / "weak.fasta")
    patterns = compress_sites(alignment)
    # 计数之和必须等于位点数；模式矩阵行数与计数长度一致。
    assert int(patterns.counts.sum()) == alignment.n_sites
    assert patterns.patterns.shape[0] == patterns.counts.size
    assert patterns.patterns.shape[1] == len(alignment.names)
    # 含 N 的模式确实保留了 4 这个编码，而不是被丢弃。
    assert np.any(patterns.patterns == 4)
    n_sites_with_n = int(np.any(patterns.patterns == 4, axis=1).sum())
    assert n_sites_with_n >= 1

    # 同一列集合重复采样后，模式计数按重复次数倍增。
    cols = np.arange(10)
    doubled = np.concatenate([cols, cols])
    rep = compress_sites(alignment, doubled)
    assert int(rep.counts.sum()) == 20
    base = compress_sites(alignment, cols)
    # 前 10 列的每个模式在双倍样本中计数恰好翻倍。
    base_lookup = {
        tuple(row): int(count)
        for row, count in zip(base.patterns, base.counts)
    }
    for row, count in zip(rep.patterns, rep.counts):
        assert int(count) == 2 * base_lookup[tuple(row)]


def test_cli_rejects_missing_case(tmp_path: Path, capsys) -> None:
    code = main([
        "compare",
        "--plan", str(FIXTURES / "plan.json"),
        "--case", "not-there.fasta",
        "--out-dir", str(tmp_path),
    ])
    assert code == 2
    err = capsys.readouterr().err
    assert "不在 plan.json cases 中" in err


def test_cli_rejects_too_many_replicates(tmp_path: Path, capsys) -> None:
    code = main([
        "compare",
        "--plan", str(FIXTURES / "plan.json"),
        "--case", "weak.fasta",
        "--replicates", "25",
        "--out-dir", str(tmp_path),
    ])
    assert code == 2
    assert "上限" in capsys.readouterr().err


def test_cli_writes_report_for_single_case(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    code = main([
        "compare",
        "--plan", str(FIXTURES / "plan.json"),
        "--case", "weak.fasta",
        "--replicates", "4",
        "--out-dir", str(out_dir),
    ])
    assert code == 0
    report_path = out_dir / "weak.report.json"
    assert report_path.is_file()
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run_parameters"]["bootstrap_replicates"] == 4
    assert len(report["bootstrap_replicates"]) == 4
