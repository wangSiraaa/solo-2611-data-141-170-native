"""端到端测试：analyze/inspect 跑通基础数据与 cases 两个分支，
参数级结论齐全，顺序不变，报告产物存在。"""

import json

import numpy as np

from mcmc_diag.cli import main
from mcmc_diag.interpret import (
    STATUS_CHAINS_DISAGREE,
    STATUS_CONSTANT,
    STATUS_OK,
)
from mcmc_diag.recovery import recover_chains
from mcmc_diag.report import run_diagnostics

from conftest import ADD_FRAGMENT, CONFLICT_PATH, FIXTURES, MANIFEST, write_manifest


def test_analyze_base_end_to_end(tmp_path):
    out = tmp_path / "report"
    assert main(["analyze", "--manifest", str(MANIFEST), "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    # 五个参数都在报告中，单个参数的问题不影响其他参数出现
    assert set(summary["parameters"]) == {
        "location", "sticky", "separated_modes", "heavy_tail", "constant",
    }
    status = {p: r["verdict"]["status"] for p, r in summary["parameters"].items()}
    assert status["location"] == STATUS_OK
    assert status["heavy_tail"] == STATUS_OK
    assert status["separated_modes"] == STATUS_CHAINS_DISAGREE
    assert status["sticky"] == STATUS_CHAINS_DISAGREE  # rhat 与 ESS 同时失败
    assert set(summary["parameters"]["sticky"]["verdict"]["failed_checks"]) == {
        "rhat", "bulk_ess", "tail_ess",
    }
    assert status["constant"] == STATUS_CONSTANT
    # 共同长度与丢弃样本明确保存
    rec = summary["chain_recovery"]
    assert rec["common_length"] == 540
    discarded = {c["chain_id"]: c["n_discarded_tail"] for c in rec["chains"]}
    assert discarded == {"c1": 60, "c2": 40, "c3": 20, "c4": 0}
    # 证据文件
    for p in summary["parameters"]:
        assert (out / f"traces_{p}.png").exists()
    assert (out / "overview.png").exists() and (out / "summary.md").exists()
    # 常量参数的指标是显式状态而非伪装的“通过”
    constant_metrics = summary["parameters"]["constant"]["metrics"]
    assert constant_metrics["all_constant"] is True
    assert np.isnan(constant_metrics["rhat_max"])


def test_analyze_fragment_order_invariant(tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    reversed_manifest = dict(manifest, files=list(reversed(manifest["files"])))
    (tmp_path / "chains").symlink_to((FIXTURES / "chains").resolve())
    shuffled_path = write_manifest(tmp_path, reversed_manifest)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    main(["analyze", "--manifest", str(MANIFEST), "--out", str(out_a)])
    main(["analyze", "--manifest", str(shuffled_path), "--out", str(out_b)])
    a = json.loads((out_a / "summary.json").read_text())["parameters"]
    b = json.loads((out_b / "summary.json").read_text())["parameters"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_analyze_overlap_case_matches_base(tmp_path):
    out_base, out_ol = tmp_path / "base", tmp_path / "overlap"
    main(["analyze", "--manifest", str(MANIFEST), "--out", str(out_base)])
    main(["analyze", "--manifest", str(MANIFEST), "--out", str(out_ol),
          "--add-fragment", str(ADD_FRAGMENT)])
    base = json.loads((out_base / "summary.json").read_text())
    overlap = json.loads((out_ol / "summary.json").read_text())
    assert json.dumps(base["parameters"], sort_keys=True) == json.dumps(
        overlap["parameters"], sort_keys=True
    )
    c1 = next(c for c in overlap["chain_recovery"]["chains"] if c["chain_id"] == "c1")
    assert c1["n_duplicates_removed"] == 4


def test_analyze_conflict_case_blocks_c1_but_keeps_all_parameters(tmp_path):
    out = tmp_path / "conflict"
    main(["analyze", "--manifest", str(MANIFEST), "--out", str(out),
          "--add-fragment", str(ADD_FRAGMENT), "--fragment-path", CONFLICT_PATH])
    summary = json.loads((out / "summary.json").read_text())
    rec = summary["chain_recovery"]
    assert rec["valid_chain_ids"] == ["c2", "c3", "c4"]
    c1 = next(c for c in rec["chains"] if c["chain_id"] == "c1")
    assert c1["status"] == "blocked"
    assert any("冲突重复" in p for p in c1["problems"])
    # 参数一个不少，仍基于剩余三条链给出诊断
    assert set(summary["parameters"]) == {
        "location", "sticky", "separated_modes", "heavy_tail", "constant",
    }
    assert summary["parameters"]["separated_modes"]["verdict"]["status"] == STATUS_CHAINS_DISAGREE
    assert summary["parameters"]["location"]["metrics"]["n_chains"] == 3


def test_inspect_runs_on_all_branches(capsys):
    assert main(["inspect", "--manifest", str(MANIFEST)]) == 0
    out = capsys.readouterr().out
    assert "共同保留长度 L=540" in out
    assert main(["inspect", "--manifest", str(MANIFEST), "--add-fragment", str(ADD_FRAGMENT),
                 "--fragment-path", CONFLICT_PATH, "--chain", "c1"]) == 0
    out = capsys.readouterr().out
    assert "冲突重复" in out and "c1" in out
    assert main(["inspect", "--manifest", str(MANIFEST), "--json"]) == 0
    json.loads(capsys.readouterr().out)


def test_warmup_boundary_actually_changes_metrics():
    """warmup 边界改变会使用样本随之改变（不是装饰性参数）。"""
    base = recover_chains(MANIFEST)
    manifest = json.loads(MANIFEST.read_text())
    for e in manifest["files"]:
        if e["chain_id"] == "c1":
            e["warmup_end"] = 200
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "chains").symlink_to((FIXTURES / "chains").resolve())
        path = write_manifest(td, manifest)
        altered = recover_chains(path)
    c1_base = next(c for c in base.chains if c.chain_id == "c1")
    c1_alt = next(c for c in altered.chains if c.chain_id == "c1")
    assert c1_alt.n_retained == 520 != c1_base.n_retained
    assert c1_alt.iterations.min() == 201
    loc_base = run_diagnostics(base)["location"]["metrics"]["mean"]
    loc_alt = run_diagnostics(altered)["location"]["metrics"]["mean"]
    assert loc_base != loc_alt
