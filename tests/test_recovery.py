"""链恢复测试：去重 / 冲突 / 缺段 / 元信息矛盾 / warmup / 共同长度 / 顺序不变性。"""

import json

import numpy as np
import pytest

from mcmc_diag.recovery import CHAIN_BLOCKED, CHAIN_VALID, recover_chains

from conftest import (
    ADD_FRAGMENT,
    CONFLICT_PATH,
    FIXTURES,
    MANIFEST,
    make_entry,
    write_chain_csv,
    write_manifest,
)

PARAMS = ["location", "sticky", "separated_modes", "heavy_tail", "constant"]


def test_base_recovery_counts():
    rec = recover_chains(MANIFEST)
    assert rec.valid_chain_ids == ["c1", "c2", "c3", "c4"]
    assert rec.common_length == 540
    retained = {c.chain_id: c.n_retained for c in rec.chains}
    assert retained == {"c1": 600, "c2": 580, "c3": 560, "c4": 540}
    discarded = {c.chain_id: c.n_discarded_tail for c in rec.chains}
    assert discarded == {"c1": 60, "c2": 40, "c3": 20, "c4": 0}
    warmups = {c.chain_id: c.warmup_end for c in rec.chains}
    assert warmups == {"c1": 120, "c2": 140, "c3": 160, "c4": 180}
    for c in rec.chains:
        assert c.n_duplicates_removed == 0
        # 保留样本的 iteration 均大于各自 warmup_end
        assert c.iterations.min() > c.warmup_end
        assert len(c.iterations) == c.n_retained


def test_used_samples_are_earliest_post_warmup():
    rec = recover_chains(MANIFEST)
    for cid, its in rec.used_iterations.items():
        chain = next(c for c in rec.chains if c.chain_id == cid)
        # 使用样本 = warmup 后最早 L 个（尾部丢弃）
        np.testing.assert_array_equal(its, chain.iterations[: rec.common_length])


def test_fragment_order_does_not_change_results(tmp_path):
    base = recover_chains(MANIFEST)
    manifest = json.loads(MANIFEST.read_text())
    rng = np.random.default_rng(0)
    (tmp_path / "chains").symlink_to((FIXTURES / "chains").resolve())
    for _ in range(5):
        entries = list(manifest["files"])
        rng.shuffle(entries)
        shuffled = dict(manifest, files=entries)
        # 相对路径相对 manifest 所在目录解析，故把打乱后的 manifest 放进 fixtures 副本结构
        path = write_manifest(tmp_path, shuffled)
        rec = recover_chains(path)
        assert rec.valid_chain_ids == base.valid_chain_ids
        assert rec.common_length == base.common_length
        for p in base.parameters:
            np.testing.assert_array_equal(rec.used_matrix(p), base.used_matrix(p))
        for c_new, c_base in zip(rec.chains, base.chains):
            assert c_new.n_duplicates_removed == c_base.n_duplicates_removed
            assert c_new.n_discarded_tail == c_base.n_discarded_tail


def test_warmup_boundary_changes_used_samples(tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    for e in manifest["files"]:
        if e["chain_id"] == "c1":
            e["warmup_end"] = 0  # 不去 warmup
    (tmp_path / "chains").symlink_to((FIXTURES / "chains").resolve())
    path = write_manifest(tmp_path, manifest)
    rec = recover_chains(path)
    c1 = next(c for c in rec.chains if c.chain_id == "c1")
    assert c1.n_retained == 720  # 原来 600
    assert c1.iterations.min() == 1
    base = recover_chains(MANIFEST)
    c1_base = next(c for c in base.chains if c.chain_id == "c1")
    assert not np.array_equal(
        rec.used_values["c1"]["location"], base.used_values["c1"]["location"]
    )


def test_overlap_case_dedupes_and_keeps_results():
    extra = [json.loads(ADD_FRAGMENT.read_text())["add_to_manifest"]]
    base = recover_chains(MANIFEST)
    rec = recover_chains(MANIFEST, extra_entries=extra)
    c1 = next(c for c in rec.chains if c.chain_id == "c1")
    assert c1.status == CHAIN_VALID
    assert c1.n_duplicates_removed == 4  # 完全相同的重复只计一次
    assert c1.n_rows_loaded == 724
    assert rec.common_length == base.common_length
    for p in base.parameters:
        np.testing.assert_array_equal(rec.used_matrix(p), base.used_matrix(p))


def test_conflict_case_blocks_only_c1():
    extra = [json.loads(ADD_FRAGMENT.read_text())["add_to_manifest"]]
    extra[0]["path"] = CONFLICT_PATH
    rec = recover_chains(MANIFEST, extra_entries=extra)
    by_id = {c.chain_id: c for c in rec.chains}
    assert by_id["c1"].status == CHAIN_BLOCKED
    assert any("冲突重复" in p and "243" in p for p in by_id["c1"].problems)
    assert rec.valid_chain_ids == ["c2", "c3", "c4"]
    assert rec.common_length == 540  # min(580, 560, 540)
    # 其余链不受影响
    assert by_id["c2"].status == CHAIN_VALID
    assert by_id["c2"].n_retained == 580


def _two_fragment_dir(tmp_path, rows_a, rows_b, warmup_b=10):
    """构造一个链两个片段的临时目录；rows: (iterations, location值)。"""
    params = ["location"]
    write_chain_csv(tmp_path, "chains/k-part1.csv", rows_a[0], {"location": rows_a[1]})
    write_chain_csv(tmp_path, "chains/k-part2.csv", rows_b[0], {"location": rows_b[1]})
    manifest = {
        "files": [
            make_entry("k", "chains/k-part1.csv", int(rows_a[0][0]), int(rows_a[0][-1]), 10),
            make_entry("k", "chains/k-part2.csv", int(rows_b[0][0]), int(rows_b[0][-1]), warmup_b),
        ],
        "parameters": params,
    }
    return write_manifest(tmp_path, manifest)


def test_internal_gap_blocks_chain(tmp_path):
    its_a = list(range(1, 11))
    its_b = list(range(13, 21))  # 缺 11, 12
    path = _two_fragment_dir(
        tmp_path, (its_a, [float(i) for i in its_a]), (its_b, [float(i) for i in its_b])
    )
    rec = recover_chains(path)
    (chain,) = rec.chains
    assert chain.status == CHAIN_BLOCKED
    assert any("缺段" in p for p in chain.problems)
    assert rec.common_length is None


def test_metadata_contradiction_blocks_chain(tmp_path):
    its_a = list(range(1, 11))
    its_b = list(range(11, 21))
    # 第二片段 warmup_end=99，与第一片段矛盾
    path = _two_fragment_dir(
        tmp_path, (its_a, [1.0] * 10), (its_b, [2.0] * 10), warmup_b=99
    )
    rec = recover_chains(path)
    (chain,) = rec.chains
    assert chain.status == CHAIN_BLOCKED
    assert any("元信息矛盾" in p for p in chain.problems)


def test_declared_range_mismatch_blocks_chain(tmp_path):
    its = list(range(1, 11))
    write_chain_csv(tmp_path, "chains/k-part1.csv", its, {"location": [0.5] * 10})
    manifest = {
        "files": [make_entry("k", "chains/k-part1.csv", 1, 12, 2)],  # 声明到 12，文件只有 1..10
        "parameters": ["location"],
    }
    path = write_manifest(tmp_path, manifest)
    rec = recover_chains(path)
    (chain,) = rec.chains
    assert chain.status == CHAIN_BLOCKED
    assert any("矛盾" in p for p in chain.problems)


def test_exact_duplicate_within_file_deduped(tmp_path):
    its = [1, 2, 3, 3, 4, 5]  # iteration 3 完全重复
    vals = [0.1, 0.2, 0.3, 0.3, 0.4, 0.5]
    write_chain_csv(tmp_path, "chains/k-part1.csv", its, {"location": vals})
    manifest = {
        "files": [make_entry("k", "chains/k-part1.csv", 1, 5, 1)],
        "parameters": ["location"],
    }
    path = write_manifest(tmp_path, manifest)
    rec = recover_chains(path)
    (chain,) = rec.chains
    assert chain.status == CHAIN_VALID
    assert chain.n_duplicates_removed == 1
    assert chain.n_retained == 4  # iteration 2..5


def test_conflicting_duplicate_within_file_blocks(tmp_path):
    its = [1, 2, 3, 3, 4, 5]
    vals = [0.1, 0.2, 0.3, 0.9, 0.4, 0.5]  # iteration 3 两次取值不同
    write_chain_csv(tmp_path, "chains/k-part1.csv", its, {"location": vals})
    manifest = {
        "files": [make_entry("k", "chains/k-part1.csv", 1, 5, 1)],
        "parameters": ["location"],
    }
    path = write_manifest(tmp_path, manifest)
    rec = recover_chains(path)
    (chain,) = rec.chains
    assert chain.status == CHAIN_BLOCKED
    assert any("冲突重复" in p for p in chain.problems)


def test_no_valid_chain_gives_none_common_length(tmp_path):
    its = list(range(1, 6))
    write_chain_csv(tmp_path, "chains/k-part1.csv", its, {"location": [1.0] * 5})
    manifest = {
        "files": [make_entry("k", "chains/k-part1.csv", 1, 5, 10)],  # warmup 吞掉全部
        "parameters": ["location"],
    }
    path = write_manifest(tmp_path, manifest)
    rec = recover_chains(path)
    assert rec.valid_chain_ids == []
    assert rec.common_length is None
