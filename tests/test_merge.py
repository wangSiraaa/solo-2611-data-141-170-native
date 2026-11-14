"""merge 的验证套件（仅标准库，python3.11 -m unittest discover -s tests -v）。

覆盖：fixtures 全部故事（反向质量、低质量冲突、高质量矛盾、重复序列、
无重叠、孤儿）、手工概率例核对共识质量、已知短片段核对重叠位置、
打乱 mate 文件顺序的不变性、坏记录诊断与来源表完整性。
"""
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import merge  # noqa: E402

FIX = ROOT / "fixtures"
PLAN = merge.load_plan(FIX / "plan.json")

# 已知短片段（固定种子，可复现）
_rng = random.Random(20261114)
FRAG120 = "".join(_rng.choice("ACGT") for _ in range(120))
FRAG60 = "".join(_rng.choice("ACGT") for _ in range(60))


def run_fixtures(outdir: Path) -> dict:
    result = merge.run_pipeline(FIX / "reads_R1.fastq", FIX / "reads_R2.fastq", PLAN)
    merge.write_outputs(result, outdir)
    return result


def write_fastq(path: Path, records: list[tuple[str, str, str]]) -> None:
    with path.open("w") as fh:
        for name, seq, qual in records:
            fh.write(f"@{name}\n{seq}\n+\n{qual}\n")


def read_fastq(path: Path) -> dict[str, tuple[str, str]]:
    lines = path.read_text().splitlines()
    assert len(lines) % 4 == 0
    return {lines[i][1:]: (lines[i + 1], lines[i + 3])
            for i in range(0, len(lines), 4)}


def fuse_q_agree(q1: int, q2: int) -> float:
    """独立手算：两条观测一致时的后验错误概率（未截断、未取整）。"""
    e1, e2 = 10 ** (-q1 / 10), 10 ** (-q2 / 10)
    return (e1 * e2 / 3) / ((1 - e1) * (1 - e2) + e1 * e2 / 3)


class FixtureStoriesTest(unittest.TestCase):
    """现有 fixtures 的端到端行为。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.outdir = Path(cls._tmp.name)
        cls.result = run_fixtures(cls.outdir)
        cls.pairs = cls.result["pairs"]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_statuses(self):
        expect = {
            "pair01": "merged", "pair02": "merged", "pair03": "no_overlap",
            "pair04": "merged", "pair05": "no_overlap", "pair06": "ambiguous",
            "pair07": "merged", "pair08": "merged",
        }
        self.assertEqual({k: p["status"] for k, p in self.pairs.items()}, expect)

    def test_overlap_positions(self):
        """已知重叠位置：d 与 overlap 逐一核对。"""
        expect = {"pair01": (50, 45), "pair02": (35, 60), "pair04": (45, 50),
                  "pair07": (70, 25), "pair08": (55, 40)}
        for key, (d, ov) in expect.items():
            c = self.pairs[key]["compatible"][0]
            self.assertEqual((c.d, c.overlap), (d, ov), key)
            self.assertEqual(c.orientation, "standard", key)

    def test_lowq_conflict_absorbed(self):
        """pair02：单个低质量冲突（R2 Q4）被吸收，不等同于高质量矛盾。"""
        pr = self.pairs["pair02"]
        self.assertEqual(pr["status"], "merged")
        c = pr["compatible"][0]
        self.assertEqual((c.n_mismatch, len(c.hq_conflicts)), (1, 0))
        prov = pr["provenance"][84]
        self.assertEqual((prov["r1_base"], prov["r1_q"]), ("A", 35))
        self.assertEqual((prov["r2_base"], prov["r2_q"]), ("T", 4))
        self.assertEqual(prov["note"], "conflict_resolved")
        # 手算融合：A 保留，质量略低于原 Q35
        e1, e2 = 10 ** -3.5, 10 ** -0.4
        lik = {"A": (1 - e1) * e2 / 3, "T": e1 / 3 * (1 - e2),
               "C": e1 * e2 / 9, "G": e1 * e2 / 9}
        p_err = 1 - lik["A"] / sum(lik.values())
        self.assertEqual(prov["consensus_base"], "A")
        self.assertEqual(prov["consensus_q"], int(-10 * math.log10(p_err) + 0.5))
        self.assertEqual(prov["consensus_q"], 32)  # 手算值 31.64 -> 32
        self.assertLess(prov["consensus_q"], 35)

    def test_hq_conflict_rejected(self):
        """pair03：高质量矛盾（Q35 vs Q40）导致候选被拒绝，不强行合并。"""
        pr = self.pairs["pair03"]
        self.assertEqual(pr["status"], "no_overlap")
        self.assertIn("高质量矛盾", pr["reason"])
        best = pr["best"]
        self.assertGreater(best.score, 0)  # 分数虽为正，硬冲突仍拒绝
        self.assertEqual(best.hq_conflicts[0]["r1_idx"], 94)

    def test_n_resolved_by_mate(self):
        """pair04：N 不提供证据，由另一端 Q35 观测补足，质量不增不减。"""
        prov = self.pairs["pair04"]["provenance"][65]
        self.assertEqual((prov["r1_base"], prov["r1_q"]), ("N", 2))
        self.assertEqual((prov["r2_base"], prov["r2_q"]), ("G", 35))
        self.assertEqual((prov["consensus_base"], prov["consensus_q"]), ("G", 35))
        self.assertEqual(prov["note"], "single_non_N")

    def test_reversed_quality(self):
        """pair07：R2 质量必须反转后再对齐（锯齿模式在手算位置兑现）。"""
        pr = self.pairs["pair07"]
        self.assertEqual(pr["status"], "merged")
        prov = pr["provenance"]
        # 输出位 70/80/94 对应原 R2 索引 94/84/70（反向坐标映射）
        self.assertEqual((prov[70]["r2_idx"], prov[70]["r2_q"]), (94, 25))
        self.assertEqual((prov[80]["r2_idx"], prov[80]["r2_q"]), (84, 25))
        self.assertEqual((prov[94]["r2_idx"], prov[94]["r2_q"]), (70, 29))
        # 手算融合 Q22+Q25 -> 52；若忘记反转质量将拿到错误的原始位置质量
        self.assertEqual(prov[80]["consensus_q"],
                         int(-10 * math.log10(fuse_q_agree(22, 25)) + 0.5))
        self.assertEqual(prov[80]["consensus_q"], 52)
        # 反证：R2 原始（未反转）在该对齐位置的质量不是 25
        raw_r2_qual = read_fastq(FIX / "reads_R2.fastq")["pair07/2"][1]
        self.assertNotEqual(ord(raw_r2_qual[10]) - 33, prov[80]["r2_q"])

    def test_repeat_ambiguous(self):
        """pair06：短重复允许 76 个同等兼容位移；最佳（最长）不救命。"""
        pr = self.pairs["pair06"]
        self.assertEqual(pr["status"], "ambiguous")
        self.assertEqual(len(pr["compatible"]), 76)
        self.assertEqual(pr["best"].overlap, 94)  # 最长重叠分数最高
        self.assertIn(pr["best"], pr["compatible"])
        self.assertIsNone(pr["consensus_seq"])  # 不伪造完整片段

    def test_no_overlap_and_orphan_preserved(self):
        unmerged = read_fastq(self.outdir / "unmerged.fastq")
        r1 = read_fastq(FIX / "reads_R1.fastq")
        r2 = read_fastq(FIX / "reads_R2.fastq")
        # 无重叠与歧义配对：原始两端原方向原名保留
        for key in ("pair03", "pair05", "pair06"):
            self.assertEqual(unmerged[f"{key}/1"], r1[f"{key}/1"])
            self.assertEqual(unmerged[f"{key}/2"], r2[f"{key}/2"])
        # 孤儿保留原始记录与来源诊断
        self.assertEqual(unmerged["orphan/1"], r1["orphan/1"])
        diags = self.result["diagnostics"]
        self.assertTrue(any("orphan" in d["name"] and "孤儿" in d["reason"]
                            for d in diags))
        self.assertNotIn("orphan", read_fastq(self.outdir / "merged.fastq"))

    def test_merged_fastq_valid_and_capped(self):
        recs = read_fastq(self.outdir / "merged.fastq")
        self.assertEqual(sorted(recs), ["pair01", "pair02", "pair04",
                                        "pair07", "pair08"])
        for name, (seq, qual) in recs.items():
            self.assertEqual(len(seq), len(qual), name)
            self.assertTrue(set(seq) <= set("ACGTN"), name)
            self.assertLessEqual(max(ord(c) - 33 for c in qual), 60, name)

    def test_provenance_integrity(self):
        """每个输出位置可回查：来源、原 mate 碱基索引、是否融合。"""
        r1 = read_fastq(FIX / "reads_R1.fastq")
        r2 = read_fastq(FIX / "reads_R2.fastq")
        merged = read_fastq(self.outdir / "merged.fastq")
        for key, (seq, qual) in merged.items():
            pr = self.pairs[key]
            prov = pr["provenance"]
            self.assertEqual(len(prov), len(seq), key)
            s1, q1 = r1[f"{key}/1"]
            s2, q2 = r2[f"{key}/2"]
            for p, pv in enumerate(prov):
                self.assertEqual(p, pv["pos"])
                self.assertEqual(seq[p], pv["consensus_base"])
                self.assertEqual(ord(qual[p]) - 33, pv["consensus_q"])
                if pv["source"] == "BOTH":  # 融合位：双端索引都能指回原记录
                    self.assertTrue(pv["fused"])
                    self.assertEqual(s1[pv["r1_idx"]], pv["r1_base"])
                    self.assertEqual(ord(q1[pv["r1_idx"]]) - 33, pv["r1_q"])
                    # r2_idx 是原始 R2 方向索引，碱基需互补回去
                    comp = str.maketrans("ACGTN", "TGCAN")
                    self.assertEqual(s2[pv["r2_idx"]].translate(comp),
                                     pv["r2_base"])
                    self.assertEqual(ord(q2[pv["r2_idx"]]) - 33, pv["r2_q"])
                else:  # 非重叠端：原碱基原质量
                    self.assertFalse(pv["fused"])
                    src, s, q = (("r1", s1, q1) if pv["source"] == "R1"
                                 else ("r2", s2, q2))
                    idx = pv[f"{src}_idx"]
                    if src == "r1":
                        self.assertEqual(s[idx], pv["consensus_base"])
                    else:
                        self.assertEqual(
                            s[idx].translate(str.maketrans("ACGTN", "TGCAN")),
                            pv["consensus_base"])
                    self.assertEqual(ord(q[idx]) - 33, pv["consensus_q"])


class HandProbabilityTest(unittest.TestCase):
    """手工概率例：直接核对 fuse_two 的共识碱基与质量。"""

    def fuse(self, b1, q1, b2, q2):
        return merge.fuse_two(b1, q1, b2, q2, cap=60)

    def test_agree_adds_confidence(self):
        # Q20+Q20 一致：手算 P_err=3.4009e-5 -> Q44.68 -> 45
        self.assertEqual(self.fuse("A", 20, "A", 20), ("A", 45, "agree"))
        # Q30+Q30 一致：手算 Q64.76 -> 封顶 60
        self.assertEqual(self.fuse("A", 30, "A", 30), ("A", 60, "agree"))
        # Q40+Q40 一致：手算 Q84.77 -> 封顶 60
        self.assertEqual(self.fuse("A", 40, "A", 40), ("A", 60, "agree"))

    def test_conflict_downgrades(self):
        # A(Q40) vs T(Q10)：手算 P(A)=0.999034 -> Q30.15 -> 30，不是简单取高质
        self.assertEqual(self.fuse("A", 40, "T", 10), ("A", 30, "conflict_resolved"))

    def test_n_rules(self):
        self.assertEqual(self.fuse("N", 2, "G", 25), ("G", 25, "single_non_N"))
        self.assertEqual(self.fuse("N", 0, "N", 0), ("N", 0, "both_N"))

    def test_tie_to_n(self):
        # A(Q30) vs T(Q30)：后验 50/50，证据不足 -> N，Q=-10log10(0.5)=3
        self.assertEqual(self.fuse("A", 30, "T", 30), ("N", 3, "tie_to_N"))

    def test_not_just_pick_higher_quality_char(self):
        # 低质量冲突拉低高质量观测：Q40 遇 Q10 矛盾后 40 -> 30
        _, q, _ = self.fuse("A", 40, "T", 10)
        self.assertLess(q, 40)
        # 一致观测互相增强：Q20+Q20 -> 45 > 20
        _, q2, _ = self.fuse("A", 20, "A", 20)
        self.assertGreater(q2, 20)


class KnownFragmentTest(unittest.TestCase):
    """已知短片段：核对重叠位置与共识序列。"""

    def run_pair(self, r1rec, r2rec):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            write_fastq(td / "r1.fq", [r1rec])
            write_fastq(td / "r2.fq", [r2rec])
            result = merge.run_pipeline(td / "r1.fq", td / "r2.fq", PLAN)
            return result["pairs"]["frag"]

    def test_standard_overlap(self):
        f = FRAG120
        pr = self.run_pair(("frag/1", f[:95], "I" * 95),
                           ("frag/2", merge.revcomp(f[25:120]), "I" * 95))
        self.assertEqual(pr["status"], "merged")
        c = pr["compatible"][0]
        self.assertEqual((c.d, c.overlap), (25, 70))
        self.assertEqual(pr["consensus_seq"], f)  # 全等质量 Q40 -> 共识即原片段
        # 重叠区 Q40+Q40 封顶 60；非重叠两端保留原质量 Q40
        self.assertEqual(pr["consensus_qual"], "I" * 25 + "]" * 70 + "I" * 25)

    def test_short_fragment(self):
        f = FRAG60
        pr = self.run_pair(("frag/1", f[:50], "I" * 50),
                           ("frag/2", merge.revcomp(f[10:60]), "I" * 50))
        self.assertEqual(pr["status"], "merged")
        self.assertEqual((pr["compatible"][0].d, pr["compatible"][0].overlap),
                         (10, 40))
        self.assertEqual(pr["consensus_seq"], f)

    def test_dovetail_orientation(self):
        f = FRAG60
        pr = self.run_pair(("frag/1", f[10:60], "I" * 50),
                           ("frag/2", merge.revcomp(f[0:50]), "I" * 50))
        self.assertEqual(pr["status"], "merged")
        c = pr["compatible"][0]
        self.assertEqual((c.orientation, c.d, c.overlap), ("dovetail", -10, 40))
        self.assertEqual(pr["consensus_seq"], f)

    def test_min_overlap_boundary(self):
        f = FRAG120
        # 重叠 20 == min_overlap：可合并
        pr = self.run_pair(("frag/1", f[:95], "I" * 95),
                           ("frag/2", merge.revcomp(f[75:120]), "I" * 45))
        self.assertEqual(pr["status"], "merged")
        self.assertEqual(pr["compatible"][0].overlap, 20)
        # 重叠 19 < min_overlap：不在允许长度内，该位移不进入候选
        pr = self.run_pair(("frag/1", f[:95], "I" * 95),
                           ("frag/2", merge.revcomp(f[76:120]), "I" * 44))
        self.assertEqual(pr["status"], "no_overlap")
        self.assertEqual(pr["compatible"], [])
        self.assertTrue(all(c.overlap >= 20 for c in pr["candidates"]))

    def test_containment_is_not_terminal(self):
        f = FRAG120
        # R2rc 完全包含于 R1 内部，不是末端重叠，不得强行合并
        pr = self.run_pair(("frag/1", f[:95], "I" * 95),
                           ("frag/2", merge.revcomp(f[40:90]), "I" * 50))
        self.assertEqual(pr["status"], "no_overlap")


class ShuffleOrderTest(unittest.TestCase):
    """打乱 mate 文件顺序：配对与结论不变。"""

    def test_shuffled_inputs_same_outputs(self):
        r1 = list(read_fastq(FIX / "reads_R1.fastq").items())
        r2 = list(read_fastq(FIX / "reads_R2.fastq").items())
        rng = random.Random(7)
        rng.shuffle(r1)
        rng.shuffle(r2)
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            t1, t2 = Path(t1), Path(t2)
            # 基线
            base = merge.run_pipeline(FIX / "reads_R1.fastq",
                                      FIX / "reads_R2.fastq", PLAN)
            merge.write_outputs(base, t1)
            # 打乱顺序重跑（同名文件，仅记录顺序不同）
            write_fastq(t2 / "reads_R1.fastq", [(n, s, q) for n, (s, q) in r1])
            write_fastq(t2 / "reads_R2.fastq", [(n, s, q) for n, (s, q) in r2])
            shuf = merge.run_pipeline(t2 / "reads_R1.fastq",
                                      t2 / "reads_R2.fastq", PLAN)
            merge.write_outputs(shuf, t2)
            for fname in ("merged.fastq", "unmerged.fastq", "pairs.tsv",
                          "candidates.tsv", "provenance.tsv"):
                self.assertEqual((t1 / fname).read_text(),
                                 (t2 / fname).read_text(), fname)
            rep1 = json.loads((t1 / "report.json").read_text())
            rep2 = json.loads((t2 / "report.json").read_text())
            for rep in (rep1, rep2):  # 路径与来源行号属输入布局，规范化后比较
                rep.pop("inputs")
                for d in rep["diagnostics"]:
                    d["file"] = Path(d["file"]).name
                    d["line"] = 0
                for o in rep["orphans"]:
                    o["file"] = Path(o["file"]).name
                    o["line"] = 0
            self.assertEqual(rep1, rep2)


class BadRecordTest(unittest.TestCase):
    """坏记录与身份唯一性：保留来源诊断并剔除。"""

    def test_bad_records_and_duplicates(self):
        frag = FRAG120
        good1 = ("k1/1", frag[:95], "I" * 95)
        good2 = ("k1/2", merge.revcomp(frag[25:120]), "I" * 95)
        r1_recs = [
            good1,
            ("k2/1", "ACGT", "IIIII"),          # 序列/质量不等长
            ("k3/1", "ACGTX", "IIIII"),          # 非法碱基
            ("k4/1", "ACGT", "IIII"),            # 重复键（两条）
            ("k4/1", "TGCA", "IIII"),
            ("k5/1", "ACGT", "I I I I".replace(" ", "\x14")),  # 非法质量字符
        ]
        r2_recs = [good2,
                   ("k2/2", "ACGT", "IIII"),
                   ("k3/2", "ACGT", "IIII"),
                   ("k4/2", "ACGT", "IIII"),
                   ("k6/2", "ACGT", "IIII")]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            write_fastq(td / "r1.fq", r1_recs)
            write_fastq(td / "r2.fq", r2_recs)
            result = merge.run_pipeline(td / "r1.fq", td / "r2.fq", PLAN)
            # 好记录不受影响
            self.assertEqual(result["pairs"]["k1"]["status"], "merged")
            self.assertEqual(result["pairs"]["k1"]["consensus_seq"], frag)
            # 坏记录全部有来源诊断（文件、行、原因）
            bad = [d for d in result["diagnostics"] if "k2/1" in d["name"]
                   or "k3/1" in d["name"] or "k5/1" in d["name"]]
            self.assertEqual(len(bad), 3)
            for d in bad:
                self.assertTrue(d["file"] and d["line"] > 0 and d["reason"])
            # 重复键两条都剔除并诊断
            dup = [d for d in result["diagnostics"] if "身份不唯一" in d["reason"]]
            self.assertEqual(len(dup), 2)
            self.assertTrue(all("k4/1" == d["name"] for d in dup))
            # 被剔除记录的 mate 成为孤儿（k2/k3/k4 的 R2、k6 的 R2）
            orphan_keys = sorted(r.key for r in result["orphans"])
            self.assertEqual(orphan_keys, ["k2", "k3", "k4", "k6"])
            self.assertNotIn("k2", result["pairs"])

    def test_truncated_file(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "r1.fq").write_text("@a/1\nACGT\n+\n")  # 3 行，不完整
            (td / "r2.fq").write_text("")
            result = merge.run_pipeline(td / "r1.fq", td / "r2.fq", PLAN)
            self.assertEqual(result["pairs"], {})
            self.assertTrue(any("4 的整数倍" in d["reason"]
                                for d in result["diagnostics"]))


if __name__ == "__main__":
    unittest.main()
