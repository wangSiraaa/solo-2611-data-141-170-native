"""注释载入测试：exon_index 按转录本方向，负链内含子不能按起点升序错算。"""
import pytest

from rnacount.annotation import load_exons

TX = {t.transcript_id: t for t in load_exons("fixtures/exons.tsv", "chrS")}


def test_plus_strand_introns():
    assert TX["A1"].introns == {(140, 200), (240, 300)}
    assert TX["A2"].introns == {(140, 250), (290, 300)}
    assert TX["A3"].introns == {(140, 200), (240, 360)}


def test_minus_strand_exon_order_and_introns():
    # B1 的 exon_index 1 在高坐标；内含子仍是转录本方向相邻外显子之间的区间
    b1 = TX["B1"]
    assert b1.exons_tx == [(700, 740), (600, 640), (500, 540)]
    assert b1.exons_genomic == [(500, 540), (600, 640), (700, 740)]
    assert b1.introns == {(540, 600), (640, 700)}
    assert TX["B2"].introns == {(540, 650), (690, 700)}


def test_single_exon_transcript_has_no_introns():
    assert TX["C1"].exons_tx == [(800, 880)]
    assert TX["C1"].introns == frozenset()


def test_covers():
    a1 = TX["A1"]
    assert a1.covers(100, 140)  # 完整外显子
    assert a1.covers(110, 130)  # 外显子内部
    assert not a1.covers(139, 141)  # 跨出外显子边界
    assert not a1.covers(140, 200)  # 内含子
    assert not a1.covers(99, 140)  # 越过左边界
    assert not a1.covers(200, 300)  # 跨两个外显子（中间是内含子）


def test_strand_and_gene():
    assert TX["B1"].strand == "-" and TX["B1"].gene_id == "B"
    assert TX["C1"].strand == "+"


def test_bad_annotation_rejected(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text(
        "transcript_id\tgene_id\tstrand\texon_index\tstart0\tend0\n"
        "X1\tX\t+\t1\t100\t200\n"
        "X1\tX\t+\t3\t300\t400\n",  # exon_index 跳号
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exon_index"):
        load_exons(str(p), "chrS")
