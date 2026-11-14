"""CIGAR 解码单元测试：M/=/X/I/D/N/S 分类与坐标推进。"""
from rnacount.cigarops import decode_cigar

ACCEPTED = frozenset({"M", "=", "X", "I", "D", "N", "S"})
# pysam 操作码: M=0 I=1 D=2 N=3 S=4 H=5 P=6 ==7 X=8 B=9


def test_plain_M():
    fp = decode_cigar([(0, 20)], 110, ACCEPTED)
    assert fp.blocks == [(110, 130)]
    assert fp.deletions == [] and fp.introns == []
    assert fp.ref_end == 130 and fp.unsupported_ops == []


def test_soft_clip_consumes_no_reference():
    fp = decode_cigar([(4, 5), (0, 20)], 315, ACCEPTED)  # 5S20M
    assert fp.blocks == [(315, 335)]
    assert fp.ref_end == 335


def test_spliced_cigar_blocks_and_intron():
    fp = decode_cigar([(0, 20), (3, 60), (0, 20)], 120, ACCEPTED)  # 20M60N20M
    assert fp.blocks == [(120, 140), (200, 220)]
    assert fp.introns == [(140, 200)]
    assert fp.deletions == []
    assert fp.ref_end == 220


def test_indel_cigar_deletion_occupies_reference():
    # 10M2I10M2D8M @810：I 不占参考，D 占 830-832
    fp = decode_cigar([(0, 10), (1, 2), (0, 10), (2, 2), (0, 8)], 810, ACCEPTED)
    assert fp.blocks == [(810, 820), (820, 830), (832, 840)]
    assert fp.deletions == [(830, 832)]
    assert fp.introns == []
    assert fp.ref_end == 840


def test_eq_and_X_are_blocks():
    fp = decode_cigar([(7, 5), (8, 5)], 100, ACCEPTED)  # 5=5X
    assert fp.blocks == [(100, 105), (105, 110)]
    assert fp.unsupported_ops == []


def test_hard_clip_flagged_unsupported_but_coordinates_intact():
    fp = decode_cigar([(5, 5), (0, 20)], 810, ACCEPTED)  # 5H20M
    assert fp.unsupported_ops == ["H"]
    assert fp.blocks == [(810, 830)]  # H 不消耗参考，坐标不受影响
    assert fp.ref_end == 830


def test_pad_and_back_flagged_unsupported():
    fp = decode_cigar([(0, 10), (6, 2), (0, 10)], 0, ACCEPTED)  # 10M2P10M
    assert fp.unsupported_ops == ["P"]
    assert fp.blocks == [(0, 10), (10, 20)]
