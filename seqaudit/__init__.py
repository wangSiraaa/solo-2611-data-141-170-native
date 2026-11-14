"""seqaudit — 有限总体无放回抽查的风险受控顺序审计工具。

对整批 N 件（<=120）单一二元属性的批次，按 plan 冻结的抽查顺序逐批查看，
用混合似然比检验鞅给出"接受/继续"结论，并保证：对任意满足零假设
（缺陷总数 D >= d0）的批次，在任意查看时点曾被错误接受的概率 <= alpha。

入口：
    python -m seqaudit audit  --plan P --lot L --observations O [--out FILE]
    python -m seqaudit verify --plan P --lot L --observations O --certificate C [--out FILE]
    python -m seqaudit verify --tiny fixtures/tiny.json [--out FILE]
"""

__version__ = "1.0.0"
