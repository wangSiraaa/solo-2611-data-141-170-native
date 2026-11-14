# 配对 FASTQ 输入

`fixtures/` 含题目输入；`merge.py` 为已实现程序。全部材料为本地合成，无真实个人或仪器数据。Python 3.11；若使用列出的库，可执行 `python -m pip install -r requirements.txt`。无需 Docker、账号或网络数据下载。

`fixtures/reads_R1.fastq` 与 R2 按去掉 `/1`、`/2` 后的完整名字配对，行序不同。不存在的 mate 应作为孤儿保留。

plan 给出支持边界，序列是 A/C/G/T/N，质量字符是 Phred+33。R2 是第二端实际测到的方向，需同时反向互补碱基并反转质量。输入有重叠冲突、重复序列和无重叠片段。位置证据与碱基质量是不同层次，无法识别位置时不能发布确定共识。

## 实现与用法

`merge.py` 为本地 merge 命令（Python 3.11，仅标准库）：

```sh
python3 merge.py            # 默认读 fixtures/，输出到 out/
python3 merge.py --r1 fixtures/reads_R1.fastq --r2 fixtures/reads_R2.fastq \
                 --plan fixtures/plan.json --outdir out
```

产物：`merged.fastq`（仅位置唯一确定的配对）、`unmerged.fastq`（歧义/无重叠
配对的原始两端与孤儿）、`pairs.tsv`（逐配对状态与候选依据）、
`candidates.tsv`（全部候选评分）、`provenance.tsv`（逐输出位置来源）、
`report.json`（机器可读汇总）。参数与规则详见 `PARAMETERS.md`。

验证：`python3 -m unittest discover -s tests -v`（23 例，覆盖 fixtures 各
故事、手工概率例、已知短片段重叠位置、打乱顺序不变性、坏记录诊断）。
