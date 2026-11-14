# 剪接 BAM 材料

本目录仅含题目输入，不含待实现程序。全部材料为本地合成，无真实个人或仪器数据。Python 3.11；若使用列出的库，可执行 `python -m pip install -r requirements.txt`。无需 Docker、账号或网络数据下载。

reads.bam 已按坐标排序，不按 query_name 邻接；reads.bam.bai 与 reference.fa.fai 是真实索引。exons.tsv 是制表符文件，start0/end0 为零基半开基因组坐标，exon_index 按转录本方向排列，负链为高坐标到低坐标。

文库约定为 read1 的比对链与转录本同向，read2 相反；两个 mate 之间未测到的区域不构成剪接证据。CIGAR 的 D 仍消耗参考位置，N 才是需要精确核对的内含子跳跃。输出 fragment 级兼容集合，不按 read 或比对记录数重复计数。
