# 字节级 BPE：增量训练、按秩编码、精确解码

纯 Python 3.11 标准库实现，不调用任何现成 tokenizer 库，不训练语言模型。

## 布局

```
bpe/
  core.py       严格 UTF-8、空白/非空白 run 预切分、符号表（字节序列即身份）
  corpus.py     语料/配置/记录文件校验（非法输入明确报错，带行号）
  trainer.py    增量 BPE 训练（链式位置身份 + 懒惰堆 + 计数器）
  encoder.py    按 merge 秩编码、字节精确解码、半开字节 offset
  artifacts.py  四个 JSON 工件的写出、预检、加载
  cli.py        train / precheck / encode / decode 子命令
tests/          50 个 unittest 用例（python3 -m unittest discover -s tests）
```

## 训练（`bpe/trainer.py`）

- 语料每行 `{"text": str, "count": 正整数}`，文件按严格 UTF-8 读取；
  非法 UTF-8、孤立代理项、非正整数 count 均明确失败。
- 预切分：连续 Unicode 空白 run 与连续非空白 run；run 铺满全文、保留全部
  原字节；绝不跨 run 合并，不做任何 Unicode 归一化。相同字节的 run 合并为
  一个加权序列（权重累加），并按字节排序建表——输入行重排不改变结果。
- 基础词表为 256 个单字节（id == 字节值）。符号身份即字节序列：相同字节
  复用已有 id，新符号按首次创建顺序取新 id。
- **位置身份**：每个序列是平行数组上的双向链表，节点下标即稳定位置身份；
  合并把右节点从链上摘除，失效位置因此永远不可达。
- **出现位置与频次**：`pair_positions` 精确维护每个活相邻对的
  `(word, left_node)` 集合，`pair_count` 维护对应加权频次。每次替换只
  动受影响的三条旧邻接（左/自身/右）与两条新邻接；重叠出现由"从左到右
  走链表 + 替换后跳到合并点之后"自然消解（`aaaaa` 中 4 个 `aa` 出现只
  替换 2 处），计数（含重叠）与替换（非重叠）不会混淆。
- **最佳候选**：懒惰最大堆，键 `(-count, left bytes, right bytes)`，并列
  按声明的原始字节对字典序裁决；计数变化只推新条目，弹出的过期条目计入
  `candidates_evicted`。
- **增量证据**：`summary.json` 记录 `positions_checked`（替换走查实际
  检查的节点数）、`adjacency_updates`、`candidates_evicted`，并与
  `full_rescan_positions_equivalent`（若每轮全量重扫需检查的位置数）
  对照。本 fixtures 上为 122 vs 765；只有初始化做了一次全量扫描
  （`initial_scan_positions`）。
- 结束条件：达到 `merge_steps`，或最佳候选加权频次低于
  `min_weighted_pair_count`，或已无相邻对（`stop_reason` 相应记录）。

## 工件（`artifacts/`，纯 JSON，无可执行代码）

- `vocab.json`：id → 十六进制字节序列（0..255 为单字节）。
- `merges.json`：rank、left/right/new_id、字节 hex、加权频次、替换出现数。
- `config.json`：`merge_steps`、`min_weighted_pair_count` 及说明字段。
- `summary.json`：停止原因、上述增量计数器、逐轮日志。

`python3 -m bpe precheck --artifacts DIR` 预检：id 连续、0..255 为对应单
字节、无重复字节序列、rank 连续、`new_id == 256 + rank`、left/right 引用
已有 id、`bytes(left)+bytes(right) == bytes(new_id)`、摘要与词表/merge
互相一致。矛盾合并引用会逐条报错并以非零码退出。

## 编码 / 解码（`bpe/encoder.py`）

- 新进程仅凭工件编码：严格 UTF-8 → 同样的 run 预切分 → 每个 run 内**按
  merge 秩序**逐轮替换（不是词表最长匹配；测试用 `aa/aaa` 工件证明
  `aaaa → [aa][aa]` 而非 `[aaa][a]`）。未见字符回退为单字节 token。
- 每个 token 给出原 UTF-8 字节串上的半开区间 `[start, end)`，铺满
  `[0, len)`；token 可以只含多字节字符的一部分（测试用"汉"前两字节
  的合并验证）。
- 解码拼接全部 token 字节后严格 UTF-8 解码，与输入逐字节一致（空白、
  换行、组合字符不变）；未知 token id 明确失败。

## 命令

```sh
python3 -m bpe train --corpus fixtures/corpus.jsonl --config fixtures/config.json --out artifacts
python3 -m bpe precheck --artifacts artifacts
python3 -m bpe encode --artifacts artifacts --input fixtures/unseen.txt --out enc.jsonl
python3 -m bpe encode --artifacts artifacts --jsonl records.jsonl --out enc.jsonl   # 按文本记录分批
python3 -m bpe decode --artifacts artifacts --input enc.jsonl [--raw out.bin]
```

编码输出每行 `{"record", "byte_length", "tokens": [{"id","start","end","hex"}]}`；
解码会校验 token 区间确实铺满整条记录。分批编码按记录独立进行，分批方式
不改变任何记录的结果（有测试锁定）。

## 测试（50 例，`python3 -m unittest discover -s tests`）

- `test_tiny_recount.py`：独立全量 recount 基准（仅用于 tiny），逐轮核对
  所选对、完整加权频次表、替换后序列；显式覆盖 `aaaaa`/`ababa` 的重叠、
  失效位置清空、过期候选淘汰、每轮后位置索引不变式、输入重排不变性。
- `test_encoder.py`：秩序 ≠ 最长匹配、offset 铺满、多字节字符被 token
  切分、组合字符往返、非法 UTF-8 失败、分批一致性。
- `test_artifacts.py`：合法工件通过；未知引用、字节不一致、rank 断档、
  重复符号、配置/摘要矛盾均失败。
- `test_cross_process.py`：子进程跑 train/precheck/encode/decode，处理
  `unseen.txt`（未见汉字、emoji、4 字节字符、连续空白）与 `empty.txt`，
  核对 offset 铺满与逐字节往返；分批 JSONL 记录跨进程结果一致。
- `test_corpus.py`：语料/配置校验与全量语料上的增量计数器 sanity。
