# 有限候选基因树

本目录仅含题目输入，不含待实现程序。全部材料为本地合成，无真实个人或仪器数据。Python 3.11；若使用列出的库，可执行 `python -m pip install -r requirements.txt`。无需 Docker、账号或网络数据下载。

aligned FASTA 的五条序列已对齐且等长，N 视为该位点未知，不视为第五种碱基。Newick 每行一棵候选，无根二叉树用三叉根序列化；所有叶名恰好为 FASTA 的同一集合。输入枝长只作初始化，应在声明界内重新拟合。

JC69 的同态转移概率为 1/4+3/4 exp(-4t/3)，异态为 1/4-1/4 exp(-4t/3)，根状态均匀。比较同一 bootstrap 位点样本的全部候选，不做拓扑搜索、序列对齐或贝叶斯后验推断。

## 实现：`compare` 命令

实现在 `phylocompare/` 包，完整规则、公开参数与输出说明见 [docs/USAGE.md](docs/USAGE.md)。

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m phylocompare compare --plan fixtures/plan.json
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/verify_enumeration.py
```

要点：对数空间 Felsenstein 剪枝（模式压缩、`logsumexp` 防下溢，N 对四碱基发射似然为 1）；枝长在公开界 `[1e-5, 1.5]` 内以 4 个公开起点重拟合并记录全部证据与触界参数；24 次共享列 bootstrap、每次重新拟合；输出全数据排名、配对似然差、严格/近乎并列支持频率（经验频率，非贝叶斯后验）与位点模式统计到 `reports/`。
