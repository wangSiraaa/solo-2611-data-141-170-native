# compare 命令实现说明与使用

## 1. 运行方式

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # 运行依赖
.venv/bin/pip install -r requirements-dev.txt      # 测试依赖（可选）

# 对 plan.json 中全部 cases 运行比较，报告写入 reports/
.venv/bin/python -m phylocompare compare --plan fixtures/plan.json

# 只跑一个数据；可用 --fasta 指向 plan 之外的 FASTA
.venv/bin/python -m phylocompare compare --plan fixtures/plan.json --case weak.fasta

# 覆盖 bootstrap 次数（计划上限 24）或种子
.venv/bin/python -m phylocompare compare --plan fixtures/plan.json --replicates 12 --seed 271828
```

演示与独立核对脚本：

```sh
.venv/bin/python scripts/verify_enumeration.py     # 枚举内部碱基核对剪枝
.venv/bin/python scripts/run_demo.py               # 端到端演示 + 共享列核对
.venv/bin/python -m pytest tests/ -q               # 14 个单元/性质测试
```

## 2. 输入校验（任何比较开始之前）

* FASTA 必须等长、叶名唯一、只含 `ACGTN`（大小写不敏感）；空序列、缺 `>` 头、
  非有限枝长等均报明确错误。
* 固定 5 个分类单元；候选树 1..4 棵；每棵树的叶集合必须与 FASTA 完全一致
  （报告缺少/多余的叶名）。
* 候选为无根二叉树，以三叉根 Newick 序列化：根恰好 3 个孩子且无根枝长，
  其余内部节点均为二叉；叶度为 1、内部度为 3、边数恰为 `2n-3 = 7`。
* 候选之间拓扑重复会直接报错，避免同一拓扑被重复计票。

## 3. 似然引擎（真正的树似然，不是两两距离之和）

* JC69：同态 `1/4 + 3/4·exp(-4t/3)`，异态 `1/4 - 1/4·exp(-4t/3)`，根状态均匀。
* Felsenstein 剪枝在**对数空间**进行：部分对数似然用 `scipy.special.logsumexp`
  组合，位点数再大也不会下溢，不依赖位点级重标定；20 万位点的长序列仍给出
  有限对数似然（见 `scripts/verify_enumeration.py`）。
* 重复位点先压缩成「模式 + 计数」（5 进制编码，N 占第 5 个符号），同一模式只
  剪枝一次、按计数加权。
* `N` 不是第五种碱基：其发射似然对四个潜隐碱基均为 1，即在该叶上对四个状态
  边际化。该性质有枚举测试与边际化等式测试覆盖。
* 边按其诱导的叶集划分（较小一侧的排序叶元组）做规范编号，因此 Newick 根孩子
  顺序、叶名排列变化都不会改变参数身份与结果（有专门的排列不变性测试）。

## 4. 枝长重拟合（公开的初值、重试与收敛规则）

* 边界 `[1e-5, 1.5]` 来自 `plan.json`，对所有候选、所有数据集、每次 bootstrap
  都相同；Newick 枝长只是初值，若落在界外先裁到界内。
* 固定 4 个多起点：裁剪后的 Newick 初值、统一 0.02、统一 0.2、统一 0.6。
* 优化器为 SciPy `L-BFGS-B`，收敛规则 `gtol=1e-8, ftol=1e-11, maxiter=200`；
  取所有成功起点中似然最高者。每个起点的初值、成功标志、消息、迭代/求值次数、
  最终似然都逐字写入报告。
* 距边界 ≤ `1e-6` 的边记录为「触界参数」（含边身份、上/下界、实际值）。
* 若某候选**全部起点失败**，标记 `optimization_ok=false` 并保留 `failure_reason`
  与各起点证据，不参与排名、不填「最差数」；全部候选失败时命令报错退出，
  绝不产出假装完整的比较。单元测试用合成数值异常验证该路径。
* 优化过程无随机数，结果确定；bootstrap 的唯一随机来源是第 5 节的 RNG。

## 5. 共享列重采样 bootstrap

* 次数固定 24（CLI 允许 1..24），种子 `271828`，RNG 为
  `numpy.random.RandomState(seed)` 的 `randint`，列矩阵形状
  `(n_replicates, n_sites)`，随 numpy 版本锁定（requirements.txt）。
* **同一组列用于该数据集的全部候选**：报告里每个 replicate 只存一份列指纹，
  演示脚本还会独立重放并核对三个候选的重采样模式指纹完全相同。
* 每次 replicate 都在重采样后的模式上**重新拟合全部枝长**，不复用全数据枝长；
  测试核对不同候选的重拟合枝长确实不同。
* 每个 replicate 记录：严格最优候选、近乎并列的领先者（阈值
  `|ΔlnL| ≤ 0.5`，公开在报告参数中）、各候选似然、重拟合枝长与触界边。

## 6. 输出与解读

报告为单个 JSON（`reports/<case>.report.json`，`allow_nan=False`，无 NaN/Infinity）：

* `run_parameters`：模型、边界、多起点、优化器与收敛规则、bootstrap 参数、
  近乎并列阈值、似然引擎、以及「支持频率不是贝叶斯后验概率」的显式声明；
* `site_patterns`：位点数、分类单元、不同模式数、含 N 位点数、恒定/可变位点、
  出现频次前 10 的模式；
* `candidates`：每候选的拓扑（规范边 key）、优化状态、全数据对数似然、实际
  枝长、触界参数、多起点证据、与最优候选的 ΔlnL；
* `full_data_ranking`、`pairwise_bootstrap`（配对 ΔlnL 的均值/标准差/胜负计数）、
  `bootstrap_support_frequencies`（严格最优频率、最优或近乎并列频率、失败次数）；
* `bootstrap_replicates`：24 次重采样的完整逐次记录与列指纹。

支持频率是「重采样样本中该候选经验最优」的频率，**不是贝叶斯后验概率**，
不做任何后验概率声明。近乎并列时两个候选都计入 `best_or_near_tied_frequency`，
从而保留短序列无法分辨的证据。

## 7. 目录结构

```
phylocompare/   包：fasta / newick / tree / model / optim / bootstrap / compare / cli
tests/          14 个测试：输入校验、枚举核对、N、叶序、排列、共享列、失败路径
scripts/        verify_enumeration.py（独立枚举核对）、run_demo.py（端到端演示）
reports/        生成的 JSON 报告
```
