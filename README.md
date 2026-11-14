# 有代价约束的可读分流规则学习工具（精确求解）

把人工标记案例压缩成**可人工编辑的有序分流规则**：每条规则 = 一个题目给定谓词
+ 一个目标队列，按顺序首次命中即停止，其余进入默认队列。最小化
「案例权重 × 错送代价」之和 + 每条规则的维护代价。

- 纯 Python 3.11 标准库，**不调用外部模型**，不执行真实业务分流；
- 不做黑箱分类器、特征搜索、在线更新或 SQL 个案放宽；
- 在 ≤12 谓词 / ≤50 案例 / ≤3 队列范围内给出**全局最优**（子集动态规划，
  非单条规则贪心）；
- 输出一份与输入行顺序无关的稳定主方案，并核验未标记探针在所有同值最优方案间
  是否存在分流歧义；
- 错误案例如实报告，绝不修改原始标签。

方法的完整状态定义、转移、最优性证明、剪枝与复杂度见 [METHOD.md](METHOD.md)。

## 输入材料

`fixtures/` 与 `cases/ambiguity/` 是题目输入，所有业务与观测数据均为本地合成
数据；`cases/` 中的文件整体替换同名基础输入，不混用基础谓词或代价。小材料用于
调试与核验，工具不通过写死文件名、行号或样例答案工作。

## 目录

- `rulelearn/` 工具源码
  - `data.py` 输入加载与校验（字段完整性、算子、代价矩阵）
  - `predicates.py` 谓词求值（仅 eq/ge/le）与案例位掩码
  - `solver.py` 子集 DP、受约束后缀 DP、规范方案重建、探针歧义 DFS
  - `replay.py` 与求解器分离的纯规则重放（apply/verify 共用）
  - `report.py` 规则 JSON 与结构化例外清单组装
  - `brute_force.py` 独立全穷举器（仅用于核验，≤4 谓词）
  - `benchmark.py` 自制 4 谓词 / 8 案例穷举基准
  - `api.py` / `cli.py` `fit` / `apply` / `verify` 三个入口
- `tests/stress_vs_bruteforce.py` 300 组随机实例 DP 对穷举压力核验
- `fixtures/` 主输入；`cases/ambiguity/` 探针歧义微型输入
- `out/` 运行产物（`rules.json`、`exceptions.json`、`report.json`、
  `verification.json`）

## 使用

```bash
# 1) 学习规则（主输入）
python3 -m rulelearn fit --input-dir fixtures --output-dir out/main

# 2) 从保存的可检查规则重新分流未标记探针（不重训、不读标签）
python3 -m rulelearn apply --rules out/main/rules.json --input-csv fixtures/probes.csv

# 3) 独立重放核验 + 自制穷举基准交叉核对
python3 -m rulelearn verify --input-dir fixtures --output-dir out/main

# 歧义微型输入（演示有限标签不能决定探针分流）
python3 -m rulelearn fit    --input-dir cases/ambiguity --output-dir out/ambiguity
python3 -m rulelearn verify --input-dir cases/ambiguity --output-dir out/ambiguity

# 随机压力核验：DP 与全穷举在 300 组小实例上逐项一致
python3 tests/stress_vs_bruteforce.py
```

也可作为库调用：`from rulelearn import fit, apply, verify`。

## 核验会证明什么

- 重放目标值与保存值一致；打乱输入行顺序后最优值与主方案不变；
- 自制基准上 DP 与独立穷举的最优值、最优方案集合、探针歧义完全一致；
- **规则顺序确实改变结果**（基准中同集合两种顺序目标值 28 vs 8）；
- **最短规则未必代价最小**（0 条规则值 16，全局最优为 6）；
- 探针歧义由两份目标值同为最优、但分流不同的方案给出证据，而非用主方案的
  单一预测冒充唯一结论。
