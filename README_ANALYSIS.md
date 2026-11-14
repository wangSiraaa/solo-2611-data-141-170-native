# 可比月份价格指数：商品更替下的离线分析

本目录给出一个可解释、可追溯的离线分析实现（Python 3.11 + NumPy/SciPy），
用于在商品更替与供应商重复报价的情况下，估计**同货币、固定物理单位**商品的
6 个月共同价格变化，并给出支持证据与不确定性。

输入为题目给定的 `fixtures/quotes.csv`、`fixtures/products.csv`、
`fixtures/plan.json`，以及 `cases/single_bridge.csv`、`cases/disconnected.csv`。
不写死文件名、行号或样例答案；所有路径由命令行或 API 传入。

## 方法（唯一一套，可解释）

对对数价格使用加性效应模型：

```
log price(s, m, i) = item_level(i) + month_effect(m) + supplier_level(s) + error
```

- `item_level(i)`：SKU 在本窗口内持续的相对价格水平；
- `month_effect(m)`：各 SKU 共同承受的月份变化；
- `supplier_level(s)`：供应商持续的报价水平；
- 误差：允许同一供应商轨迹内相关。

**权重**：每条供应商轨迹获得相等的总权重。某供应商的一条报价基础权重为
`1 / n_s`（`n_s` 为该供应商的报价条数），再整体缩放到均值 1。这样报价更多
SKU/月份的供应商不会主导共同月份变化；权重同样施加于每个重抽样复制。

**误差假设**：`Var(error) = sigma^2 / base_weight`；残差尺度用 MAD 估计。

**异常报价处置**：不删除任何报价，而是在 IRLS 中用 Huber 权重
（调节常数 1.345，基于 MAD 标准化残差）对异常报价降权；最终权重 = 基础权重 ×
Huber 权重。降权报价仍保留在其设计位置，因此观测支持不会被悄悄移除。

**归一化**：仅有两个常规归一化——基期月份效应固定为 0（指数=100），
参考供应商水平固定为 0（`supplier-1`，按标识符排序的第一个）。除此之外，
不做任何额外约束。

**商品语义**：`products.csv` 的 `family` 只声明替代关系，**绝不用作等价商品键**。
新旧 SKU 的质量/价格差异只由实际观测到的重叠报价识别；没有重叠时报告
“不可识别”，不用外部商品价格补桥。

### 可识别性诊断（精确有理数）

`identification.py` 在有理数（`fractions.Fraction`）上对（去重后的）
供应商×月份×商品单元做精确 RREF：

- 给出精确零空间与极大无关列集（pivot 列）；拟合只在这组列上进行，
  **不让数值库自动丢列、也不以矩阵秩阈值判定可识别性**；
- 对每个月份对比做精确可识别性检验；
- 若某月份不可识别，输出具体的**混淆方向**（商品水平与月份效应如何共同平移
  而观测报价不变），并给出**见证报价编号**；
- 输出商品–月份二部图的连通分量（边仅来自真实报价，不含家族名）；
- 逐一删除整个供应商，报告哪些月份失去可识别性、新增分量，以及作为唯一
  跨组连接的**桥接报价**（含 `quote_id`、月份、商品、价格）。

### 不确定性（保留供应商内依赖）

以**整条供应商轨迹**为重抽样单位做 bootstrap（有放回）。每个复制都重新做精确
可识别性检验：

- 若复制中某对比不可识别，记为**失败**，绝不填零、绝不用外部信息补桥、
  绝不悄悄跳过；
- 只有可识别复制比例达到 80% 且至少 20 个复制时，才报告百分位区间；
- 否则输出点估计（若全样本可识别）与状态 `insufficient_replicate_support`，
  区间**不予报告**，并列出失败原因、可用复制比例；
- 同时给出**留一供应商**稳定性表与关键桥接供应商列表，供判断区间是否可报告。

### 独立微型核对（`verify.py`）

1. **固定 SKU + 给定共同倍数涨价**：无噪声微型数据，恢复的指数必须精确等于注入
   的倍数（误差 < 1e-6 指数点）。
2. **整体乘常数 / 行顺序变化**：所有价格乘同一常数、或打乱报价行顺序，指数不变。
3. **唯一桥接供应商删除**：仅一名供应商连接新旧两组 SKU 的微型设计中，删除该
   整条轨迹后，轮换后月份必须变为 `insufficient_support` 且区间消失——若只是
   指数轻微变动即为失败。
4. **家族标签改名**：只改 `family` 文本不改变任何指数（证明未把家族名当键）。
5. **案例级结论**：base / single_bridge / disconnected 的状态与桥接证据。

## 目录

```
price_link/
  __init__.py        # 公共 API: load_inputs / fit / explain
  data.py            # 输入加载与严格校验
  identification.py  # 精确有理数可识别性、图证据、桥接证据
  estimator.py       # 供应商平衡 + Huber 稳健 WLS、bootstrap、稳定性
  explain.py         # 文本/JSON/Markdown/HTML 报告与逐报价追溯 CSV
  verify.py          # 微型数据独立核对
  cli.py, __main__.py# 命令行驱动
tests/test_index.py  # pytest 测试
```

## 运行

```sh
python3 -m venv .runtime/venv
.runtime/venv/bin/python -m pip install -r requirements.txt pytest

# 基础案例
.runtime/venv/bin/python -m price_link \
    --quotes fixtures/quotes.csv --products fixtures/products.csv \
    --plan fixtures/plan.json --out outputs

# 基础 + 单桥 + 断桥 + 微型核对，并生成跨案例对比
.runtime/venv/bin/python -m price_link --run-all --out outputs

# 测试
.runtime/venv/bin/python -m pytest tests -q
```

主要输出（`outputs/`）：

- `<case>.result.json`：完整结构化结果（模型、可识别性、估计、不确定性、稳定性、
  逐报价追溯）；
- `<case>.explain.txt` / `<case>.report.md` / `<case>.report.html`：静态报告；
- `<case>.quote_trace.csv`：每条报价 → 商品水平 + 月份效应 + 供应商水平 + 残差；
- `micro_checks.json`：微型核对结果；
- `case_comparison.{json,md}`：base / single_bridge / disconnected 结论对比。

## 主要结论（本数据）

- **base**：六个月完全连接，所有月份可识别；指数约
  100 → 102.1 → 105.3 → 106.6 → 108.8 → 112.9（2025-01 起基期 100），
  全部复制保持可识别，区间可报告。旧/新 SKU 的实测水平比：
  A-new/A-old ≈ 1.22、B-new/B-old ≈ 1.09、C-new/C-old ≈ 0.96（**不是质量等价声明**）。
- **single_bridge**：仅 `supplier-1` 在 2025-03/04 同时报旧/新 SKU（见
  `q0008`、`q0010`、`q0012`、`q0014`）。全样本仍可识别，但删除该供应商会使
  2025-04/05/06 失去识别；重抽样中只有约 68% 复制保持可识别，故这些月份
  点估计保留、**区间撤回**，并标注关键桥接供应商。
- **disconnected**：旧 SKU（01–03）与新 SKU（04–06）之间没有任何共同报价，
  2025-04/05/06 相对基期**不可识别**，指数与区间均不报告；给出精确混淆方向、
  见证报价与“家族跨分量”证据。2025-01–03 仍可识别。

## 局限

- 加性对数模型假设窗口内各 SKU 相对价格水平持续；若存在窗口内相对漂移，
  共同月份效应会被拉向平均漂移。
- 家族仅表示替代关系；报告的新旧价差是实测持续水平差，不是质量或等价结论。
- 可识别性是对观测报价模式的条件性精确结论；无重叠时明确报告支持不足，
  不用外部价格补桥。
- 供应商内相关仅通过整轨迹重抽样处理，未拟合参数化协方差模型。
- 有限样本百分位区间为描述性结果，不声称精确覆盖；当桥接仅由单一供应商
  支撑时，留一稳定性表会直接暴露这一点。
- 不做预测、享乐价格回归、收入 Shapley 分解、滚动预测或统计平台。
