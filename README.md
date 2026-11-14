# 调查非响应的有界补权与不确定性报告

本目录是题目输入材料，未包含答案实现。所有业务与观测数据均为本地合成数据。

## 文件

- `fixtures/respondents.csv`
- `fixtures/margins.csv`
- `fixtures/replicate_weights.csv`
- `fixtures/plan.json`
- `cases/replicate_no_older.csv`
- `cases/infeasible_margins.csv`

以题面规定的字段语义与边界为准。cases/ 中的文件只替换说明中指明的输入，其他基础输入保持不变。小材料用于调试与核验，不能通过写死文件名、行号或样例答案完成任务。

运行环境：Python 3.11。可用 `python -m pip install -r requirements.txt` 安装普通依赖；不需要 Docker。

---

# 有界熵补权与设计重复方差工具（实现）

离线、确定性、非响应的**有界加权**工具：在给定上下界内让调整权重满足已知人口边际，
并以熵距离最小化偏离基础设计权重。附带静态报告、约束/界支持证据、设计重复
（replicate）方差，以及独立核验。

不抽取新样本、不补造未响应者、不做多重插补/回归建模/通用统计平台。

## 安装与运行

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pytest
```

三个子命令共享同一确定性管线，各自在控制台聚焦自己的产物，并都会写出静态报告：

```sh
# 基础数据（fixtures）
.venv/bin/python -m reweight fit      --out out/base
.venv/bin/python -m reweight estimate --out out/base
.venv/bin/python -m reweight verify   --out out/base

# 不可行边际（cases/）
.venv/bin/python -m reweight fit --margins cases/infeasible_margins.csv --out out/infeasible

# 缺少 older 组的重复权重（cases/）
.venv/bin/python -m reweight fit --replicate cases/replicate_no_older.csv --out out/no_older
```

退出码：`0` 可行并求解成功；`2` 基础数据不可行；`1` 求解器失败（收敛性/残差/
最优性检查未通过）。输入路径均可通过 `--respondents/--margins/--plan/--replicate` 覆盖。

## 产物

每个输出目录包含：

- `weights.csv` — 每名响应者的基础权重、上下界、调整权重、是否命中上下界，可逐人追溯。
- `margin_residuals.csv` — 每个边际的目标、实现、残差、逐约束容差、独立/冗余标记、状态。
- `replicates.csv` — 每个设计重复的状态、最大边际残差、诊断、领域估计。
- `summary.json` — 状态、目标值、最优性、领域估计/方差/标准误、逐重复诊断、约束与界复核、
  非响应假设。
- `report.md` — 静态报告（状态、容差、约束证据、界命中、估计与方差、独立复核、假设边界）。

不可行时另写 `constraint_bounds.csv`（各约束在盒式界下的可达区间）与 `bounds.csv`
（逐人上下界），并在 `report.md` 给出可独立核对的冲突/上下界证据，不输出权重。

## 方法要点

- **目标**：`Σ_i [w_i log(w_i/d_i) − w_i + d_i]`，线性边际，盒式界
  `[relative_lower·d_i, relative_upper·d_i]`；`d_i=0` 强制 `w_i=0`。
- **冗余边际**：以**精确有理数**把每个冗余总量表示成独立基的组合，并核对给定总量；
  不因矩阵冗余直接删除业务约束。
- **预检**：结构空单元、冗余一致性、逐行盒式可达性、以及交叠约束的联合 LP 可行性；
  不可行时给出最小冲突子集与上下界证据。
- **求解**：在对偶（盒约束）上做带线搜索的 Newton，Hessian 为 `Aᵀdiag(w)A`；
  收敛判据、残差、KKT 符号、上下界均显式检查，任一失败即标 `failed`，不靠放宽容差。
- **容差**：约束容差 `REL_TOL·max(|总量|,1)`，界容差 `BOUND_REL_TOL·max(上界,1)`，
  最优性 `OPT_TOL`；全部在报告中公开。均随人口总量缩放。
- **重复方差**：对每组设计重复基础权重，用**相同边际与相对界**重新求解，再按 plan 的
  系数 `c` 计算 `c·Σ_r(estimate_r − full_estimate)²`。失败重复逐条记录；若有失败，
  受影响领域的标准误按可用项合成并**显式标为不完整**，不静默跳过、不给完整 SE。
- **顺序无关**：加载时按 `respondent_id` / `constraint_id` 规范排序；行顺序、约束顺序
  不影响权重与估计。
- **独立核验**：`verify` 仅用返回权重从头重算所有约束、界、总和、目标、KKT 与方差，
  不复用求解/估计/方差函数的算术。

## 非响应假设与边界

`plan.json` 给出：`after conditioning on supplied groups, response selection does not
further depend on score; this cannot be established from these response-only data`。
该假设无法由仅有响应者的数据检验。边际配平只修正已知分组结构，**不等于**消除所有
非响应偏差，也不保证组内评分分布相同。

## 测试

```sh
.venv/bin/python -m pytest -q
.venv/bin/python examples/hand_check.py   # 两个手算小例，断言闭式解
```

测试覆盖：与独立 SLSQP 求解的交叉核对；逐边际/逐人上界与零权重复核；方差直接代回
公式（不复用汇总函数）；冗余边际精确证书；不可行证据可独立复算；`replicate_no_older`
保留点估计并标不完整 SE；两个手算可核对的小例；输入行/约束顺序无关；CLI 状态分离。

## 手算核对小例

**例 1（两组闭式解）**：两名响应者 `d=(1,3)`，`total=4`，`young=2`（故 older=2），
界 `[0.5d, 2d]`。熵解为 `w=(2,2)`；均值 `(2·10+2·20)/4 = 15`。
目标 `= (2ln2−2+1) + (2ln(2/3)−2+3)`。

**例 2（上界命中）**：三人基础权重均 1，`total=4`，`young=3.4`，界 `[0.5,1.7]`。
两名 young 响应者上界 `1.7` 命中，older 取剩余 `0.6`，总重 `4`。
