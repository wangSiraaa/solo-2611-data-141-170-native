# 部分观测下的共同失效原因与维修收益范围

实现位于 `reliability/`，测试位于 `tests/`。所有分析为精确边缘化（枚举原始事件
配置），不做仿真或寿命建模。

## 运行

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 两个快照各自独立分析，不合并
.venv/bin/python -m reliability all            # fit/assess/verify + summary.json
.venv/bin/python -m reliability fit   --snapshots fixtures/snapshots.csv --name fit.json
.venv/bin/python -m reliability assess --snapshots fixtures/snapshots.csv --name assess.json
.venv/bin/python -m reliability verify --snapshots fixtures/snapshots.csv --name verify.json

.venv/bin/python -m unittest discover -s tests -v
```

报告写入 `.runtime/reports/`（已忽略，不提交）。`all` 读取 `plan.input_cases`，
按文件名后缀分别生成 `snapshots_*.json` 与 `weak_snapshots_*.json`。

## 模型与似然

原始事件（`local_*`、`controller`）概率固定；`surge`、`heat` 概率未知，分别记为
`p_surge`、`p_heat`。同周期原始事件相互独立，跨门可共享同一事实（例如 `surge`
同时驱动 `p1`、`p2`、`c3`、`control_lost`），同门重复 input 在建模时拒绝。

对每个周期，把未观测内部部件的原始事件配置全部边缘化，只保留观测约束：

```
P(pattern | θ) = Σ_x [∏_fixed p^xi(1-p)^(1-xi)] · basis_{b(x)}(θ)
```

其中 `basis` 是 `(1-p_s)(1-p_h), (1-p_s)p_h, p_s(1-p_h), p_sp_h` 四项。因此

```
loglik(θ) = Σ_patterns count · log( n_pattern · basis(θ) )
```

`n_pattern` 只依赖固定概率与图，与 `θ` 无关；同周期多个部件不被当作独立样本，
周期之间才相乘。`NA` 始终作为未观测（内部编码 `-1`），绝不补成健康。若某观测
模式没有任何原始配置能同时满足，直接报图/快照矛盾（`assess` 抛
`ValidationError`，`verify` 记录失败）。

## 拟合、支持区与可识别性

- 在 `plan.parameters` 的盒约束内，以 401×401 网格做多起点 L-BFGS-B 求 MLE。
- 联合支持区：`loglik ≥ loglik_mle - support_loglik_drop`（题目为 2.0）。这是
  数值支持标准，不是校准置信区间。
- 报告网格分辨率、支持区切片、单参数剖面曲线，以及每个参数在盒边界上的覆盖
  情况（`covered` 标志）。
- 可识别性：MLE 处 Fisher 信息矩阵的特征值/条件数/秩，并沿最弱特征方向做探针。
  仅观测 `system` 时秩为 1，存在近等价参数族；报告明确写
  `identified_separately=false`，不因优化器返回一个点就宣称两种原因已分清。

## 维修语义

`one_repair` 唯一地把 `c1` 夹为健康并切断该节点输入：`c1` 的取值被强制为 0，
其上游 `local_c1`、`heat` 不再通过 `c1` 生效；但 `heat` 仍作用于 `c2`，`surge`
仍作用于 `p1/p2/c3/control_lost`。这**不是**在观察到 `c1` 健康后的条件概率，
也不把 `p_heat` 清零。

风险是参数的平滑函数 `R(θ)=c·basis(θ)`（系数由枚举得到）。基线风险、维修后
风险、绝对收益都在**同一联合支持区**上求极值：网格定位 + 有界 SLSQP 在
`loglik≥阈值` 约束下精化，每个极值都给出可代回的参数对，并再次验证其在支持区
内、且能重现所报数值。绝对收益逐参数对做 `baseline-repaired` 差值，不对两个
独立极值端点做拼接。报告还列出边际区间四个角的组合，仅用于展示它们可能落在
联合支持区之外。

## 结果（`support_loglik_drop = 2.0`）

| 快照 | MLE `(p_surge, p_heat)` | p_surge 剖面 | p_heat 剖面 | 可分辨 | 基线风险 | 维修后风险 | 绝对收益 |
|---|---|---|---|---|---|---|---|
| `snapshots` | (0.06231, 0.12893) | [0.03113, 0.10791] | [0.08172, 0.18839] | 是（秩 2） | [0.14156, 0.25967] | [0.04874, 0.12429] | [0.07463, 0.16886] |
| `weak_snapshots` | (0.18450, 0.00375) | [0.0, 0.25653] | [0.0, 0.25653] | **否**（秩 1） | [0.14201, 0.26791] | [0.01830, 0.26556] | [0.00235, 0.24350] |

弱快照只观测系统状态（160 周期中 32 次失效），支持区触及 `p_surge=0` 与
`p_heat=0` 两条边并按参数曲线延伸；维修收益范围因此很宽：在
`p_surge` 高、`p_heat≈0` 处收益接近 0（`surge` 仍打击其它支路），在
`p_heat` 高、`p_surge≈0` 处收益接近 0.243。基础快照观测到内部部件，两个原因
可分辨，收益范围明显更窄。

## 核验

`verify` 逐项检查并在报告中给出通过标志：

- `NA_not_imputed`、`observed_patterns_positive_at_mle`、`mle_loglik_consistent`；
- `mle_inside_bounds`、`mle_in_support`、`support_mask_reproduces_drop_rule`；
- `risk_witnesses_valid`、`benefit_{min,max}_matched_pair`；
- `profile_endpoints_on_boundary`（端点可行、越界 1e-6 即不可行）；
- `finer_grid_not_narrower`、`fine_grid_risk_consistency`（803² 独立细网格复核）；
- `graph_has_no_contradictions`、`identifiability_not_overclaimed`。

`tests/test_shared_shock.py` 用两部件共享冲击的小例手算联合健康概率与修复语义；
`tests/test_independent_grid.py` 用不调用包内聚合的另一套枚举、181×181 独立二维
网格复核似然、极值与支持区传播，并逐参数对验证见证。

## 边界与限制

- 仅两个未知概率、一个静态模型、一次固定维修；无寿命/在线监控/长时仿真，
  也不做割集枚举或维修组合优化。
- 支持区为数值支持标准，不能声称精确置信水平。
- 弱快照下两种原因不可分辨，报告给出近等价族说明；风险范围宽是数据导致的，
  不是数值噪声。
