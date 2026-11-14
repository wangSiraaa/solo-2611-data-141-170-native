# 分片 MCMC 样本链恢复与估计可信度诊断 — 实现说明

离线诊断工具：不运行采样器、不拟合原统计模型，只基于 `fixtures/manifest.json` 与
分片 CSV 恢复链，并回答“这些抽样是否足以支持参数均值与尾部估计”。

## 运行

```bash
python -m pip install -r requirements.txt          # numpy / scipy / matplotlib
python -m mcmc_diag analyze --out report           # 诊断 + 报告（JSON/Markdown/PNG）
python -m mcmc_diag inspect                        # 只看链恢复记录
# cases 分支：重叠片段（完全重复 → 去重）
python -m mcmc_diag analyze --out report_overlap  --add-fragment cases/add_fragment.json
# cases 分支：冲突片段（冲突重复 → 阻断 c1）
python -m mcmc_diag analyze --out report_conflict --add-fragment cases/add_fragment.json \
    --fragment-path ../cases/c1-conflict.csv
python -m pip install -r requirements-dev.txt      # 测试与独立核对（pytest、arviz）
python -m pytest tests/ -q
```

## 链恢复语义（所有参数一致）

1. 文件顺序无意义：片段按 `(chain_id, start_iteration, end_iteration, path)` 排序，
   链按 `chain_id` 排序；manifest 条目乱序不改变结果（有测试）。
2. 同一 `chain_id` 的片段合并为**一条**链；片段不会被当作独立链，链之间不拼接。
3. 元信息矛盾阻断该链：同一链各片段 `warmup_end`/`thin` 必须一致；manifest 声明的
   iteration 区间必须与文件实际 iteration 集合一致（完全重复行在合并阶段处理）。
4. 合并按 iteration 对齐：**完全相同**的重复观察（iteration 与全部参数值逐一相等）
   只计一次并记录去重数；iteration 相同但取值不同即**冲突重复**，阻断该链诊断；
   链内缺段同样阻断。被阻断的链从多链诊断中剔除，其余链继续，报告中显式列出。
5. 各链去除各自 warmup（`iteration <= warmup_end`）；warmup 边界确实改变使用样本
   （有测试）。
6. 多链诊断使用公开的共同保留长度 `L = min(各有效链保留数)`，每链取 warmup 后
   **最早** L 个样本，尾部多余样本丢弃并逐链记录（`n_discarded_tail`），
   存于 `summary.json` 的 `chain_recovery`。截断规则固定，不允许挑选片段。

基础数据恢复结果：c1–c4 保留 600/580/560/540，共同长度 L=540，
丢弃尾部 60/40/20/0。cases：`c1-overlap.csv` 4 行完全重复 → 去重，结果与基础
完全一致；`c1-conflict.csv` iteration 243 取值冲突 → c1 被阻断，c2–c4 继续诊断。

## 诊断口径（核心自行实现，仅调用 NumPy/SciPy 数值原语）

遵循 Vehtari et al. (2021) 的改进 R-hat 框架：

- **并列秩**：平均秩（`scipy.stats.rankdata(method="average")`），相同值映射到同一
  z；再经 Blom 变换 `z = Φ⁻¹((r − 3/8)/(S + 1/4))`。
- **split R-hat**：秩正态化在全体合并样本上进行，随后每链对半 split（奇数长度丢
  末样本）得 2m 条半链，代入经典 Gelman 方差比公式。
- **folded R-hat**：先取 `|x − median|`（中位数为全体合并样本），再秩正态化 +
  split。它专门发现“均值相近但尺度/形状不同”的链间不一致（测试：同均值异尺度
  链 rank R-hat≈1.00 而 folded≈1.19）。判定用 `rhat_max = max(rank, folded)`。
- **ESS 自相关截断**：多链合并自相关 `ρ_t = 1 − (W − mean_m acov_{m,t})/var⁺`，
  `var⁺ = (n−1)/n·W + B/n` 与 R-hat 共用同一方差分解；自协方差为 FFT 计算的有偏
  估计（除以 n）。截断用 Geyer 初始正序列：成对和 `Γ_k = ρ_{2k}+ρ_{2k+1}` 在首个
  负值处截断并强制单调不增；`τ = −1 + 2ΣΓ_k`，`ESS = m·n/τ`，反相关时上限
  `m·n·log10(m·n)`（与 Stan 相同）。ESS 不对链 split（Stan/posterior 口径；
  arviz 先 split，病态数据上数值可差数倍但结论一致，见测试）。
- **bulk ESS**：秩正态化 z 值的多链 ESS。
- **tail ESS**：p ∈ {5%, 95%}，用全体合并样本经验分位数 q_p 构造指示序列
  `I(x ≤ q_p)`，取两个指示序列多链 ESS 的较小者——衡量尾部（分位数）估计的
  有效信息量。
- **均值 MCSE**：`sd(合并样本, ddof=1) / √ESS(原始值)`。

## 判读规则（多链一致性 × 单链有效信息量）

阈值来自 manifest `decision_limits`（R-hat 1.01、bulk/tail ESS 400），仅为复核辅助：

| 状态 | 含义 |
|---|---|
| `OK_NO_ISSUE_DETECTED` | 全部通过：这些有限诊断未发现问题，**不证明**收敛或后验模型正确 |
| `CHAINS_DISAGREE` | rhat_max 超限：各链可能卡在不同模式或未平稳；rank≫folded 时标注“位置/模式漂移”特征 |
| `INSUFFICIENT_ESS` | R-hat 达标但 ESS 不足：均值看似接近但自相关强，MC 误差仍不可控 |
| `CONSTANT_ALL_CHAINS` | 全链同一常量：R-hat/ESS 无定义（非“通过”），均值即常量、MCSE=0 |
| `CONSTANT_DIFFERS_ACROSS_CHAINS` | 各链各自常量但链间不同：R-hat=inf |
| `NONFINITE_VALUES` / `INSUFFICIENT_DATA` / `DIAGNOSTICS_BLOCKED` | 非有限值 / 样本不足 / 有效链不足，均有显式状态，NaN 绝不当作通过 |

R-hat 与 ESS 同时失败时状态取 `CHAINS_DISAGREE`，但 `failed_checks` 与解释中保留
全部失败项。逐参数独立判定，单个参数的问题不影响其他参数出现在报告中。

## 基础数据结论（L=540，4 链）

| 参数 | 状态 | 关键证据 |
|---|---|---|
| location | 未发现问题 | R-hat 1.001，ESS bulk 636 / tail 1161，MCSE 0.037 |
| sticky | 链间不一致 + ESS 不足 | R-hat 1.19，ESS bulk 22（lag-1 自相关≈0.97），MCSE 0.19 |
| separated_modes | 链间不一致（模式漂移） | rank R-hat 1.73 ≫ folded 1.009；c1/c2 卡在 +3，c3/c4 卡在 −3 |
| heavy_tail | 未发现问题 | R-hat ≤1.001，ESS bulk 2260 / tail 2180，MCSE 0.034 |
| constant | 全链同一常量 | 恒为 5，R-hat/ESS 无定义，MCSE=0 |

## 验证

- 方向性：独立生成的短独立链（R-hat≈1、ESS≈N、MCSE≈sd/√N）与刻意卡在不同均值
  的链（R-hat≫1）确认诊断方向；强 AR(1) 链 ESS 骤降；完全反相关链 ESS 取上限。
- 独立核对：R-hat 与 arviz 0.23 逐位一致（<2e-3）；ESS 与测试内显式循环（非 FFT）
  复算一致（<1e-10），并与 arviz 同量级、阈值同侧；非同一函数自证。
- 工程性质：manifest 片段乱序结果不变；warmup 边界改变使用样本；重叠片段去重后
  结果与基础完全一致；冲突片段仅阻断 c1，其余链与其余参数照常出报告。
