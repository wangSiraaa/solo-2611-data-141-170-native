# 不规则光变曲线周期分析（本地 `analyze` 命令）

单目标、单波段、相对日时间的光变曲线周期分析。只用真实观测时刻与逐点 σ 做
加权最小二乘，不插补规则时间轴、不做 FFT、不联网、不做历法换算或星表查询。

## 安装与运行

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m period_analysis.cli analyze \
    fixtures/spread.csv fixtures/daily_alias.csv fixtures/half_period.csv \
    --plan fixtures/plan.json --out-dir output
```

每份 CSV 独立分析，输出到 `output/<case名>/`；周期范围与最大谐波数也可用
`--period-bounds P_MIN P_MAX --max-harmonics 2` 直接给出。

## 方法与口径

1. **数据校验**（`io_utils.py`）：重复 `observation_id`、非有限数、非正 σ
   按 CSV 物理行号定位；存在问题的输入不参与拟合，只写带问题清单的 JSON。
2. **模型**（`model.py`）：μ(t) = c + Σₖ [aₖ cos(2πkf(t−t₀)) + bₖ sin(…)]，
   k = 1…2，权重 w = 1/σ²，WLS 解 β = (X′WX)⁻¹X′Wy，协方差即 (X′WX)⁻¹。
   `design_matrix()` 可在任意单个频率独立构建以复算残差。
3. **频率网格与细化**（`periodogram.py`）：频率范围由周期范围换算；
   Rayleigh 分辨率 1/T（T 为真实时间跨度），粗网格步长 1/(10T)，
   局部极小做 3 轮细化；候选按 1 个 Rayleigh 容差去重，谐波阶数按 BIC 选择。
4. **采样窗审计**（`window.py`）：夜间隔、夜内本地时刻 t mod 1 散布，
   以及只由时刻决定的谱窗 W(ν)=|Σe^{2πiνt}|²/N²；候选按
   |fᵢ−fⱼ−k/天| < 1 Rayleigh 连每日别名族。
5. **P / 2P 近等价**（`holdout.py`）：识别 f_hi ≈ 2 f_lo 的配对，
   综合 χ²、BIC、整夜留出 χ² 与长周期模型 A1/A2 振幅比判定是否近等价；
   不可分辨时两个周期同时保留。
6. **整夜留出**（`holdout.py`）：12 个观测夜逐夜留出，每折只用其余夜拟合、
   预测整条留出夜；不随机拆散夜内密集点。
7. **分数口径**：`explained_score = 1 − χ²/χ²_常数` 只是描述性拟合分数，
   不是后验概率，候选间分数差不得解释为概率差。

## 输出（每个案例目录）

| 文件 | 内容 |
|---|---|
| `candidates.json` | 全部候选、逐 K 系数/χ²/BIC、谐波参数与误差、留出结果、别名族、半周期配对、数据问题 |
| `sampling_window.json` | 采样统计、谱窗峰、逐候选别名证据 |
| `window_function.csv` | 完整谱窗曲线 |
| `periodogram.csv` | 粗网格上 K=1/K=2 的逐频率 χ² |
| `candidate_N_phase.csv` | 逐点相位、预测、残差、加权残差 |
| `report.md` | 静态中文报告，数值与上述文件同口径 |

## 独立复算

```sh
.venv/bin/python scripts/recompute_check.py fixtures/spread.csv \
    --freqs 0.36621 0.5 1.35883 --harmonics 2
```

绕过候选搜索，在给定频率上直接构建设计矩阵、用 NumPy 正规方程独立求解 WLS，
与包内 `weighted_fit` 的 χ²、系数、残差和 (X′WX)⁻¹ 逐项比对。

## 测试

```sh
.venv/bin/python -m unittest discover -s tests -v
```

覆盖：独立设计矩阵复算加权残差与协方差、与 astropy LombScargle 峰位交叉核对、
频率分辨率随时间跨度变化、精确每日采样合成例（别名必须显式出现）、
脏数据行级定位、整夜留出的折完整性、半周期近等价识别。
