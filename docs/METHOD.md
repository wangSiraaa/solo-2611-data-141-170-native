# 扫频法房间脉冲响应与分频衰减测量

离线工具：对**已知对数扫频**的**一次**单通道录音（16 kHz、PCM16、单声道）做反卷积，
定位主线性响应，按 plan 给定的三个固定频带给出带内衰减曲线与 T20（条件满足时），
并输出可信/不足状态。不接麦克风、不做任意非线性系统辨识或声源识别。

## 运行

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m sweeproom analyze  --out output
.venv/bin/python -m sweeproom verify   --out output/verify
.venv/bin/python -m pytest tests -q
```

- `analyze` 默认读取 `fixtures/{plan.json,sweep.wav,recording.wav}`。
  用 `--recording cases/noisy_recording.wav --out output/noisy` 处理高噪声录音。
- `verify` 做三项独立核对（见下），退出码 0 表示前两项通过。

## 输出文件

`analyze`（`--out DIR`）：

| 文件 | 内容 |
| --- | --- |
| `impulse_response.csv` | `lag_samples,time_s,impulse_response`，物理增益（未归一化） |
| `band_decay.csv` | 每带一行：状态、噪声地板、动态范围、−5/−15/−25 dB 穿越点、拟合起止、残差、斜率、积分截断、噪声裕量 |
| `decay_curves.csv` | 各带反向积分衰减曲线（电平 dB 与噪声门控后的原始平方响应） |
| `evidence.json` | 全部数值证据：时基、定位、谐波镜像、噪声两套估计、每带判定、阈值 |
| `report.png` | 原录音 / 反卷积 / 直达放大 / 首带衰减叠加报告 |
| `report_decay.png` | 三个频带的衰减曲线叠加图 |

`verify`（`--out DIR`）：`verify.json`、`base_vs_high_noise.png`。

## 方法

### 1. 激励模型与逆滤波

plan 给定指数正弦（对数）扫频：

```
f(t) = f1·exp(t/K),  K = T / ln(f2/f1)
s(t) = A·sin(2π f1 K (exp(t/K) − 1)),  带 fades_seconds 线性淡入淡出
```

逆滤波器由**实际提供的 sweep.wav** 构造（不是只用公式重建），为 Farina 式：

```
g[n] = s[N−1−n] · exp(−n / (sr·K))
```

`exp(−n/(sr·K))` 包络使 `|S(f)·G(f)|` 在扫频带内基本平坦，**这正是保证各频带相对
幅度不被反卷积无意改变的关键**（若用反向扫频但配错包络，频带间相对电平会带 ±数十 dB
斜率）。`g` 再归一化，使 `s ⊛ g` 的峰值在滞后 0 处为 1。于是对"延迟 d、缩放 g₀ 的
扫频"反卷积直接得到延迟 d、幅度 g₀ 的冲激——该性质由 `verify` 用
`reference_delay.wav`（0.050 s、0.7）独立核对。

### 2. 线性卷积补零与时轴

用**全线性卷积** `deconvolve = full_conv(recording, g)`，再取从 `zero_lag_index = N−1`
开始的因果段，长度与录音相同。滞后 0 对应录音第 0 个采样。

录音长度 `M`、激励长度 `N` 时，滞后 `L` 只有在 `L ≤ M−N` 时才被激励**完整支撑**；
更晚的滞后只由部分激励支撑，不作解释。`valid_lag = M−N`（本数据约 1.95 s）会写入
`evidence.json` 的时间基。时基：`time_s = lag_samples / 16000`。

### 3. 主线性响应定位与谐波排除

未知起始延迟：在完整有效滞后范围内取 `|deconv|` 全局峰（排除滞后 0 附近的逆滤波器
自身残差）作为直达 `τ₁`。判定为**歧义**的情况：

- 直达峰相对最强竞争峰（排除直达簇与谐波窗口）的优势 < `direct_peak_dominance_db`（6 dB）；
- 直达峰信噪比 < `direct_peak_snr_min_db`（20 dB）。

对数扫频下，前置记忆less 非线性（二次/三次）产生的 `k·f(t)` 谐波，经同一逆滤波后
成像于

```
τ_k = τ₁ + K·ln(1/k) = τ₁ − K·ln(k)     （k ≥ 2 在直达之前）
```

本数据 `K≈0.463 s`，`k=2` 落在 `τ₁ − K ln2 ≈ 463` 采样（约 0.029 s），`k≥3` 落到
负滞后（因果范围之外、不可观测）。处理方式：

- 显式预测并把每个 `k` 的预测滞后、是否在因果范围、窗口能量与峰值写入
  `evidence.json`；
- 选择主峰时**排除**落在因果范围内的谐波窗口（并用 `harmonic_exclusion_halfwidth_ms`
  覆盖 chirp 压缩后的展宽）；
- 分析窗口从直达起始 `onset` 开始，前置谐波能量按构造落在窗口之外；报告"窗口前被
  排除能量"与"谐波窗口能量"两项证据。

要求只是**定位主线性响应并排除前置谐波影响**，不做任意非线性辨识。

### 4. 噪声估计与传播

反卷积会给背景噪声**上色**，直接把平方响应整段反向积分等于对噪声平台积分。工具并行
给出两套每带噪声功率：

1. **经验尾窗**：有效滞后后段（`noise_control_frac` = 72%–97%）的带内功率中位数，
   并检查前后半窗一致性（`noise_stationarity_max_db` = 3 dB）；
2. **传播模型**：用 onset 前纯噪声段做 Welch PSD，合成同谱有色噪声，经**实际逆滤波器**
   推到有效滞后区，取确定性种子的中位数功率。

每带取二者**较大者**作为噪声地板（保守），并把经验/传播 dB 差异写入证据；差异超过
`noise_floor_agreement_max_db`（6 dB）会作为告警记录。

### 5. 反向积分、截断与拟合

对每带：

- 4 阶 Butterworth 带通（`sosfiltfilt`，零相移），平方得带内能量 `e(t)`；
- 10 ms 平滑；`t_cut` = 最后一个平滑能量仍高于噪声地板 `noise_gate_db`（10 dB）的时刻，
  **绝不对平台积分**；
- 噪声补偿反向积分：`E(t) = Σ_{t'=t}^{t_cut} max(e(t') − Nf, 0)`，
  `L(t) = 10log10(E(t)/E(onset))`；
- 在 `L` 的 −5…−25 dB 区间做最小二乘直线拟合，`T20 = −60/slope`。

### 6. 可信度判据（公开阈值）

在弱端（−25 dB）计算**累计噪声裕量**

```
clearance = 10log10( E_signal(t25) / (Nf · (t_cut − t25)) )
```

即该点剩余信号能量相对剩余积分噪声能量的倍数。

| 判定 | 条件 |
| --- | --- |
| `credible_t20` | −25 dB 裕量 ≥ `t20_clearance_min_db`（20 dB）**且** −5…−25 拟合残差 ≤ `fit_resid_max_db`（3 dB） |
| `qualified_short_range` | T20 不满足，但 −15 dB 裕量 ≥ `t10_clearance_min_db`（12 dB）；此时报告的是**观测到的 −5…−15 dB 衰减时间**（10 dB 衰减时间，不外推） |
| `unavailable` | 两者都不满足：不给数值，只给原因 |

要求 T20 端 20 dB 裕量，是因为 T20 把斜率外推 2.4 倍；20 dB 意味着 −25 dB 处剩余信号
能量是其后积分噪声的 100 倍（噪声贡献 <1%）。**当裕量不足时绝不外推 `−60/slope`**，
因此不会出现"对噪声平台拟合直线再报精确混响时间"。

此外：
- 对可报告结果额外给出**噪声地板敏感性区间**（把噪声地板 `Nf` 乘 0.5 与 2 重算同一
  统计量）：`credible_t20` 时为 T20，`qualified_short_range` 时为观测 T10。这是稳健性
  读数，**不是**统计置信区间；高噪声时区间明显变宽，直接体现可信区间收窄；
- 全部每带拟合起止、点数、残差、斜率、`t_cut`、噪声裕量写入 `band_decay.csv`
  与 `evidence.json`。

### 7. 诊断

- **数字满幅**：统计 |sample| ≥ 32767/32768 的样本数（录音与激励分别统计），
  出现即 `diagnostic_failure`，不静默裁剪；
- **无可用激励频段**：若某请求带落在扫频范围外或激励能量占比 < `min_band_excitation_fraction`
  （1%），该带直接 `unavailable` 并给出原因；
- **定位歧义**：见 §3，触发时整体 `diagnostic_failure`；
- **扫频模型核对**：把提供的 `sweep.wav` 与 plan 公式重建比对（相关系数），
  逆滤波仍以实际波形为准；
- **样本归一化**：统一除以 32768 是单一全局常数，不改变频带间相对关系。

## verify 的三项独立核对（不读生成器内部真值）

1. **参考延迟/缩放**：反卷积 `reference_delay.wav`，核对滞后 = 0.050 s、幅度 = 0.7
   （容差 ±1 采样、±0.05）。
2. **已知指数衰减合成响应**：构造 RT60 已知（0.600 s）的指数包络冲激响应，与**实际提供的
   sweep** 卷积、加少量噪声，走同一条分析链，核对反向积分斜率恢复的 T20 与已知 RT60
   相对误差 ≤ 15%。
3. **基础 vs 高噪声并列**：同一条链分析 `recording.wav` 与 `cases/noisy_recording.wav`，
   输出每带判定、T20、裕量、动态范围与判定变化，并画 `base_vs_high_noise.png`。
   本数据上高噪声把三个频带从 `credible_t20` 降级为 `qualified_short_range`。

## 依赖

- Python 3.11
- NumPy ≥1.26,<3
- SciPy ≥1.11,<2
- matplotlib ≥3.8,<4（仅绘图）
- （测试）pytest

## 范围与局限

- 只支持题面所述单一情形：一次**已知**对数扫频、16 kHz、PCM16、单声道、离线。
- 不声称现场认证精度，不提供统计置信区间（无重复测量依据）；
  `noise_sensitivity_range_s` 是噪声地板敏感性，不是 CI。
- 不处理未知激励、任意非线性系统辨识、降噪播放器、音频切片语料或声源识别。
