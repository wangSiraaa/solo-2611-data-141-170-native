# 二维直线射线层析重建工具（recon2d）

从无量纲对数衰减观测，在规则矩形网格上重建非负密度，并严格区分
**数据直接支持的结果**与**平滑先验补出来的结果**。只处理直线线性
积分（无散射、无三维、无神经网络；网格 ≤12×12、射线 ≤100 条）。

## 1. 构建与运行

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # numpy、scipy，Python 3.11

make test          # 23 个单元测试（含 tiny 手算与独立积分交叉校验）
make build         # 测试 + 完整/有限角度/tiny 三组重建，产物在 out/
# 或单独运行：
.venv/bin/python -m recon2d --rays fixtures/rays.csv \
    --grid fixtures/grid.json --out out/full --name full
.venv/bin/python -m recon2d --rays fixtures/limited-rays.csv \
    --grid fixtures/grid.json --out out/limited --name limited
.venv/bin/python -m recon2d --rays fixtures/tiny/rays.csv \
    --grid fixtures/tiny/grid.json --out out/tiny --name tiny
```

无射线格需要显式先验锚定时（默认不锚定）：

```bash
.venv/bin/python -m recon2d --rays fixtures/limited-rays.csv \
    --grid fixtures/grid.json --out out/pa --name pa \
    --prior-weight 0.5 --prior-value 0.7
```

## 2. 前向模型与几何约定

单元 `(row, col)`（行沿 +y、列沿 +x，扁平索引 `idx = row*nx + col`）：

- x 区间 `[x_min + col*dx, x_min + (col+1)*dx)`，
  y 区间 `[y_min + row*dy, y_min + (row+1)*dy)`；最外侧最大边界取闭。
- **半开归属**：沿内部网格线的射线归高坐标一侧（`x=x_k` 归 `col=k`，
  `y=y_k` 归 `row=k`），由段中点的 `floor()` 天然实现。
- **角点不双计**：收集裁切段内全部 x/y 网格线跨越事件并按参数 `t`
  归并，穿过角点时两个跨越同 `t` 只产生一个事件，角点两侧各算一段、
  各归一个单元。

每条射线先用 Liang–Barsky 式 slab 求交**裁到区域**，再在事件之间按
欧氏长度累计到对应单元，得到稀疏前向算子 `A`（单位：米）：

```
attenuation_r = mass_attenuation_mu * sum_j A[r,j] * density_j
```

完全不相交的射线（含仅角点相切、内部长度为零）得到 **A 的全零行**
并被排除出拟合——绝不会被误当成一条穿满区域的有效观测。零长度
射线、NaN/Inf 端点、非正 `sigma`、非法网格、超规模输入都有显式
诊断（见 `diagnostics_*.json` 的 `rays.input_diagnostics` 与
`geometry_diagnostics`，以及测试中的相关用例）。

## 3. 目标量、误差加权与正则尺度

非负密度约束下的加权最小二乘（增广稀疏线性系统，`scipy.optimize.lsq_linear`
的 TRF 求解，下界为 0）：

```
minimize_{x >= 0}  J(x) = J_data + lambda * J_smooth + J_prior

J_data   = Σ_r ((atten_r - μ(Ax)_r) / σ_r)^2     # 误差 1/σ 加权，即卡方量
J_smooth = ||D x||^2                              # 邻格平滑（见下）
J_prior  = ||w_prior P (x - x_prior)||^2          # 仅无射线格，默认 w_prior=0
```

- `D` 是公开的最近邻一阶差分矩阵：每对共享边的相邻单元一行
  `(e_i - e_j)`，即标准 Tikhonov 邻域平滑（二次 TV 替代）。fixtures
  全部为 1 m 等宽网格，单位差分与除以格宽的物理差分数值相同。
- **正则尺度**：`lambda` 直接乘在“单位邻格差分平方”上；诊断 JSON
  同时报告归一化强度 `lambda_eff = lambda * n_data / n_edges`，
  以及 `regularization_to_data_ratio = (λ·J_smooth + J_prior)/J_data`，
  避免只给一个无量纲数字。两个强度（0.01、1.0）来自 `grid.json`。
- 默认 `w_prior=0`：无射线格不凭空锚定数值，其解完全是平滑项从
  邻近有数据区域“渗漏”过来的，并在输出中单独标记。给定先验权重时，
  权重、先验值、`J_prior` 全部显式写入 JSON。

## 4. 输出清单（每个案例的 out 目录）

| 文件 | 内容 |
| --- | --- |
| `operator_<case>.npz` | 稀疏前向算子 A（scipy CSR，路径长度，米） |
| `operator_manifest_<case>.csv` | 稀疏算子逐项清单：射线、行列、扁平索引、路径长度；不相交射线显式标注 `DISJOINT_ZERO_LENGTH_INSIDE_REGION` |
| `results_<case>.npz` | 两个 λ 的密度（2D，行=y+）、λ=0 纯数据解、每格射线条数/路径长度、支持掩码、近零空间能量、正则扫描摆动、单元可信度标签 |
| `predictions_<case>.csv` | 逐射线观测、预测衰减、残差、加权残差、裁切段长、有效/剔除状态 |
| `diagnostics_<case>.json` | 完整诊断：输入问题、算子规模与 nnz、秩与全部奇异值、近零空间、角度覆盖、有限角度探测、目标量分解、先验依赖、独立校验结果 |

单元可信度标签（`cell_labels`）：

- `data_determined`：有射线支持且不在近零空间、正则下稳定；
- `ray_supported_but_near_nullspace`：有射线但该格变化主要落在
  算子近零空间，数据约束很弱；
- `ray_supported_unstable_under_regularization`：正则扫描中相对摆动 >25%；
- `no_ray_support_prior_only`：**无任何射线穿过，数值只来自平滑/先验**。

## 5. 可辨识性诊断：success ≠ 唯一真值

`lsq_linear` 返回成功只说明最小二乘收敛。当加权算子 `WμA` 秩亏时，
满足数据的密度有无穷多个，正则只是在零空间里挑了一个最平滑的——
**不能把它称为数据证明的真实密度**。工具因此额外给出：

1. **每格射线支持**：射线条数与累计路径长度（`ray_count_per_cell` 等）；
2. **秩与近零空间**：完整 SVD（注意必须 `full_matrices=True`，紧 SVD
   会丢掉右零空间）、全部奇异值、数值秩阈值、零空间维数；
3. **零空间能量图**：每个单元的变化落在不可约束方向上的比例 [0,1]；
4. **正则稳定性扫描**：在两个给定 λ 之间对数取点重复求解，报告逐格
   标准差与相对摆动，直接暴露“哪些区域随先验变化而不稳定”；
5. **有限角度探测**：行内/列内零和交替模式在加权算子下的可见比
   `||Bv||/||v||`，为 0 表示该空间变化与数据完全正交、无法区分；
6. **角度覆盖**：10° 扇区直方图、最大无射线角度空隙。

## 6. 独立校验（不是同一段代码跑两遍）

- **段长和 = 裁切段长**：每条射线各格路径长度之和与 slab 裁切段
  欧氏长度逐线对比（实测最大相对差 0，机器精度）。
- **另一条积分路径**：`verify.py` 用固定步长中点采样法（与事件法
  共享的只有裁切函数和半开归属约定）独立累计逐格长度并重算预测
  衰减；两套实现的预测最大相对差约 1e-4（采样误差随步长收敛）。
- **tiny 手算**：2×2 场景的水平/垂直/对角（含角点 `√2` 不双计）
  路径长度全部在测试中手算断言；手算线性方程组的密度解
  `[1,2,3,4]` 与 λ=0 重建一致。

## 7. 两个输入案例的结论摘要

- **完整角度（45 条，R045 完全不相交被剔除）**：44 条有效射线，
  加权算子秩 36/36、零空间 0、条件数约 72；36 格全部有射线支持，
  两个 λ 下卡方均远小于自由度（数据噪声水平内），正则扫描逐格摆动
  ≤10%。密度由数据确定，正则只做轻微平滑。
- **有限角度（5 条纯水平射线）**：秩 5/36、零空间 31 维；行内交替
  模式可见比为 **0**（同一行内的密度重新分布完全不可区分，平滑解
  把每行拉成常数）；最上行 6 格无任何射线，零空间能量为 1.0，其
  数值纯粹是从邻行平滑渗漏（默认配置下等于相邻行均值），占总质量
  约 13%，已单独标记，不得与测量充分区域同可信度解读。

## 8. 边界与不做的事

仅限 ≤12×12 网格、≤100 条射线、二维直线、单能线性衰减；不包含
医学影像流程、散射/束硬化模型、神经网络或三维大体积重建。
