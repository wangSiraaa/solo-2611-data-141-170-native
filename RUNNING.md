# 联合刚体配准工具（jointreg）运行说明

多站位、已知标靶身份的小型联合配准。固定 A 站位姿为世界基准，求各站刚体变换
与共有标靶世界坐标，约定

```
world_point = R @ local_point + t,   R 正交且 det(R) = +1
```

仅依赖 NumPy / SciPy（见 `requirements.txt`），Python 3.11。**不做** ICP、SLAM、
非刚体形变、自动标靶识别或大点云压测；对应关系由 `target_id` 直接给定。

## 1. 环境准备

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 2. 一键演示（五个案例）

```bash
.venv/bin/python run_demo.py
```

| 案例 | 输入 | 演示内容 |
|---|---|---|
| `main` | `fixtures/observations.csv` | 四站 A/B/C/D 正常闭环；数据内置 1 个错配离群（C/T05，约 56σ）与共线站 D |
| `bad-observation` | `fixtures/cases/bad-observation.csv` | 在 B/T06 再注入 0.30 m 粗差（约 30σ），验证两处离群同时被稳健降权 |
| `collinear` | 同 main | 重点展示共线站 D 绕直线轴旋转自由度不可识别 |
| `disconnected` | `fixtures/cases/disconnected.csv` | E 站与 A 无共有标靶路径，分量间 6 自由度不可识别 |
| `exact-small` | `fixtures/cases/exact-small.csv` | 两站四点精确小例，残差达机器精度量级 |

`bad-observation.csv` 由 `fixtures/generate_bad_observation_case.py` 从主数据生成，
可随时重新生成：

```bash
.venv/bin/python fixtures/generate_bad_observation_case.py
```

## 3. 单独运行一个案例

```bash
.venv/bin/python run_joint_reg.py <observations.csv> <anchor.json> <out_dir>
# 例如
.venv/bin/python run_joint_reg.py fixtures/observations.csv fixtures/anchor.json outputs/main
```

## 4. 独立验证

```bash
.venv/bin/python verify.py
```

验证内容（全部从输入 CSV 与输出 JSON 重新计算，不调用求解器内部状态）：

1. 用报告的 R/t 独立代入 `R@local+t` 重算每条拟合残差，与输出逐分量一致；
2. 每站 `RᵀR=I`、`det(R)=+1`，锚站 `R=I,t=0`；
3. 打乱 CSV 记录行序后重跑，可识别量逐元素不变（身份只由 station_id/target_id 决定）；
   共线站只核对可识别自由度与可检验残差，绕直线轴转角本就不可识别；
4. 3+1 个手工点验证变换方向（含已知 Rz(90°)+平移，确认是 world=R·local+t 而非反向）；
5. 反射防护：镜面对应点必须返回 det=+1 并标注 `reflection_corrected`；
6. 解析雅可比对中心有限差分（零旋转点与非零旋转点），相对误差 < 1e-6；
7. exact-small 残差 < 1e-8 m；disconnected 的 E 站标注为不可识别。

## 5. 输出文件（每案例一个 `outputs/<case>/` 目录）

| 文件 | 内容 |
|---|---|
| `station_poses.csv` | 每站联合解 R/t 与初始化 R/t，并附 `RTR_dev`、`det_R` 自检列 |
| `targets.csv` | 拟合标靶世界坐标（含初始化坐标、拟合观测数、`holdout_only` 标记） |
| `observation_residuals.csv` | 逐观测三分量残差（米）、范数、σ 归一化、稳健权重与处置依据；三套解并列 |
| `closure_summary.csv` / `closure_details.csv` | 站间共有标靶闭环差，含全部 RMS 与**内点 RMS** 两口径 |
| `holdout_checks.csv` | holdout 跨站一致性（holdout 从不参与拟合） |
| `report.json` | 上述全部内容 + 连接图、几何支撑、初始化挂载记录、联合雅可比 SVD 可识别性 |

## 6. 方法与设计要点（为什么这样做）

### 6.1 两类问题分开报告，不拿小残差掩盖不可识别

* **图连接**（`graph.py`）：以共有拟合标靶构建站间图，BFS 求锚站连通分量与跳数。
  未连到 A 的站（如 E）与 A 系之间 6 个分量间基准自由度没有任何观测约束。
* **几何支撑**（`geometry.support_report`）：对每站加权中心化点坐标做 SVD，
  区分完整 3D / 共面（rank 2）/ 共线（rank 1，绕直线轴 1 个旋转自由度不可识别）/
  单点（rank 0，3 个旋转自由度全自由）；近退化（奇异值比 < 1e-3）单列弱轴，
  不与精确缺秩混为一谈。
* **解点可识别性**（`adjust.JointModel.identifiability`）：在解点对加权雅可比做 SVD，
  零空间/弱方向映射回具体站的旋转、平移或标靶参数。优化器收敛只代表迭代停止，
  不代表解唯一；唯一性由零空间判定。主数据中该 SVD 恰好给出 1 个零方向，
  由 D 站绕其共线轴的旋转主导，并与几何直线轴数值对齐。

### 6.2 可解释初始化（`init.py`）

锚站 A 固定 `R=I,t=0`；沿"共有标靶最多、支撑秩最高"的 BFS 生成树，逐站用
**加权 Kabsch**（权重 1/σ²）挂载，新标靶按权重累积平均。闭环边初始化时不参与
姿态求解，因此初始化的闭环误差可与联合调整后的闭环一致性直接对比。
不做"每站独立对 A 求变换后平均"——D 站与 A 没有直接共有标靶，且闭环约束必须
进入联合阶段才有意义。

### 6.3 反射防护与正交性

Kabsch 的无约束 SVD 解若 `det<0`（最优是反射），翻转最小奇异方向强制 `det=+1`
并置 `reflection_corrected=True`。联合平差用旋转向量参数化 `R=SO3_exp(r)`，
反射矩阵在该参数空间中不存在，从机制上无法以小残差混入旋转。

### 6.4 稳健联合调整（`adjust.py`）

高斯-牛顿 + IRLS，Huber 损失拐点 `f_scale=2σ`。观测方程
`e = R·l + t − X`（米），解析雅可比（含完整 SO(3) 左雅可比，非仅小角近似）。

* 离群观测**保留不删除**，仅按 Huber 规则降权，残差表给出 σ 倍数、权重与依据；
* 同时输出普通加权最小二乘对照解。主数据上：总 RMS 稳健解看似更大（0.130 vs 0.101），
  因为它拒绝向 56σ 粗差弯曲；但同一内点集合上稳健 RMS 0.0068 m，普通 LSQ 被
  拖到 0.068 m（约 10 倍），holdout 一致性 0.0135 m vs 0.0767 m；
* 秩亏法方程的增量显式投影到可识别子空间，零空间保持初始化代表姿态，
  保证输出对记录顺序确定，同时不掩盖不可识别自由度。

### 6.5 holdout 纪律

`holdout=true` 的观测不进入初始化、Kabsch、IRLS 或标靶估计的任何一步，
只在三套变换（初始化 / 联合稳健 / 普通 LSQ）下做跨站一致性检查。
只在 holdout 中出现的标靶（如 T10）输出看到它的各站两两变换差。
拟合改善而 holdout 恶化（或反之）都会在摘要中如实显示，不做粉饰。

## 7. 残差与单位

所有残差、闭环差、holdout 差均回到原始观测坐标系、以米（`anchor.json` 的 units）
给出；σ 归一化列单独标注倍数，不替代物理量残差。
