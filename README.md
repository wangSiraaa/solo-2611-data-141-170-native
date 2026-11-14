# celltrack — 相邻帧细胞谱系追踪

对**已分割**的逐帧标签 TIFF，恢复连续 track 身份与母女关系。不训练模型，不做原始图像分割，
只处理小规模相邻帧（无长缺口重识别、无多细胞融合修复）。

## 快速开始

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m celltrack track --series fixtures/series.json --out-dir out
.venv/bin/python -m celltrack validate --out-dir out
.venv/bin/python -m celltrack explain --series fixtures/series.json --obs-id f2:l141
```

`explain` 打印某个观测的来龙去脉：它如何出现（延续/出生/分裂女儿）、之后发生什么
（延续/分裂/消失），以及每个候选事件的完整证据与代价分项。

`track` 输出到 `out/`：

| 文件 | 内容 |
|------|------|
| `objects.csv` | 每个观测（单帧单个正标签）：位置/面积/形态、`frame_id`+`label` 溯源、`track_id`、`mother_track_id` |
| `events.csv` | 相邻帧事件（延续/分裂/出生/消失），含帧、标签、代价、`undecided`、证据 JSON |
| `tracks.csv` | 连续 track 身份：帧区间、观测序列、母/子 track、命运 |
| `mother_daughters.csv` | 每次分裂的母体与两女果、代价与完整证据 |
| `candidates.csv` | 全部候选事件（含未选中），便于审计候选边界 |
| `alternatives.csv` | 每个被标记为未决的帧对中，所有近乎等价的事件组合 |
| `summary.json` | 校验结果、输入异常、每对帧的求解/备选/被拒候选 |

退出码：`0` 校验通过（未决帧对仅作标记，不算校验失败）；`2` 校验失败。未决信息写入
`alternatives.csv` / `summary.json` 并由 `track` 在终端提示，供人工复核。

## 输入与单位

* TIFF 为二维 `uint16`，`0` 为背景；正整数**仅在单帧内**标识对象，**不**当作跨帧身份。
* `series.json` 提供 `pixel_size_um = [x, y]`、每帧 `time_minutes`、
  `max_centroid_speed_um_per_min`、`division_area_ratio_range`。
* 所有距离统一为 **微米（um）**，所有时间统一为 **分钟（minute）**；像素坐标在输出中保留，
  同时给出 um 版本。
* 帧顺序由 `time_minutes` 决定（相同时间按 `frame_id` 稳定排序）；**旁观的 `frames` 数组
  顺序与目录枚举顺序不影响结果**。

## 输入异常报告

`parse_series` 遇到下列情况会记录结构化的 `[severity] kind: message` 并继续处理可用的帧：

* `missing_file` / `unreadable_file` — 文件缺失或无法读取
* `illegal_ndim` — 非二维
* `illegal_dimensions` — 形状为零，或与首帧不一致
* `non_integer_labels` / `negative_labels` — 标签类型非法
* `duplicate_frame_id` / `duplicate_frame_file` — 重复帧身份（该条目被跳过）
* `duplicate_frame_time` — 两帧同一时刻，无法排序
* `empty_frame` — 空帧（仍参与流程，只是没有对象）
* `illegal_pixel_size` / `illegal_speed` / `illegal_area_ratio` — 元数据非法（回退到默认并报错）

## 事件模型（候选边界）

相邻帧对 (P, N)，`dt = time(N) - time(P)` 分钟：

| 事件 | 支持门（硬性可行条件） | 代价 |
|------|----------------------|------|
| **延续** continuation | 质心位移 ≤ `max_speed * dt` | `w_move·(预测位移/max_speed·dt) + w_area·|log 面积比| + w_shape·形态变化` |
| **分裂** division | 每个女儿距母体 ≤ `division_max_mother_daughter_dist_um`；两女间距 ≤ `division_max_daughter_sep_um`；**两女面积之和 / 母体面积** 落在 `division_area_ratio_range`；两女面积比可计算 | `base + w_area·|log 面积守恒比| + w_balance·|log 两女面积比| + w_sep·(间距/上限) + w_place·(中点偏移/上限) + w_mother·max(0,log(最大女儿/母体)) + w_shape·伸长惩罚` |
| **出生** birth | 无前帧对象满足延续门 | 平坦 `birth_cost` |
| **消失** disappearance | 无后帧对象满足延续门 | 平坦 `disappearance_cost` |

* **延续的“预测位移”**：当上一相邻帧对已求解且该对象有延续链时，用恒速外推打分：
  由上一间隔 `prev_dt` 得到速度 `v = Δpos / prev_dt`，再按当前间隔预测
  `pred = pos + v * dt`，因此**帧间隔不均匀时也物理正确**。这正是近距离**交叉**时保持身份的原因：交叉瞬间
  “最近质心”会互换两个对象，但各自的速度方向不会。可用 `--no-velocity` 关闭，此时退化为
  纯最近质心代价，并且近似等价的交换会被如实标为未决。
* **分裂不是“附近多两个对象”**：面积守恒 + 母体大于等于女儿 + 两女间距 + 中点位置 +
  面积平衡共同构成代价；若面积守恒比越界，候选直接被删。两女面积之和必须与母体面积相符，
  母体分裂后不会再以第三个细胞身份延续。
* 出生/消失是**正常事件**，不是失败。求解器只有在该链接代价确实优于出生/消失时才连线。

## 联合裁决（为什么用整数规划）

延续与分裂争用**同一批后帧观测**，多个前帧对象也可能争同一后帧对象。逐对贪心无法仲裁。
本实现为每个相邻帧对建立一个小型 0/1 整数规划（SciPy `milp`，HiGHS 后端）：

```
最小化   Σ_c cost(c)·x_c
约束     ∀ 前帧对象 a： Σ_{c: 亲本(c)=a} x_c = 1     （恰好一个出边事件）
         ∀ 后帧对象 b： Σ_{c: b∈子代(c)} x_c = 1     （恰好一个入边事件）
         x_c ∈ {0,1}
```

由此保证：

* **观测唯一归属**：每个前帧对象恰好一个出边事件，每个后帧对象恰好一个入边事件；
* **亲本唯一**：每个对象至多一个亲本（入边唯一）；母体只有一个出边，因此**分裂后不可能
  再延续**为第三个细胞；
* 每个后帧对象要么被延续、要么是某个分裂的女儿、要么是出生；每个前帧对象要么延续、
  要么分裂、要么消失。

**目标**：相邻帧对事件的代价和最小化。**候选边界**：即上表的支持门；门外的观测对不进入
整数规划（例如超速位移、面积不守恒的“假分裂”）。

### 近乎等价与未决

找到最优事件集后，依次加入 no-good 割 `Σ_{c∈S} x_c ≤ |S|-1` 再求解，收集所有总代价
与最优值相差 ≤ `ambiguity_cost_margin` 的不同事件集（上限 `max_alternatives`）。
若多于一个：

* 该帧对标记 `undecided`，所有备选写入 `alternatives.csv` 与 `summary.json`；
* **只有**在不同备选中出现不一致的事件才逐事件标 `undecided=True`；在所有备选都相同的
  事件（例如一次稳定的分裂）不标未决；
* 为了输出连续，按**纯几何**签名（物理质心/面积，与标签编号无关）选出代表，
  `chosen_is_representative=True` 明示它只是代表，不是“真实谱系”；
* 若达到备选/节点上限，如实写“可能还有更多等价解”。

## 输出字段与溯源

`objects.csv` 中每行都有 `frame_id` + `label`（可回到源 TIFF 的像素）以及
`bbox_min_x/…`、像素质心与 um 质心；`warnings` 标记 `sliver_object`/`tiny_object`。
`track_id` 是**规范编号**：先按“起始帧 + 质心轨迹几何”排序再编号，因此**重排标签数字
或文件列举顺序不会改变 track 编号、分组、母女关系**。`mother_track_id` 对女儿 track 的
每个观测都一致（谱系是 track 的属性）。

`events.csv` 的 `evidence_json` 给出该事件被选中/评估的全部证据，例如延续的位移、速度上限、
预测位移、面积比、形态差与**代价分项**；分裂的母/女面积、面积守恒比、间距、中点偏移、
母体是否大于女儿与**代价分项**。这就是“为什么延续/为什么分裂”的回答。

## 校验（独立于求解器）

`lineage.validate_result` 在最终事件集上重新检查：

1. 观测唯一归属（首帧无入边、末帧无出边视为序列边界，非失败）；
2. 每个对象至多一个亲本；每个前帧对象只有一个出边事件（母体不会又分又续）；
3. 谱系无环（沿母链回溯不重复）；
4. 每个女儿 track 必有母 track，且母 track 恰有两个女儿；
5. track 观测帧连续；
6. 每个观测恰好属于其所属 track。

任何一项失败 → `validation.ok=false`，列出错误，退出码 `2`。

## 测试

```sh
.venv/bin/python -m pytest tests -q
```

覆盖：

* 附带的**交叉**（f2→f3 最近质心打平，但恒速外推保持身份）、**分裂**（面积 91→49+49）、
  **边界消失**（右侧出界）与**出生**；
* 极小例**独立枚举**全部事件组合，核对整数规划最优值、观测唯一归属与可行集一致；
* **不因附近多两个对象**就判分裂（面积不守恒）；
* 小女儿/母体面积差过大时的分裂裁决；
* 只保留相邻帧延续，超速位移变成出生+消失，不强行连线；
* 重排每帧标签数字、反转文件列举顺序 → 有意义结论与 track 编号不变；
* 关闭速度预测时交叉帧对标为未决，且**稳定的分裂不标未决**；
* 重复 `frame_id`、空帧、非法维度、缺失文件均被明确报告；
* 每个对象可回到源 TIFF 像素（像素标签与质心逐一核对）。

## 参数覆盖

`track` 支持覆盖候选门与权重，例如：

```sh
.venv/bin/python -m celltrack track --series fixtures/series.json --out-dir out \
  --max-speed-um-per-min 2.5 --divide-ratio-low 0.65 --divide-ratio-high 1.6 \
  --ambiguity-margin 0.05 --birth-cost 1.2 --disappearance-cost 1.2 --no-velocity
```

所有默认权重集中在 `celltrack/config.py`，每项都对应上面的可解释分项。

## 模块

```
celltrack/
  series.py      # series.json 解析、TIFF 读取、帧校验与异常报告
  features.py    # 逐对象位置/面积/形态 + (frame_id,label) 溯源
  candidates.py  # 相邻帧候选事件、支持门与代价
  solve.py       # 整数规划联合裁决 + 近似等价枚举与未决标记
  lineage.py     # 连续 track 身份、母女关系、速度预测、校验
  report.py      # 输出表
  cli.py         # track / validate 命令
```
