# SVG 折线图数值恢复工具 — 交付说明与边界

`svgrec.py`（Python 3.11，仅标准库）从导出的 SVG 折线图恢复可核查数据。
不做截图、OCR 或网页采集，只处理 manifest 声明的真实矢量元素。

## 运行

```sh
python3 svgrec.py extract --manifest fixtures/manifest.json --out out
python3 svgrec.py verify  --manifest fixtures/manifest.json --out out
```

`extract` 逐图输出：`out/<图>.points.csv`（曲线点）、`out/<图>.axes.json`
（轴标定证据）、`out/<图>.segments.json`（可见片段与隐藏顶点）、
`out/<图>.report.json`（可见区间与诊断）、`out/<图>.overlay.svg`（叠加证据）。
刻度不一致或输入越界的图只写 `<图>.rejected.json`，**不产出可信数值**，
且命令以非零码退出。`verify` 重跑提取并执行全部核验，打印 PASS/FAIL，
写 `out/verify.json`，失败时退出码为 1。

## 坐标与证据链设计

1. **统一空间**：所有几何（曲线顶点、刻度锚点、裁剪边界）都变换到"共享
   绘图区坐标"——manifest `plot_frame` 元素的局部坐标系；`plot_frame` 为空
   时取全部刻度元素最深公共祖先的局部坐标。图内的旋转与非等比缩放只存在
   于该空间之外（或其内部嵌套层），通过矩阵链 `inv(CTM_frame)·CTM_el·p`
   统一消除；**不**用屏幕横纵坐标做两个独立比例，也不忽略变换顺序
   （transform 列表按声明顺序复合，嵌套 g 自根向下累积）。
2. **刻度标定**：刻度数值只读 `text/tspan` 文本（必须整体为纯数值，否则
   报错）；text 的 `(x,y)` 属性是锚点，变换到绘图区坐标后取本轴分量参与
   拟合，垂直方向的标签偏移不参与。每轴用**全部**刻度做最小二乘
   （log10 轴对 `log10(值)` 拟合），残差同时以绘图区单位和值单位报告；
   任一残差超过声明容差（`max(1e-9, 1e-6×刻度跨度)` 绘图单位）即整图
   拒绝——不任选两个刻度掩盖矛盾。log 轴遇零/负刻度直接诊断。
3. **裁剪**：`clipPath` 按 `userSpaceOnUse` 语义在其声明空间（引用元素的
   用户空间，含其自身 transform）解释；曲线顶点先变换到该空间，用
   Liang–Barsky 与矩形求交，幸存点再变换到绘图区坐标。嵌套变换下该顺序
   不变。交点标记为 `intersection`（注明所在线段与边界），只表示图形截断
   位置，**不冒充采样观测**；原始顶点全部保留并标注可见性，隐藏顶点用于
   解释片段端点从何而来。
4. **数值恢复**：片段点的绘图区坐标经拟合函数映射为值；每个片段记录连续
   顺序、来源元素/点索引、`from`/`to` 端点说明与诊断。

## verify 的核验

- `roundtrip`：把写出的 CSV 数值经 `axes.json` 的拟合参数映回绘图区，
  与 CSV 中的几何坐标比对（容差 1e-9×跨度）。
- `hand_two_point`：用首末两个刻度做两点式（log 轴在对数上）独立手算
  每个恢复值，与联合拟合结果比对——独立于主拟合代码路径。
- `log_decade`：log 轴相邻刻度的位置间距必须等于 `log10(值比)/a`；
  沿斜率方向相隔一个 decade 宽度的两点，恢复值之比恰为 10。
- `geometry`：交点带合法边界标记；隐藏顶点不出现在任何片段中。
- `rigid_invariance`：对整图外套 `translate(37.5 -12.25) rotate(23)` 刚性
  变换后重新提取，恢复值、可见性、片段结构应在 1e-9 内不变（证明未使用
  屏幕坐标）。
- `negative:inconsistent_ticks`：`cases/inconsistent_ticks.svg`（x 刻度
  0/7/10 在等距位置）必须抛出 `AxisInconsistent` 且不产生可信输出。

## 精度与误差说明

- SVG 数字按 IEEE-754 双精度解析；源文件坐标保留 k 位小数时，量化误差
  ≤ 0.5×10⁻ᵏ（元素局部单位），经变换链最大奇异值放大后进入绘图区坐标
  （`report.json` 的 `coordinate_quantization` 给出该事实性上界）。
- 恢复值误差 ≈ |拟合斜率| × 绘图区坐标误差（线性轴）；log10 轴的相对误差
  ≈ ln(10) × |斜率| × 坐标误差。刻度拟合残差以同样方式传播。
- 不从文件推断原始数据未知的绘图/采样精度；恢复值仅描述图形的绘制几何。

## 明确支持的特性（白名单）

单一 SVG 命名空间；`translate/scale/rotate`（含 `rotate(a,cx,cy)`）嵌套
变换；`text/tspan`；`polyline`；仅含绝对 `M/L`（含隐式 lineto）的 `path`；
`clipPathUnits=userSpaceOnUse` 的单矩形 `clipPath`（无圆角、无变换、
不嵌套求交）；嵌套 `g`。`title/desc` 及表现属性（fill/stroke/font-size/
text-anchor 等）被忽略；根元素 `viewBox/width/height` 仅属屏幕映射，
不参与恢复。

## 明确拒绝（抛出 `UnsupportedFeature`/`DataError`，不建设通用渲染器）

- 任意其他元素（`circle/image/use/style/...`）、外命名空间、
  `display/visibility` 可见性控制；
- `matrix/skewX/skewY` 变换；
- 相对或曲线 path 命令（`m/l/C/Q/S/T/A/Z/...`）；
- `objectBoundingBox` 裁剪、非矩形/多形状/带变换的 clipPath、嵌套裁剪求交；
- 非数值刻度文本、缺失的引用 id、不成对坐标、退化（不可逆）变换；
- 不识别任意图表类型，不处理贝塞尔曲线与位图。
