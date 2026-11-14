# 方法说明（METHOD）

离线、纯 Python 3.11 + NumPy 实现；仅地图报告额外用 matplotlib（Agg 后端）。
不做多流向、降雨径流模型、河网拟合，也不做投影变换、点云地面估计、镶嵌或
大栅格性能工程。输入固定为米制方格 ESRI ASCII Grid。

## 1. 输入与语义

- 行列：零基，行 0 最北；`xllcorner/yllcorner` 是栅格西南外侧角点；
  格心坐标由头推算：`x = xllcorner + (col+0.5)*cellsize`，
  `y = yllcorner + (nrows-row-0.5)*cellsize`。
- 无效判定只看 `NODATA_value`（NaN 一并视为无效）。**真实零高程是有效高程**。
- nodata = 封闭无效区：不是接收格、不是自动出口，水流不可穿过。
- 有效外边界单元允许出流；图外没有任何来水。

## 2. 连通组件与“未解决”判定

对有效单元做 8 邻接（与 D8 邻域一致）迭代 BFS 标记组件。组件中只要含一个
外边界有效单元即为“可解决”；被 nodata 完全包围、没有任何有效外边界出口的
组件标记为“未解决”（`status.asc = 2`）。未解决组件：

- 不参与填洼抬升（高程原样保留，填深为 0）；
- 不赋予接收格/方向，不累计、不允许出流；
- 不开洞、不凭空选内部终点。该结论在 `report.json` 的
  `unresolved_components` 中留痕，`cases/enclosed.asc` 即对应场景。

## 3. 最小填洼（Priority Flood）

只在可解决组件上运行 Barnes et al. (2014) 风格的优先洪水：

1. 所有可解决的外边界有效单元以**原高程**入最小堆（堆项
   `(水位, 线性编号)`，编号是确定性并列裁决）。
2. 弹出堆顶，向 8 邻居扩散：邻居的填后水位 = `max(原高程, 当前水位)`，
   每个单元只入堆一次。
3. 首次被漫到的水位就是该单元能排向外边界所需的**最低**高程，因此
   `filled >= z` 恒成立，`fill_depth = filled - z` 即最小填深。

产物保留 `dem_original.asc`、`dem_filled.asc`、`fill_depth.asc` 三件套，
填方量（m³）= Σ填深 × cellsize²，写入 `report.json`。

## 4. 单一 D8 流向

方向编码（北朝上）：`1=N 2=NE 3=E 4=SE 5=S 6=SW 7=W 8=NW`。
相邻中心距离：正交 = cellsize，对角 = √2·cellsize。

- **严格下坡**：在有效可解决邻居中取坡度
  `(z_self − z_nbr)/中心距离` 最大者；坡度完全并列时取最小方向码。
  对角与正交按真实中心距离公平比较（见测试 `test_d8_uses_center_distance_slope`）。
- 没有严格更低邻居的可解决单元称为 **platter（平坦台面单元）**，
  进入第 5 节的平坦路由。同高程但自身有严格下坡的单元记为 **spill（泄水点）**。

## 5. 平坦区路由（无环、可回查，不微加高程）

不使用“同高邻居固定方向”，也不通过加微小人造坡度掩盖问题。对每个
platter 单元计算两个 8 邻接 BFS 距离（只在同高程 platter 间扩散，跳数为单位）：

- `dist_lower`：到泄水点的距离。泄水点两类：
  (a) 邻接同高程 spill 的 platter；(b) 位于外边界可直接出流的 platter。
- `dist_higher`：到高地边缘（邻接严格更高单元）的距离；全域高台无更高
  邻居时为无穷大（落盘按 nodata 写）。

路由采用**带父指针的多源 BFS**：

- 种子（`dist_lower=0`）：邻接 spill 者直接指向 spill（并列取最小编号）；
  否则作为外边界单元向图外出流。
- 其余单元的接收格 = 它在 `dist_lower` BFS 中首次被发现时的父单元；
  逐层 BFS 在所有同层父候选间按公开规则裁决：
  `dist_higher` 较大者优先（远离台地坎边），再并列取线性编号较小者。
- 父单元的 `dist_lower` 严格小 1，故每条平坦链距离严格递减、必终止于
  种子，**结构上无环**（不依赖局部比较的运气）。

辅助顺序与距离证据全部落盘：`flat_dist_lower.asc`、
`flat_dist_higher.asc`、`flat_bfs_parent.asc`、`flat_evidence.csv`，
可逐格回查；`fixtures/tiny_flat.asc` 用于手工核对。

## 6. 边界退出规则

- 仅当单元没有栅格内接收格且位于外边界时允许出流；出流方向取**指向
  图外的最小方向码**（角点在两个朝外方向中取小码）。
- 接收格编码 `receiver_id.asc`：非负 = 接收单元线性编号
  `row*ncols+col`；负数 = 出流，码值 `-(10+方向码)`；`-1` = 无接收
  （nodata/未解决）。`flow_direction.asc` 出流处为 0。
- 每个出流单元在 `boundary_exits.csv` 登记行列、方向、格心坐标与
  出口累计值，可回查。

## 7. 汇流累计与面积

- 每个有效可解决单元贡献 1 个单位（含自身）。流向图是以出流单元为根的
  有向森林；建反向邻接表后从各出流根迭代 DFS 取后序，节点退出时把自身
  计数加到接收格，共享下游由所有上游分支完整累加。
- `accum_count.asc` = 上游累计单元数；`accum_area_m2.asc` = 计数 ×
  cellsize²。未解决与 nodata 位置保持 nodata 语义。

## 8. 指定出口与流域

- 出口按零基 `(row, col)` 定位，必须是有效可解决单元，否则报错拒绝。
- 流域 = 接收格链经过该出口格的全部单元，即在反向图上从出口 DFS 可达的
  全部单元（含出口自身）。每个出口独立追踪，**允许嵌套/重叠，不做互斥
  切分**；重叠关系写入 `report.json` 的 `overlaps`，成员逐格列入
  `basin_members.csv`，并各出一个 `basin_<id>.asc`。

## 9. 守恒口径

- 总贡献 = 可解决单元数（每单元 1 单位）。
- **总出流 = 各边界出流单元累计值之和**，二者必须相等。
- 明确不把所有内部单元的累计值相加当总流量（该口径在核验中作为反例
  打印，数量级明显更大）。未解决单元不计入守恒两侧。

## 10. 独立核验（hydro.verify，不复用构建期内存结果）

核验从磁盘重读全部产物并独立重算：

| 编号 | 内容 |
| --- | --- |
| V1 | 全部栅格头/形状/nodata 语义一致；零高程不被当无效 |
| V2 | 独立组件标记 + 独立优先洪水，逐格比对填后高程/填深；封闭组件必须未解决且高程未改动 |
| V3a | 独立按中心距离坡度重算严格下坡 D8 与方向码并列 |
| V3b | 独立多源 BFS 复算 `dist_lower/dist_higher` |
| V3c | 逐条接收链走查：终止于合法边界出流或未解决状态、无环、不穿 nodata、出流方向确实朝外 |
| V4 | 逐源沿链独立累加，核对计数与面积（含自身贡献） |
| V5 | 守恒：出流之和 == 可解决单元数；拒绝内部累计求和口径 |
| V6 | 出口合法性 + 独立反向追踪复核流域成员，确认嵌套重叠未被互斥化 |

任一失败即以非零码退出并写 `verify_report.json`。

## 11. 产物清单（每个 outdir）

栅格（ASC，全部保留原网格与 nodata 语义）：
`dem_original, dem_filled, fill_depth, status(1可解决/2未解决), component,
flow_direction, receiver_id, flat_class(0普通/1platter/2spill),
flat_dist_lower, flat_dist_higher, flat_bfs_parent, accum_count,
accum_area_m2, basin_<id>`。

表格/报告：
`flow_records.csv`（逐格接收格/坐标/状态/证据）、`boundary_exits.csv`、
`flat_evidence.csv`、`basin_members.csv`、`report.json`、
`verify_report.json`、`flow_arrows.txt`（小图终端箭头手查）、
`map_report.png`（六面板：原高程、填后高程、填深、状态、平坦区+D8+出口、
汇流量；灰=nodata 洞，红=未解决，白=未解决留白）。

## 12. 运行

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# 单场景
.venv/bin/python -m hydro.build  --dem fixtures/dem.asc --outlets fixtures/outlets.csv --outdir outputs/dem
.venv/bin/python -m hydro.verify --dem fixtures/dem.asc --outlets fixtures/outlets.csv --outdir outputs/dem
.venv/bin/python -m hydro.report --outdir outputs/dem
# 一次跑完三个场景
.venv/bin/python run_all.py
# 单元测试（合成栅格，检验算法性质而非背答案）
.venv/bin/python -m unittest discover -s tests -v
```
