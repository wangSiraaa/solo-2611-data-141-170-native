# 含洼地与平坦区的小DEM汇流及流域产物

本目录是题目输入材料，未包含答案实现。所有业务与观测数据均为本地合成数据。

## 文件

- `fixtures/dem.asc`
- `fixtures/tiny_flat.asc`
- `fixtures/outlets.csv`
- `fixtures/plan.json`
- `cases/enclosed.asc`

以题面规定的字段语义与边界为准。cases/ 中的文件只替换说明中指明的输入，其他基础输入保持不变。小材料用于调试与核验，不能通过写死文件名、行号或样例答案完成任务。

运行环境：Python 3.11。可用 `python -m pip install -r requirements.txt` 安装普通依赖；不需要 Docker。

## 实现与运行

本目录已包含完整离线实现（仅 NumPy；地图用 matplotlib Agg）：

- `hydro/`：`gridio`（ASC 读写/地理元数据）、`fill`（组件标记+优先洪水最小填洼）、
  `flow`（D8 + BFS 父指针无环平坦路由）、`accumulation`（汇流单元数/面积）、
  `basins`（出口流域，允许嵌套）、`verify`（独立核验）、`report`（PNG/箭头图）、`build`（编排）。
- `tests/test_hydro.py`：合成栅格单元测试；`run_all.py`：三场景一键构建/核验/出图。
- `METHOD.md`：算法、并列规则、边界退出规则、平坦路由证据与守恒口径的完整说明。

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_all.py                                   # dem / tiny_flat / enclosed
.venv/bin/python -m unittest discover -s tests -v             # 单元测试
# 或分步：python -m hydro.build|verify|report --dem ... --outlets ... --outdir ...
```

产物写入 `outputs/<场景>/`：ASC 栅格（原高程/填后高程/填深/状态/组件/方向/接收格/
平坦证据/汇流计数/面积/流域）、逐格与边界退出 CSV、守恒 `report.json`、
独立 `verify_report.json`、`flow_arrows.txt` 与 `map_report.png`。
封闭组件（如 `cases/enclosed.asc`）标记未解决，不开洞、不伪造排水。
