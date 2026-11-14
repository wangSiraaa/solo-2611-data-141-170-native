# 部分部件观测下的共同失效原因与维修收益范围

本目录是题目输入材料，未包含答案实现。所有业务与观测数据均为本地合成数据。

## 文件

- `fixtures/system.json`
- `fixtures/events.csv`
- `fixtures/plan.json`
- `fixtures/snapshots.csv`
- `fixtures/weak_snapshots.csv`

以题面规定的字段语义与边界为准。cases/ 中的文件只替换说明中指明的输入，其他基础输入保持不变。小材料用于调试与核验，不能通过写死文件名、行号或样例答案完成任务。

运行环境：Python 3.11。可用 `python -m pip install -r requirements.txt` 安装普通依赖；不需要 Docker。

快照字段中 `0` 表示健康（该节点未失效），`1` 表示失效，`NA` 表示未观测，不能按健康处理。每行是一个独立运行周期，列名是 system.json 中的可观测节点。snapshots.csv 与 weak_snapshots.csv 分别单独分析，不合并。
