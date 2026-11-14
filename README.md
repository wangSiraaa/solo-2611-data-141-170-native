# 商品更替与重复供应商报价的可比价格链接

本目录是题目输入材料，未包含答案实现。所有业务与观测数据均为本地合成数据。

## 文件

- `fixtures/quotes.csv`
- `fixtures/products.csv`
- `fixtures/plan.json`
- `cases/single_bridge.csv`
- `cases/disconnected.csv`

以题面规定的字段语义与边界为准。cases/ 中的文件只替换说明中指明的输入，其他基础输入保持不变。小材料用于调试与核验，不能通过写死文件名、行号或样例答案完成任务。

运行环境：Python 3.11。可用 `python -m pip install -r requirements.txt` 安装普通依赖；不需要 Docker。
