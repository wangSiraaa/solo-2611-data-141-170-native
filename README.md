# order-exclusion-diagnoser（只读订单报表排除诊断工具）

在受限 SQL 范围内诊断：某订单在基础表中存在，却没有出现在报表中，
是在**哪一步**被排除；以及在 `allowed_changes.json` 声明的有限范围内，
哪些**最少条件调整**能让它出现，并逐一用真实 SQL 重跑核实。

只读：数据库以 SQLite URI `mode=ro` + `PRAGMA query_only=ON` 打开，
诊断过程绝不修改源数据；方案仅作**建议**，不自动改写为正式 SQL。

## 支持的 SQL 子集（超出即明确拒绝）

- 普通 `SELECT`（列引用与 `AS` 别名）
- 最多 2 次 `INNER JOIN` / `LEFT [OUTER] JOIN`，`ON` 只能是等值列比较
- `WHERE` 由比较（`= <> != < <= > >=`）与 `AND` / `OR` / 括号组成
- 命名绑定参数 `:name` / `@name`（不支持位置参数 `?`）
- 明确拒绝：聚合、子查询、`GROUP BY`/`ORDER BY`/`LIMIT`、`UNION`、
  `IN`/`BETWEEN`/`LIKE`/`IS`/`NOT`/`CASE`、3 次及以上连接等；返回码 3。

解析是真正的递归下降解析（`diagtool/parser.py`），保留源码区段与完整布尔结构；
求值复现 SQLite 三值逻辑与类型亲和性，并与 SQLite 逐项对拍（见测试）。

## 快速开始

```bash
# 1) 建立独立只读输入库（prepare.py 会复制 fixtures 并建 report.sqlite）
python3 fixtures/prepare.py work-input

# 2) 人类可读报告
python3 -m diagtool explain \
  --db work-input/report.sqlite \
  --query work-input/query.sql \
  --plan work-input/plan.json \
  --allowed work-input/allowed_changes.json

# 3) 结构化 JSON 报告（可写入文件）
python3 -m diagtool explain \
  --db work-input/report.sqlite \
  --query work-input/query.sql \
  --plan work-input/plan.json \
  --allowed work-input/allowed_changes.json \
  --format json --output report.json

# 可选：限制枚举成本上限（默认到单操作总数，足以覆盖所有组合）
python3 -m diagtool explain ... --max-cost 2
```

退出码：`0` 成功；`2` 输入/计划不一致；`3` 不支持的 SQL 语法。

## 报告内容

1. **原查询解释**：基础表、连接类型与 `ON`、SELECT 列、WHERE 原子谓词
   （按源码顺序编号并给出源码区段）、绑定参数。
2. **原报表基线**：真实 SQL 执行得到的完整结果行与各目标的可见次数。
3. **逐目标排除诊断**，分类包括：
   - `base_row_missing`：基础表无该主键；
   - `inner_join_no_match`：`INNER JOIN` 某步无匹配（含 NULL 键），WHERE 尚未参与；
   - `where_false`：所有候选行的 WHERE 为 FALSE（可能含 UNKNOWN 原子，
     但只要 AND 链上有一处 FALSE 即整体 FALSE）；
   - `where_unknown`：候选行 WHERE 为 UNKNOWN（如 LEFT 补出的 NULL 参与比较）；
   - `visible`：目标本就在报表中（给出重复次数）。
   每条候选关联行都有独立证据：涉及的物理行（表、主键、原值）、每个原子的
   三值结果与左右操作数来源、完整布尔树（含 OR 分支）。**右表多行会逐行给出证据**，
   不会只挑一行解释整个订单。LEFT 补出的 NULL 行按真实 SQL 语义标注
   `null_extended` 并参与三值求值。
4. **最少条件调整方案**（逐目标）：
   - 成本：一个参数改值或删除一个已声明原子谓词各计 1；删除等价于把该
     AST 位置替换为 `TRUE`，**保留其余括号结构**；
   - 按成本递增枚举，返回该目标**全部并列最小**方案；每个方案都用实际 SQL
     在只读库上重跑，核实目标可见；
   - 报告相对原报表**新增/减少的完整结果行及重复次数**（多重集比较）；
   - 若放宽声明范围内所有 WHERE 条件仍无法补回缺失关联（如 INNER 无匹配、
     基础行缺失），明确报告**无可行方案**，且不发明新表行或新连接。

## 现成样例覆盖（fixtures）

| 目标 | 情况 | 诊断类别 | 最少调整 |
|------|------|----------|----------|
| O2 | 客户 active=0，金额不足，付款 pending | `where_false` | 3：改 `min_amount`+删 2 谓词 |
| O3 | 付款 status 为 NULL（UNKNOWN 原子） | `where_false`（FALSE 主导） | 2：删 2 谓词 |
| O4 | 无付款（LEFT 补 NULL），走 OR 分支 | `where_unknown` | 1：删 `payment_status` |
| O5 | 无客户（`o.customer_id` 为 NULL） | `inner_join_no_match` | 无可行方案 |

另含 O1（多付款，重复 2 次）、O6（可见）、base_row_missing 等测试。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

覆盖：解析与未知语法拒绝、四类排除路径、OR 完整结构、LEFT NULL 语义、
右表多行证据、最小性与并列方案、方案真实重跑与影响差集、行序不变性、
只读保证，以及 Python 三值/亲和性与 SQLite 的逐项对拍。

## 模块结构

```
diagtool/
  tokenizer.py   词法分析（拒绝未知记号/关键字）
  parser.py      递归下降解析，保留源码区段与布尔结构
  ast_nodes.py   AST 节点与原子/参数遍历
  semantics.py   SQL 三值逻辑与类型亲和性（与 SQLite 对拍）
  db.py          只读 SQLite 访问与模式解析
  analysis.py    候选关联行重建、逐行三值求值与证据
  diagnose.py    目标排除路径分类与证据组装
  rewrite.py     删除谓词 => 将该 AST 位置替换为 TRUE
  solutions.py   逐目标最小方案枚举（含并列）与真实 SQL 验证
  runner.py      真实 SQL 执行与结果多重集差集
  report.py      报告编排与结构化输出
  cli.py         命令行（explain 命令）
```

## 明确不做

不做列级血缘平台、不自动修复业务数据、不做通用数据库调优；
不猜测未知语法，不修改输入库，不把建议方案自动落成正式 SQL。
