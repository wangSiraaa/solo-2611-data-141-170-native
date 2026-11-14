# sqlcomplete：编辑器 SQL 补全核心

纯 Python 3.11 标准库实现，不连接数据库、不依赖完整可解析的 SQL。
输入 schema（catalog）、整段 SQL（含游标右侧文本）和从 0 开始的 Unicode
码点游标，输出带来源的候选与半开替换范围。

## 快速开始

### API

```python
import json
from sqlcomplete import complete, apply_completion

catalog = json.load(open("fixtures/catalog.json", encoding="utf-8"))
sql = "SELECT o.am FROM orders o"
result = complete(catalog, sql, cursor=11)

result.state                 # "ok"
result.start, result.end     # (9, 11) 半开码点区间，覆盖整个 "am" 片段
result.candidates[0].label   # "amount_cents"
result.candidates[0].kind    # "column"
result.candidates[0].origin  # {"type":"column","relation":"o", ...}
result.to_dict()             # 可直接 JSON 序列化

apply_completion(sql, result)  # "SELECT o.amount_cents FROM orders o"
```

### 命令行

```sh
# 单条补全（输出含 applied，便于核对替换范围）
python -m sqlcomplete.cli complete fixtures/catalog.json "SELECT o.am FROM orders o" 11

# 批量跑 cases.json
python -m sqlcomplete.cli fixtures fixtures/catalog.json fixtures/cases.json
```

### 测试

```sh
python -m unittest discover -s tests -v
```

## 候选与替换范围约定

每条候选包含：

| 字段 | 含义 |
| --- | --- |
| `label` | 展示名（列名、`别名.列`、别名、CTE 或表名） |
| `kind` | `column` / `alias` / `cte` / `table` |
| `insert_text` | 应用到替换范围时插入的文本（别名候选自带尾随 `.`） |
| `origin` | 来源：关系别名、关系种类（table/cte/derived）、物理表名、列名 |
| `replacement` | `[start, end)` 半开区间，Unicode 码点偏移 |

- 游标位于标识符中间时，替换范围覆盖该标识符**整段**，应用后不会留下
  右半截重复字符；游标不在词内时范围退化为 `[cursor, cursor)`（纯插入）。
- 标识符比较不区分大小写；输入字符串本身不改写，插入文本使用 catalog
  的规范拼写。
- 排序稳定：先按 kind（column → alias → cte → table），再按标签与来源。

## 作用域与歧义规则

- **帧链**：每个 SELECT 是一帧；派生表与 EXISTS/IN 子查询产生子帧。
  普通分组括号 `(...)` 不产生作用域。
- **派生表非 LATERAL**：子帧看不到外层别名，链在自身终止；对不可见别名
  的限定补全返回 `unknown_qualifier`，不猜测。
- **EXISTS/IN 可相关**：子帧沿父帧链回退，内层同名别名遮蔽外层。
- **CTE 按定义顺序**登记，后者可引用前者；CTE 同名遮蔽物理表；只暴露
  可解析的输出列（普通/限定列、`*`、`alias.*`、显式 `AS` 别名、
  双标识符隐式别名）。无法解释的无别名表达式不产出列。
- **裸列歧义**：同一列名被两个及以上可见关系暴露时，不给出裸列候选，
  只给 `别名.列` 限定形式，避免绑定错误对象；无歧义列同时给裸列与限定形式。

## 局部语法恢复

- 整段 SQL 都会参与分析，**不截断游标右侧文本**，右侧已有的 FROM/JOIN
  上下文照常生效。
- 允许缺表达式、缺右括号、缺后续子句；不闭合的字符串/括号/CTE 都有
  确定行为，解析器在期望外的词上跳过并继续。
- 无法恢复确定作用域时显式标状态（`no_relations` / `unknown_qualifier`
  / `other_clause` / `alias_definition`），候选为空，绝不把整个 schema
  的字段塞进结果。

## 词法护栏

游标处于单引号字符串、`-- 行注释`、`/* 块注释 */`（含未闭合块注释）或
双引号标识符内部时，状态为 `in_string_or_comment` / `in_quoted`，
不推荐业务标识符。字符串支持 `''` 转义，双引号标识符支持 `""` 转义。

## 支持边界

**支持**：SELECT、FROM、INNER/LEFT（及 RIGHT/FULL/CROSS，按连接解析）
JOIN、ON、WHERE、USING、有序非递归 CTE、一层派生表与 EXISTS/IN 子查询、
投影别名参与建议、`schema.table` 关系名、无引号 ASCII 标识符（大小写
不敏感）与双引号标识符。

**明确不做**：SQL 执行、列级血缘、缺行/错误诊断、查询修复平台、全 SQL
方言兼容；不解释任意表达式（函数/运算表达式仅在带显式别名时作为输出列）；
不支持递归 CTE、窗口函数专用补全、存储过程/DDL、LATERAL、超过一层嵌套的
派生表相关引用。`GROUP BY/HAVING/ORDER BY/LIMIT` 与集合操作子句可被
词法跳过不报错，但其中不提供列建议（`other_clause`）。
