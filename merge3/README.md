# merge3 — 固定两表的一次离线三方合并工具

面向 `customers` / `orders` 两张固定表，以 `base` 为基线合并 `left`、`right` 两个离线分支。
只做**一次离线合并**：不做实时协作、CRDT、schema 迁移，也不猜测“正确值”——所有取舍都来自用户裁决文件。
仅依赖 Python 3.11 标准库。

## 三个入口

```bash
# 1) 计划：识别真实变更，组织冲突组，保留三侧值/自动合并内容/待决选择
python -m merge3 plan --input fixtures/input.json --schema fixtures/schema.sql --out work/plan.json

# 2) 裁决：套用用户 decisions，全量复查约束；失败只返回残留原因，不写产物
python -m merge3 resolve --plan work/plan.json --decisions work/decisions.json --out work/resolved.json

# 3) 导出：完整 SQLite + 逐变更来源表，重新打开做主外键/唯一/非负金额校验
python -m merge3 export --resolved work/resolved.json --out work/merged.sqlite
```

## 差异分类（以 base 为基线，逐字段三方比较）

| 情况 | 分类 | 处理 |
| --- | --- | --- |
| 两侧都没动 | `unchanged` | 自动保留 |
| 各改不同字段 | `auto_merged` / `left_only`、`right_only` | 字段级自动合并 |
| 同字段改成同一值 | `both_same` | 自动取该值，不算冲突 |
| 同字段改成不同值 | `field_conflict` | 冲突组，逐字段 `pick` 一侧 |
| 同主键各自新增、内容不同 | `insert_collision` | 可取整侧或 `field_mix` 按字段混合 |
| 一侧删行、另一侧保留/修改 | `delete_vs_modify` | 显式 `keep_row` 或 `delete_row` |
| 两侧都删 | `deleted` | 自动删除并在来源表留痕 |

**三态严格区分**：SQL `NULL`、空字符串 `""` 是真实值；整行不存在用内部哨兵
`{"$missing": true}` 表示，绝不与前两者混淆。不会把“两边都变了”统称为冲突，
也没有任何按更新时间取胜的逻辑。

## 冲突组（可解释、带约束说明与合法选项）

- `field_conflict` — 同字段异值，选项只列取值互不相同的侧；
- `unique_conflict` — 跨行唯一碰撞（如 C5、C6 的 code 都变成 `COMMON`），
  成员行必须一起裁决，裁决后再次检查 UNIQUE；
- `delete_vs_modify` — 删除与修改互斥，禁止静默决定；
- `insert_collision` — 主键身份相同的新增碰撞，主键本身不可修改；
- `fk_delete_reference` — 删客户与引用该客户的订单并存（含“一边删客户、
  另一边新增引用订单”）。选项只有**显式保留/恢复客户**或**显式列出要丢弃的订单**，
  工具不会自动删订单、自动恢复删除行或编造业务值。

组之间的外键依赖在计划里双向登记。一个裁决引出新冲突时（例如保留订单却删了其客户），
resolve 返回派生残留组（稳定 id，如 `fk:customers:C3`）及仍需裁决的原因，
可在第二轮 decisions 中继续裁决；任何半成品都不会落盘。

## 摘要绑定（防旧裁决套用）

- 计划内含 base/left/right/schema 四个文件的 SHA-256 与自身 `plan_digest`；
- 裁决文件必须显式点名 `plan_digest`，resolve/export 每次重新计算输入摘要；
- 任一源文件变化即报 `InputChangedError` 拒绝执行。

## 裁决文件形状

```json
{
  "plan_digest": "计划文件中的 plan_digest",
  "decisions": [
    {"group_id": "field:customers:C2:name", "decision": {"action": "pick", "side": "left"}},
    {"group_id": "dvm:customers:C3", "decision": {"action": "keep_row", "side": "right"}},
    {"group_id": "fk:customers:C4", "decision": {"action": "keep_customer", "side": "right"}},
    {"group_id": "insert:orders:O9",
     "decision": {"action": "field_mix", "fields": {"amount_cents": "right", "note": "left"}}},
    {"group_id": "unique:customers:code:COMMON",
     "decision": {"action": "unique_pick", "members": [
       {"table": "customers", "key": "C5", "column": "code", "side": "left"},
       {"table": "customers", "key": "C6", "column": "code", "side": "base"}]}}
  ]
}
```

无冲突计划允许空 `decisions` 列表。每个计划冲突组都必须被显式裁决；未知组、
未知动作、重复裁决、成员覆盖不完整、丢弃订单清单不完整都会被拒绝。

## 输出库

与 `schema.sql` 同构的 `customers`、`orders` 两张表，外加审计表：

```sql
CREATE TABLE merge_provenance(
  table_name TEXT, row_key TEXT, column_name TEXT,
  change_kind TEXT,       -- unchanged/auto_merged/insert/conflict_resolved/deleted/identity
  sources TEXT,           -- base,left,right 逗号分隔
  value_json TEXT, note TEXT)
```

导出后关闭再以 `PRAGMA foreign_keys=ON` 重开，执行 `foreign_key_check`、
全部 UNIQUE 重复值检查、NOT NULL、`amount_cents` 非负整数检查，并逐行比对裁决文档；
失败即删除输出文件。输入库全程以 `mode=ro` 只读打开。

## 测试与演示

```bash
python -m unittest tests.test_merge3 -v   # 24 个回归测试
python tests/tiny_example.py              # 极小合成例，按变更意图逐项核对
python tests/demo_fixtures.py             # fixtures 端到端 plan/resolve/export 演示
```

覆盖：独立字段修改、删除/改动、同字段冲突、新增碰撞、跨行唯一冲突、外键引用冲突、
NULL/空串/行缺失三态、左右交换对称解释、行序无关、摘要防篡改、输入不可变、
派生冲突两轮裁决、导出重开校验。
