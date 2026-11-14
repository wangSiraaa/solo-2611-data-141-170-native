# 资格解释工具（survey_eligibility）

把**一个冻结问卷**上**已有回答**解释成逐成员的资格数据。工具只读输入：不编辑问卷、不采集新回答、不做答案插补、不做统计推断。

- Python 3.11，仅用标准库。
- 处理一个问卷、一层成员名册；实例身份是 `(household_id, member_id)`，不同家庭里的同名 `member_id` 是不同实例。

## 运行

```sh
# 样例（fixtures）
python3 -m survey_eligibility \
  --questionnaire fixtures/questionnaire.json \
  --households    fixtures/households.jsonl \
  --roster        fixtures/roster.jsonl \
  --responses     fixtures/responses.jsonl \
  --output-dir    out/sample

# 边界样例（未知年龄、同意未知、跨分支汇合、残留养老金、两户同名成员）
python3 -m survey_eligibility \
  --questionnaire fixtures/questionnaire.json \
  --households    examples/edge-cases/households.jsonl \
  --roster        examples/edge-cases/roster.jsonl \
  --responses     examples/edge-cases/responses.jsonl \
  --output-dir    out/edge

# 非法循环问卷：在处理任何回答之前被拒绝，退出码 2
python3 -m survey_eligibility \
  --questionnaire fixtures/cases/cyclic-questionnaire.json \
  --households    fixtures/households.jsonl \
  --roster        fixtures/roster.jsonl \
  --responses     fixtures/responses.jsonl \
  --output-dir    out/cyclic
```

`examples/run_demo.sh` 一次跑完上面三步。

## 输出（output-contract.json）

| 文件 | 内容 |
| --- | --- |
| `long_table.csv` | 长表，键为 `(household_id, member_id, question)`；含 `eligibility`、`answer_state`、路径计数与分母说明 |
| `path_evidence.jsonl` | 每题一条：至少一条支持路径、至少一条反证路径；含节点序列、逐步判定、见证取值与取值来源 |
| `summary.json` | 各题 MUST/MAY/NEVER 人数、回答类别计数、显式分母与响应率、隔离错误清单 |

`answer_state` ∈ `valid_answer / asked_missing / asked_refused / conditional_answer / structurally_not_asked / stray_answer`。

## 语义要点与实现对应

1. **路径枚举**（`engine.py`）。对每个成员枚举所有与已知回答相容的完整路径。未知值（missing / refused / 无记录）按其声明域枚举，并在分支处收窄；同一变量在一条路径上始终是同一个值，因此不会出现“年龄既 <18 又 ≥65”。
2. **只有实际经过的问题才用其回答**。到达 ask 节点才把该题的回答并入域；从未经过的路径完全不看该回答。因此“响应存在 ≠ 被合法问到”：不可达的残留答案记为 `stray_answer`，不驱动任何前置判断。
3. **MUST / MAY / NEVER**。必经 = 所有相容完整路径都经过；部分 = 部分经过；从不 = 无路径经过。跨分支汇合的后续题按“所有路径都经过”判定为 MUST，而不是把未知分支简单传播成未问。
4. **资格与回答分开**。`valid_answer`（MUST 且有回答）、`conditional_answer`（MAY 且有回答）、`asked_missing`、`asked_refused`、`structurally_not_asked`、`stray_answer`。零小时、枚举首值等均为有效值。
5. **分母显式**。`summary.json` 对每题分别报告 MUST/MAY/NEVER 人数；`valid_answer_among_must` 与 `conditional_answer_among_may` 各有自己的分母，MAY 不会被静默并入 MUST 或排除。
6. **错误隔离**（`loader.py`）。未知名册成员、错误作用域题目、越界值、冲突重复回答都带源文件与行号单独报告；冲突键整体不用于资格推导（不挑选“赢家”）。合法原文件不被改写。
7. **结构校验先于数据**（`cli.py`）。先校验类型、引用作用域、跳转目标、支配性与环；非法问卷（如循环图）在打开任何回答文件之前以退出码 2 拒绝。
8. **独立穷举核对**（`exhaustive.py`）。不复用引擎的路径枚举，在极小声明域上对具体赋值空间逐一路走并判定资格；测试用它交叉验证引擎（`tests/test_engine.py`）。

## 测试

```sh
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

覆盖：问卷结构/类型/作用域/跳转/环、加载错误隔离、未知值一致性、MUST/MAY/NEVER、六种回答类别、分母口径、样例与边界样例的穷举一致性、CLI 先拒绝后读数据。
