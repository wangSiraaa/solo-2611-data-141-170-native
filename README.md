# 有限批次的风险受控顺序抽查审计

本目录是题目输入材料，未包含答案实现。所有业务与观测数据均为本地合成数据。

## 文件

- `fixtures/lot.csv`
- `fixtures/plan.json`
- `fixtures/observations.csv`
- `fixtures/tiny.json`
- `cases/defects.csv`
- `cases/duplicate.csv`

以题面规定的字段语义与边界为准。cases/ 中的文件只替换说明中指明的输入，其他基础输入保持不变。小材料用于调试与核验，不能通过写死文件名、行号或样例答案完成任务。

运行环境：Python 3.11 标准库；不需要 Docker 或外部服务。

---

## 实现（seqaudit）

`seqaudit/` 为本题实现，方法依据见 `docs/METHOD.md`。

```sh
# 顺序审计：出具证书（资格失败时仅出诊断，退出码 2）
python3 -m seqaudit audit --plan fixtures/plan.json --lot fixtures/lot.csv \
    --observations fixtures/observations.csv --out out/certificate_base.json

# 独立重算并核对证书（哈希、资格、证据、批次不变性、判定自洽）
python3 -m seqaudit verify --plan fixtures/plan.json --lot fixtures/lot.csv \
    --observations fixtures/observations.csv --certificate out/certificate_base.json

# tiny 独立枚举：穷举零假设配置 × 全部排列，核查整路径曾接受概率 <= alpha
python3 -m seqaudit verify --tiny fixtures/tiny.json

# 单元测试（32 项）
python3 -m unittest discover -s tests
```

替换观察文件即可审计 cases/ 场景（`cases/defects.csv` 不接受、
`cases/duplicate.csv` 资格阻断）。生成的证书与验证记录写入 `out/`（已忽略，不入库）。
