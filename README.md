# 实验运行设计工具（题 144）

在尚无响应数据时，按声明因素/模型/约束/计划，用**条件 D 准则的候选坐标交换**
选出使指定主效应、平方项、二阶交互与块效应可估计的有限次运行设计，并给出
秩/混淆证据、朴素可行基线对照、微型枚举核对与等价编码不变性验证。

方法细节、准则定义与失败分类见 [METHODS.md](METHODS.md)。

## 输入

`fixtures/` 下的 `factors.json`、`model.json`、`constraints.json`、`plan.json`
与 `locked_runs.csv`（三个因素、两个实验日、禁用组合 `(temperature=60, catalyst=B)`、
两条必须保留的锁定运行）。替代案例位于 `fixtures/cases/`（小预算、混淆因素）、
`fixtures/micro/`（可全枚举的微缩候选集）、`fixtures/noblock/`（无块效应）。

## 安装与运行

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pytest
PYTHONPATH=src .venv/bin/python -m expdesign.cli
```

退出码：`0` 成功；`2` 配置错误（未知项/矛盾输入，先拒绝）；
`3` 有证据的不可行（结构混淆/预算不足/锁定冲突）；
`4` 有限搜索未找到方案（不是不可行证明）。

## 测试与独立核对

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
PYTHONPATH=src .venv/bin/python scripts/independently_verify.py outputs
```

## 代码结构

```
src/expdesign/
  errors.py        三类异常与退出码
  factors.py       因素声明、编码、合法候选集（禁用组合剔除）
  model.py         模型项 schema、列身份、设计矩阵
  constraints.py   禁用/锁定/块内重复
  plan.py          块名额、预算与结构可行性（秩、零空间证据）
  criteria.py      条件 D 准则、可估计性、相关诊断
  baseline.py      确定性贪心可行基线
  search.py        多起点坐标交换 + QR 秩救援 + 审计轨迹
  enumerate.py     微型候选集全枚举
  invariance.py    等价编码下的可估计性不变性
  design.py        端到端编排、独立重建核对、工件
  cli.py / io_utils.py
scripts/independently_verify.py   仅从落盘工件独立重建核对
tests/test_expdesign.py           45 项测试
```

所有数据为合成数据，不含响应数据；工具不做效果分析、功效平台、在线自适应或
多模型筛选。
