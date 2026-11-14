"""输入加载与校验。

所有数值按「能转 int 就转 int，否则 float，否则字符串」解析；
案例与探针的字段完整性、谓词算子、代价矩阵完整性均在此处校验。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

VALID_OPS = ("eq", "ge", "le")


def parse_scalar(token: str):
    """把 CSV 单元格解析为 int / float / str。"""
    s = token.strip()
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


@dataclass(frozen=True)
class Predicate:
    id: str
    field: str
    op: str
    value: object

    def to_dict(self) -> dict:
        return {"id": self.id, "field": self.field, "op": self.op, "value": self.value}


@dataclass(frozen=True)
class Case:
    id: str
    fields: dict
    target: str
    weight: float


@dataclass(frozen=True)
class Plan:
    max_rules: int
    rule_penalty: float
    allow_default: tuple


@dataclass(frozen=True)
class CostMatrix:
    # cost[true_queue][assigned_queue]
    cost: dict
    queues: tuple


def load_plan(path: str | Path) -> Plan:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data["max_rules"], int) or data["max_rules"] < 0:
        raise ValueError("plan.max_rules 必须是非负整数")
    if not data["allow_default"]:
        raise ValueError("plan.allow_default 不能为空")
    return Plan(
        max_rules=data["max_rules"],
        rule_penalty=parse_scalar(str(data["rule_penalty"])),
        allow_default=tuple(data["allow_default"]),
    )


def load_predicates(path: str | Path) -> list[Predicate]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("predicates.json 必须是非空列表")
    preds = []
    seen_ids = set()
    for item in data:
        pid = item["id"]
        if pid in seen_ids:
            raise ValueError(f"谓词 id 重复: {pid}")
        seen_ids.add(pid)
        op = item["op"]
        if op not in VALID_OPS:
            raise ValueError(f"谓词 {pid} 使用了不支持的算子: {op}（仅支持 {VALID_OPS}）")
        preds.append(
            Predicate(
                id=pid,
                field=item["field"],
                op=op,
                value=parse_scalar(str(item["value"])),
            )
        )
    return preds


def load_costs(path: str | Path) -> CostMatrix:
    cost: dict = {}
    queues: set = set()
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            true_q, assigned = row["true_queue"], row["assigned_queue"]
            queues.add(true_q)
            queues.add(assigned)
            cost.setdefault(true_q, {})[assigned] = parse_scalar(row["cost"])
    ordered = tuple(sorted(queues))
    # 代价矩阵必须是完整方阵，避免静默缺项
    for t in ordered:
        if t not in cost:
            raise ValueError(f"costs.csv 缺少真实队列 {t} 的行")
        for q in ordered:
            if q not in cost[t]:
                raise ValueError(f"costs.csv 缺少 {t} -> {q} 的代价")
    return CostMatrix(cost=cost, queues=ordered)


def _read_csv_rows(path: str | Path) -> tuple[list, list]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} 缺少表头")
        return list(reader), list(reader.fieldnames)


def load_cases(path: str | Path, costs: CostMatrix) -> list[Case]:
    rows, header = _read_csv_rows(path)
    for required in ("case_id", "target", "weight"):
        if required not in header:
            raise ValueError(f"{path} 缺少必需列 {required}")
    field_names = [c for c in header if c not in ("case_id", "target", "weight")]
    cases = []
    seen_ids = set()
    for row in rows:
        cid = row["case_id"]
        if cid in seen_ids:
            raise ValueError(f"case_id 重复: {cid}")
        seen_ids.add(cid)
        target = row["target"]
        if target not in costs.queues:
            raise ValueError(f"案例 {cid} 的 target={target} 不在代价矩阵队列中")
        fields = {}
        for name in field_names:
            if row[name] is None or row[name] == "":
                raise ValueError(f"案例 {cid} 字段 {name} 缺失，题面保证字段完整")
            fields[name] = parse_scalar(row[name])
        cases.append(
            Case(id=cid, fields=fields, target=target, weight=parse_scalar(row["weight"]))
        )
    if not cases:
        raise ValueError(f"{path} 没有任何案例")
    return cases


def load_unlabeled_rows(path: str | Path) -> list[dict]:
    """加载探针等未标记行，返回 [{id, fields}]。"""
    rows, header = _read_csv_rows(path)
    if "case_id" not in header:
        raise ValueError(f"{path} 缺少 case_id 列")
    field_names = [c for c in header if c != "case_id"]
    out = []
    for row in rows:
        fields = {}
        for name in field_names:
            if row[name] is None or row[name] == "":
                raise ValueError(f"行 {row['case_id']} 字段 {name} 缺失")
            fields[name] = parse_scalar(row[name])
        out.append({"id": row["case_id"], "fields": fields})
    return out
