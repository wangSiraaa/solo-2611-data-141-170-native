"""plan.json 与 allowed_changes.json 的加载与一致性校验。"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import PlanError


@dataclass
class Plan:
    parameters: dict
    target_table: str
    target_pk: str
    targets: list[str]
    result_identity_column: str
    raw: dict = field(default_factory=dict)


@dataclass
class RemovablePredicate:
    id: str
    sql: str
    atom_index: int = -1
    atom_node_id: str = ""
    atom_sql: str = ""


@dataclass
class AllowedChanges:
    parameter_alternatives: dict[str, list]
    removable_predicates: list[RemovablePredicate]
    raw: dict = field(default_factory=dict)


def _load_json(path: Path, label: str) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlanError(f"无法读取 {label} {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlanError(f"{label} 不是合法 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PlanError(f"{label} 顶层必须是对象")
    return data


def load_plan(path: Path) -> Plan:
    data = _load_json(path, "plan")
    for key in ("parameters", "target_table", "target_pk", "targets",
                "result_identity_column"):
        if key not in data:
            raise PlanError(f"plan 缺少字段 {key}")
    if not isinstance(data["parameters"], dict):
        raise PlanError("plan.parameters 必须是对象")
    if not isinstance(data["targets"], list) or not data["targets"]:
        raise PlanError("plan.targets 必须是非空数组")
    for t in data["targets"]:
        if not isinstance(t, str):
            raise PlanError("plan.targets 元素必须是字符串主键")
    if len(set(data["targets"])) != len(data["targets"]):
        raise PlanError("plan.targets 存在重复主键")
    if not isinstance(data["result_identity_column"], str):
        raise PlanError("plan.result_identity_column 必须是字符串")
    return Plan(
        parameters=dict(data["parameters"]),
        target_table=data["target_table"],
        target_pk=data["target_pk"],
        targets=list(data["targets"]),
        result_identity_column=data["result_identity_column"],
        raw=data,
    )


def load_allowed_changes(path: Path) -> AllowedChanges:
    data = _load_json(path, "allowed_changes")
    alts = data.get("parameter_alternatives", {})
    if not isinstance(alts, dict):
        raise PlanError("allowed_changes.parameter_alternatives 必须是对象")
    for name, values in alts.items():
        if not isinstance(values, list) or not values:
            raise PlanError(f"参数 {name} 的备选值必须是非空数组")
    preds_raw = data.get("removable_predicates", [])
    if not isinstance(preds_raw, list):
        raise PlanError("allowed_changes.removable_predicates 必须是数组")
    preds = []
    seen = set()
    for item in preds_raw:
        if not isinstance(item, dict) or "id" not in item or "sql" not in item:
            raise PlanError("removable_predicates 每项必须含 id 与 sql")
        if item["id"] in seen:
            raise PlanError(f"可删谓词 id 重复: {item['id']}")
        seen.add(item["id"])
        preds.append(RemovablePredicate(id=item["id"], sql=item["sql"]))
    return AllowedChanges(parameter_alternatives=alts,
                          removable_predicates=preds, raw=data)
