"""硬约束：禁用组合、锁定运行与块内重复策略。

约束语义：

* ``forbidden_all`` —— 部分指定的因素取值，命中即全局禁用，构建候选集时已剔除；
  任何环节都不会为了提高信息准则得分把它们重新引入。
* ``locked_runs.csv`` —— 必须原样保留的运行（块与因素组合都不能移动或修改），
  若引用未知水平、命中禁用组合或落在未声明块中，属于矛盾配置，直接拒绝。
* 重复策略 —— 每个块内至少有一个完全相同的因素组合出现 >=2 次；重复行占用
  运行名额，只是对同一候选 id 的多次引用，不是新的设计方向。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import Any

from .errors import ConfigError
from .factors import CandidateSet, FactorSet

REPLICATE_POLICY = (
    "at least one identical factor combination repeated within each block; "
    "repetitions consume run slots"
)
LOCKED_HEADER_REQUIRED = ("run_id", "block")


@dataclass(frozen=True)
class LockedRun:
    run_id: str
    block: str
    combo: tuple[Any, ...]
    candidate_id: int


@dataclass
class Constraints:
    forbidden_rules: list[dict[str, Any]]
    replicate_required: bool
    raw_replicates: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Constraints":
        if not isinstance(payload, dict):
            raise ConfigError("constraints.json 顶层必须是对象")
        extra = set(payload) - {"forbidden_all", "replicates"}
        if extra:
            raise ConfigError(f"constraints.json 含未知键: {sorted(extra)}")
        rules = payload.get("forbidden_all", [])
        if not isinstance(rules, list):
            raise ConfigError("forbidden_all 必须是列表")
        replicate_text = payload.get("replicates", "")
        if not isinstance(replicate_text, str) or not replicate_text.strip():
            raise ConfigError("必须显式声明 replicates 策略字符串")
        if "repeated within each block" not in replicate_text:
            raise ConfigError(
                f"不支持的 replicates 策略: {replicate_text!r}；"
                f"当前仅支持 {REPLICATE_POLICY!r}"
            )
        if "consume run slots" not in replicate_text:
            raise ConfigError("replicates 策略必须明确重复占用运行名额")
        return cls(
            forbidden_rules=[dict(r) for r in rules],
            replicate_required=True,
            raw_replicates=replicate_text,
        )

    @classmethod
    def from_json_file(cls, path: str) -> "Constraints":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def load_locked_runs(
    path: str,
    factor_set: FactorSet,
    candidates: CandidateSet,
    block_ids: set[str] | None = None,
) -> list[LockedRun]:
    """读取并严格校验锁定运行 CSV。"""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ConfigError(f"锁定运行文件 {path} 为空")
        required = list(LOCKED_HEADER_REQUIRED) + factor_set.names
        if reader.fieldnames != required:
            raise ConfigError(
                f"锁定运行表头必须恰好为 {required}，实际为 {reader.fieldnames}"
            )
        locked: list[LockedRun] = []
        seen_ids: set[str] = set()
        legal = set(candidates.combos)
        for line_no, row in enumerate(reader, start=2):
            run_id = (row.get("run_id") or "").strip()
            block = (row.get("block") or "").strip()
            if not run_id or not block:
                raise ConfigError(f"锁定运行第 {line_no} 行缺少 run_id 或 block")
            if run_id in seen_ids:
                raise ConfigError(f"锁定运行 run_id 重复: {run_id!r}")
            seen_ids.add(run_id)
            if block_ids is not None and block not in block_ids:
                raise ConfigError(
                    f"锁定运行 {run_id} 引用了未声明的块 {block!r}"
                )
            combo = _parse_combo(row, factor_set)
            if combo not in legal:
                if combo in set(candidates.forbidden):
                    raise ConfigError(
                        f"锁定运行 {run_id} 的组合 {dict(zip(factor_set.names, combo))} "
                        f"命中禁用组合，配置矛盾"
                    )
                raise ConfigError(
                    f"锁定运行 {run_id} 的组合 {dict(zip(factor_set.names, combo))} "
                    f"含非法水平，配置矛盾"
                )
            locked.append(LockedRun(
                run_id=run_id,
                block=block,
                combo=combo,
                candidate_id=candidates.index_of_combo(combo),
            ))
        if not locked:
            raise ConfigError(f"锁定运行文件 {path} 没有任何数据行")
    return locked


def _parse_combo(row: dict[str, str], factor_set: FactorSet) -> tuple[Any, ...]:
    values: list[Any] = []
    for factor in factor_set.factors:
        text = row.get(factor.name)
        if text is None or text == "":
            raise ConfigError(f"锁定运行缺少因素 {factor.name!r} 的取值")
        if factor.is_numeric:
            try:
                num = float(text)
            except ValueError as exc:
                raise ConfigError(
                    f"数值因素 {factor.name!r} 的取值无法解析: {text!r}"
                ) from exc
            declared = {float(v): v for v in factor.levels}
            if num not in declared:
                raise ConfigError(
                    f"数值因素 {factor.name!r} 的取值 {text} 不在声明水平 {factor.levels}"
                )
            values.append(declared[num])
        else:
            if text not in factor.levels:
                raise ConfigError(
                    f"类别因素 {factor.name!r} 的取值 {text!r} 不在声明水平 {factor.levels}"
                )
            values.append(text)
    return tuple(values)


def block_replicate_satisfied(candidate_ids: list[int]) -> bool:
    """块内是否存在重复候选（同一组合出现 >= 2 次）。"""
    if len(candidate_ids) < 2:
        return False
    return len(set(candidate_ids)) < len(candidate_ids)
