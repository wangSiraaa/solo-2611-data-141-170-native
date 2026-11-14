"""声明式模型：项 schema 校验、列身份与设计矩阵构造。

模型只允许四类项，任何未知结构或多余键都会触发 :class:`ConfigError`：

1. ``{"kind": "intercept"}`` —— 单列 ``Intercept``，恒为 1；
2. ``{"factor": "f"}`` —— 数值因素一次项（列名 ``f``）或类别因素全部虚拟列
   （列名 ``f[level]``，参考水平无列）；
3. ``{"factor": "f", "power": p}`` —— 仅数值因素，编码值的 p 次幂，列名 ``f^p``；
4. ``{"interaction": ["f1", "f2"]}`` —— 两个不同因素的二阶交互，按双方基础列
   逐列相乘（数值×数值 1 列；数值×类别按虚拟列展开；类别×类别按虚拟列对展开）。

块效应（``include_block_effect=true``）是不参与目标效应的噪声列：每个块一列
one-hot，列名 ``block[<id>]``。注意截距与块 one-hot 张成同一子空间，全矩阵秩
按整体计算，可估计性结论不依赖选择哪一种参数化。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import ConfigError
from .factors import Factor, FactorSet

INTERCEPT_NAME = "Intercept"
BLOCK_PREFIX = "block["


@dataclass(frozen=True)
class Term:
    """解析后的单个模型项。"""

    kind: str  # intercept / factor / interaction
    factor: str | None = None
    power: int | None = None
    factors: tuple[str, str] | None = None
    raw: dict[str, Any] | None = None


class Model:
    """已通过校验的固定模型，负责把候选组合展开成设计矩阵。"""

    def __init__(self, factor_set: FactorSet, terms: list[Term],
                 include_block_effect: bool):
        self.factor_set = factor_set
        self.terms = terms
        self.include_block_effect = include_block_effect
        self._columns = self._compile_columns()
        self.treatment_names: list[str] = [name for name, _ in self._columns]
        self.n_treatment = len(self.treatment_names)

    # ------------------------------------------------------------------ #
    # 解析
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, factor_set: FactorSet, payload: dict[str, Any]) -> "Model":
        if not isinstance(payload, dict) or "terms" not in payload:
            raise ConfigError("model.json 顶层必须是含 'terms' 的对象")
        extra = set(payload) - {"terms", "include_block_effect", "target_terms"}
        if extra:
            raise ConfigError(f"model.json 含未知顶层键: {sorted(extra)}")
        raw_terms = payload["terms"]
        if not isinstance(raw_terms, list) or not raw_terms:
            raise ConfigError("模型 'terms' 必须是非空列表")

        terms: list[Term] = []
        saw_intercept = False
        for raw in raw_terms:
            term = _parse_term(raw, factor_set)
            if term.kind == "intercept":
                if saw_intercept:
                    raise ConfigError("模型中截距项重复声明")
                saw_intercept = True
            terms.append(term)
        if not saw_intercept:
            raise ConfigError("模型必须显式声明 intercept 项")

        flag = payload.get("include_block_effect", False)
        if not isinstance(flag, bool):
            raise ConfigError("include_block_effect 必须是布尔值")
        return cls(factor_set, terms, flag)

    @classmethod
    def from_json_file(cls, factor_set: FactorSet, path: str) -> "Model":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(factor_set, json.load(fh))

    # ------------------------------------------------------------------ #
    # 列编译：把项展开成 (列名, 取值函数) 的有序列表
    # ------------------------------------------------------------------ #
    def _compile_columns(self) -> list[tuple[str, Any]]:
        columns: list[tuple[str, Any]] = []
        for term in self.terms:
            if term.kind == "intercept":
                columns.append((INTERCEPT_NAME, lambda cols, num: 1.0))
            elif term.kind == "factor":
                factor = self.factor_set.get(term.factor)
                if factor.is_numeric:
                    power = term.power or 1
                    name = factor.name if power == 1 else f"{factor.name}^{power}"
                    columns.append((
                        name,
                        lambda cols, num, f=factor, p=power: num[f.name] ** p,
                    ))
                else:
                    if term.power is not None:
                        raise ConfigError(
                            f"类别因素 {factor.name!r} 不支持 power 多项式项"
                        )
                    for col_name, _level in factor.dummy_columns():
                        columns.append((
                            col_name,
                            lambda cols, num, c=col_name: cols[c],
                        ))
            else:  # interaction
                f1, f2 = (self.factor_set.get(name) for name in term.factors)
                for col_name, key1, key2 in _interaction_specs(f1, f2):
                    columns.append((
                        col_name,
                        lambda cols, num, k1=key1, k2=key2:
                            _base_value(k1, cols, num) * _base_value(k2, cols, num),
                    ))
        names = [name for name, _ in columns]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ConfigError(f"模型展开后出现重复列: {dupes}")
        return columns

    # ------------------------------------------------------------------ #
    # 矩阵构造
    # ------------------------------------------------------------------ #
    def treatment_row(self, combo: tuple[Any, ...]) -> np.ndarray:
        base_cols = self.factor_set.coded_columns(combo)
        numeric = self.factor_set.numeric_coded(combo)
        return np.array([fn(base_cols, numeric) for _, fn in self._columns],
                        dtype=float)

    def treatment_matrix(self, combos: list[tuple[Any, ...]]) -> np.ndarray:
        if not combos:
            return np.empty((0, self.n_treatment), dtype=float)
        return np.vstack([self.treatment_row(combo) for combo in combos])

    def block_names(self, block_ids: list[str]) -> list[str]:
        if not self.include_block_effect:
            return []
        return [f"{BLOCK_PREFIX}{bid}]" for bid in block_ids]

    def full_matrix(self, x_treat: np.ndarray, blocks: list[str],
                    block_ids: list[str]) -> np.ndarray:
        """拼接处理列与块 one-hot 列。"""
        if not self.include_block_effect:
            return x_treat
        if len(blocks) != x_treat.shape[0]:
            raise ConfigError("块标签行数与设计矩阵不一致")
        index = {bid: j for j, bid in enumerate(block_ids)}
        z = np.zeros((len(blocks), len(block_ids)), dtype=float)
        for i, bid in enumerate(blocks):
            if bid not in index:
                raise ConfigError(f"运行引用了未声明的块 {bid!r}")
            z[i, index[bid]] = 1.0
        return np.hstack([x_treat, z])

    def full_column_names(self, block_ids: list[str]) -> list[str]:
        return list(self.treatment_names) + self.block_names(block_ids)


def _base_value(key: tuple[str, Any], base_cols: dict[str, float],
                numeric: dict[str, float]) -> float:
    """交互取基础列值：数值键取标准化值，类别虚拟键取 0/1 列。"""
    kind, name = key
    if kind == "n":
        return numeric[name]
    return base_cols[name]


def _interaction_specs(f1: Factor, f2: Factor) -> list[tuple[str, tuple, tuple]]:
    """生成交互项的列名与两侧基础列键。"""
    specs: list[tuple[str, tuple, tuple]] = []
    keys1 = _base_keys(f1)
    keys2 = _base_keys(f2)
    for (label1, key1) in keys1:
        for (label2, key2) in keys2:
            specs.append((f"{label1}*{label2}", key1, key2))
    return specs


def _base_keys(factor: Factor) -> list[tuple[str, tuple[str, Any]]]:
    if factor.is_numeric:
        return [(factor.name, ("n", factor.name))]
    return [(col_name, ("c", col_name)) for col_name, _ in factor.dummy_columns()]


def _parse_term(raw: Any, factor_set: FactorSet) -> Term:
    if not isinstance(raw, dict):
        raise ConfigError(f"模型项必须是对象，得到 {type(raw).__name__}")
    if raw.get("kind") == "intercept":
        extra = set(raw) - {"kind"}
        if extra:
            raise ConfigError(f"intercept 项含未知键: {sorted(extra)}")
        return Term(kind="intercept", raw=dict(raw))

    if "interaction" in raw:
        extra = set(raw) - {"interaction"}
        if extra:
            raise ConfigError(f"interaction 项含未知键: {sorted(extra)}")
        pair = raw["interaction"]
        if not isinstance(pair, list) or len(pair) != 2:
            raise ConfigError("interaction 必须是恰好两个因素名的列表")
        if len(set(pair)) != 2 or not all(isinstance(n, str) for n in pair):
            raise ConfigError("interaction 必须是两个不同的因素名")
        for name in pair:
            factor_set.get(name)  # 未知因素直接抛错
        return Term(kind="interaction", factors=tuple(pair), raw=dict(raw))

    if "factor" in raw:
        extra = set(raw) - {"factor", "power"}
        if extra:
            raise ConfigError(f"因素项含未知键: {sorted(extra)}")
        name = raw["factor"]
        if not isinstance(name, str):
            raise ConfigError("模型项 'factor' 必须是字符串")
        factor = factor_set.get(name)
        power = raw.get("power")
        if power is not None:
            if not isinstance(power, int) or isinstance(power, bool) or power < 1:
                raise ConfigError(f"因素 {name!r} 的 power 必须是 >=1 的整数")
            if not factor.is_numeric:
                raise ConfigError(f"类别因素 {name!r} 不能声明 power={power}")
        elif factor.is_numeric:
            power = 1  # 数值因素省略 power 即一次项；类别因素保持 None
        return Term(kind="factor", factor=name, power=power, raw=dict(raw))

    raise ConfigError(f"无法识别的模型项: {raw!r}（仅支持 intercept/factor/interaction）")
