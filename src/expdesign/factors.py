"""因素声明、水平编码与合法候选集。

编码身份（列含义）在此固定并对外公开：

* 数值因素 ``f``（水平经 ``(x - center) / scale`` 标准化到编码值）：
    一次项列名 ``f``，幂次 ``p`` 的多项式列名 ``f^p``；
* 类别因素 ``f``：参考水平由 ``reference`` 给出，采用 treatment（虚拟）编码，
    参考水平对应 0，其余每个非参考水平一列，列名 ``f[level]``。

候选集是全部声明水平的笛卡尔积，再剔除禁用组合。候选用 *组合元组*
（各因素按声明顺序的原始水平值）唯一标识，重复运行只是对同一候选的重复引用，
不会产生新的设计方向。
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import ConfigError

NUMERIC = "numeric"
CATEGORICAL = "categorical"


@dataclass(frozen=True)
class Factor:
    """单个因素的不可变声明。"""

    name: str
    kind: str
    levels: tuple[Any, ...]
    center: float | None = None
    scale: float | None = None
    reference: Any = None

    @property
    def is_numeric(self) -> bool:
        return self.kind == NUMERIC

    def coded_value(self, raw: Any) -> float:
        """数值因素：按声明的中心与尺度标准化。"""
        if not self.is_numeric:
            raise ConfigError(f"因素 {self.name!r} 不是数值因素，不能标准化")
        return (float(raw) - float(self.center)) / float(self.scale)

    def dummy_columns(self) -> list[tuple[str, Any]]:
        """类别因素的非参考列：``(列名, 水平)`` 列表，列顺序固定。"""
        if self.is_numeric:
            raise ConfigError(f"因素 {self.name!r} 不是类别因素")
        return [(f"{self.name}[{lv}]", lv) for lv in self.levels if lv != self.reference]


class FactorSet:
    """因素集合：负责解析、校验与逐行编码。"""

    def __init__(self, factors: list[Factor]):
        if not factors:
            raise ConfigError("至少需要声明一个因素")
        names = [f.name for f in factors]
        if len(set(names)) != len(names):
            raise ConfigError(f"因素名重复: {names}")
        self.factors: list[Factor] = list(factors)
        self.names: list[str] = names
        self._index = {n: i for i, n in enumerate(self.names)}

    # ------------------------------------------------------------------ #
    # 构造与解析
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactorSet":
        if not isinstance(payload, dict) or "factors" not in payload:
            raise ConfigError("factors.json 顶层必须是含 'factors' 的对象")
        factors: list[Factor] = []
        for raw in payload["factors"]:
            factors.append(_parse_factor(raw))
        return cls(factors)

    @classmethod
    def from_json_file(cls, path: str) -> "FactorSet":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def get(self, name: str) -> Factor:
        if name not in self._index:
            raise ConfigError(f"未知因素 {name!r}，已声明因素: {self.names}")
        return self.factors[self._index[name]]

    def index_of(self, name: str) -> int:
        return self._index[name]

    # ------------------------------------------------------------------ #
    # 编码
    # ------------------------------------------------------------------ #
    def coded_columns(self, combo: tuple[Any, ...]) -> dict[str, float]:
        """把一个原始水平组合展开成 *基础编码列*（一次/虚拟列，不含幂与交互）。

        键即列身份，数值一次项与类别虚拟列同名规则见模块 docstring。
        另为数值因素保留原始标准化值，供幂次/交互列复用。
        """
        cols: dict[str, float] = {}
        for factor, raw in zip(self.factors, combo):
            if factor.is_numeric:
                cols[factor.name] = factor.coded_value(raw)
            else:
                for col_name, level in factor.dummy_columns():
                    cols[col_name] = 1.0 if raw == level else 0.0
                # 参考水平未声明在列里：所有虚拟列自然为 0。
        return cols

    def numeric_coded(self, combo: tuple[Any, ...]) -> dict[str, float]:
        """仅返回数值因素的标准化值。"""
        out = {}
        for factor, raw in zip(self.factors, combo):
            if factor.is_numeric:
                out[factor.name] = factor.coded_value(raw)
        return out

    def all_combinations(self) -> list[tuple[Any, ...]]:
        """声明水平的完整笛卡尔积，顺序按因素声明顺序确定。"""
        return list(itertools.product(*[f.levels for f in self.factors]))


def _parse_factor(raw: Any) -> Factor:
    if not isinstance(raw, dict):
        raise ConfigError(f"因素声明必须是对象，得到 {type(raw).__name__}")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError("因素缺少非空字符串 'name'")
    kind = raw.get("type")
    levels = raw.get("levels")
    if not isinstance(levels, list) or len(levels) < 2:
        raise ConfigError(f"因素 {name!r} 的 levels 必须是长度 >= 2 的列表")
    if len(set(map(_hashable, levels))) != len(levels):
        raise ConfigError(f"因素 {name!r} 的水平存在重复: {levels}")
    levels_t = tuple(levels)

    if kind == NUMERIC:
        center = raw.get("center")
        scale = raw.get("scale")
        if center is None or scale is None:
            raise ConfigError(f"数值因素 {name!r} 必须声明 center 与 scale")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in levels):
            raise ConfigError(f"数值因素 {name!r} 的水平必须全部是数字")
        if float(scale) == 0.0:
            raise ConfigError(f"数值因素 {name!r} 的 scale 不能为 0")
        # center 是编码基准，允许落在离散水平之间（如两水平因素取中点），
        # 只要求是有限数值，不强制属于 levels。
        if not np.isfinite(float(center)):
            raise ConfigError(f"数值因素 {name!r} 的 center 必须是有限数值")
        return Factor(name, NUMERIC, levels_t, center=float(center), scale=float(scale))

    if kind == CATEGORICAL:
        reference = raw.get("reference")
        if reference is None:
            raise ConfigError(f"类别因素 {name!r} 必须声明 reference 参考水平")
        if reference not in levels_t:
            raise ConfigError(f"类别因素 {name!r} 的 reference {reference!r} 不在 levels 中")
        return Factor(name, CATEGORICAL, levels_t, reference=reference)

    raise ConfigError(f"因素 {name!r} 的 type 未知: {kind!r}（仅支持 numeric/categorical）")


def _hashable(value: Any) -> Any:
    return value if isinstance(value, (str, int, float, bool, tuple)) else json.dumps(
        value, sort_keys=True, default=str
    )


@dataclass
class CandidateSet:
    """合法候选集及其与原始组合的一一对应。

    ``combos[i]`` 是第 ``i`` 个候选的原始水平元组；候选矩阵的行号即候选 id，
    设计内部只存候选 id，重复 = 同一 id 出现多次。
    """

    factor_set: FactorSet
    combos: list[tuple[Any, ...]]
    forbidden: list[tuple[Any, ...]] = field(default_factory=list)
    forbidden_rules: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_candidates(self) -> int:
        return len(self.combos)

    def index_of_combo(self, combo: tuple[Any, ...]) -> int:
        try:
            return self._combo_index[combo]
        except KeyError as exc:  # pragma: no cover - 仅内部调用保护
            raise ConfigError(f"组合 {combo} 不在合法候选集中") from exc

    def __post_init__(self) -> None:
        self._combo_index = {combo: i for i, combo in enumerate(self.combos)}

    def combo_matches(self, combo: tuple[Any, ...], partial: dict[str, Any]) -> bool:
        """组合是否命中某条（部分指定的）规则。"""
        for fname, value in partial.items():
            if combo[self.factor_set.index_of(fname)] != value:
                return False
        return True


def build_candidate_set(
    factor_set: FactorSet, forbidden_rules: list[dict[str, Any]] | None = None
) -> CandidateSet:
    """笛卡尔积后逐条剔除禁用组合；被剔除的组合单独留档，永不重新引入。"""
    rules = forbidden_rules or []
    for rule in rules:
        if not isinstance(rule, dict) or not rule:
            raise ConfigError(f"禁用组合必须是非空对象，得到 {rule!r}")
        unknown = set(rule) - set(factor_set.names)
        if unknown:
            raise ConfigError(f"禁用组合引用了未知因素: {sorted(unknown)}")

    legal: list[tuple[Any, ...]] = []
    forbidden: list[tuple[Any, ...]] = []
    for combo in factor_set.all_combinations():
        hit = any(_matches(factor_set, combo, rule) for rule in rules)
        (forbidden if hit else legal).append(combo)
    return CandidateSet(
        factor_set=factor_set,
        combos=legal,
        forbidden=forbidden,
        forbidden_rules=list(rules),
    )


def _matches(factor_set: FactorSet, combo: tuple[Any, ...], rule: dict[str, Any]) -> bool:
    for fname, value in rule.items():
        if combo[factor_set.index_of(fname)] != value:
            return False
    return True
