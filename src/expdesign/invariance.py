"""等价编码下的可估计性不变性验证。

对同一组运行分别用两套等价参数化构造设计矩阵：

* 声明编码：数值标准化 ``(x-center)/scale`` + 类别 treatment 虚拟编码；
* 替代编码：数值改用中心原始单位（``x-center``，不除以 scale）+ 类别
  改用 sum（effect）编码 ``{-1, +1}``，参考水平取 -1。

两套矩阵的每一列都能由另一套的列经可逆线性变换得到（数值列差常数倍；
虚拟编码与 effect 编码张成同一子空间；幂/交互随基础列同步变换），
因此列空间、秩、零空间与"哪些处理效应可估计"的实质结论必须完全一致。
若不一致，说明实现把编码细节与可估计性错误地耦合了。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .factors import CandidateSet, Factor, FactorSet
from .model import Model, Term
from .plan import matrix_rank


@dataclass
class InvarianceResult:
    declared_treatment_rank: int
    alternative_treatment_rank: int
    declared_full_rank: int
    alternative_full_rank: int
    n_treatment_columns: int
    n_full_columns: int
    same_column_space: bool
    estimability_consistent: bool
    max_subspace_distance: float


def _base_numeric(factor: Factor, value: Any) -> float:
    """替代数值编码：中心原始单位（不除以 scale）。"""
    return float(value) - float(factor.center)


def _base_categorical(factor: Factor, value: Any) -> np.ndarray:
    """替代类别编码：参考水平 -1 的 effect 编码，与声明虚拟列逐列对应。"""
    out = []
    for _col_name, level in factor.dummy_columns():
        if value == level:
            out.append(1.0)
        elif value == factor.reference:
            out.append(-1.0)
        else:
            out.append(0.0)
    return np.array(out, dtype=float)


def _alternative_row(
    factor_set: FactorSet, model: Model, combo: tuple[Any, ...]
) -> np.ndarray:
    """严格按模型项的展开顺序（与 Model.treatment_names 同序）构造替代行。"""
    values: list[float] = []
    numeric_cache: dict[str, float] = {}
    dummy_cache: dict[str, np.ndarray] = {}
    for factor, value in zip(factor_set.factors, combo):
        if factor.is_numeric:
            numeric_cache[factor.name] = _base_numeric(factor, value)
        else:
            dummy_cache[factor.name] = _base_categorical(factor, value)

    for term in model.terms:
        values.extend(_term_columns(term, factor_set, numeric_cache, dummy_cache))
    return np.array(values, dtype=float)


def _term_columns(
    term: Term, factor_set: FactorSet,
    numeric_cache: dict[str, float], dummy_cache: dict[str, np.ndarray],
) -> list[float]:
    if term.kind == "intercept":
        return [1.0]
    if term.kind == "factor":
        factor = factor_set.get(term.factor)
        if factor.is_numeric:
            return [numeric_cache[factor.name] ** (term.power or 1)]
        return list(dummy_cache[factor.name])
    # interaction：按两侧基础列做外积，与 model._interaction_specs 同序
    f1, f2 = (factor_set.get(name) for name in term.factors)
    left = _base_vector(f1, numeric_cache, dummy_cache)
    right = _base_vector(f2, numeric_cache, dummy_cache)
    cols: list[float] = []
    for v1 in left:
        for v2 in right:
            cols.append(float(v1 * v2))
    return cols


def _base_vector(
    factor: Factor, numeric_cache: dict[str, float],
    dummy_cache: dict[str, np.ndarray],
) -> np.ndarray:
    if factor.is_numeric:
        return np.array([numeric_cache[factor.name]], dtype=float)
    return dummy_cache[factor.name]


def alternative_treatment_matrix(
    factor_set: FactorSet, model: Model, combos: list[tuple[Any, ...]]
) -> np.ndarray:
    return np.vstack([
        _alternative_row(factor_set, model, combo) for combo in combos
    ])


def check_coding_invariance(
    factor_set: FactorSet, model: Model, candidates: CandidateSet,
    design_combos: list[tuple[Any, ...]],
    blocks: list[str], block_ids: list[str],
) -> InvarianceResult:
    """比较两套编码在同一设计上的秩与列空间。"""
    x_declared = model.treatment_matrix(design_combos)
    x_alt = alternative_treatment_matrix(factor_set, model, design_combos)
    if x_alt.shape != x_declared.shape:
        raise AssertionError(
            f"替代编码列数 {x_alt.shape[1]} 与声明编码 {x_declared.shape[1]} 不一致"
        )

    r_decl = matrix_rank(x_declared)
    r_alt = matrix_rank(x_alt)

    if model.include_block_effect:
        full_decl = model.full_matrix(x_declared, blocks, block_ids)
        # 块 one-hot 不涉及因素编码，两套矩阵的块列完全相同。
        full_alt = model.full_matrix(x_alt, blocks, block_ids)
        r_z = matrix_rank(full_decl[:, model.n_treatment:])
        n_target = model.n_treatment - 1  # 截距与块列共线，剔除
        expected_full = r_z + n_target
    else:
        full_decl = x_declared
        full_alt = x_alt
        expected_full = model.n_treatment
    fr_decl = matrix_rank(full_decl)
    fr_alt = matrix_rank(full_alt)

    space = _same_column_space(x_declared, x_alt)
    estimable = (
        r_decl == model.n_treatment
        and r_alt == model.n_treatment
        and fr_decl == expected_full
        and fr_alt == expected_full
    )
    return InvarianceResult(
        declared_treatment_rank=r_decl,
        alternative_treatment_rank=r_alt,
        declared_full_rank=fr_decl,
        alternative_full_rank=fr_alt,
        n_treatment_columns=model.n_treatment,
        n_full_columns=full_decl.shape[1],
        same_column_space=space["same"],
        estimability_consistent=bool(
            r_decl == r_alt and fr_decl == fr_alt and estimable
        ),
        max_subspace_distance=space["distance"],
    )


def _same_column_space(a: np.ndarray, b: np.ndarray,
                       tol: float = 1e-8) -> dict[str, Any]:
    """用正交投影差的谱范数量化两个列空间是否相同。"""
    def projector(x: np.ndarray) -> np.ndarray:
        q, _ = np.linalg.qr(x, mode="reduced")
        return q @ q.T

    distance = float(np.linalg.norm(projector(a) - projector(b), ord=2))
    return {"same": distance <= tol, "distance": distance}
