"""最少条件调整的枚举与真实 SQL 验证（逐目标）。

一次操作 = 给某个已声明参数换一个备选值，或删除一个已声明原子谓词
（把该 AST 位置替换为 TRUE）。成本均为 1。只允许在 allowed_changes.json
声明的范围内调整。

对每个目标订单：按成本 1,2,3... 枚举组合，用真实 SQL 重跑；
保留能让该目标可见的组合；某个成本层一旦出现可行方案，即停止并
返回该层全部并列最小方案。若穷尽范围内所有组合仍无可行方案，
报告该目标无可行方案（通常因 INNER 关联无匹配，WHERE 调整无法补回）。
"""

import itertools
from dataclasses import dataclass, field

from . import ast_nodes as ast
from .errors import PlanError
from .rewrite import rewrite_remove_atoms
from .runner import diff_multisets, execute_rows


@dataclass
class Operation:
    kind: str                # change_parameter / remove_predicate
    cost: int = 1
    parameter: str = ""
    old_value: object = None
    new_value: object = None
    predicate_id: str = ""
    atom_index: int = -1
    sql: str = ""

    def key(self) -> tuple:
        if self.kind == "change_parameter":
            return ("param", self.parameter, repr(self.new_value))
        return ("remove", self.predicate_id)

    def describe(self) -> dict:
        if self.kind == "change_parameter":
            return {"kind": self.kind, "cost": self.cost,
                    "parameter": self.parameter,
                    "from": self.old_value, "to": self.new_value}
        return {"kind": self.kind, "cost": self.cost,
                "predicate_id": self.predicate_id,
                "atom_index": self.atom_index, "removed_sql": self.sql,
                "replacement": "TRUE"}


@dataclass
class Solution:
    operations: list
    cost: int
    params: dict
    removed_ids: tuple
    sql: str
    target: str = ""
    visible: bool = False
    revalidation: dict = field(default_factory=dict)

    def signature(self) -> tuple:
        return tuple(sorted(op.key() for op in self.operations))


@dataclass
class TargetSearch:
    target: str
    feasible: bool
    min_cost: int
    solutions: list = field(default_factory=list)  # 全部并列最小可行方案
    exhausted: bool = True
    reason: str = ""


def resolve_removable(allowed, atoms):
    """把 allowed_changes 声明的 sql 映射到 WHERE 中的具体原子节点。"""
    resolved = []
    used = set()
    for decl in allowed.removable_predicates:
        matches = [a for a in atoms if _normalize(a.sql_text) == _normalize(decl.sql)]
        if not matches:
            raise PlanError(
                f"allowed_changes 声明的可删谓词 {decl.id!r} ({decl.sql!r}) "
                "没有在 WHERE 中找到匹配的原子谓词"
            )
        if len(matches) > 1:
            raise PlanError(
                f"可删谓词 {decl.id!r} 在 WHERE 中匹配到多个位置，无法唯一定位"
            )
        atom = matches[0]
        if atom.atom_index in used:
            raise PlanError(f"多个可删谓词声明指向同一原子谓词: {decl.id!r}")
        used.add(atom.atom_index)
        decl.atom_index = atom.atom_index
        decl.atom_node_id = atom.node_id
        decl.atom_sql = atom.sql_text
        resolved.append((decl, atom))
    return resolved


def _normalize(sql: str) -> str:
    return " ".join(sql.split())


def _param_options(ctx, allowed):
    base = dict(ctx.plan_params)
    options = {}
    for name, alts in allowed.parameter_alternatives.items():
        if name not in base:
            raise PlanError(
                f"allowed_changes 为参数 {name!r} 声明了备选值，但该参数未被查询绑定"
            )
        opts = [(base[name], False)]
        seen = {repr(base[name])}
        for v in alts:
            if repr(v) not in seen:
                opts.append((v, True))
                seen.add(repr(v))
        options[name] = opts
    for p in ast.collect_params(ctx.select.where):
        options.setdefault(p.name, [(base[p.name], False)])
    return options


def build_single_ops(ctx, allowed, resolved):
    base = dict(ctx.plan_params)
    ops = []
    for name, opts in sorted(_param_options(ctx, allowed).items()):
        for value, changed in opts:
            if changed:
                ops.append(Operation(kind="change_parameter", parameter=name,
                                     old_value=base[name], new_value=value))
    for decl, atom in resolved:
        ops.append(Operation(kind="remove_predicate", predicate_id=decl.id,
                             atom_index=atom.atom_index, sql=atom.sql_text))
    return ops


def search_target(ctx, allowed, target, baseline_rows, max_cost=None,
                  diagnosis=None):
    """逐目标搜索全部并列最小可行方案，并用真实 SQL 验证。"""
    atoms = ctx.atoms
    resolved = resolve_removable(allowed, atoms)
    single_ops = build_single_ops(ctx, allowed, resolved)
    base_params = dict(ctx.plan_params)
    columns = [item.alias or item.expression.name for item in ctx.select.items]
    id_index = identity_index(ctx)

    def target_visible(rows):
        if id_index is None:
            return False
        col = columns[id_index]
        return any(str(r.get(col)) == str(target) for r in rows)

    limit = max_cost if max_cost is not None else len(single_ops)
    exhausted = True
    for cost in range(0, limit + 1):
        found = []
        for combo in _combos(single_ops, cost):
            params = dict(base_params)
            removed_atoms = []
            for op in combo:
                if op.kind == "change_parameter":
                    params[op.parameter] = op.new_value
                else:
                    removed_atoms.append(ctx.select.atom_by_index[op.atom_index])
            sql = rewrite_remove_atoms(ctx.select.original_sql, removed_atoms)
            rows = execute_rows(ctx.db, sql, ctx.bindings(params))
            if not target_visible(rows):
                continue
            diff = diff_multisets(baseline_rows, rows, columns)
            sol = Solution(
                operations=list(combo), cost=cost, params=params,
                removed_ids=tuple(sorted(a.atom_index for a in removed_atoms)),
                sql=sql, target=target, visible=True,
                revalidation={
                    "row_count": len(rows),
                    "target_visible": {target: True},
                    "impact": {
                        "columns": columns,
                        "added": diff["added"],
                        "removed": diff["removed"],
                        "added_row_count": diff["added_row_count"],
                        "removed_row_count": diff["removed_row_count"],
                        "baseline_row_count": diff["old_row_count"],
                        "solution_row_count": diff["new_row_count"],
                    },
                })
            found.append(sol)
        if found:
            # 同一层内的并列方案，按操作签名去重并稳定排序。
            uniq = {}
            for s in found:
                uniq.setdefault(s.signature(), s)
            ordered = [uniq[k] for k in sorted(uniq)]
            return TargetSearch(target=target, feasible=True, min_cost=cost,
                                solutions=ordered, exhausted=True)
    return TargetSearch(target=target, feasible=False, min_cost=-1,
                        solutions=[], exhausted=exhausted,
                        reason=_no_solution_reason(ctx, diagnosis))


def _no_solution_reason(ctx, diagnosis) -> str:
    base = "在声明范围内穷尽所有调整组合仍无法使该目标可见。"
    if not diagnosis:
        return base
    cat = diagnosis.get("category")
    if cat == "base_row_missing":
        return ("目标在基础表中不存在，放宽 WHERE 或改参数都无法补回；"
                "不发明新表行，故无可行方案。")
    if cat == "inner_join_no_match":
        return ("目标在 INNER 关联阶段即无匹配（WHERE 尚未参与），"
                "仅调整 WHERE 或参数无法补回缺失关联；不发明新表行或新连接，"
                "故无可行方案。")
    return base


def _combos(ops, k):
    if k == 0:
        yield ()
        return
    for combo in itertools.combinations(ops, k):
        params = [o.parameter for o in combo if o.kind == "change_parameter"]
        if len(params) != len(set(params)):
            continue
        preds = [o.predicate_id for o in combo if o.kind == "remove_predicate"]
        if len(preds) != len(set(preds)):
            continue
        yield combo


def identity_index(ctx) -> int:
    name = ctx.identity_column
    if not name:
        return None
    for i, item in enumerate(ctx.select.items):
        if (item.alias or item.expression.name) == name:
            return i
    return None
