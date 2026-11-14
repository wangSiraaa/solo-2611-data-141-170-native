"""候选关联行的重建、原值证据与三值逻辑排除路径。"""

from dataclasses import dataclass, field

from . import ast_nodes as ast
from .db import Database, TableInfo
from .errors import PlanError, UnsupportedSyntaxError
from .semantics import (affinity_of_type, sql_compare, tri_and, tri_label,
                        tri_or)


@dataclass
class RowSnapshot:
    """某张物理表的一行，附带可回溯的原值。"""
    table: TableInfo
    values: dict
    alias: str

    def evidence(self) -> dict:
        return {
            "table": self.table.name,
            "alias": self.alias,
            "primary_key": ({c: self.values.get(c) for c in self.table.pk_columns}
                            if self.table.pk_columns else None),
            "values": {k: self.values.get(k) for k in self.table.column_names},
        }


@dataclass
class CandidateRow:
    """一条候选关联行（join tuple）及其 WHERE 求值结果。"""
    sources: dict            # alias -> RowSnapshot 或 None（LEFT 补出的 NULL 扩展行）
    synthetic_aliases: tuple  # 哪些别名是 LEFT 补 NULL 产生的
    atom_values: dict        # atom_index -> True/False/None
    where_value: object      # True/False/None
    order: tuple             # 稳定排序键（按连接顺序的物理行序），不用于身份
    where_trace: dict = field(default_factory=dict)
    atom_details: dict = field(default_factory=dict)  # atom_index -> 证据
    identity: dict = field(default_factory=dict)

    @property
    def visible(self) -> bool:
        return self.where_value is True


@dataclass
class JoinStep:
    """连接链上的一步，用于说明基础行/INNER 无匹配的排除。"""
    alias: str
    table: TableInfo
    join_type: str            # BASE / INNER / LEFT
    on_text: str = ""
    target_key_value: object = None
    matched_rows: list = field(default_factory=list)
    synthetic: bool = False
    step_index: int = 0


class AnalysisContext:
    """预处理一次查询，供目标诊断与方案验证共享。"""

    def __init__(self, db: Database, select: ast.Select, plan_params: dict,
                 target_pk: str = "", identity_column: str = ""):
        self.db = db
        self.select = select
        self.plan_params = plan_params
        self.target_pk = target_pk
        self.identity_column = identity_column
        self.alias_map: dict[str, TableInfo] = {}
        self.alias_order: list[str] = []
        self._build_alias_map()
        self.atoms = [select.atom_by_index[i] for i in sorted(select.atom_by_index)]
        self.target_table = self.db.table(select.base_table.name)
        self.identity_col = None

    def _build_alias_map(self):
        sel = self.select
        base_alias = sel.base_table.alias or sel.base_table.name
        self.alias_map[base_alias] = self.db.table(sel.base_table.name)
        self.alias_order.append(base_alias)
        for j in sel.joins:
            alias = j.right.alias or j.right.name
            self.alias_map[alias] = self.db.table(j.right.name)
            self.alias_order.append(alias)

    # ---- 参数绑定 ----
    def bindings(self, params: dict) -> dict:
        """把 :name 参数映射到实际绑定值；@name 形式也接受。"""
        out = {}
        for p in ast.collect_params(self.select.where):
            if p.name not in params:
                raise PlanError(f"参数 {p.raw} 在 plan.parameters 中未提供")
            out[p.name] = params[p.name]
            out["@" + p.name] = params[p.name]
        return out

    # ---- 列求值 ----
    def operand_value(self, node, sources: dict, params: dict):
        """返回 (值, 亲和性, 证据描述)。sources: alias -> RowSnapshot|None。"""
        if node.kind == "literal":
            # SQLite 中字面量本身无亲和性；由另一侧的列亲和性决定转换。
            return node.value, "NONE", {"kind": "literal", "raw": node.raw,
                                        "value": node.value}
        if node.kind == "param":
            if node.name not in params:
                raise PlanError(f"参数 {node.raw} 无绑定值")
            val = params[node.name]
            # 绑定参数同样无亲和性。
            return val, "NONE", {"kind": "parameter", "raw": node.raw,
                                 "name": node.name, "value": val}
        if node.kind == "column":
            alias, info, col = self.db.resolve(node, self.alias_map, "WHERE")
            snap = sources.get(alias)
            if snap is None:
                # LEFT JOIN 补出的 NULL 扩展行：列值为 NULL，但亲和性仍来自表定义。
                return None, affinity_of_type(self._col_type(info, col)), {
                    "kind": "null_extended_column",
                    "alias": alias,
                    "table": info.name,
                    "column": col,
                    "value": None,
                }
            val = snap.values.get(col)
            return val, affinity_of_type(self._col_type(info, col)), {
                "kind": "column",
                "alias": alias,
                "table": info.name,
                "column": col,
                "value": val,
                "row_primary_key": ({c: snap.values.get(c) for c in info.pk_columns}
                                    if info.pk_columns else None),
            }
        raise UnsupportedSyntaxError(f"不支持的表达式节点 {node.kind}")

    @staticmethod
    def _col_type(info: TableInfo, col: str) -> str:
        for c in info.columns:
            if c.name == col:
                return c.type
        return ""

    # ---- WHERE 求值（三值）----
    def eval_atom(self, atom, sources, params):
        if atom.is_constant:
            # 改写产生的 TRUE/FALSE 常量，直接按字面量求值。
            res = bool(atom.left.value)
            return res, {
                "atom_index": atom.atom_index,
                "sql": atom.sql_text,
                "op": "constant",
                "left": {"kind": "literal", "raw": atom.left.raw,
                         "value": atom.left.value},
                "right": None,
                "result": tri_label(res),
            }
        lv, la, ldesc = self.operand_value(atom.left, sources, params)
        rv, ra, rdesc = self.operand_value(atom.right, sources, params)
        res = sql_compare(lv, la, rv, ra, atom.op)
        return res, {
            "atom_index": atom.atom_index,
            "sql": atom.sql_text,
            "op": atom.op,
            "left": ldesc,
            "right": rdesc,
            "result": tri_label(res),
        }

    def eval_expr(self, node, sources, params, atom_trace=None, removed_ids=frozenset()):
        if node.kind == "comparison":
            if node.is_constant:
                # 常量原子（改写出的 TRUE）不进入可诊断原子轨迹。
                res = bool(node.left.value)
                return res, {"kind": "constant", "sql": node.sql_text,
                              "result": tri_label(res)}
            if node.node_id in removed_ids:
                # 删除谓词等价于把该 AST 位置替换为 TRUE，其余括号结构不变。
                res = True
                desc = {"atom_index": node.atom_index, "sql": node.sql_text,
                        "removed": True, "result": "TRUE"}
            else:
                res, desc = self.eval_atom(node, sources, params)
            if atom_trace is not None:
                atom_trace[node.atom_index] = (res, desc)
            return res, {"kind": "comparison", "sql": node.sql_text,
                         "atom_index": node.atom_index,
                         "removed": node.node_id in removed_ids,
                         "result": tri_label(res)}
        if node.kind == "boolop":
            lres, ldesc = self.eval_expr(node.left, sources, params, atom_trace,
                                         removed_ids)
            # 三值逻辑：AND/OR 正常可短路，但解释时两侧结果都要给出，
            # 因此不短路求值。这样 UNKNOWN 与短路分支都能区分。
            rres, rdesc = self.eval_expr(node.right, sources, params, atom_trace,
                                         removed_ids)
            if node.operator == "AND":
                res = tri_and(lres, rres)
            else:
                res = tri_or(lres, rres)
            return res, {"kind": "boolop", "operator": node.operator,
                         "result": tri_label(res),
                         "left": ldesc, "right": rdesc}
        raise UnsupportedSyntaxError(f"布尔结构中出现不支持的节点 {node.kind}")

    # ---- 候选关联行枚举 ----
    def enumerate_candidates(self, target_values: dict = None, params=None,
                             removed_ids=frozenset()):
        """重建候选关联行。

        target_values 非空时只沿目标订单这一条链展开（用于目标诊断）；
        为空时展开全部基础行。
        params 控制绑定值；removed_ids 为被视为 TRUE 的原子 node_id 集合。
        返回 (candidates, traces)。traces 按基础行分组，逐步记录连接匹配证据；
        LEFT 无匹配时产生 NULL 扩展行（synthetic_aliases 记录）。
        """
        tuples_, traces = self.enumerate_tuples(target_values)
        params = self.plan_params if params is None else params
        candidates = []
        for trace in traces:
            if not trace["alive"]:
                continue
            for srcs, synth, ordkey in trace["tuples"]:
                cand = self.evaluate_tuple(srcs, synth, ordkey, params, removed_ids)
                candidates.append(cand)
                trace["candidates"].append(cand)
        return candidates, traces

    def enumerate_tuples(self, target_values: dict = None):
        """枚举满足全部 ON 条件的连接元组，返回 [(sources, synth, order), ...]。"""
        sel = self.select
        base_alias = self.alias_order[0]
        base_table = self.alias_map[base_alias]
        if target_values is not None:
            base_rows = [RowSnapshot(base_table, target_values, base_alias)]
        else:
            base_rows = [RowSnapshot(base_table, r, base_alias)
                         for r in self.db.rows(base_table.name)]

        traces = []
        for order_idx, base in enumerate(base_rows):
            partial = [({base_alias: base}, tuple(), (order_idx,))]
            steps = [JoinStep(alias=base_alias, table=base_table, join_type="BASE",
                              step_index=0, matched_rows=[base])]
            alive = True
            for j_idx, join in enumerate(sel.joins):
                alias = join.right.alias or join.right.name
                info = self.alias_map[alias]
                step = JoinStep(alias=alias, table=info,
                                join_type=join.join_type, on_text=join.on_text,
                                step_index=j_idx + 1)
                next_partial = []
                step_matches = []
                for srcs, synth, ordkey in partial:
                    matched = self._match_join(join, srcs, params=self.plan_params,
                                               info=info)
                    if matched:
                        for r_idx, (row, _lv, _rv) in enumerate(matched):
                            ns = dict(srcs)
                            ns[alias] = RowSnapshot(info, row, alias)
                            next_partial.append((ns, synth, ordkey + (r_idx,)))
                            step_matches.append(RowSnapshot(info, row, alias))
                    elif join.join_type == "LEFT":
                        ns = dict(srcs)
                        ns[alias] = None
                        next_partial.append((ns, synth + (alias,), ordkey + (-1,)))
                        step.synthetic = True
                step.matched_rows = step_matches
                steps.append(step)
                if not next_partial:
                    alive = False
                    break
                partial = next_partial
            traces.append({"base": base.evidence(), "base_snapshot": base,
                           "steps": steps, "alive": alive,
                           "tuples": partial if alive else [],
                           "candidates": []})
        return [t for tr in traces for t in tr["tuples"]], traces

    def evaluate_tuple(self, sources, synthetic_aliases, order, params,
                       removed_ids=frozenset()):
        atom_trace = {}
        if self.select.where is None:
            wval, wdesc = True, {"kind": "no_where", "result": "TRUE"}
        else:
            wval, wdesc = self.eval_expr(self.select.where, sources, params,
                                         atom_trace, removed_ids)
        ov = {i: v[0] for i, v in atom_trace.items()}
        details = {i: v[1] for i, v in atom_trace.items()}
        cand = CandidateRow(
            sources=sources, synthetic_aliases=synthetic_aliases,
            atom_values=ov, where_value=wval, order=order,
            where_trace=wdesc, atom_details=details)
        cand.identity = self.project_identity(cand)
        return cand

    def _match_join(self, join, srcs, params, info):
        """在给定已连接元组上尝试连接右表，返回 [(row, left_val, right_val)]。

        ON 两侧都按完整元组求值，因此不依赖哪一侧引用新表。
        只有比较结果为 TRUE 才匹配；UNKNOWN/FALSE 都不匹配（真实 SQL 语义）。
        """
        alias = join.right.alias or join.right.name
        matched = []
        for row in self.db.rows(info.name):
            rs = dict(srcs)
            rs[alias] = RowSnapshot(info, row, alias)
            lv, la, _ = self.operand_value(join.left_col, rs, params)
            rv, ra, _ = self.operand_value(join.right_col, rs, params)
            if sql_compare(lv, la, rv, ra, "=") is True:
                matched.append((row, lv, rv))
        return matched

    # ---- 结果投影 ----
    def project_identity(self, cand: CandidateRow) -> dict:
        """按 SELECT 列表投影一行结果，形成可用作结果身份的值向量。"""
        columns = []
        values = []
        for idx, item in enumerate(self.select.items):
            alias, info, col = self.db.resolve(item.expression, self.alias_map, "SELECT")
            snap = cand.sources.get(alias)
            value = None if snap is None else snap.values.get(col)
            name = item.alias or item.expression.name
            columns.append(name)
            values.append(value)
        return {"columns": columns, "values": tuple(values)}

    def identity_key(self, cand: CandidateRow) -> tuple:
        return tuple(cand.identity["values"])
