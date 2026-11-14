"""递归下降 SQL 解析器。

文法（超出该文法的任何输入都明确拒绝）：

    select := SELECT item (, item)*
              FROM table_ref
              (join_clause){0,2}
              [WHERE or_expr] [';']
    join_clause := (INNER JOIN | LEFT [OUTER] JOIN) table_ref ON col '=' col
    or_expr  := and_expr (OR and_expr)*
    and_expr := factor (AND factor)*
    factor   := '(' or_expr ')' | comparison
    comparison := operand cmp_op operand
    operand := col_ref | literal | named_param
"""

import dataclasses

from . import ast_nodes as ast
from .errors import UnsupportedSyntaxError
from .tokenizer import Tokenizer

MAX_JOINS = 2


class Parser:
    def __init__(self, sql: str):
        self.original_sql = sql
        self.tokens = Tokenizer(sql).tokenize()
        self.pos = 0
        self._seq = 0

    # ---- 基础工具 ----
    def _nid(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}{self._seq}"

    @property
    def cur(self):
        return self.tokens[self.pos]

    def _eat(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect_kw(self, word: str):
        tok = self.cur
        if tok.kind != "KEYWORD" or tok.upper != word:
            raise UnsupportedSyntaxError(
                f"期望关键字 {word}，实际得到 {tok.text!r}（位置 {tok.line}:{tok.col}）"
            )
        return self._eat()

    def _match_kw(self, word: str) -> bool:
        tok = self.cur
        if tok.kind == "KEYWORD" and tok.upper == word:
            self._eat()
            return True
        return False

    def _expect_op(self, op: str):
        tok = self.cur
        if tok.kind != "OP" or tok.text != op:
            raise UnsupportedSyntaxError(
                f"期望运算符 {op!r}，实际得到 {tok.text!r}（位置 {tok.line}:{tok.col}）"
            )
        return self._eat()

    # ---- 入口 ----
    def parse(self) -> ast.Select:
        self._expect_kw("SELECT")
        items = self._parse_select_items()
        self._expect_kw("FROM")
        base = self._parse_table_ref()
        joins = []
        while self.cur.kind == "KEYWORD" and self.cur.upper in (
                "INNER", "LEFT", "JOIN"):
            joins.append(self._parse_join())
            if len(joins) > MAX_JOINS:
                raise UnsupportedSyntaxError(
                    f"连接次数超过上限：最多 {MAX_JOINS} 次 INNER/LEFT 连接"
                )
        where = None
        where_start = where_end = -1
        if self.cur.kind == "KEYWORD" and self.cur.upper == "WHERE":
            self._eat()
            where_start = self.cur.pos
            where = self._parse_or()
            where_end = self.tokens[self.pos - 1].end
        if self.cur.kind == "OP" and self.cur.text == ";":
            self._eat()
        if self.cur.kind != "EOF":
            tok = self.cur
            raise UnsupportedSyntaxError(
                f"WHERE 之后存在不支持的输入 {tok.text!r}（位置 {tok.line}:{tok.col}）；"
                "本工具不接受 GROUP BY/ORDER BY/LIMIT/子查询等"
            )
        select = ast.Select(
            items=tuple(items),
            base_table=base,
            joins=tuple(joins),
            where=where,
            original_sql=self.original_sql,
            where_start=where_start,
            where_end=where_end,
        )
        select = self._index_atoms(select)
        self._validate(select)
        return select

    # ---- SELECT 列表 ----
    def _parse_select_items(self):
        items = []
        while True:
            start = self.cur.pos
            col = self._parse_column()
            alias = None
            if self._match_kw("AS"):
                tok = self._eat()
                if tok.kind != "IDENT":
                    raise UnsupportedSyntaxError("AS 后必须是列别名标识符")
                alias = tok.text
            end = self.tokens[self.pos - 1].end
            items.append(
                ast.SelectItem(expression=col, alias=alias,
                               raw_text=self.original_sql[start:end].strip())
            )
            if self.cur.kind == "OP" and self.cur.text == ",":
                self._eat()
                continue
            break
        return items

    # ---- 表与连接 ----
    def _parse_table_ref(self) -> ast.TableRef:
        tok = self._eat()
        if tok.kind != "IDENT":
            raise UnsupportedSyntaxError(
                f"表名必须是标识符，实际得到 {tok.text!r}（位置 {tok.line}:{tok.col}）"
            )
        name, start = tok.text, tok.pos
        alias = None
        if self._match_kw("AS"):
            at = self._eat()
            if at.kind != "IDENT":
                raise UnsupportedSyntaxError("AS 后必须是表别名标识符")
            alias = at.text
        elif self.cur.kind == "IDENT":
            alias = self._eat().text
        end = self.tokens[self.pos - 1].end
        return ast.TableRef(name=name, alias=alias, node_id=self._nid("t"),
                            start=start, end=end)

    def _parse_join(self) -> ast.Join:
        first = self._eat()
        join_type = "INNER"
        if first.upper == "JOIN":
            pass                       # 裸 JOIN 等价 INNER JOIN
        elif first.upper == "LEFT":
            join_type = "LEFT"
            self._match_kw("OUTER")
            self._expect_kw("JOIN")
        else:                          # INNER
            self._expect_kw("JOIN")
        right = self._parse_table_ref()
        self._expect_kw("ON")
        on_start = self.cur.pos
        left_col = self._parse_column()
        self._expect_op("=")
        right_col = self._parse_column()
        on_text = self.original_sql[on_start:self.cur.pos].strip()
        return ast.Join(join_type=join_type, right=right, left_col=left_col,
                        right_col=right_col, on_text=on_text, node_id=self._nid("j"))

    # ---- 布尔表达式 ----
    def _parse_or(self):
        node = self._parse_and()
        while self._match_kw("OR"):
            right = self._parse_and()
            node = ast.BoolOp("OR", node, right, self._nid("b"))
        return node

    def _parse_and(self):
        node = self._parse_factor()
        while self._match_kw("AND"):
            right = self._parse_factor()
            node = ast.BoolOp("AND", node, right, self._nid("b"))
        return node

    def _parse_factor(self):
        tok = self.cur
        if tok.kind == "KEYWORD" and tok.upper == "NOT":
            raise UnsupportedSyntaxError(
                f"不支持 NOT 运算（位置 {tok.line}:{tok.col}）；WHERE 仅允许比较与 AND/OR/括号"
            )
        if tok.kind == "OP" and tok.text == "(":
            self._eat()
            node = self._parse_or()
            close = self.cur
            if not (close.kind == "OP" and close.text == ")"):
                raise UnsupportedSyntaxError("括号未闭合")
            self._eat()
            return node
        if tok.kind == "KEYWORD" and tok.upper in ("TRUE", "FALSE"):
            # 布尔常量原子：主要出现在“删除谓词 => TRUE”的改写 SQL 中。
            self._eat()
            lit = ast.Literal(value=(tok.upper == "TRUE"), raw=tok.upper,
                              node_id=self._nid("l"), start=tok.pos, end=tok.end)
            return ast.Comparison(
                op="=", left=lit, right=lit, sql_text=tok.text,
                node_id=self._nid("t"), start=tok.pos, end=tok.end,
                is_constant=True)
        return self._parse_comparison()

    def _parse_comparison(self) -> ast.Comparison:
        left = self._parse_operand()
        op_tok = self.cur
        if not (op_tok.kind == "OP" and op_tok.text in ast.COMPARE_OPS):
            raise UnsupportedSyntaxError(
                f"WHERE 因子必须是二元比较，实际得到 {op_tok.text!r}"
                f"（位置 {op_tok.line}:{op_tok.col}）"
            )
        self._eat()
        op = "<>" if op_tok.text == "!=" else op_tok.text
        right = self._parse_operand()
        start = self._span_start(left)
        end = self._span_end(right)
        return ast.Comparison(
            op=op, left=left, right=right,
            sql_text=self.original_sql[start:end].strip(),
            node_id=self._nid("a"), start=start, end=end,
        )

    def _parse_operand(self):
        tok = self.cur
        if tok.kind == "IDENT":
            return self._parse_column()
        if tok.kind == "KEYWORD" and tok.upper in ("NULL", "TRUE", "FALSE"):
            self._eat()
            if tok.upper == "NULL":
                value, raw = None, "NULL"
            else:
                value, raw = (tok.upper == "TRUE"), tok.upper
            return ast.Literal(value=value, raw=raw, node_id=self._nid("l"),
                               start=tok.pos, end=self.tokens[self.pos - 1].end)
        if tok.kind == "NUMBER":
            self._eat()
            text = tok.text
            value = float(text) if "." in text else int(text)
            return ast.Literal(value=value, raw=text, node_id=self._nid("l"),
                               start=tok.pos, end=self.tokens[self.pos - 1].end)
        if tok.kind == "STRING":
            self._eat()
            return ast.Literal(value=tok.upper, raw=tok.text, node_id=self._nid("l"),
                               start=tok.pos, end=self.tokens[self.pos - 1].end)
        if tok.kind == "PARAM":
            self._eat()
            if tok.text == "?":
                raise UnsupportedSyntaxError(
                    "不支持位置参数 ?；请使用 :name 或 @name 命名绑定参数"
                )
            name = tok.text[1:]
            return ast.Param(raw=tok.text, name=name, node_id=self._nid("p"),
                             start=tok.pos, end=self.tokens[self.pos - 1].end)
        if tok.kind == "OP" and tok.text == "(":
            raise UnsupportedSyntaxError(
                "比较的操作数不允许括号表达式；只允许列、字面量或绑定参数"
            )
        raise UnsupportedSyntaxError(
            f"无法识别的比较操作数 {tok.text!r}（位置 {tok.line}:{tok.col}）"
        )

    def _parse_column(self) -> ast.ColumnRef:
        tok = self._eat()
        if tok.kind != "IDENT":
            raise UnsupportedSyntaxError(
                f"期望列名标识符，实际得到 {tok.text!r}（位置 {tok.line}:{tok.col}）"
            )
        table_alias, name = None, tok.text
        if self.cur.kind == "OP" and self.cur.text == ".":
            self._eat()
            col_tok = self._eat()
            if col_tok.kind != "IDENT":
                raise UnsupportedSyntaxError(
                    f"点号后必须是列名，实际得到 {col_tok.text!r}"
                    f"（位置 {col_tok.line}:{col_tok.col}）"
                )
            table_alias, name = name, col_tok.text
        return ast.ColumnRef(table_alias=table_alias, name=name,
                             node_id=self._nid("c"), start=tok.pos,
                             end=self.tokens[self.pos - 1].end)

    # ---- 后处理 ----
    def _span_start(self, node) -> int:
        return node.start

    def _span_end(self, node) -> int:
        return node.end

    def _index_atoms(self, select: ast.Select) -> ast.Select:
        atoms = ast.collect_atoms(select.where)
        indexed = []
        id_to_index = {}
        for i, atom in enumerate(atoms):
            new_atom = dataclasses.replace(atom, atom_index=i)
            indexed.append(new_atom)
            id_to_index[atom.node_id] = (i, new_atom)
        if select.where is not None:
            def rebuild(node):
                if node.kind == "comparison":
                    if node.is_constant:
                        return node
                    return id_to_index[node.node_id][1]
                if node.kind == "boolop":
                    return dataclasses.replace(
                        node, left=rebuild(node.left), right=rebuild(node.right))
                return node
            where = rebuild(select.where)
        else:
            where = None
        return dataclasses.replace(
            select, where=where,
            atom_by_index={i: a for i, a in enumerate(indexed)})

    def _validate(self, select: ast.Select) -> None:
        aliases = {select.base_table.alias or select.base_table.name}
        for j in select.joins:
            aliases.add(j.right.alias or j.right.name)
        for item in select.items:
            self._check_column_alias(item.expression, aliases)
        for j in select.joins:
            self._check_column_alias(j.left_col, aliases)
            self._check_column_alias(j.right_col, aliases)
        for atom in ast.collect_atoms(select.where):
            for side in (atom.left, atom.right):
                if side.kind == "column":
                    self._check_column_alias(side, aliases)
        # 连接表名/别名不允许重复。
        seen = set()
        for ref in (select.base_table, *[j.right for j in select.joins]):
            key = ref.alias or ref.name
            if key in seen:
                raise UnsupportedSyntaxError(f"表别名 {key} 重复")
            seen.add(key)

    def _check_column_alias(self, col: ast.ColumnRef, aliases: set) -> None:
        if col.table_alias is not None and col.table_alias not in aliases:
            raise UnsupportedSyntaxError(
                f"列 {col.table_alias}.{col.name} 引用了未在 FROM/JOIN 中声明的表别名"
            )


def parse_sql(sql: str) -> ast.Select:
    return Parser(sql).parse()
