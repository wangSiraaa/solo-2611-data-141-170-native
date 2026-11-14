"""容错的局部 SQL 解析与作用域构建。

设计原则：
1. 不要求整段 SQL 可解析——遇到期望之外的词跳过并继续，输入可以缺
   表达式、右括号或后续子句。
2. 不截断游标右侧文本：整段 SQL 都参与词法/语法分析，游标右侧已有的
   FROM/JOIN 等上下文不会丢失。
3. 作用域以“帧(Frame)”表示：每个 SELECT 查询一帧，派生表/EXISTS/IN
   子查询产生子帧并记录父帧。普通括号不产生新作用域。
4. CTE 按定义顺序登记到所在帧，后者可引用前者；CTE 名称遮蔽同名表。
5. 派生表帧不可见外层别名（非 LATERAL）；EXISTS/IN 子查询帧标记为
   可相关，名称解析时沿父帧链回退，内层同名别名遮蔽外层。

只支持 scope.json 声明的语法子集；无法解释的投影表达式不产生输出列，
但不影响其余结构的恢复。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .lexer import Token

CLAUSE_KEYWORDS = frozenset(
    {"SELECT", "FROM", "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT",
     "UNION", "EXCEPT", "INTERSECT"}
)
JOIN_KEYWORDS = frozenset(
    {"JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS", "ON", "USING"}
)


@dataclass
class Relation:
    """FROM/JOIN 中一个可见的关系（表、CTE 或派生表）及其别名。"""

    alias: str                       # 解析后的别名（小写）；无别名时为关系名
    alias_span: tuple[int, int] | None
    kind: str                        # table / cte / derived
    source_name: str | None          # table/cte 的名称（小写）
    columns: list[str]               # 该关系暴露的输出列（小写原名）
    derived_outputs: list[tuple[str, str | None]] | None = None
    # 派生表输出列的 (列名, 原始来源描述)，用于 origin 透传


@dataclass
class Frame:
    parent: "Frame | None"
    correlated: bool                 # True=EXISTS/IN 子查询，可回退父帧
    start_tok: int
    end_tok: int = -1                # 词法 token 下标（不含），-1 表示到末尾
    relations: list[Relation] = field(default_factory=list)
    ctes: "dict[str, list[tuple[str, str | None]]]" = field(default_factory=dict)
    # cte 名称 -> 输出列 [(列名, 来源描述)]
    cte_order: list[str] = field(default_factory=list)
    select_tok: int | None = None
    from_tok: int | None = None
    where_tok: int | None = None
    on_spans: list[tuple[int, int]] = field(default_factory=list)
    # ON 表达式的 token 下标区间（不含 ON 关键字本身）
    children: list["Frame"] = field(default_factory=list)


class Parser:
    def __init__(self, tokens: list[Token], catalog: dict, sql_len: int):
        self.toks = tokens
        self.catalog = catalog
        self.sql_len = sql_len

    # ---------- 基础工具 ----------

    def _is(self, idx: int, word: str) -> bool:
        return (0 <= idx < len(self.toks)
                and self.toks[idx].kind == "word"
                and self.toks[idx].text.upper() == word)

    def _word(self, idx: int) -> str | None:
        if 0 <= idx < len(self.toks) and self.toks[idx].kind == "word":
            return self.toks[idx].text
        return None

    def _skip_noise(self, idx: int) -> int:
        while idx < len(self.toks) and self.toks[idx].is_comment:
            idx += 1
        return idx

    def _table_columns(self, name_lower: str) -> list[str]:
        tbl = self.catalog.get("tables", {}).get(name_lower)
        return list(tbl["columns"].keys()) if tbl else []

    def _cte_lookup(self, frame: Frame, name_lower: str):
        cur: Frame | None = frame
        while cur is not None:
            if name_lower in cur.ctes:
                return cur.ctes[name_lower]
            cur = cur.parent
        return None

    # ---------- 入口 ----------

    def parse(self) -> Frame:
        root = Frame(parent=None, correlated=False, start_tok=0,
                     end_tok=len(self.toks))
        idx = self._skip_noise(0)
        if self._is(idx, "WITH") or self._is(idx, "SELECT") or self._is(idx, "("):
            self.parse_query(idx, root)
        else:
            # 无法识别的起始结构：保留空帧，补全侧按 unknown 处理
            pass
        return root

    # ---------- 查询帧 ----------

    def parse_query(self, idx: int, frame: Frame) -> int:
        idx = self._skip_noise(idx)

        # WITH cte_name AS ( ... ) [, ...]
        if self._is(idx, "WITH"):
            idx += 1
            while True:
                idx = self._skip_noise(idx)
                name = self._word(idx)
                if name is None:
                    break
                idx += 1
                idx = self._skip_noise(idx)
                if self._is(idx, "AS"):
                    idx += 1
                idx = self._skip_noise(idx)
                if idx < len(self.toks) and self.toks[idx].text == "(":
                    open_idx = idx
                    close_idx, inner = self._scan_paren(idx)
                    # 括号不闭合时仍建帧，帧尾延伸到 token 流末尾（局部恢复）
                    sub_end = close_idx if inner else len(self.toks)
                    sub = Frame(parent=frame.parent, correlated=False,
                                start_tok=open_idx + 1,
                                end_tok=max(sub_end, open_idx + 1))
                    frame.children.append(sub)
                    self.parse_query(open_idx + 1, sub)
                    outputs = self._derive_outputs(sub)
                    frame.ctes[name.lower()] = outputs
                    frame.cte_order.append(name.lower())
                    idx = close_idx if inner else len(self.toks)
                idx = self._skip_noise(idx)
                if idx < len(self.toks) and self.toks[idx].text == ",":
                    idx += 1
                    continue
                break

        idx = self._skip_noise(idx)
        if self._is(idx, "SELECT"):
            frame.select_tok = idx
            idx += 1

        # SELECT ... 直到 FROM（投影内容只记录 token，输出列延迟推导）
        while idx < len(self.toks):
            idx = self._skip_noise(idx)
            if idx >= len(self.toks):
                break
            if self._is(idx, "FROM"):
                frame.from_tok = idx
                idx += 1
                break
            if self.toks[idx].text == "(":
                idx, _ = self._scan_paren(idx)
                continue
            idx += 1

        # FROM 关系序列
        if frame.from_tok is not None:
            idx = self._skip_noise(idx)
            idx, rel = self.parse_source(idx, frame)
            if rel is not None:
                frame.relations.append(rel)
            idx = self._skip_noise(idx)
            while idx < len(self.toks):
                # JOIN 前置关键字：[INNER|LEFT|RIGHT|FULL|OUTER|CROSS] JOIN
                join_seen = False
                while idx < len(self.toks) and (
                    self._is(idx, "INNER") or self._is(idx, "LEFT")
                    or self._is(idx, "RIGHT") or self._is(idx, "FULL")
                    or self._is(idx, "OUTER") or self._is(idx, "CROSS")
                ):
                    join_seen = True
                    idx += 1
                if not self._is(idx, "JOIN"):
                    break
                join_seen = True
                idx += 1
                idx = self._skip_noise(idx)
                idx, rel = self.parse_source(idx, frame)
                if rel is not None:
                    frame.relations.append(rel)
                idx = self._skip_noise(idx)
                if self._is(idx, "ON"):
                    on_start = idx + 1
                    idx = self.scan_expr(on_start, frame)
                    frame.on_spans.append((on_start, idx))
                elif self._is(idx, "USING"):
                    idx += 1
                    if idx < len(self.toks) and self.toks[idx].text == "(":
                        idx, _ = self._scan_paren(idx)
                idx = self._skip_noise(idx)

        # 尾部子句：WHERE 与其它（GROUP/ORDER/...）只扫描，不深入建模
        idx = self._skip_noise(idx)
        if self._is(idx, "WHERE"):
            frame.where_tok = idx
            idx = self.scan_expr(idx + 1, frame)
        # 集合操作 / GROUP BY 等：跳过到帧结束，保持局部恢复
        return idx

    # ---------- FROM 中的单个关系 ----------

    def parse_source(self, idx: int, frame: Frame) -> tuple[int, Relation | None]:
        idx = self._skip_noise(idx)
        if idx >= len(self.toks):
            return idx, None

        tok = self.toks[idx]

        # 派生表 / 括号子查询
        if tok.text == "(":
            open_idx = idx
            close_idx, inner = self._scan_paren(idx)
            if not inner:
                # 括号不闭合：仍按派生表建帧，帧尾取到 token 流末尾
                close_idx = len(self.toks)
            sub = Frame(parent=frame.parent, correlated=False,
                        start_tok=open_idx + 1, end_tok=max(close_idx, open_idx + 1))
            frame.children.append(sub)
            self.parse_query(open_idx + 1, sub)
            outputs = self._derive_outputs(sub)
            after = self._skip_noise(max(close_idx, open_idx + 1))
            alias_idx, alias_name, alias_span = self.take_alias(after)
            rel = Relation(
                alias=alias_name or f"__derived_{open_idx}",
                alias_span=alias_span,
                kind="derived",
                source_name=None,
                columns=[c for c, _ in outputs],
                derived_outputs=outputs,
            )
            return alias_idx, rel

        # 普通表名或 CTE 名（允许 schema.table 形式，取最后一段）
        if tok.kind in ("word", "quoted"):
            name = tok.text.strip('"')
            start_idx = idx
            idx += 1
            idx = self._skip_noise(idx)
            if idx < len(self.toks) and self.toks[idx].text == ".":
                dot = idx
                idx += 1
                idx = self._skip_noise(idx)
                if idx < len(self.toks) and self.toks[idx].kind in ("word", "quoted"):
                    name = self.toks[idx].text.strip('"')
                    start_idx = dot + 1
                    idx += 1
            name_lower = name.lower()
            cte_cols = self._cte_lookup(frame, name_lower)
            if cte_cols is not None:
                kind, cols, source = "cte", [c for c, _ in cte_cols], None
            else:
                kind = "table"
                cols = self._table_columns(name_lower)
                source = name_lower
            idx = self._skip_noise(idx)
            alias_idx, alias_name, alias_span = self.take_alias(idx)
            rel = Relation(
                alias=alias_name or name_lower,
                alias_span=alias_span,
                kind=kind,
                source_name=name_lower if kind == "cte" else source,
                columns=cols,
            )
            return alias_idx, rel

        # 期望关系却遇到其它词（破损输入）：跳过一个词继续恢复
        return idx + 1, None

    def take_alias(self, idx: int) -> tuple[int, str, tuple[int, int] | None]:
        """消费可选的 [AS] alias，返回 (新下标, 别名小写, 别名 span)。"""
        idx = self._skip_noise(idx)
        if self._is(idx, "AS"):
            idx = self._skip_noise(idx + 1)
        if idx < len(self.toks) and self.toks[idx].kind in ("word", "quoted"):
            text = self.toks[idx].text
            upper = text.upper()
            if upper not in CLAUSE_KEYWORDS and upper not in JOIN_KEYWORDS:
                tok = self.toks[idx]
                return idx + 1, text.strip('"').lower(), (tok.start, tok.end)
        return idx, "", None

    # ---------- 表达式扫描（WHERE/ON/投影中的括号） ----------

    def scan_expr(self, idx: int, frame: Frame) -> int:
        """扫描一个表达式，为其中的 EXISTS/IN 子查询建立可相关子帧。"""
        depth = 0
        while idx < len(self.toks):
            idx = self._skip_noise(idx)
            if idx >= len(self.toks):
                break
            tok = self.toks[idx]
            if depth == 0 and tok.kind == "word" and tok.text.upper() in (
                    "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT",
                    "UNION", "EXCEPT", "INTERSECT"):
                break
            if tok.text == "(":
                # 仅 EXISTS ( ... ) / IN ( ... ) 建立可相关子帧；
                # 普通分组括号不产生新作用域，跳过配对括号即可。
                prev = self._skip_noise(idx - 1) if idx > 0 else -1
                correlated = (
                    prev >= 0 and self.toks[prev].kind == "word"
                    and self.toks[prev].text.upper() in ("EXISTS", "IN")
                )
                open_idx = idx
                close_idx, inner = self._scan_paren(idx)
                if not correlated:
                    idx = close_idx if inner else len(self.toks)
                    continue
                end = close_idx if inner else len(self.toks)
                sub = Frame(parent=frame, correlated=correlated,
                            start_tok=open_idx + 1, end_tok=max(end, open_idx + 1))
                frame.children.append(sub)
                self.parse_query(open_idx + 1, sub)
                idx = max(end, open_idx + 1)
                continue
            idx += 1
        return idx

    # ---------- 括号匹配 ----------

    def _scan_paren(self, idx: int) -> tuple[int, bool]:
        """从 '(' 开始扫描，返回 (右括号后下标, 是否闭合)。

        不闭合时返回 (len(toks), False)，调用方据此做局部恢复。
        """
        depth = 0
        i = idx
        while i < len(self.toks):
            t = self.toks[i].text
            if t == "(":
                depth += 1
            elif t == ")":
                depth -= 1
                if depth == 0:
                    return i + 1, True
            i += 1
        return len(self.toks), False

    # ---------- 投影输出列推导 ----------

    def collect_projection(self, frame: Frame) -> list[list[Token]]:
        if frame.select_tok is None:
            return []
        start = frame.select_tok + 1
        end = frame.from_tok if frame.from_tok is not None else frame.end_tok
        items: list[list[Token]] = []
        cur: list[Token] = []
        depth = 0
        i = start
        while i < end:
            tok = self.toks[i]
            if tok.is_comment:
                i += 1
                continue
            if tok.text == "(":
                depth += 1
                cur.append(tok)
            elif tok.text == ")":
                depth -= 1
                cur.append(tok)
            elif tok.text == "," and depth == 0:
                if cur:
                    items.append(cur)
                cur = []
            else:
                cur.append(tok)
            i += 1
        if cur:
            items.append(cur)
        return items

    def _resolve_relation(self, frame: Frame, alias_lower: str) -> Relation | None:
        for rel in frame.relations:
            if rel.alias == alias_lower:
                return rel
        return None

    def _derive_outputs(self, frame: Frame) -> list[tuple[str, str | None]]:
        """推导 SELECT 帧暴露的输出列：(列名, 来源描述)。

        只解释可解析的形态：*、alias.*、普通/限定列、显式 AS 别名、
        双标识符隐式别名；函数与任意表达式无别名时跳过（不要求解释
        任意表达式），有别名时以别名作为输出列。
        """
        outputs: list[tuple[str, str | None]] = []
        for item in self.collect_projection(frame):
            atoms = [t for t in item if t.kind != "number"]
            words = [t for t in item if t.kind in ("word", "quoted")]
            if len(item) == 1 and item[0].text == "*":
                for rel in frame.relations:
                    for col in rel.columns:
                        outputs.append((col, f"{rel.alias}.{col}"))
                continue

            # 显式 AS 别名：别名即输出名
            as_pos = next(
                (k for k, t in enumerate(item)
                 if t.kind == "word" and t.text.upper() == "AS"), -1)
            if as_pos >= 0 and as_pos + 1 < len(item):
                alias_tok = item[as_pos + 1]
                if alias_tok.kind in ("word", "quoted"):
                    name = alias_tok.text.strip('"')
                    origin_expr = "".join(t.text for t in item[:as_pos]).strip()
                    outputs.append((name, origin_expr or None))
                    continue

            # 限定 * ：t.*
            if (len(item) == 3 and item[0].kind in ("word", "quoted")
                    and item[1].text == "." and item[2].text == "*"):
                rel = self._resolve_relation(frame, item[0].text.lower())
                if rel is not None:
                    for col in rel.columns:
                        outputs.append((col, f"{rel.alias}.{col}"))
                continue

            # 双标识符隐式别名：expr alias（仅限简单列形态 col alias）
            if len(words) == 2 and len(item) == 2:
                first, second = words
                if first.kind in ("word", "quoted") and second.kind in ("word", "quoted"):
                    outputs.append((second.text.strip('"'), first.text))
                    continue

            # 单个普通/限定列：沿用列名
            if len(item) == 1 and item[0].kind in ("word", "quoted"):
                outputs.append((item[0].text.strip('"'), None))
                continue
            if (len(item) == 3 and item[0].kind in ("word", "quoted")
                    and item[1].text == "." and item[2].kind in ("word", "quoted")):
                outputs.append((item[2].text.strip('"'),
                                f"{item[0].text.lower()}.{item[2].text}"))
                continue
            # 其它无法解释的表达式：不产出输出列
        return outputs


def all_frames(root: Frame) -> list[Frame]:
    """深度优先枚举全部帧（含 CTE、派生表、EXISTS/IN 子查询帧）。"""
    out: list[Frame] = []

    def walk(f: Frame) -> None:
        out.append(f)
        for ch in f.children:
            walk(ch)

    walk(root)
    return out


def frame_at(root: Frame, tok_index: int) -> Frame:
    """返回 token 下标所在的最深帧；未落入任何子帧时返回根帧。

    all_frames 为 DFS 先序，嵌套帧的父帧先于子帧出现，因此包含该
    token 的最后一个帧即最深帧（兄弟帧区间互不重叠）。
    """
    best = root
    for f in all_frames(root):
        if f.start_tok <= tok_index < f.end_tok:
            best = f
    return best
