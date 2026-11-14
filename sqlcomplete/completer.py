"""补全引擎：根据 schema、整段 SQL 与游标位置产出带来源的候选。

核心决策链：
1. 词法护栏：游标处于字符串/注释（含未闭合块注释）内时不推荐业务标识符。
2. 容错解析整段 SQL（含游标右侧文本）得到作用域帧树，定位游标所在最深帧。
3. 在帧内重放游标左侧的 token，判定当前语法位置：
   关系名槽 / 别名定义槽 / 列槽（SELECT 投影、ON、WHERE）。
4. 列槽再区分限定补全（a.）与非限定补全；非限定同名列有歧义时只给
   限定形式，绝不给出会绑定错误对象的裸列。
5. 无法恢复确定作用域时 state 标为 unknown_*，候选为空，不把整个
   schema 的字段塞进结果。
"""

from __future__ import annotations

from dataclasses import dataclass

from .lexer import Token, tokenize, token_at, word_span_at
from .parser import Parser, Frame, Relation, frame_at

KIND_RANK = {"column": 0, "alias": 1, "cte": 2, "table": 3}
CLAUSE_AFTER_KEYWORD = frozenset(
    {"GROUP", "HAVING", "ORDER", "LIMIT", "UNION", "EXCEPT", "INTERSECT"}
)


@dataclass
class Candidate:
    label: str           # 展示名
    kind: str            # column / alias / cte / table
    insert_text: str     # 应用替换范围时插入的文本
    origin: dict         # 来源描述
    start: int           # 替换范围（半开，码点偏移）
    end: int

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "kind": self.kind,
            "insert_text": self.insert_text,
            "origin": self.origin,
            "replacement": [self.start, self.end],
        }


@dataclass
class CompletionResult:
    state: str
    candidates: list[Candidate]
    start: int
    end: int
    clause: str | None = None

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "clause": self.clause,
            "replacement": [self.start, self.end],
            "candidates": [c.to_dict() for c in self.candidates],
        }


# ---------------------------------------------------------------- 公共 API

def complete(catalog: dict, sql: str, cursor: int) -> CompletionResult:
    if not (0 <= cursor <= len(sql)):
        raise ValueError("cursor 超出 SQL 码点范围")

    tokens = tokenize(sql)

    # 1. 词法护栏：字符串/注释内不推荐业务标识符
    here = token_at(tokens, cursor)
    if here is not None and (here.kind in ("string", "line_comment",
                                           "block_comment", "quoted")):
        state = "in_string_or_comment" if here.kind != "quoted" else "in_quoted"
        return CompletionResult(state, [], cursor, cursor)

    root = Parser(tokens, catalog, len(sql)).parse()

    span = word_span_at(tokens, cursor)
    cursor_word = _cursor_word(tokens, cursor, span)
    start, end = span if span is not None else (cursor, cursor)
    prefix = cursor_word.lower() if cursor_word is not None else ""

    # 2. 定位游标所在最深帧（以游标左侧最近 token 归属）
    anchor = _anchor_index(tokens, cursor)
    frame = frame_at(root, anchor) if anchor >= 0 else root

    # 3. 语法位置分类（先分类，才能区分列限定与 schema 链）
    position = _classify_position(tokens, frame, cursor, cursor_word)

    # 4. 尾部 ident . 结构：列槽中按限定列处理，关系槽中按 schema 链处理
    trailing = _trailing_qualifier(tokens, frame, cursor)
    if trailing is not None and position != "relation":
        qualifier, _ = trailing
        if _resolve_alias(_visible_chain(frame), qualifier) is None:
            # 列槽里点号左侧不是可见别名：作用域不可恢复，不猜测
            return CompletionResult("unknown_qualifier", [], start, end)
        return _complete_qualified(frame, qualifier, prefix, start, end)

    if position == "relation":
        return _complete_relation(catalog, frame, prefix, start, end)
    if position == "alias_def":
        return CompletionResult("alias_definition", [], start, end, clause="from")
    if position == "other_clause":
        return CompletionResult("other_clause", [], start, end)
    # 列槽
    return _complete_column(frame, prefix, start, end)


def apply_completion(sql: str, result: CompletionResult | dict) -> str:
    """把补全结果应用回字符串；游标范围之外的字节保持不变。"""
    if isinstance(result, dict):
        start, end = result["replacement"]
        text = result["candidates"][0]["insert_text"] if result["candidates"] else ""
    else:
        start, end, text = result.start, result.end, (
            result.candidates[0].insert_text if result.candidates else "")
    return sql[:start] + text + sql[end:]


# ---------------------------------------------------------------- 词法辅助

def _cursor_word(tokens: list[Token], cursor: int,
                 span: tuple[int, int] | None) -> str | None:
    if span is None:
        return None
    for tok in tokens:
        if (tok.start, tok.end) == span and tok.kind in ("word", "quoted"):
            return tok.text.strip('"')
    return None


def _anchor_index(tokens: list[Token], cursor: int) -> int:
    """游标左侧（含游标所在词的起点）最近的 token 下标。"""
    anchor = -1
    for i, tok in enumerate(tokens):
        if tok.start < cursor or (tok.start <= cursor <= tok.end):
            anchor = i
        else:
            break
    return anchor


def _trailing_qualifier(tokens: list[Token], frame: Frame,
                        cursor: int) -> tuple[str, Token] | None:
    """检测游标左侧（限当前帧内）尾部的 ident . 结构。

    返回 (名称小写, ident token)；该 ident 是可见别名还是 schema 名由上层
    结合语法位置判定：FROM/JOIN 槽中的不可解析 ident 属于 schema 链，
    SELECT/WHERE/ON 槽中的才是未知列限定。
    """
    tail: list[Token] = []
    for i in range(frame.start_tok, min(frame.end_tok, len(tokens))):
        tok = tokens[i]
        if tok.end <= cursor and not tok.is_comment:
            tail.append(tok)
    if len(tail) >= 2 and tail[-1].text == "." and tail[-2].kind in (
            "word", "quoted"):
        return tail[-2].text.strip('"').lower(), tail[-2]
    return None


# ---------------------------------------------------------------- 位置分类

def _classify_position(tokens: list[Token], frame: Frame, cursor: int,
                       cursor_word: str | None) -> str:
    """重放帧内游标左侧 token，返回 relation / alias_def / column / other_clause。"""
    clause = "select"
    from_sub = None           # None | expect_rel | after_rel | after_alias
    from_depth = 0

    lo, hi = frame.start_tok, min(frame.end_tok, len(tokens))
    for i in range(lo, hi):
        tok = tokens[i]
        if tok.end > cursor or tok.is_comment:
            continue
        if cursor_word is not None and (tok.start, tok.end) == word_span_at(
                tokens, cursor):
            continue  # 游标自身所在的词不参与状态机

        if tok.kind == "word":
            kw = tok.text.upper()
            if kw == "SELECT":
                clause, from_sub = "select", None
            elif kw == "FROM":
                clause, from_sub, from_depth = "from", "expect_rel", 0
            elif kw == "WHERE":
                clause, from_sub = "where", None
            elif kw == "ON":
                clause, from_sub = "on", None
            elif kw in CLAUSE_AFTER_KEYWORD:
                clause, from_sub = "other", None
            elif kw in ("JOIN", "INNER", "LEFT", "RIGHT", "FULL",
                        "OUTER", "CROSS"):
                if clause == "from":
                    from_sub = "expect_rel"
            elif clause == "from" and from_depth == 0:
                if from_sub == "expect_rel":
                    from_sub = "after_rel"
                elif from_sub == "after_rel" and kw != "AS":
                    from_sub = "after_alias"
        elif tok.text == "(" and clause == "from":
            from_depth += 1
        elif tok.text == ")" and clause == "from":
            from_depth -= 1
            if from_depth == 0:
                from_sub = "after_rel"
        elif tok.text == "." and clause == "from" and from_depth == 0:
            # schema.table 的点号：后续标识符仍是关系名的一部分
            if from_sub == "after_rel":
                from_sub = "expect_rel"
        elif tok.text == "," and clause == "from" and from_depth == 0:
            from_sub = "expect_rel"

    if clause == "other":
        return "other_clause"
    if clause == "from":
        if from_sub == "expect_rel":
            return "relation"
        if from_sub == "after_rel":
            return "alias_def"
    return "column"


# ---------------------------------------------------------------- 作用域链

def _visible_chain(frame: Frame) -> list[Frame]:
    """从当前帧向外的可见帧链。

    派生表帧的父帧是查询根（非相关），链在自身终止；
    EXISTS/IN 帧 correlated=True，可继续回退到外层帧。
    """
    chain: list[Frame] = []
    cur: Frame | None = frame
    while cur is not None:
        chain.append(cur)
        if cur.correlated:
            cur = cur.parent
        else:
            break
    return chain


def _resolve_alias(chain: list[Frame], alias_lower: str) -> Relation | None:
    for f in chain:
        for rel in f.relations:
            if rel.alias == alias_lower:
                return rel  # 内层先于外层：同名别名自动遮蔽
    return None


def _all_visible_relations(frame: Frame) -> list[tuple[Frame, Relation]]:
    out: list[tuple[Frame, Relation]] = []
    for f in _visible_chain(frame):
        for rel in f.relations:
            out.append((f, rel))
    return out


def _relation_origin(rel: Relation) -> dict:
    if rel.kind == "derived":
        return {"relation": rel.alias, "relation_kind": "derived"}
    return {
        "relation": rel.alias,
        "relation_kind": rel.kind,
        "source": rel.source_name,
    }


# ---------------------------------------------------------------- 候选生成

def _complete_qualified(frame: Frame, qualifier: str, prefix: str,
                        start: int, end: int) -> CompletionResult:
    chain = _visible_chain(frame)
    rel = _resolve_alias(chain, qualifier)
    if rel is None:
        # 限定名无法解析：作用域不可恢复，不猜测
        return CompletionResult("unknown_qualifier", [], start, end)

    cands: list[Candidate] = []
    for col in rel.columns:
        if not col.startswith(prefix.lower()):
            continue
        origin = _relation_origin(rel)
        origin.update({"column": col, "qualified_by": qualifier})
        cands.append(Candidate(
            label=col, kind="column", insert_text=col,
            origin={"type": "column", **origin}, start=start, end=end))
    cands.sort(key=lambda c: (c.label, c.origin.get("relation", "")))
    return CompletionResult("ok", cands, start, end)


def _complete_column(frame: Frame, prefix: str,
                     start: int, end: int) -> CompletionResult:
    visible = _all_visible_relations(frame)
    if not visible:
        # 作用域内没有任何可解析关系：明确标未知，不倾倒 schema 字段
        return CompletionResult("no_relations", [], start, end)

    # 同名列计数：按“暴露该列的关系数”计；>1 说明裸列有歧义，只能给限定形式。
    # 键用关系对象身份（自连接即便别名不同也是两个关系；同别名的两个关系
    # 同样必须分别计数）。
    name_rels: dict[str, set[int]] = {}
    for _, rel in visible:
        for col in rel.columns:
            name_rels.setdefault(col, set()).add(id(rel))
    name_counts = {col: len(rels) for col, rels in name_rels.items()}

    cands: list[Candidate] = []
    seen: set[tuple] = set()

    # 别名候选（输入限定名的快捷方式，插入自带点号）
    for f, rel in visible:
        if rel.alias.startswith(prefix):
            key = ("alias", rel.alias)
            if key not in seen:
                seen.add(key)
                cands.append(Candidate(
                    label=rel.alias, kind="alias",
                    insert_text=rel.alias + ".",
                    origin={"type": "alias", **_relation_origin(rel)},
                    start=start, end=end))

    for f, rel in visible:
        for col in rel.columns:
            qualified = f"{rel.alias}.{col}"
            if qualified.startswith(prefix):
                key = ("column", qualified)
                if key not in seen:
                    seen.add(key)
                    origin = _relation_origin(rel)
                    origin.update({"column": col})
                    cands.append(Candidate(
                        label=qualified, kind="column", insert_text=qualified,
                        origin={"type": "column", **origin},
                        start=start, end=end))
            # 无歧义的裸列：count==1 且前缀匹配
            if name_counts[col] == 1 and col.startswith(prefix):
                key = ("column", col)
                if key not in seen:
                    seen.add(key)
                    origin = _relation_origin(rel)
                    origin.update({"column": col})
                    cands.append(Candidate(
                        label=col, kind="column", insert_text=col,
                        origin={"type": "column", **origin},
                        start=start, end=end))

    cands.sort(key=lambda c: (KIND_RANK[c.kind], c.label.lower(),
                              str(c.origin)))
    return CompletionResult("ok", cands, start, end)


def _complete_relation(catalog: dict, frame: Frame, prefix: str,
                       start: int, end: int) -> CompletionResult:
    # 名称 -> (kind, source)；CTE 同名遮蔽物理表
    names: dict[str, tuple[str, str | None]] = {}
    for tbl in catalog.get("tables", {}):
        names[tbl.lower()] = ("table", tbl.lower())

    chain = _visible_chain(frame)
    for f in chain:
        for cte in f.cte_order:
            names[cte] = ("cte", cte)  # 后写覆盖：CTE 遮蔽表

    cands: list[Candidate] = []
    for name in sorted(names):
        if not name.startswith(prefix):
            continue
        kind, source = names[name]
        cands.append(Candidate(
            label=name, kind=kind, insert_text=name,
            origin={"type": kind, "name": source},
            start=start, end=end))
    cands.sort(key=lambda c: (KIND_RANK[c.kind], c.label))
    return CompletionResult("ok", cands, start, end, clause="from")
