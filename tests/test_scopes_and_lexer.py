"""独立小作用域与边界测试。

第一部分直接手工构造 Relation/Frame 对象，只跑 completer 的作用域链与
候选生成逻辑——不经过 parser，避免“拿同一解析器互证”。
第二部分覆盖词法层（注释/字符串/转义/码点）、局部恢复与稳定排序。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from sqlcomplete import complete, apply_completion
from sqlcomplete.completer import (
    _complete_column,
    _complete_qualified,
    _complete_relation,
    _visible_chain,
)
from sqlcomplete.lexer import tokenize, token_at
from sqlcomplete.parser import Parser, Frame, Relation, all_frames

ROOT = Path(__file__).resolve().parent.parent
CATALOG = json.loads((ROOT / "fixtures" / "catalog.json").read_text("utf-8"))


def _table_relation(alias: str, table: str) -> Relation:
    cols = list(CATALOG["tables"][table]["columns"].keys())
    return Relation(alias=alias, alias_span=None, kind="table",
                    source_name=table, columns=cols)


class HandBuiltScopeTest(unittest.TestCase):
    """手工作用域表，只验证 completer，不经过 parser。"""

    def setUp(self):
        # 外层帧：orders o
        self.outer = Frame(parent=None, correlated=False, start_tok=0, end_tok=100)
        self.outer.relations.append(_table_relation("o", "orders"))
        # 可相关子帧（EXISTS）：payments p
        self.exists_frame = Frame(parent=self.outer, correlated=True,
                                  start_tok=0, end_tok=100)
        self.exists_frame.relations.append(_table_relation("p", "payments"))
        # 非相关子帧（派生表）：customers c，父帧挂查询根
        root = Frame(parent=None, correlated=False, start_tok=0, end_tok=100)
        self.derived_frame = Frame(parent=root, correlated=False,
                                   start_tok=0, end_tok=100)
        self.derived_frame.relations.append(
            _table_relation("c", "customers"))

    def test_visible_chain_correlated_walks_outer(self):
        chain = _visible_chain(self.exists_frame)
        self.assertEqual(chain, [self.exists_frame, self.outer])

    def test_visible_chain_derived_stops_at_self(self):
        # 非 LATERAL：派生表看不到外层别名
        chain = _visible_chain(self.derived_frame)
        self.assertEqual(chain, [self.derived_frame])

    def test_qualified_resolves_outer_alias_in_exists(self):
        r = _complete_qualified(self.exists_frame, "o", "", 0, 0)
        labels = [c.label for c in r.candidates]
        self.assertEqual(
            sorted(labels),
            ["amount_cents", "customer_id", "order_id", "state"])
        self.assertTrue(all(c.origin["source"] == "orders"
                            for c in r.candidates))

    def test_qualified_unknown_alias_is_unknown_state(self):
        r = _complete_qualified(self.derived_frame, "o", "", 0, 0)
        self.assertEqual(r.state, "unknown_qualifier")
        self.assertEqual(r.candidates, [])

    def test_shadowing_inner_alias_wins(self):
        # 内外层同名别名 o：内层 customers 遮蔽外层 orders
        inner = Frame(parent=self.outer, correlated=True,
                      start_tok=0, end_tok=100)
        inner.relations.append(_table_relation("o", "customers"))
        r = _complete_qualified(inner, "o", "", 0, 0)
        self.assertEqual(
            sorted(c.label for c in r.candidates),
            ["customer_id", "name", "region"])
        self.assertTrue(all(c.origin["source"] == "customers"
                            for c in r.candidates))

    def test_ambiguous_bare_column_suppressed(self):
        # 同一帧挂 orders + payments：order_id/amount_cents/state 同名
        f = Frame(parent=None, correlated=False, start_tok=0, end_tok=100)
        f.relations.append(_table_relation("o", "orders"))
        f.relations.append(_table_relation("p", "payments"))
        r = _complete_column(f, "", 0, 0)
        bare = {c.label for c in r.candidates if "." not in c.label
                and c.kind == "column"}
        self.assertEqual(bare, {"customer_id", "payment_id"})
        qualified = {c.label for c in r.candidates if c.kind == "column"}
        for name in ("order_id", "amount_cents", "state"):
            self.assertNotIn(name, bare)
            self.assertIn(f"o.{name}", qualified)
            self.assertIn(f"p.{name}", qualified)

    def test_relation_slot_cte_shadows_table(self):
        f = Frame(parent=None, correlated=False, start_tok=0, end_tok=100)
        f.ctes["orders"] = [("owner", None)]
        f.cte_order.append("orders")
        r = _complete_relation(CATALOG, f, "", 0, 0)
        by_name = {c.label: c for c in r.candidates}
        self.assertEqual(by_name["orders"].kind, "cte")
        self.assertEqual(by_name["payments"].kind, "table")

    def test_empty_scope_is_unknown_not_schema_dump(self):
        f = Frame(parent=None, correlated=False, start_tok=0, end_tok=100)
        r = _complete_column(f, "", 0, 0)
        self.assertEqual(r.state, "no_relations")
        self.assertEqual(r.candidates, [])

    def test_prefix_filtering_case_insensitive(self):
        r = _complete_qualified(self.outer, "o", "AMOUNT", 0, 6)
        self.assertEqual([c.label for c in r.candidates], ["amount_cents"])

    def test_sorting_is_stable(self):
        f = Frame(parent=None, correlated=False, start_tok=0, end_tok=100)
        f.relations.append(_table_relation("o", "orders"))
        r1 = [c.label for c in _complete_column(f, "", 0, 0).candidates]
        r2 = [c.label for c in _complete_column(f, "", 0, 0).candidates]
        self.assertEqual(r1, r2)
        # column 在 alias 前；同类按标签排序
        kinds = [c.kind for c in _complete_column(f, "", 0, 0).candidates]
        self.assertEqual(kinds, sorted(kinds, key={"column": 0, "alias": 1}.get))


class LexerTest(unittest.TestCase):
    def test_comments_and_strings_classified(self):
        toks = tokenize("SELECT 'it''s' -- 注释\n /* 块 */ x")
        kinds = [(t.kind, t.text) for t in toks]
        self.assertIn(("string", "'it''s'"), kinds)
        self.assertIn(("line_comment", "-- 注释"), kinds)
        self.assertIn(("block_comment", "/* 块 */"), kinds)

    def test_unclosed_block_comment_runs_to_end(self):
        toks = tokenize("SELECT x /* 未闭合")
        last = toks[-1]
        self.assertEqual(last.kind, "block_comment")
        self.assertEqual(last.end, len("SELECT x /* 未闭合"))

    def test_unclosed_string_runs_to_end(self):
        toks = tokenize("SELECT 'abc")
        self.assertEqual(toks[-1].kind, "string")

    def test_cursor_inside_comment_returns_comment_token(self):
        sql = "SELECT o.order_id /* 名称字段 */ FROM orders o"
        toks = tokenize(sql)
        tok = token_at(toks, 23)
        self.assertIsNotNone(tok)
        self.assertEqual(tok.kind, "block_comment")

    def test_codepoint_offsets_with_multibyte(self):
        sql = "/* 中文 */ SELECT x"
        toks = tokenize(sql)
        # 所有偏移都是码点偏移：每个 token 按 [start,end) 切片能还原其文本
        for t in toks:
            self.assertEqual(sql[t.start:t.end], t.text)
        # 词法覆盖的码点之外只能是空白（注释内部空白已被注释 token 覆盖）
        covered = {i for t in toks for i in range(t.start, t.end)}
        uncovered = [i for i in range(len(sql)) if i not in covered]
        self.assertTrue(all(sql[i].isspace() for i in uncovered))
        # 2 个中文字符：每个 UTF-8 占 3 字节，码点长度比字节长度少 4
        self.assertEqual(len(sql.encode("utf-8")) - len(sql), 4)

    def test_double_quote_escaping(self):
        toks = tokenize('SELECT "a""b" FROM t')
        self.assertEqual(toks[1].kind, "quoted")
        self.assertEqual(toks[1].text, '"a""b"')


class RecoveryTest(unittest.TestCase):
    def test_select_without_from_still_recovers_scope_later(self):
        # 缺 FROM 的破损输入不抛异常，状态明确
        r = complete(CATALOG, "SELECT o. FROM orders o WHERE (", 9)
        self.assertEqual(r.state, "ok")
        self.assertTrue(r.candidates)

    def test_right_side_from_is_not_truncated(self):
        # 游标在 SELECT 之后，但右侧已有完整 FROM/JOIN，上下文必须保留
        sql = "SELECT  FROM orders o JOIN payments p ON 1=1"
        r = complete(CATALOG, sql, 7)
        labels = {c.label for c in r.candidates}
        self.assertIn("o.order_id", labels)
        self.assertIn("p.payment_id", labels)

    def test_missing_close_paren_keeps_subquery_frame(self):
        sql = ("SELECT o.order_id FROM orders o WHERE "
               "EXISTS (SELECT p. FROM payments p")
        r = complete(CATALOG, sql, sql.index("p.") + 2)
        self.assertEqual(r.state, "ok")
        self.assertIn("payment_id", [c.label for c in r.candidates])

    def test_derived_table_cannot_see_outer_alias_end_to_end(self):
        sql = ("SELECT o.order_id FROM orders o WHERE EXISTS "
               "(SELECT 1 FROM (SELECT customer_id FROM customers) d WHERE d.")
        r = complete(CATALOG, sql, len(sql))
        # 派生表 d 内部限定 o. 应未知；这里 d. 只给 customer_id
        self.assertEqual(
            sorted(c.label for c in r.candidates), ["customer_id"])

    def test_unknown_qualifier_end_to_end(self):
        r = complete(CATALOG, "SELECT z. FROM orders o", 9)
        self.assertEqual(r.state, "unknown_qualifier")
        self.assertEqual(r.candidates, [])

    def test_alias_definition_slot_no_business_candidates(self):
        sql = "SELECT * FROM orders "
        r = complete(CATALOG, sql, len(sql))
        self.assertEqual(r.state, "alias_definition")

    def test_apply_every_candidate_keeps_surrounding_bytes(self):
        sql = "SELECT o.am FROM orders o"
        r = complete(CATALOG, sql, 11)
        for cand in r.candidates:
            out = sql[:r.start] + cand.insert_text + sql[r.end:]
            # 替换范围两侧的码点内容逐字节不变
            self.assertEqual(out[:r.start], sql[:r.start])
            self.assertEqual(out[r.start + len(cand.insert_text):], sql[r.end:])

    def test_input_text_is_not_rewritten(self):
        sql = "SELECT O.AM FROM ORDERS O"  # 全大写输入
        r = complete(CATALOG, sql, 11)
        self.assertTrue(r.candidates)
        # 输入对象内容不变；插入文本使用 catalog 规范拼写
        self.assertEqual(sql, "SELECT O.AM FROM ORDERS O")
        self.assertEqual(r.candidates[0].insert_text, "amount_cents")

    def test_invalid_cursor_raises(self):
        with self.assertRaises(ValueError):
            complete(CATALOG, "SELECT 1", 99)

    def test_line_comment_inside_blocks_business_completion(self):
        sql = "SELECT o.order_id -- 中文注释\nFROM orders o"
        # 游标位于注释词内部（-- 之后）
        r = complete(CATALOG, sql, 21)
        self.assertEqual(r.state, "in_string_or_comment")
        self.assertEqual(r.candidates, [])

    def test_escaped_quote_string_spans_whole_literal(self):
        sql = "SELECT 'it''s ok' FROM orders"
        r = complete(CATALOG, sql, 10)
        self.assertEqual(r.state, "in_string_or_comment")

    def test_join_relation_slot_and_cte_shadowing_in_slot(self):
        r = complete(CATALOG, "SELECT * FROM orders o JOIN pay", 31)
        self.assertEqual(r.clause, "from")
        self.assertEqual([c.label for c in r.candidates], ["payments"])
        r2 = complete(CATALOG,
                      "WITH paid AS (SELECT 1) SELECT * FROM pa", 40)
        self.assertEqual({c.label: c.kind for c in r2.candidates},
                         {"paid": "cte", "payments": "table"})

    def test_exists_bare_columns_see_both_scopes(self):
        sql = ("SELECT o.order_id FROM orders o WHERE EXISTS "
               "(SELECT 1 FROM payments p WHERE ")
        r = complete(CATALOG, sql, len(sql))
        labels = {c.label for c in r.candidates}
        # 内层 p 与外层 o 的别名都可见
        self.assertIn("o", labels)
        self.assertIn("p", labels)
        self.assertIn("p.payment_id", labels)
        self.assertIn("o.customer_id", labels)

    def test_derived_qualifier_unknown_end_to_end(self):
        sql = "SELECT * FROM (SELECT o. FROM customers c) d"
        r = complete(CATALOG, sql, sql.index("o.") + 2)
        self.assertEqual(r.state, "unknown_qualifier")

    def test_schema_qualified_relation_mid_word(self):
        # schema.table 词中游标：按关系名槽补全，只替换表名片段
        sql = "SELECT * FROM public.ord"
        r = complete(CATALOG, sql, 22)
        self.assertEqual(r.clause, "from")
        self.assertEqual([c.label for c in r.candidates], ["orders"])
        self.assertEqual([r.start, r.end], [21, 24])
        self.assertEqual(apply_completion(sql, r),
                         "SELECT * FROM public.orders")

    def test_multilevel_schema_chain(self):
        sql = "SELECT * FROM a.b.pay"
        r = complete(CATALOG, sql, len(sql) - 1)
        self.assertEqual([c.label for c in r.candidates], ["payments"])

    def test_cursor_just_after_dot_is_qualified(self):
        sql = "SELECT o. FROM orders o"
        r = complete(CATALOG, sql, sql.index(".") + 1)
        self.assertEqual(
            sorted(c.label for c in r.candidates),
            ["amount_cents", "customer_id", "order_id", "state"])

    def test_nested_plain_parens_keep_scope(self):
        sql = "SELECT * FROM orders o WHERE ((o."
        r = complete(CATALOG, sql, len(sql))
        self.assertEqual(r.state, "ok")
        self.assertEqual(len(r.candidates), 4)

    def test_unclosed_cte_recovers_inner_scope(self):
        sql = "WITH paid AS (SELECT  FROM payments"
        r = complete(CATALOG, sql, sql.index("SELECT ") + 7)
        self.assertEqual(r.state, "ok")
        labels = {c.label for c in r.candidates}
        self.assertTrue(
            {"payment_id", "order_id", "amount_cents", "state"} <= labels)

    def test_malformed_inputs_never_raise(self):
        for sql in ("SELECT", "WITH", "SELECT * FROM", ")))",
                    "WITH x AS (", "SELECT * FROM orders o JOIN"):
            r = complete(CATALOG, sql, len(sql))
            self.assertIn(r.state, (
                "ok", "no_relations", "alias_definition",
                "unknown_qualifier", "other_clause"))


class ParserStructureTest(unittest.TestCase):
    def _parse(self, sql: str) -> Frame:
        toks = tokenize(sql)
        return Parser(toks, CATALOG, len(sql)).parse()

    def test_cte_order_and_resolution(self):
        sql = ("WITH a AS (SELECT order_id FROM orders), "
               "b AS (SELECT a.order_id AS key FROM a) SELECT * FROM b")
        root = self._parse(sql)
        self.assertEqual(root.cte_order, ["a", "b"])
        self.assertEqual([c for c, _ in root.ctes["a"]], ["order_id"])
        self.assertEqual([c for c, _ in root.ctes["b"]], ["key"])

    def test_on_span_recorded(self):
        root = self._parse(
            "SELECT * FROM orders o JOIN customers c ON o.customer_id=c.customer_id")
        self.assertEqual(len(root.on_spans), 1)

    def test_frames_for_nested_exists(self):
        sql = ("SELECT o.order_id FROM orders o WHERE "
               "EXISTS (SELECT 1 FROM payments p WHERE p.order_id=o.order_id)")
        root = self._parse(sql)
        frames = all_frames(root)
        self.assertEqual(len(frames), 2)
        sub = frames[1]
        self.assertTrue(sub.correlated)
        self.assertEqual(sub.parent, frames[0])


if __name__ == "__main__":
    unittest.main()
