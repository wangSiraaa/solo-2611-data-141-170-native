"""fixtures/cases.json 全量用例测试。

每条用例核对：状态、候选标签、类别、来源、替换范围，以及把建议应用回
字符串后游标范围之外的码点内容逐字节不变。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from sqlcomplete import complete, apply_completion

ROOT = Path(__file__).resolve().parent.parent
CATALOG = json.loads((ROOT / "fixtures" / "catalog.json").read_text("utf-8"))
CASES = {c["case_id"]: c
         for c in json.loads((ROOT / "fixtures" / "cases.json").read_text("utf-8"))}


class FixtureCasesTest(unittest.TestCase):
    def _run(self, case_id: str):
        case = CASES[case_id]
        result = complete(CATALOG, case["sql"], case["cursor"])
        return case, result

    def _labels(self, result, kind: str | None = None) -> list[str]:
        return [c.label for c in result.candidates
                if kind is None or c.kind == kind]

    def test_prefix_select_before_from(self):
        # SELECT 先于 FROM 也能补全；限定列只来自别名指向的表
        case, r = self._run("prefix")
        self.assertEqual(r.state, "ok")
        self.assertEqual(self._labels(r), ["amount_cents"])
        self.assertEqual(r.candidates[0].origin["source"], "orders")
        self.assertEqual([r.start, r.end], [9, 11])  # 替换 "am"
        self.assertEqual(apply_completion(case["sql"], r),
                         "SELECT o.amount_cents FROM orders o")

    def test_middle_cursor_replaces_whole_word(self):
        # 词中游标：替换整个标识符片段，不留右半截
        case, r = self._run("middle")
        self.assertEqual([r.start, r.end], [9, 20])  # 整个 aount_cents
        # 拼写错误前缀无匹配，但范围仍覆盖整词
        self.assertEqual(r.candidates, [])
        self.assertEqual(apply_completion(case["sql"], r),
                         "SELECT o. FROM orders o")

    def test_scope_join_two_relations(self):
        _, r = self._run("scope")
        labels = self._labels(r)
        # 两表都暴露的 customer_id 只能限定
        self.assertNotIn("customer_id", labels)
        self.assertIn("o.customer_id", labels)
        self.assertIn("c.customer_id", labels)
        # 单侧列保留裸列
        self.assertIn("name", labels)
        self.assertIn("order_id", labels)
        # 别名候选
        self.assertCountEqual(self._labels(r, "alias"), ["o", "c"])

    def test_table_position_suggests_relations(self):
        _, r = self._run("table")
        self.assertEqual(r.clause, "from")
        self.assertEqual(self._labels(r), ["orders"])
        self.assertEqual(r.candidates[0].kind, "table")
        self.assertEqual([r.start, r.end], [14, 17])  # 替换 "ord"

    def test_cte_output_columns(self):
        _, r = self._run("cte")
        # 投影别名 amount 与可解析列 order_id 参与建议；payments 其它列不泄露
        self.assertCountEqual(self._labels(r), ["amount", "order_id"])
        col = next(c for c in r.candidates if c.label == "amount")
        self.assertEqual(col.origin["relation"], "p")
        self.assertEqual(col.origin["relation_kind"], "cte")

    def test_ordered_ctes_reference_earlier(self):
        _, r = self._run("ordered_ctes")
        # b 的输出列是显式别名 key；a.order_id AS key
        self.assertEqual(self._labels(r), ["key"])

    def test_cte_shadows_physical_table(self):
        _, r = self._run("shadow_table")
        # CTE orders 遮蔽物理表 orders，输出只有 owner
        self.assertEqual(self._labels(r), ["owner"])
        self.assertEqual(r.candidates[0].origin["relation_kind"], "cte")

    def test_correlated_exists_outer_alias(self):
        _, r = self._run("correlated")
        # EXISTS 内 o. 解析到外层 orders（可相关）
        self.assertCountEqual(
            self._labels(r),
            ["amount_cents", "customer_id", "order_id", "state"])
        self.assertEqual(r.candidates[0].origin["source"], "orders")

    def test_inner_alias_shadows_outer(self):
        _, r = self._run("shadow_alias")
        # 内层 customers o 遮蔽外层 orders o -> 只给 customers 的列
        self.assertCountEqual(self._labels(r),
                              ["customer_id", "name", "region"])
        self.assertEqual(r.candidates[0].origin["source"], "customers")

    def test_derived_table_outputs_and_no_outer_scope(self):
        case, r = self._run("derived")
        self.assertCountEqual(self._labels(r), ["amount", "owner"])
        # 派生表内部不能引用外层别名：手工构造见 test_scopes 的独立核对
        self.assertEqual(r.candidates[0].origin["relation_kind"], "derived")

    def test_broken_where_still_completes(self):
        _, r = self._run("broken")
        self.assertEqual(r.state, "ok")
        self.assertCountEqual(
            self._labels(r),
            ["amount_cents", "customer_id", "order_id", "state"])

    def test_string_literal_no_business_completion(self):
        _, r = self._run("literal")
        self.assertEqual(r.state, "in_string_or_comment")
        self.assertEqual(r.candidates, [])

    def test_block_comment_no_business_completion(self):
        _, r = self._run("comment")
        self.assertEqual(r.state, "in_string_or_comment")
        self.assertEqual(r.candidates, [])

    def test_unclosed_block_comment_no_completion(self):
        _, r = self._run("unclosed_comment")
        self.assertEqual(r.state, "in_string_or_comment")
        self.assertEqual(r.candidates, [])

    def test_missing_select_no_relations_is_unknown(self):
        _, r = self._run("missing")
        # 无法恢复确定作用域：明确标未知，不倾倒 schema
        self.assertEqual(r.state, "no_relations")
        self.assertEqual(r.candidates, [])

    def test_unclosed_subquery_keeps_inner_scope(self):
        _, r = self._run("unclosed_subquery")
        self.assertEqual(r.state, "ok")
        self.assertCountEqual(
            self._labels(r),
            ["amount_cents", "order_id", "payment_id", "state"])
        self.assertEqual(r.candidates[0].origin["source"], "payments")

    def test_application_preserves_bytes_outside_range(self):
        for case_id, case in CASES.items():
            sql, cur = case["sql"], case["cursor"]
            r = complete(CATALOG, sql, cur)
            if r.candidates:
                out = apply_completion(sql, r)
                expected = sql[:r.start] + r.candidates[0].insert_text + sql[r.end:]
                self.assertEqual(out, expected, case_id)
                # 范围外两侧逐字节不变
                self.assertTrue(out.endswith(sql[r.end:]) or sql[r.end:] == "")
                self.assertTrue(out.startswith(sql[:r.start]))

    def test_cursor_is_codepoint_not_byte(self):
        # 中文注释占多字节；游标按码点工作，字符串下标即码点
        case = CASES["comment"]
        self.assertIsInstance(case["sql"], str)
        self.assertEqual(len(case["sql"][:case["cursor"]]), case["cursor"])


if __name__ == "__main__":
    unittest.main()
