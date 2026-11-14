"""SQL 词法分析器。

职责：把 SQL 文本切成带 Unicode 码点偏移的词法片段，识别
- 普通标识符 / 关键字（同一类，关键字由 parser 按词面判断）
- 单引号字符串（支持 '' 转义）与双引号标识符（支持 "" 转义）
- -- 行注释与 /* 块注释（允许不闭合）
- 数字、点号、逗号、括号与其它运算符

游标以 Unicode 码点为单位；Python str 的索引天然就是码点索引，
因此本模块所有偏移直接使用字符串下标，不做 UTF-8/UTF-16 换算。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
OPERATORS = ("<>", "!=", ">=", "<=", "||", "::")


@dataclass(frozen=True)
class Token:
    kind: str          # word / number / string / quoted / line_comment / block_comment / punct
    text: str
    start: int         # 含
    end: int           # 不含（半开区间，均为码点偏移）

    @property
    def is_comment(self) -> bool:
        return self.kind in ("line_comment", "block_comment")


def tokenize(sql: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]

        if ch in " \t\r\n":
            i += 1
            continue

        # -- 行注释：到行尾或文本结束
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            j = i + 2
            while j < n and sql[j] != "\n":
                j += 1
            tokens.append(Token("line_comment", sql[i:j], i, j))
            i = j
            continue

        # /* 块注释：允许不闭合（局部恢复的一部分）
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            j = i + 2
            closed = False
            while j < n:
                if sql[j] == "*" and j + 1 < n and sql[j + 1] == "/":
                    j += 2
                    closed = True
                    break
                j += 1
            tokens.append(Token("block_comment", sql[i:j], i, j))
            i = j
            continue

        # 单引号字符串：'' 为转义引号；允许结尾缺引号
        if ch == "'":
            j = i + 1
            while j < n:
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            tokens.append(Token("string", sql[i:j], i, j))
            i = j
            continue

        # 双引号标识符："" 为转义；允许结尾缺引号
        if ch == '"':
            j = i + 1
            while j < n:
                if sql[j] == '"':
                    if j + 1 < n and sql[j + 1] == '"':
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            tokens.append(Token("quoted", sql[i:j], i, j))
            i = j
            continue

        m = WORD_RE.match(sql, i)
        if m:
            tokens.append(Token("word", m.group(0), i, m.end()))
            i = m.end()
            continue

        m = NUMBER_RE.match(sql, i)
        if m:
            tokens.append(Token("number", m.group(0), i, m.end()))
            i = m.end()
            continue

        # 多字符运算符优先
        two = sql[i:i + 2]
        if two in OPERATORS:
            tokens.append(Token("punct", two, i, i + 2))
            i += 2
            continue

        tokens.append(Token("punct", ch, i, i + 1))
        i += 1

    return tokens


def token_at(tokens: list[Token], cursor: int) -> Token | None:
    """返回游标所处的词法片段。

    判定：start < cursor < end（严格位于片段内部）；
    对于不闭合片段（未闭合注释/字符串），cursor == end 也算处于其中。
    游标停在两个片段的边界上时返回 None。
    """
    for tok in tokens:
        if tok.start < cursor < tok.end:
            return tok
        if cursor == tok.end and tok.kind in ("string", "block_comment"):
            # 未闭合片段的 end 等于文本长度；闭合片段的游标可停在引号外侧
            if not tok.text.endswith(("'", '"', "*/")):
                return tok
    return None


def word_span_at(tokens: list[Token], cursor: int) -> tuple[int, int] | None:
    """若游标落在某个标识符（word/quoted）内部或边界上，返回其整段范围。

    边界规则：cursor 处于 [start, end] 即归属该标识符，
    从而词中游标替换整个词，不会留下右半截。
    """
    for tok in tokens:
        if tok.kind in ("word", "quoted") and tok.start <= cursor <= tok.end:
            return tok.start, tok.end
    return None
