"""SQL 词法分析器。

只产出本工具支持的记号；不属于受限语法词法集合的输入直接拒绝。
支持：普通/双引号标识符、单引号字符串、数字、命名/位置绑定参数、
括号、比较与逻辑运算符、空白与 SQL 行/块注释。
"""

from dataclasses import dataclass

from .errors import UnsupportedSyntaxError

KEYWORDS = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "INNER", "LEFT",
    "OUTER", "JOIN", "ON", "AS", "NULL", "TRUE", "FALSE",
}
# 出现即说明查询超出本工具能力范围的词法关键字（聚合、子查询等）。
FORBIDDEN_KEYWORDS = {
    "GROUP", "HAVING", "ORDER", "LIMIT", "OFFSET", "UNION", "EXCEPT",
    "INTERSECT", "WITH", "DISTINCT", "ALL", "EXISTS", "BETWEEN", "LIKE",
    "IN", "IS", "CASE", "WHEN", "THEN", "ELSE", "END", "CAST",
}


@dataclass(frozen=True)
class Token:
    kind: str          # KEYWORD / IDENT / STRING / NUMBER / PARAM / OP / EOF
    text: str
    upper: str
    pos: int
    end: int
    line: int
    col: int


class Tokenizer:
    def __init__(self, sql: str):
        self.sql = sql
        self.i = 0
        self.line = 1
        self.col = 1

    def _advance(self, n: int = 1) -> str:
        chunk = self.sql[self.i:self.i + n]
        for ch in chunk:
            if ch == "\n":
                self.line += 1
                self.col = 1
            else:
                self.col += 1
        self.i += n
        return chunk

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        sql = self.sql
        n = len(sql)
        while self.i < n:
            ch = sql[self.i]
            if ch.isspace():
                self._advance()
                continue
            if ch == "-" and self.i + 1 < n and sql[self.i + 1] == "-":
                while self.i < n and sql[self.i] != "\n":
                    self._advance()
                continue
            if ch == "/" and self.i + 1 < n and sql[self.i + 1] == "*":
                self._advance(2)
                while self.i + 1 < n and not (sql[self.i] == "*" and sql[self.i + 1] == "/"):
                    self._advance()
                if self.i + 1 >= n:
                    raise UnsupportedSyntaxError("块注释缺少结束符 */")
                self._advance(2)
                continue
            start, sline, scol = self.i, self.line, self.col
            if ch == '"':
                tokens.append(self._read_quoted_ident(start, sline, scol))
                continue
            if ch == "'":
                tokens.append(self._read_string(start, sline, scol))
                continue
            if ch.isdigit() or (ch == "." and self.i + 1 < n and sql[self.i + 1].isdigit()):
                tokens.append(self._read_number(start, sline, scol))
                continue
            if ch.isalpha() or ch == "_":
                tokens.append(self._read_word(start, sline, scol))
                continue
            if ch in (":", "@", "?"):
                tokens.append(self._read_param(ch, start, sline, scol))
                continue
            if ch in "(),.=":
                self._advance()
                tokens.append(Token("OP", ch, ch, start, self.i, sline, scol))
                continue
            if ch in "<>!":
                tokens.append(self._read_compare_op(start, sline, scol))
                continue
            raise UnsupportedSyntaxError(
                f"不支持的字符 {ch!r}（位置 {sline}:{scol}）；本工具只接受普通 SELECT、"
                "等值连接与由比较及 AND/OR/括号组成的 WHERE"
            )
        tokens.append(Token("EOF", "", "", n, n, self.line, self.col))
        return tokens

    def _read_quoted_ident(self, start, sline, scol) -> Token:
        self._advance()  # 开引号
        buf = []
        sql = self.sql
        while True:
            if self.i >= len(sql):
                raise UnsupportedSyntaxError("双引号标识符缺少结束引号")
            ch = sql[self.i]
            if ch == '"':
                if self.i + 1 < len(sql) and sql[self.i + 1] == '"':
                    buf.append('"')
                    self._advance(2)
                    continue
                self._advance()
                break
            buf.append(ch)
            self._advance()
        text = "".join(buf)
        return Token("IDENT", text, text.upper(), start, self.i, sline, scol)

    def _read_string(self, start, sline, scol) -> Token:
        self._advance()  # 开引号
        buf = []
        sql = self.sql
        while True:
            if self.i >= len(sql):
                raise UnsupportedSyntaxError("字符串字面量缺少结束单引号")
            ch = sql[self.i]
            if ch == "'":
                if self.i + 1 < len(sql) and sql[self.i + 1] == "'":
                    buf.append("'")
                    self._advance(2)
                    continue
                self._advance()
                break
            buf.append(ch)
            self._advance()
        return Token("STRING", "'" + "".join(buf) + "'", "".join(buf),
                     start, self.i, sline, scol)

    def _read_number(self, start, sline, scol) -> Token:
        sql = self.sql
        seen_dot = False
        while self.i < len(sql):
            ch = sql[self.i]
            if ch.isdigit():
                self._advance()
            elif ch == "." and not seen_dot:
                seen_dot = True
                self._advance()
            else:
                break
        text = sql[start:self.i]
        if text == ".":
            raise UnsupportedSyntaxError(f"非法数字字面量（位置 {sline}:{scol}）")
        return Token("NUMBER", text, text, start, self.i, sline, scol)

    def _read_word(self, start, sline, scol) -> Token:
        sql = self.sql
        while self.i < len(sql) and (sql[self.i].isalnum() or sql[self.i] == "_"):
            self._advance()
        text = sql[start:self.i]
        upper = text.upper()
        if upper in FORBIDDEN_KEYWORDS:
            raise UnsupportedSyntaxError(
                f"不支持的关键字 {upper}（位置 {sline}:{scol}）；本工具不含聚合、子查询、"
                "GROUP BY、IN/BETWEEN/LIKE/IS 等能力"
            )
        kind = "KEYWORD" if upper in KEYWORDS else "IDENT"
        return Token(kind, text, upper, start, self.i, sline, scol)

    def _read_param(self, marker, start, sline, scol) -> Token:
        sql = self.sql
        if marker == "?":
            self._advance()
            return Token("PARAM", "?", "?", start, self.i, sline, scol)
        self._advance()
        name_start = self.i
        while self.i < len(sql) and (sql[self.i].isalnum() or sql[self.i] == "_"):
            self._advance()
        if self.i == name_start:
            raise UnsupportedSyntaxError(f"绑定参数 {marker} 后缺少名字（位置 {sline}:{scol}）")
        name = sql[name_start:self.i]
        text = marker + name
        return Token("PARAM", text, text, start, self.i, sline, scol)

    def _read_compare_op(self, start, sline, scol) -> Token:
        sql = self.sql
        two = sql[self.i:self.i + 2]
        if two in ("<=", ">=", "<>", "!="):
            self._advance(2)
            return Token("OP", two, two, start, self.i, sline, scol)
        ch = sql[self.i]
        if ch in "<>":
            self._advance()
            return Token("OP", ch, ch, start, self.i, sline, scol)
        raise UnsupportedSyntaxError(f"不支持的运算符 {ch!r}（位置 {sline}:{scol}）")
