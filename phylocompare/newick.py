"""最小 Newick 解析器。

只实现本题需要的子集：每行一棵树、三叉根表示无根二叉树、
节点名为裸标识符、枝长为浮点数、以分号结尾。
不依赖第三方 Newick 库，解析过程确定且可独立测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import CompareError


@dataclass
class Node:
    name: str | None = None
    length: float | None = None
    children: list["Node"] = field(default_factory=list)
    parent: "Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return not self.children


class _Cursor:
    def __init__(self, text: str, where: str) -> None:
        self.text = text
        self.pos = 0
        self.where = where

    def peek(self) -> str | None:
        return self.text[self.pos] if self.pos < len(self.text) else None

    def take(self) -> str:
        ch = self.text[self.pos]
        self.pos += 1
        return ch

    def expect(self, ch: str) -> None:
        if self.peek() != ch:
            raise CompareError(
                f"{self.where}: 位置 {self.pos} 处应为 {ch!r}，实际为 {self.peek()!r}"
            )
        self.pos += 1

    def read_token(self, delimiters: str) -> str:
        start = self.pos
        while self.peek() is not None and self.peek() not in delimiters:
            self.pos += 1
        return self.text[start:self.pos].strip()


def _parse_node(cur: _Cursor, parent: Node | None = None) -> Node:
    node = Node(parent=parent)
    if cur.peek() == "(":
        cur.take()
        while True:
            node.children.append(_parse_node(cur, node))
            sep = cur.peek()
            if sep == ",":
                cur.take()
                continue
            if sep == ")":
                cur.take()
                break
            raise CompareError(
                f"{cur.where}: 位置 {cur.pos} 处应为 ',' 或 ')'，实际为 {sep!r}"
            )

    label = cur.read_token("(),;")
    if ":" in label:
        name_part, _, length_part = label.partition(":")
        node.name = name_part.strip() or None
        length_part = length_part.strip()
        try:
            node.length = float(length_part)
        except ValueError as exc:
            raise CompareError(
                f"{cur.where}: 枝长无法解析为浮点数: {length_part!r}"
            ) from exc
        if not (node.length == node.length) or node.length in (
            float("inf"),
            float("-inf"),
        ):
            raise CompareError(f"{cur.where}: 枝长必须是有限数，得到 {length_part!r}")
    else:
        node.name = label or None
    return node


def parse_newick(text: str, source: str = "<newick>") -> list[Node]:
    """解析包含一棵或多棵树的 Newick 文本，返回各树的根节点。"""
    trees: list[Node] = []
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        where = f"{source}:{line_no}"
        if not line.endswith(";"):
            raise CompareError(f"{where}: 树声明必须以 ';' 结尾")
        cur = _Cursor(line[:-1], where)
        root = _parse_node(cur)
        if cur.peek() is not None:
            raise CompareError(f"{where}: 位置 {cur.pos} 之后存在多余记号")
        trees.append(root)
    if not trees:
        raise CompareError(f"{source}: 没有解析到任何树")
    return trees
