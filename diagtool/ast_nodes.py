"""受限 SELECT 的 AST 节点定义。

表达式只有四种：列引用、字面量、绑定参数、二元比较；
布尔结构只有 AND / OR 与括号（括号在递归下降解析中天然保留）。
每个节点带解析期分配的稳定 id 与源码位置，证据用它定位到具体 AST 位置。
"""

from dataclasses import dataclass, field
from typing import Optional

COMPARE_OPS = ("=", "<>", "!=", "<", "<=", ">", ">=")


@dataclass(frozen=True)
class ColumnRef:
    table_alias: Optional[str]
    name: str
    node_id: str
    start: int = -1
    end: int = -1

    @property
    def kind(self) -> str:
        return "column"


@dataclass(frozen=True)
class Literal:
    value: object          # int / float / str / None / bool
    raw: str                # 原始 SQL 文本
    node_id: str
    start: int = -1
    end: int = -1

    @property
    def kind(self) -> str:
        return "literal"


@dataclass(frozen=True)
class Param:
    raw: str                # 如 :min_amount；? 位置参数由解析器拒绝
    name: Optional[str]
    node_id: str
    start: int = -1
    end: int = -1

    @property
    def kind(self) -> str:
        return "param"


@dataclass(frozen=True)
class Comparison:
    op: str
    left: object
    right: object
    sql_text: str
    node_id: str
    start: int = -1
    end: int = -1
    atom_index: int = -1    # 在 WHERE 原子谓词中的序号（解析后回填）
    is_constant: bool = False  # TRUE/FALSE 常量原子（改写产物）

    @property
    def kind(self) -> str:
        return "comparison"


@dataclass(frozen=True)
class BoolOp:
    operator: str           # AND / OR
    left: object
    right: object
    node_id: str

    @property
    def kind(self) -> str:
        return "boolop"


@dataclass(frozen=True)
class TableRef:
    name: str
    alias: Optional[str]
    node_id: str
    start: int = -1
    end: int = -1


@dataclass(frozen=True)
class Join:
    join_type: str          # INNER / LEFT
    right: TableRef
    left_col: ColumnRef     # ON 左列
    right_col: ColumnRef    # ON 右列
    on_text: str
    node_id: str


@dataclass(frozen=True)
class SelectItem:
    expression: ColumnRef   # 本工具只允许纯列引用
    alias: Optional[str]
    raw_text: str


@dataclass(frozen=True)
class Select:
    items: tuple[SelectItem, ...]
    base_table: TableRef
    joins: tuple[Join, ...]
    where: Optional[object]
    original_sql: str
    atom_by_index: dict = field(default_factory=dict)
    where_start: int = -1
    where_end: int = -1


def walk_expr(node):
    """深度优先遍历表达式节点。"""
    yield node
    if node.kind == "comparison":
        yield from walk_expr(node.left)
        yield from walk_expr(node.right)
    elif node.kind == "boolop":
        yield from walk_expr(node.left)
        yield from walk_expr(node.right)


def collect_atoms(where) -> list[Comparison]:
    """按从左到右的源码顺序收集 WHERE 中的原子比较谓词。

    改写产生的 TRUE/FALSE 常量原子（is_constant）不是可诊断/可删除的谓词，
    不计入原子集合。
    """
    if where is None:
        return []
    return [n for n in walk_expr(where)
            if n.kind == "comparison" and not n.is_constant]


def collect_params(node) -> list[Param]:
    if node is None:
        return []
    return [n for n in walk_expr(node) if n.kind == "param"]
