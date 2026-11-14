"""sqlcomplete：面向编辑器的局部 SQL 补全核心（不连接数据库）。"""

from .completer import (
    Candidate,
    CompletionResult,
    complete,
    apply_completion,
)
from .lexer import tokenize

__all__ = [
    "Candidate",
    "CompletionResult",
    "complete",
    "apply_completion",
    "tokenize",
]
