"""把可删除的原子谓词在原 SQL 的对应位置替换为 TRUE。

删除一个原子谓词 = 仅把该 AST 节点（及其精确源码区段）替换为 TRUE，
周围括号、AND/OR 结合结构与其他谓词文本保持原样。
"""

from .errors import PlanError

TRUE_TOKEN = "TRUE"


def rewrite_remove_atoms(sql: str, atoms_to_remove) -> str:
    """atoms_to_remove: 需要替换为 TRUE 的 Comparison 节点列表。

    区段互不重叠；按位置从右向左替换，避免偏移失效。
    只删除完整原子比较表达式，绝不删除括号或 AND/OR 记号。
    """
    spans = []
    for atom in atoms_to_remove:
        if atom.start < 0 or atom.end <= atom.start:
            raise PlanError(f"原子谓词 {atom.sql_text!r} 缺少源码区段，无法安全删除")
        spans.append((atom.start, atom.end, atom))
    spans.sort(key=lambda x: x[0])
    for i in range(1, len(spans)):
        if spans[i][0] < spans[i - 1][1]:
            raise PlanError("待删除谓词的源码区段相互重叠，拒绝改写")
    out = sql
    for start, end, atom in reversed(spans):
        out = out[:start] + TRUE_TOKEN + out[end:]
    return out


def render_atom_removals(atoms) -> list[dict]:
    return [
        {
            "atom_index": a.atom_index,
            "sql": a.sql_text,
            "start": a.start,
            "end": a.end,
            "replacement": TRUE_TOKEN,
        }
        for a in atoms
    ]
