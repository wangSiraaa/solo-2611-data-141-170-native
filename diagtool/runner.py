"""真实 SQL 重跑与结果行多重集比较。

所有验证都直接交给 SQLite 执行改写后的 SQL；Python 侧的求值仅用于解释，
不作为方案有效性的最终依据。
"""

from collections import Counter

from .errors import UnsupportedSyntaxError


def execute_rows(db, sql: str, bindings: dict) -> list[dict]:
    """执行查询，返回 [{列名: 值}, ...]，保留列名与重复行。"""
    try:
        cur = db.conn.execute(sql, bindings)
    except Exception as exc:  # sqlite3.Error
        raise UnsupportedSyntaxError(f"验证 SQL 执行失败: {exc}") from exc
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def multiset_counter(rows: list[dict], columns: list[str]) -> Counter:
    def key(row):
        return tuple(_hashable(row.get(c)) for c in columns)
    return Counter(key(r) for r in rows)


def _hashable(v):
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v)
    return v


def diff_multisets(old_rows, new_rows, columns):
    """返回新增/减少的完整结果行及各自出现次数。"""
    old = multiset_counter(old_rows, columns)
    new = multiset_counter(new_rows, columns)
    added = []
    removed = []
    for key in new:
        delta = new[key] - old.get(key, 0)
        if delta > 0:
            added.append({"row": dict(zip(columns, key)), "count": delta})
    for key in old:
        delta = old[key] - new.get(key, 0)
        if delta > 0:
            removed.append({"row": dict(zip(columns, key)), "count": delta})
    return {
        "added": _sorted_rows(added, columns),
        "removed": _sorted_rows(removed, columns),
        "added_row_count": sum(x["count"] for x in added),
        "removed_row_count": sum(x["count"] for x in removed),
        "old_row_count": sum(old.values()),
        "new_row_count": sum(new.values()),
    }


def _sorted_rows(items, columns):
    def sort_key(item):
        return tuple((v is None, str(type(v)), repr(v))
                     for v in (item["row"][c] for c in columns))
    return sorted(items, key=sort_key)


def identity_rows(rows, identity_column):
    """把结果行按身份列聚合成 (目标 -> 出现次数)。"""
    counts = Counter()
    for row in rows:
        counts[row.get(identity_column)] += 1
    return counts
