"""SQL 比较与三值逻辑语义。

本模块在 Python 中复现 SQLite 的比较语义，用于逐行解释 WHERE：
  * 任一操作数为 NULL 时，比较结果为 UNKNOWN（Python None）；
  * 比较前按操作数来源套用列的类型亲和性（TEXT/NUMERIC/INTEGER/REAL/BLOB）；
  * 不同存储类之间按 SQLite 的存储类次序比较。
基线结果会用真实 SQLite 全量查询做一致性核对，见 analysis.py。
"""

import re

_NUMERIC_FAMILY = {"INTEGER", "NUMERIC", "REAL"}
_INT_RE = re.compile(r"^[+-]?\d+$")
_REAL_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")

NUM = 1
TEXT = 2
BLOB = 3


def affinity_of_type(declared_type: str) -> str:
    t = (declared_type or "").upper()
    if "INT" in t:
        return "INTEGER"
    if "CHAR" in t or "CLOB" in t or "TEXT" in t:
        return "TEXT"
    if "BLOB" in t or t.strip() == "":
        return "BLOB"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "NUMERIC"


def storage_class(value) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return NUM
    if isinstance(value, str):
        return TEXT
    if isinstance(value, (bytes, bytearray, memoryview)):
        return BLOB
    return TEXT


def _to_numeric(value):
    if not isinstance(value, str):
        return value
    s = value.strip()
    if _INT_RE.match(s):
        try:
            return int(s)
        except ValueError:
            return value
    if _REAL_RE.match(s):
        try:
            return float(s)
        except ValueError:
            return value
    return value


def _to_text(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # 与 SQLite 的 REAL->TEXT 保持一致：整数值保留 ".0"，科学计数法用
        # 形如 "1.0e+20" 的格式。SQLite 使用 "%!.15g"；这里用 repr 并做必要修正。
        s = repr(value)
        if "e" in s and "." not in s.split("e")[0]:
            mant, exp = s.split("e")
            s = mant + ".0e" + exp
        return s
    return value


def coerce_pair(v1, a1, v2, a2):
    """按 SQLite 亲和性规则对比较双方做一次转换。a1/a2 为 'NONE' 表示无亲和性。"""
    if a1 in _NUMERIC_FAMILY and a2 in ("TEXT", "BLOB", "NONE"):
        v2 = _to_numeric(v2)
    elif a1 == "TEXT" and a2 == "NONE":
        v2 = _to_text(v2)
    if a2 in _NUMERIC_FAMILY and a1 in ("TEXT", "BLOB", "NONE"):
        v1 = _to_numeric(v1)
    elif a2 == "TEXT" and a1 == "NONE":
        v1 = _to_text(v1)
    return v1, v2


def _order_key(value):
    cls = storage_class(value)
    if cls == NUM:
        return (cls, float(value))
    if cls == TEXT:
        return (cls, value.encode("utf-8"))
    if cls == BLOB:
        return (cls, bytes(value))
    return (0, 0)


def _compare_order(v1, v2) -> int:
    # INTEGER 与 REAL 混合：按精确数值比较，避免 float 往返丢精度
    # （例如 REAL 列存 1.2345678901234568e17 与整数 123456789012345678 不等）。
    if _is_num(v1) and _is_num(v2):
        if isinstance(v1, float) or isinstance(v2, float):
            exact = _cmp_int_real(v1, v2)
            if exact is not None:
                return exact
        a, b = float(v1), float(v2)
        return (a > b) - (a < b)
    k1, k2 = _order_key(v1), _order_key(v2)
    if k1 < k2:
        return -1
    if k1 > k2:
        return 1
    return 0


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _cmp_int_real(a, b):
    """精确比较 INTEGER 与 REAL；返回 -1/0/1，无法精确判定时返回 None。"""
    if isinstance(a, float) and isinstance(b, float):
        return (a > b) - (a < b)
    if isinstance(a, int) and isinstance(b, float):
        return _cmp_int_float(a, b)
    if isinstance(a, float) and isinstance(b, int):
        r = _cmp_int_float(b, a)
        return None if r is None else -r
    return None


def _cmp_int_float(i, f):
    if f != f:  # NaN
        return None
    if f in (float("inf"), float("-inf")):
        return -1 if f == float("inf") else 1
    fi = f.__trunc__()
    # 先比较截断后的整数部分，再比较小数部分，全程精确。
    if i < fi:
        return -1
    if i > fi:
        return 1
    if f == fi:
        return 0
    return -1 if 0 < (f - fi) else 1


def sql_compare(v1, aff1, v2, aff2, op: str):
    """返回 True / False / None（UNKNOWN）。"""
    if v1 is None or v2 is None:
        return None
    v1, v2 = coerce_pair(v1, aff1, v2, aff2)
    c = _compare_order(v1, v2)
    if op == "=":
        return c == 0
    if op in ("<>", "!="):
        return c != 0
    if op == "<":
        return c < 0
    if op == "<=":
        return c <= 0
    if op == ">":
        return c > 0
    if op == ">=":
        return c >= 0
    raise ValueError(f"未知比较运算符 {op}")


def tri_and(a, b):
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return True


def tri_or(a, b):
    if a is True or b is True:
        return True
    if a is None or b is None:
        return None
    return False


def tri_label(value) -> str:
    if value is True:
        return "TRUE"
    if value is False:
        return "FALSE"
    return "UNKNOWN"
