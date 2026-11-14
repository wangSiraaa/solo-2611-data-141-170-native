"""规范化 JSON 与输入摘要。

计划与裁决都绑定三份输入的 SHA-256 摘要：源数据库一旦变化，旧裁决直接拒绝。
所有 JSON 落盘均使用相同的键排序与分隔符，保证摘要稳定、与行序无关。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# 行不存在的哨兵。null（None）与空字符串是真实值，绝不能用它们表示“整行缺失”。
MISSING = {"$missing": True}


def canonical_dumps(obj: Any) -> str:
    """与读取顺序无关的规范 JSON 文本（ensure_ascii=False 保留中文说明）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_of(obj: Any) -> str:
    return hashlib.sha256(canonical_dumps(obj).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
