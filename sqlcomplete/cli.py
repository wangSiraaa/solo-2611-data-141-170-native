"""命令行入口。

用法：
  python -m sqlcomplete.cli complete <catalog.json> <sql> <cursor>
  python -m sqlcomplete.cli fixtures <catalog.json> <cases.json>

complete 输出单条补全的 JSON（含应用后的字符串，便于核对替换范围）；
fixtures 批量跑 cases.json，逐条输出候选与替换范围。
"""

from __future__ import annotations

import json
import sys

from .completer import complete, apply_completion


def _load(path: str) -> dict | list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cmd_complete(argv: list[str]) -> int:
    catalog_path, sql, cursor_str = argv
    catalog = _load(catalog_path)
    cursor = int(cursor_str)
    result = complete(catalog, sql, cursor)
    payload = result.to_dict()
    payload["applied"] = apply_completion(sql, result)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_fixtures(argv: list[str]) -> int:
    catalog_path, cases_path = argv
    catalog = _load(catalog_path)
    cases = _load(cases_path)
    report = []
    for case in cases:
        sql, cursor = case["sql"], case["cursor"]
        result = complete(catalog, sql, cursor)
        report.append({
            "case_id": case["case_id"],
            **result.to_dict(),
            "applied_first": apply_completion(sql, result),
        })
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 1
    command, rest = argv[0], argv[1:]
    if command == "complete" and len(rest) == 3:
        return cmd_complete(rest)
    if command == "fixtures" and len(rest) == 2:
        return cmd_fixtures(rest)
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
