"""Shared test helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
TINY = Path(__file__).resolve().parent / "fixtures" / "tiny"
EDGE = ROOT / "examples" / "edge-cases"


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def base_tiny_questionnaire() -> Dict[str, Any]:
    """Very small declared domain: consent yes/no, age only 0..3."""
    return {
        "version": "tiny-1",
        "household_fields": {"consent": {"type": "enum", "values": ["yes", "no"]}},
        "member_questions": {
            "age": {"type": "integer", "minimum": 0, "maximum": 3},
            "schooling": {"type": "enum", "values": ["yes", "no"]},
            "pension": {"type": "enum", "values": ["yes", "no"]},
        },
        "start": "gate",
        "nodes": {
            "gate": {
                "kind": "branch",
                "test": {"ref": "household.consent", "op": "eq", "value": "yes"},
                "true": "q_age",
                "false": "end",
            },
            "q_age": {"kind": "ask", "question": "age", "next": "adult"},
            "adult": {
                "kind": "branch",
                "test": {"ref": "member.age", "op": "ge", "value": 2},
                "true": "old",
                "false": "q_school",
            },
            "q_school": {"kind": "ask", "question": "schooling", "next": "old"},
            "old": {
                "kind": "branch",
                "test": {"ref": "member.age", "op": "ge", "value": 3},
                "true": "q_pension",
                "false": "end",
            },
            "q_pension": {"kind": "ask", "question": "pension", "next": "end"},
            "end": {"kind": "end"},
        },
    }
