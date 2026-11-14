"""Corpus and configuration input validation.

All failures raise :class:`CorpusError` with the file and line number so the
CLI can report them clearly.  Files are read with strict UTF-8; invalid byte
sequences surface as ``UnicodeDecodeError``.
"""
from __future__ import annotations

import json
from pathlib import Path


class CorpusError(ValueError):
    """Raised for malformed training corpora, record files, or configs."""


def _validate_text(obj: object, where: str) -> str:
    if not isinstance(obj, dict):
        raise CorpusError(f"{where}: expected a JSON object")
    text = obj.get("text")
    if not isinstance(text, str):
        raise CorpusError(f"{where}: 'text' must be a string")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CorpusError(f"{where}: 'text' is not valid UTF-8 ({exc})") from exc
    return text


def read_corpus(path: str | Path) -> list[tuple[str, int]]:
    """Read a JSONL training corpus of ``{"text": str, "count": positive int}``.

    The file is decoded with strict UTF-8; ``UnicodeDecodeError`` propagates.
    Records with an empty ``text`` are legal and simply contribute no segments.
    """
    records: list[tuple[str, int]] = []
    with open(path, "r", encoding="utf-8", errors="strict") as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            where = f"{path}:{lineno}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{where}: invalid JSON: {exc}") from exc
            text = _validate_text(obj, where)
            count = obj.get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise CorpusError(f"{where}: 'count' must be a positive integer")
            records.append((text, count))
    return records


def read_text_records(path: str | Path) -> list[str]:
    """Read a JSONL file of ``{"text": str}`` records for batch encoding."""
    texts: list[str] = []
    with open(path, "r", encoding="utf-8", errors="strict") as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            where = f"{path}:{lineno}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{where}: invalid JSON: {exc}") from exc
            texts.append(_validate_text(obj, where))
    return texts


def read_config(path: str | Path) -> dict:
    """Read a training configuration JSON object."""
    with open(path, "r", encoding="utf-8", errors="strict") as handle:
        try:
            obj = json.load(handle)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise CorpusError(f"{path}: config must be a JSON object")
    return obj


def _check_non_negative_int(value: object, name: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorpusError(f"{where}: '{name}' must be a non-negative integer")
    return value


def params_from_config(config: dict, where: str = "config") -> "TrainParams":
    """Build validated training parameters from a config mapping."""
    from .trainer import TrainParams

    if "merge_steps" not in config:
        raise CorpusError(f"{where}: 'merge_steps' is required")
    merge_steps = _check_non_negative_int(config["merge_steps"], "merge_steps", where)
    min_count = config.get("min_weighted_pair_count", 1)
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise CorpusError(f"{where}: 'min_weighted_pair_count' must be an integer >= 1")
    return TrainParams(merge_steps=merge_steps, min_weighted_pair_count=min_count)
