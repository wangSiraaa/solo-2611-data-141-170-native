"""Readable JSON artifacts: vocabulary, merge order, parameters, summary.

The artifacts are pure data — no executable code is stored.  ``precheck``
validates a directory of artifacts before it is used; contradictory merge
references (unknown ids, non-contiguous ranks, byte-sequence mismatches) fail
loudly with a description of every problem found.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .core import BASE_VOCAB_SIZE
from .trainer import STOP_REASONS

VOCAB_FILE = "vocab.json"
MERGES_FILE = "merges.json"
CONFIG_FILE = "config.json"
SUMMARY_FILE = "summary.json"

VOCAB_FORMAT = "bpe-vocab/1"
MERGES_FORMAT = "bpe-merges/1"
CONFIG_FORMAT = "bpe-config/1"
SUMMARY_FORMAT = "bpe-summary/1"

SUMMARY_INT_KEYS = (
    "rounds_completed",
    "vocab_size",
    "num_merges",
    "training_records",
    "unique_segments",
    "initial_tokens",
    "final_tokens",
    "initial_scan_positions",
    "positions_checked",
    "adjacency_updates",
    "candidates_evicted",
    "full_rescan_positions_equivalent",
)


class ArtifactError(ValueError):
    """Raised when artifacts are missing, malformed, or self-contradictory."""


@dataclass
class Model:
    symbols: list[bytes]
    merges: list[tuple[int, int, int]]  # (left_id, right_id, new_id) by rank
    config: dict
    summary: dict


# --------------------------------------------------------------------- save


def save_artifacts(
    directory: str | Path,
    *,
    symbols: list[bytes],
    merges: list[dict],
    config: dict,
    summary: dict,
) -> None:
    """Write the four readable JSON artifacts (no executable code)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    vocab = {
        "format": VOCAB_FORMAT,
        "symbols": [
            {"id": token_id, "hex": data.hex(), "length": len(data)}
            for token_id, data in enumerate(symbols)
        ],
    }
    merge_entries = []
    for merge in merges:
        left, right, new_id = merge["left"], merge["right"], merge["new_id"]
        merge_entries.append(
            {
                "rank": merge["rank"],
                "left": left,
                "right": right,
                "new_id": new_id,
                "left_hex": symbols[left].hex(),
                "right_hex": symbols[right].hex(),
                "new_hex": symbols[new_id].hex(),
                "weighted_count": merge["weighted_count"],
                "occurrences_replaced": merge["occurrences_replaced"],
            }
        )
    merges_doc = {"format": MERGES_FORMAT, "merges": merge_entries}
    config_doc = {"format": CONFIG_FORMAT, **config}

    _write_json(directory / VOCAB_FILE, vocab)
    _write_json(directory / MERGES_FILE, merges_doc)
    _write_json(directory / CONFIG_FILE, config_doc)
    _write_json(directory / SUMMARY_FILE, summary)


def _write_json(path: Path, doc: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


# ----------------------------------------------------------------- precheck


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def precheck(directory: str | Path) -> list[str]:
    """Validate an artifact directory; return a list of problems ([] == ok)."""
    directory = Path(directory)
    problems: list[str] = []
    docs: dict[str, dict] = {}
    for name, fmt in (
        (VOCAB_FILE, VOCAB_FORMAT),
        (MERGES_FILE, MERGES_FORMAT),
        (CONFIG_FILE, CONFIG_FORMAT),
        (SUMMARY_FILE, SUMMARY_FORMAT),
    ):
        path = directory / name
        if not path.is_file():
            problems.append(f"missing artifact file: {name}")
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="strict") as handle:
                doc = json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            problems.append(f"{name}: not valid strict-UTF-8 JSON: {exc}")
            continue
        if not isinstance(doc, dict):
            problems.append(f"{name}: top level must be a JSON object")
            continue
        if doc.get("format") != fmt:
            problems.append(f"{name}: expected format {fmt!r}, got {doc.get('format')!r}")
            continue
        docs[name] = doc
    if problems:
        return problems

    symbols = _check_vocab(docs[VOCAB_FILE], problems)
    merges = docs[MERGES_FILE]["merges"]
    if symbols is not None:
        _check_merges(merges, symbols, problems)
    _check_config(docs[CONFIG_FILE], problems)
    _check_summary(docs[SUMMARY_FILE], symbols, merges, problems)
    return problems


def _check_vocab(doc: dict, problems: list[str]) -> list[bytes] | None:
    entries = doc.get("symbols")
    if not isinstance(entries, list):
        problems.append("vocab.json: 'symbols' must be a list")
        return None
    symbols: list[bytes] = []
    seen: dict[bytes, int] = {}
    ok = True
    for expect_id, entry in enumerate(entries):
        where = f"vocab.json symbols[{expect_id}]"
        if not isinstance(entry, dict):
            problems.append(f"{where}: must be an object")
            ok = False
            continue
        token_id = entry.get("id")
        hex_bytes = entry.get("hex")
        if not _is_int(token_id) or token_id != expect_id:
            problems.append(f"{where}: id must be {expect_id}, got {token_id!r}")
            ok = False
            continue
        if not isinstance(hex_bytes, str):
            problems.append(f"{where}: 'hex' must be a string")
            ok = False
            continue
        try:
            data = bytes.fromhex(hex_bytes)
        except ValueError:
            problems.append(f"{where}: invalid hex {hex_bytes!r}")
            ok = False
            continue
        if expect_id < BASE_VOCAB_SIZE and data != bytes((expect_id,)):
            problems.append(
                f"{where}: base id {expect_id} must be the single byte {expect_id:02x}"
            )
            ok = False
            continue
        if data in seen:
            problems.append(
                f"{where}: duplicate byte sequence already used by id {seen[data]}"
            )
            ok = False
            continue
        if entry.get("length") != len(data):
            problems.append(f"{where}: length does not match hex")
            ok = False
            continue
        seen[data] = expect_id
        symbols.append(data)
    if len(entries) < BASE_VOCAB_SIZE:
        problems.append("vocab.json: fewer than 256 base symbols")
        ok = False
    return symbols if ok else None


def _check_merges(merges: object, symbols: list[bytes], problems: list[str]) -> None:
    if not isinstance(merges, list):
        problems.append("merges.json: 'merges' must be a list")
        return
    for expect_rank, merge in enumerate(merges):
        where = f"merges.json merges[{expect_rank}]"
        if not isinstance(merge, dict):
            problems.append(f"{where}: must be an object")
            continue
        rank = merge.get("rank")
        left = merge.get("left")
        right = merge.get("right")
        new_id = merge.get("new_id")
        if rank != expect_rank:
            problems.append(f"{where}: rank must be {expect_rank}, got {rank!r}")
            continue
        for name, value in (("left", left), ("right", right), ("new_id", new_id)):
            if not _is_int(value):
                problems.append(f"{where}: '{name}' must be an integer")
        if not all(_is_int(v) for v in (left, right, new_id)):
            continue
        if new_id != BASE_VOCAB_SIZE + expect_rank:
            problems.append(
                f"{where}: new_id must be {BASE_VOCAB_SIZE + expect_rank} "
                f"(first-creation order), got {new_id}"
            )
            continue
        if not (0 <= left < new_id and 0 <= right < new_id):
            problems.append(
                f"{where}: contradictory merge reference: left={left}, "
                f"right={right} must be existing ids < new_id={new_id}"
            )
            continue
        if new_id >= len(symbols):
            problems.append(f"{where}: new_id {new_id} not present in vocab.json")
            continue
        if symbols[left] + symbols[right] != symbols[new_id]:
            problems.append(
                f"{where}: contradictory merge reference: "
                f"bytes(left)+bytes(right) != bytes(new_id)"
            )
            continue
        if (
            merge.get("left_hex") != symbols[left].hex()
            or merge.get("right_hex") != symbols[right].hex()
            or merge.get("new_hex") != symbols[new_id].hex()
        ):
            problems.append(f"{where}: hex fields contradict vocab.json")
            continue
        if not _is_int(merge.get("weighted_count")) or merge["weighted_count"] < 1:
            problems.append(f"{where}: 'weighted_count' must be a positive integer")


def _check_config(doc: dict, problems: list[str]) -> None:
    merge_steps = doc.get("merge_steps")
    if not _is_int(merge_steps) or merge_steps < 0:
        problems.append("config.json: 'merge_steps' must be a non-negative integer")
    min_count = doc.get("min_weighted_pair_count")
    if not _is_int(min_count) or min_count < 1:
        problems.append(
            "config.json: 'min_weighted_pair_count' must be an integer >= 1"
        )


def _check_summary(
    doc: dict,
    symbols: list[bytes] | None,
    merges: object,
    problems: list[str],
) -> None:
    for key in SUMMARY_INT_KEYS:
        value = doc.get(key)
        if not _is_int(value) or value < 0:
            problems.append(f"summary.json: '{key}' must be a non-negative integer")
    if doc.get("stop_reason") not in STOP_REASONS:
        problems.append(f"summary.json: unknown stop_reason {doc.get('stop_reason')!r}")
    rounds = doc.get("rounds")
    if not isinstance(rounds, list):
        problems.append("summary.json: 'rounds' must be a list")
        rounds = None
    if problems:
        return
    if symbols is not None and doc["vocab_size"] != len(symbols):
        problems.append("summary.json: vocab_size contradicts vocab.json")
    if isinstance(merges, list):
        if doc["num_merges"] != len(merges):
            problems.append("summary.json: num_merges contradicts merges.json")
        if doc["rounds_completed"] != len(merges):
            problems.append("summary.json: rounds_completed contradicts merges.json")
    if rounds is not None and doc["rounds_completed"] != len(rounds):
        problems.append("summary.json: rounds_completed contradicts 'rounds'")
    if rounds is not None:
        replaced = 0
        for index, entry in enumerate(rounds):
            if not isinstance(entry, dict) or not _is_int(entry.get("occurrences_replaced")):
                problems.append(f"summary.json: rounds[{index}] lacks occurrences_replaced")
                return
            replaced += entry["occurrences_replaced"]
        if doc["final_tokens"] != doc["initial_tokens"] - replaced:
            problems.append(
                "summary.json: final_tokens contradicts initial_tokens and replacements"
            )


# --------------------------------------------------------------------- load


def load_model(directory: str | Path) -> Model:
    """Load artifacts after a full precheck; raise :class:`ArtifactError`."""
    problems = precheck(directory)
    if problems:
        raise ArtifactError(
            "invalid artifacts in "
            + str(directory)
            + ":\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )
    directory = Path(directory)
    with open(directory / VOCAB_FILE, "r", encoding="utf-8") as handle:
        vocab_doc = json.load(handle)
    with open(directory / MERGES_FILE, "r", encoding="utf-8") as handle:
        merges_doc = json.load(handle)
    with open(directory / CONFIG_FILE, "r", encoding="utf-8") as handle:
        config_doc = json.load(handle)
    with open(directory / SUMMARY_FILE, "r", encoding="utf-8") as handle:
        summary_doc = json.load(handle)
    symbols = [bytes.fromhex(entry["hex"]) for entry in vocab_doc["symbols"]]
    merges = [
        (entry["left"], entry["right"], entry["new_id"])
        for entry in merges_doc["merges"]
    ]
    return Model(symbols=symbols, merges=merges, config=config_doc, summary=summary_doc)
