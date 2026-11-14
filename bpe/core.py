"""Shared primitives: strict UTF-8, whitespace-run segmentation, symbol identity.

Pretokenization splits text into maximal runs of Unicode whitespace and
maximal runs of non-whitespace.  Every byte of the input belongs to exactly
one run, so the original text is preserved exactly; pairs are never merged
across run boundaries and no Unicode normalization is ever applied.
"""
from __future__ import annotations

BASE_VOCAB_SIZE = 256


def whitespace_run_spans(text: str) -> list[tuple[int, int]]:
    """Half-open character spans of maximal whitespace / non-whitespace runs.

    The spans tile ``[0, len(text))`` with no gaps, so nothing is dropped.
    """
    spans: list[tuple[int, int]] = []
    n = len(text)
    i = 0
    while i < n:
        want_space = text[i].isspace()
        j = i + 1
        while j < n and text[j].isspace() == want_space:
            j += 1
        spans.append((i, j))
        i = j
    return spans


def segment_byte_spans(data: bytes) -> list[tuple[int, int]]:
    """Half-open UTF-8 byte spans of the whitespace/non-whitespace runs.

    ``data`` must be valid UTF-8; ``UnicodeDecodeError`` propagates to the
    caller unchanged.  No normalization is performed.
    """
    text = data.decode("utf-8", errors="strict")
    spans: list[tuple[int, int]] = []
    byte_pos = 0
    for i, j in whitespace_run_spans(text):
        width = len(text[i:j].encode("utf-8"))
        spans.append((byte_pos, byte_pos + width))
        byte_pos += width
    return spans


class SymbolTable:
    """Token identity is the exact byte sequence.

    The base vocabulary is the 256 single bytes (id == byte value).  Whenever
    the same byte sequence would be produced through any path, the existing id
    is reused; genuinely new sequences take the next free id in first-creation
    order.
    """

    def __init__(self) -> None:
        self.symbols: list[bytes] = [bytes((i,)) for i in range(BASE_VOCAB_SIZE)]
        self._ids: dict[bytes, int] = {s: i for i, s in enumerate(self.symbols)}

    def __len__(self) -> int:
        return len(self.symbols)

    def bytes_of(self, token_id: int) -> bytes:
        return self.symbols[token_id]

    def intern(self, data: bytes) -> tuple[int, bool]:
        """Return ``(id, created)`` for ``data``, allocating a new id if unseen."""
        hit = self._ids.get(data)
        if hit is not None:
            return hit, False
        token_id = len(self.symbols)
        self.symbols.append(data)
        self._ids[data] = token_id
        return token_id, True
