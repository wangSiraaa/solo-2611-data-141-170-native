"""Rank-based BPE encoding and byte-exact decoding.

Encoding replays the learned merges in rank order — never a longest match
over the vocabulary.  Any byte is representable because the base vocabulary
is all 256 single bytes, so unseen text (new CJK characters, emoji, runs of
whitespace) simply falls back to byte tokens plus whichever learned merges
happen to apply.

Every token carries the half-open byte range ``[start, end)`` it covers in
the original UTF-8 encoding of the input; the ranges tile ``[0, len(data))``
with no gaps.  A token may cover only part of a multi-byte character — tokens
are byte strings, not necessarily valid Unicode.  Decoding concatenates the
token byte strings, so it reproduces the input exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core import segment_byte_spans


class DecodeError(ValueError):
    """Raised when token ids cannot be decoded with the loaded vocabulary."""


@dataclass(frozen=True)
class Token:
    id: int
    start: int  # half-open byte offset into the UTF-8 encoding of the record
    end: int


class BPECodec:
    """Encode and decode using only a vocabulary and an ordered merge list."""

    def __init__(self, symbols: list[bytes], merges: list[tuple[int, int, int]]) -> None:
        self.symbols = symbols
        # merges[rank] == (left_id, right_id, new_id); applied in rank order.
        self.merges = merges

    @classmethod
    def from_artifacts(cls, directory: str) -> "BPECodec":
        from .artifacts import load_model

        model = load_model(directory)
        return cls(model.symbols, model.merges)

    # --------------------------------------------------------------- encode

    def encode_bytes(self, data: bytes) -> list[Token]:
        """Encode raw UTF-8 bytes.  ``UnicodeDecodeError`` propagates."""
        tokens: list[Token] = []
        for span_start, span_end in segment_byte_spans(data):
            run = [
                (byte, span_start + k, span_start + k + 1)
                for k, byte in enumerate(data[span_start:span_end])
            ]
            for left, right, new_id in self.merges:
                if len(run) < 2:
                    break
                run = self._merge_pass(run, left, right, new_id)
            tokens.extend(Token(id=t[0], start=t[1], end=t[2]) for t in run)
        return tokens

    def encode_text(self, text: str) -> list[Token]:
        """Encode a string; lone surrogates raise ``UnicodeEncodeError``."""
        return self.encode_bytes(text.encode("utf-8"))

    def encode_records(self, texts: list[str]) -> list[list[Token]]:
        """Encode records independently; batching never changes the result."""
        return [self.encode_text(text) for text in texts]

    @staticmethod
    def _merge_pass(
        run: list[tuple[int, int, int]], left: int, right: int, new_id: int
    ) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        i = 0
        n = len(run)
        while i < n:
            if (
                i + 1 < n
                and run[i][0] == left
                and run[i + 1][0] == right
            ):
                out.append((new_id, run[i][1], run[i + 1][2]))
                i += 2
            else:
                out.append(run[i])
                i += 1
        return out

    # --------------------------------------------------------------- decode

    def decode_ids_to_bytes(self, ids: list[int]) -> bytes:
        parts = []
        for token_id in ids:
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < len(self.symbols)
            ):
                raise DecodeError(f"unknown token id: {token_id!r}")
            parts.append(self.symbols[token_id])
        return b"".join(parts)

    def decode_ids(self, ids: list[int]) -> str:
        """Decode token ids to text; invalid UTF-8 raises ``UnicodeDecodeError``."""
        return self.decode_ids_to_bytes(ids).decode("utf-8", errors="strict")
