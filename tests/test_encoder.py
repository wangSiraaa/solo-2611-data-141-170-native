"""Encoder/decoder unit tests: rank order, offsets, fallback, round-trip."""
from __future__ import annotations

import unittest

from bpe.core import SymbolTable
from bpe.encoder import BPECodec, DecodeError


def make_codec(extra_symbols=(), merges=()):
    symbols = [bytes([i]) for i in range(256)] + list(extra_symbols)
    return BPECodec(symbols, list(merges))


class RankOrderTest(unittest.TestCase):
    def test_rank_order_is_not_longest_match(self):
        # Trained on "aaa": merge (a,a) first, then (aa,a).  A longest-match
        # tokenizer would emit [aaa][a] for "aaaa"; rank order gives [aa][aa].
        codec = make_codec(
            extra_symbols=[b"aa", b"aaa"],
            merges=[(97, 97, 256), (256, 97, 257)],
        )
        tokens = codec.encode_text("aaaa")
        self.assertEqual([t.id for t in tokens], [256, 256])
        self.assertEqual([(t.start, t.end) for t in tokens], [(0, 2), (2, 4)])

    def test_merges_applied_in_rank_sequence(self):
        # (b,c) outranks (a,bc): "abc" must become [a][bc], not [ab? ...].
        codec = make_codec(
            extra_symbols=[b"bc", b"abc"],
            merges=[(98, 99, 256), (97, 256, 257)],
        )
        self.assertEqual([t.id for t in codec.encode_text("abc")], [257])
        self.assertEqual([t.id for t in codec.encode_text("ab")], [97, 98])
        self.assertEqual([t.id for t in codec.encode_text("bcbc")], [256, 256])


class OffsetAndFallbackTest(unittest.TestCase):
    def assert_tiling(self, codec, text):
        data = text.encode("utf-8")
        tokens = codec.encode_text(text)
        cursor = 0
        for token in tokens:
            self.assertEqual(token.start, cursor)
            self.assertGreater(token.end, token.start)
            self.assertEqual(data[token.start:token.end], codec.symbols[token.id])
            cursor = token.end
        self.assertEqual(cursor, len(data))
        return tokens

    def test_unseen_bytes_fall_back_to_single_bytes(self):
        codec = make_codec()  # no merges at all
        tokens = self.assert_tiling(codec, "汉字🙂🙂  café")
        self.assertTrue(all(t.end - t.start == 1 for t in tokens))
        self.assertEqual(codec.decode_ids([t.id for t in tokens]), "汉字🙂🙂  café")

    def test_token_may_split_a_multibyte_character(self):
        # Merge only the first two bytes of 汉 (E6 B1 89).
        codec = make_codec(extra_symbols=[b"\xe6\xb1"], merges=[(0xE6, 0xB1, 256)])
        tokens = codec.encode_text("汉")
        self.assertEqual([(t.id, t.start, t.end) for t in tokens],
                         [(256, 0, 2), (0x89, 2, 3)])
        # the merged token alone is not valid Unicode, and that is fine
        self.assertRaises(UnicodeDecodeError, codec.symbols[256].decode, "utf-8")
        self.assertEqual(codec.decode_ids([t.id for t in tokens]), "汉")

    def test_whitespace_runs_are_preserved(self):
        codec = make_codec(extra_symbols=[b"  "], merges=[(0x20, 0x20, 256)])
        tokens = self.assert_tiling(codec, "a \t a  a\n\nb")
        self.assertEqual(codec.decode_ids([t.id for t in tokens]), "a \t a  a\n\nb")
        # the double space merged, the tab/newline runs did not cross runs
        self.assertIn((256, 5, 7), [(t.id, t.start, t.end) for t in tokens])

    def test_combining_characters_round_trip(self):
        codec = make_codec()
        text = "café café́"  # second é is e + combining acute
        tokens = self.assert_tiling(codec, text)
        self.assertEqual(codec.decode_ids([t.id for t in tokens]), text)

    def test_empty_input(self):
        codec = make_codec()
        self.assertEqual(codec.encode_text(""), [])
        self.assertEqual(codec.decode_ids([]), "")

    def test_invalid_utf8_fails(self):
        codec = make_codec()
        with self.assertRaises(UnicodeDecodeError):
            codec.encode_bytes(b"abc\xff\xfe")

    def test_batch_encoding_is_record_independent(self):
        codec = make_codec(extra_symbols=[b"aa"], merges=[(97, 97, 256)])
        records = ["aaaa", "a a", "", "汉字", "aa"]
        batched = codec.encode_records(records)
        one_by_one = [codec.encode_text(t) for t in records]
        self.assertEqual(batched, one_by_one)
        shuffled = codec.encode_records([records[2], records[0], records[4]])
        self.assertEqual(shuffled, [batched[2], batched[0], batched[4]])


class DecodeTest(unittest.TestCase):
    def test_unknown_token_id_fails(self):
        codec = make_codec()
        with self.assertRaises(DecodeError):
            codec.decode_ids([256])
        with self.assertRaises(DecodeError):
            codec.decode_ids([-1])
        with self.assertRaises(DecodeError):
            codec.decode_ids([True])

    def test_symbol_identity_reuse(self):
        table = SymbolTable()
        first, created = table.intern(b"ab")
        self.assertTrue(created)
        again, created = table.intern(b"ab")
        self.assertFalse(created)
        self.assertEqual(first, again)
        other, created = table.intern(b"abc")
        self.assertTrue(created)
        self.assertEqual(other, first + 1)


if __name__ == "__main__":
    unittest.main()
