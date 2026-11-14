"""Corpus/config validation and full-corpus training sanity."""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from bpe.corpus import CorpusError, params_from_config, read_config, read_corpus
from bpe.trainer import IncrementalBPETrainer, TrainParams


class CorpusValidationTest(unittest.TestCase):
    def write(self, tmp: Path, content: bytes) -> Path:
        path = tmp / "corpus.jsonl"
        path.write_bytes(content)
        return path

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_invalid_utf8_fails(self):
        path = self.write(self.tmp, b'{"text": "ok\xff", "count": 1}\n')
        with self.assertRaises(UnicodeDecodeError):
            read_corpus(path)

    def test_lone_surrogate_fails(self):
        line = json.dumps({"text": "\ud800", "count": 1})
        path = self.write(self.tmp, (line + "\n").encode("utf-8"))
        with self.assertRaises(CorpusError):
            read_corpus(path)

    def test_bad_counts_fail(self):
        for bad in (0, -3, 1.5, "2", True, None):
            line = json.dumps({"text": "a", "count": bad})
            path = self.write(self.tmp, (line + "\n").encode("utf-8"))
            with self.assertRaises(CorpusError, msg=f"count={bad!r}"):
                read_corpus(path)

    def test_missing_fields_and_bad_json_fail(self):
        for raw in (
            '{"count": 1}',
            '{"text": 3, "count": 1}',
            '[1, 2]',
            '{"text": "a", "count": 1',
        ):
            path = self.write(self.tmp, (raw + "\n").encode("utf-8"))
            with self.assertRaises(CorpusError, msg=raw):
                read_corpus(path)

    def test_empty_text_record_is_legal(self):
        path = self.write(self.tmp, b'{"text": "", "count": 4}\n')
        self.assertEqual(read_corpus(path), [("", 4)])

    def test_config_validation(self):
        config = read_config("fixtures/config.json")
        params = params_from_config(config)
        self.assertEqual((params.merge_steps, params.min_weighted_pair_count), (24, 2))
        with self.assertRaises(CorpusError):
            params_from_config({"min_weighted_pair_count": 2})
        with self.assertRaises(CorpusError):
            params_from_config({"merge_steps": -1})
        with self.assertRaises(CorpusError):
            params_from_config({"merge_steps": 1, "min_weighted_pair_count": 0})
        with self.assertRaises(ValueError):
            TrainParams(merge_steps=-1)
        with self.assertRaises(ValueError):
            TrainParams(merge_steps=1, min_weighted_pair_count=0)


class FullCorpusTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = read_corpus("fixtures/corpus.jsonl")
        cls.params = params_from_config(read_config("fixtures/config.json"))
        cls.trainer = IncrementalBPETrainer(cls.params).fit(cls.records)

    def test_summary_is_internally_consistent(self):
        s = self.trainer.summary()
        self.assertEqual(s["rounds_completed"], len(self.trainer.merges))
        self.assertEqual(s["vocab_size"], 256 + s["num_merges"])
        self.assertEqual(
            s["final_tokens"],
            s["initial_tokens"] - sum(m["occurrences_replaced"] for m in self.trainer.merges),
        )
        self.assertIn(s["stop_reason"], {"merge_steps_reached", "no_pairs", "below_threshold"})
        self.assertGreater(s["rounds_completed"], 0)

    def test_incremental_counters_prove_no_full_rescan(self):
        s = self.trainer.stats
        self.assertGreater(s.positions_checked, 0)
        self.assertGreater(s.adjacency_updates, 0)
        self.assertGreater(s.candidates_evicted, 0)
        # the incremental walk examines strictly fewer positions than a
        # per-round full recount would have
        self.assertLess(s.positions_checked, s.full_rescan_positions_equivalent)
        self.assertGreater(s.initial_scan_positions, 0)

    def test_new_ids_in_first_creation_order(self):
        for rank, merge in enumerate(self.trainer.merges):
            self.assertEqual(merge["new_id"], 256 + rank)

    def test_multibyte_segments_merge_within_runs_only(self):
        # every learned merge must be a byte string that occurred inside one
        # whitespace-delimited run of the corpus
        runs = set()
        for text, _count in self.records:
            i, n = 0, len(text)
            while i < n:
                j = i + 1
                while j < n and text[j].isspace() == text[i].isspace():
                    j += 1
                runs.add(text[i:j].encode("utf-8"))
                i = j
        for token_id in range(256, len(self.trainer.symbols.symbols)):
            data = self.trainer.symbols.bytes_of(token_id)
            self.assertTrue(
                any(data in run for run in runs),
                f"merge {data!r} crosses a run boundary",
            )

    def test_input_reorder_is_irrelevant(self):
        baseline = self.trainer
        shuffled = list(self.records)
        random.Random(7).shuffle(shuffled)
        other = IncrementalBPETrainer(self.params).fit(shuffled)
        self.assertEqual(baseline.symbols.symbols, other.symbols.symbols)
        self.assertEqual(baseline.merges, other.merges)


if __name__ == "__main__":
    unittest.main()
