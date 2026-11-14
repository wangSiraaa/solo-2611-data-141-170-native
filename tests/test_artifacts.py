"""Artifact save/precheck/load tests, including contradictory merges."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bpe.artifacts import ArtifactError, load_model, precheck, save_artifacts
from bpe.corpus import read_corpus
from bpe.trainer import IncrementalBPETrainer, TrainParams


def train_tiny(tmp: Path) -> None:
    records = read_corpus("fixtures/tiny-corpus.jsonl")
    trainer = IncrementalBPETrainer(
        TrainParams(merge_steps=24, min_weighted_pair_count=2)
    ).fit(records)
    save_artifacts(
        tmp,
        symbols=trainer.symbols.symbols,
        merges=trainer.merges,
        config={"merge_steps": 24, "min_weighted_pair_count": 2},
        summary=trainer.summary(),
    )


def rewrite(tmp: Path, name: str, mutate) -> None:
    path = tmp / name
    doc = json.loads(path.read_text(encoding="utf-8"))
    mutate(doc)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        train_tiny(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_valid_artifacts_pass_and_load(self):
        self.assertEqual(precheck(self.tmp), [])
        model = load_model(self.tmp)
        self.assertEqual(len(model.symbols), 256 + 5)
        self.assertEqual(model.symbols[256], b"aa")
        self.assertEqual(model.merges[0], (97, 97, 256))
        self.assertEqual(model.summary["stop_reason"], "below_threshold")

    def test_artifacts_are_pure_json(self):
        names = sorted(p.name for p in self.tmp.iterdir())
        self.assertEqual(names, ["config.json", "merges.json", "summary.json", "vocab.json"])
        for name in names:
            json.loads((self.tmp / name).read_text(encoding="utf-8"))

    def test_missing_file_fails(self):
        (self.tmp / "merges.json").unlink()
        problems = precheck(self.tmp)
        self.assertTrue(any("merges.json" in p for p in problems))
        with self.assertRaises(ArtifactError):
            load_model(self.tmp)

    def test_unknown_merge_reference_fails(self):
        def mutate(doc):
            doc["merges"][1]["left"] = 999
        rewrite(self.tmp, "merges.json", mutate)
        problems = precheck(self.tmp)
        self.assertTrue(any("contradictory merge reference" in p for p in problems))

    def test_byte_sequence_mismatch_fails(self):
        def mutate(doc):
            doc["merges"][2]["right"] = 98  # claims ab + b, but new bytes are aba
        rewrite(self.tmp, "merges.json", mutate)
        problems = precheck(self.tmp)
        self.assertTrue(any("contradictory merge reference" in p for p in problems))

    def test_non_contiguous_rank_fails(self):
        def mutate(doc):
            doc["merges"][2]["rank"] = 7
        rewrite(self.tmp, "merges.json", mutate)
        self.assertTrue(any("rank" in p for p in precheck(self.tmp)))

    def test_new_id_order_fails(self):
        def mutate(doc):
            doc["merges"][1]["new_id"] = 300
        rewrite(self.tmp, "merges.json", mutate)
        self.assertTrue(any("first-creation order" in p for p in precheck(self.tmp)))

    def test_base_vocab_corruption_fails(self):
        def mutate(doc):
            doc["symbols"][65]["hex"] = "62"
            doc["symbols"][65]["length"] = 1
        rewrite(self.tmp, "vocab.json", mutate)
        problems = precheck(self.tmp)
        self.assertTrue(any("base id 65" in p for p in problems))

    def test_duplicate_symbol_fails(self):
        def mutate(doc):
            doc["symbols"][256]["hex"] = doc["symbols"][255]["hex"]
        rewrite(self.tmp, "vocab.json", mutate)
        self.assertTrue(any("duplicate" in p for p in precheck(self.tmp)))

    def test_bad_config_fails(self):
        def mutate(doc):
            doc["min_weighted_pair_count"] = 0
        rewrite(self.tmp, "config.json", mutate)
        self.assertTrue(any("min_weighted_pair_count" in p for p in precheck(self.tmp)))

    def test_summary_contradiction_fails(self):
        def mutate(doc):
            doc["final_tokens"] = doc["initial_tokens"]
        rewrite(self.tmp, "summary.json", mutate)
        self.assertTrue(any("final_tokens" in p for p in precheck(self.tmp)))


if __name__ == "__main__":
    unittest.main()
