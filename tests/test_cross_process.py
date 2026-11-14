"""Cross-process tests: train once, then encode/decode in fresh processes.

A new process must be able to encode unseen text (new CJK characters, emoji,
consecutive whitespace) using only the JSON artifacts, with correct UTF-8
byte offsets and exact round-trips.  Batch encoding with the fixed artifacts
must agree with one-by-one encoding.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bpe.corpus import read_config, read_corpus, params_from_config
from bpe.encoder import BPECodec
from bpe.trainer import IncrementalBPETrainer
from bpe.artifacts import save_artifacts

ROOT = Path(__file__).resolve().parent.parent
UNSEEN = ROOT / "fixtures" / "unseen.txt"
EMPTY = ROOT / "fixtures" / "empty.txt"


def run_cli(*args, expect=0):
    proc = subprocess.run(
        [sys.executable, "-m", "bpe", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != expect:
        raise AssertionError(
            f"cli {args} exited {proc.returncode} (expected {expect})\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc


class CrossProcessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.artifacts = Path(cls._tmp.name) / "artifacts"
        records = read_corpus(ROOT / "fixtures" / "corpus.jsonl")
        params = params_from_config(read_config(ROOT / "fixtures" / "config.json"))
        trainer = IncrementalBPETrainer(params).fit(records)
        save_artifacts(
            cls.artifacts,
            symbols=trainer.symbols.symbols,
            merges=trainer.merges,
            config={
                "merge_steps": params.merge_steps,
                "min_weighted_pair_count": params.min_weighted_pair_count,
            },
            summary=trainer.summary(),
        )
        cls.codec = BPECodec.from_artifacts(str(cls.artifacts))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def encode_file(self, path: Path) -> list[dict]:
        proc = run_cli(
            "encode", "--artifacts", str(self.artifacts), "--input", str(path)
        )
        return [json.loads(line) for line in proc.stdout.splitlines()]

    def check_tokens_tile(self, record: dict, data: bytes) -> None:
        self.assertEqual(record["byte_length"], len(data))
        cursor = 0
        for token in record["tokens"]:
            self.assertEqual(token["start"], cursor)
            self.assertGreater(token["end"], token["start"])
            self.assertEqual(
                data[token["start"]:token["end"]].hex(),
                token["hex"],
                "token bytes must equal the covered input slice",
            )
            self.assertEqual(
                bytes.fromhex(token["hex"]), self.codec.symbols[token["id"]]
            )
            cursor = token["end"]
        self.assertEqual(cursor, len(data))

    # ------------------------------------------------------------- pipeline

    def test_cli_train_precheck_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "art"
            run_cli(
                "train",
                "--corpus", "fixtures/tiny-corpus.jsonl",
                "--config", "fixtures/config.json",
                "--out", str(out),
            )
            proc = run_cli("precheck", "--artifacts", str(out))
            self.assertIn("ok:", proc.stdout)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["stop_reason"], "below_threshold")
            self.assertEqual(summary["rounds_completed"], 5)
            self.assertGreater(summary["candidates_evicted"], 0)
            self.assertLess(
                summary["positions_checked"],
                summary["full_rescan_positions_equivalent"] + summary["initial_scan_positions"],
            )

    def test_precheck_of_trained_artifacts(self):
        proc = run_cli("precheck", "--artifacts", str(self.artifacts))
        self.assertIn("ok:", proc.stdout)

    # -------------------------------------------------------------- unseen

    def test_unseen_file_round_trip_in_new_process(self):
        data = UNSEEN.read_bytes()
        (record,) = self.encode_file(UNSEEN)
        self.check_tokens_tile(record, data)
        # decode in yet another process, byte-exact
        with tempfile.TemporaryDirectory() as tmp:
            enc = Path(tmp) / "enc.jsonl"
            enc.write_text(
                json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            raw = Path(tmp) / "raw.bin"
            run_cli(
                "decode", "--artifacts", str(self.artifacts),
                "--input", str(enc), "--raw", str(raw),
            )
            self.assertEqual(raw.read_bytes(), data)

    def test_unseen_characters_fall_back_to_bytes(self):
        # 汉 字 🙂 use bytes that never appear in the training corpus
        tokens = self.codec.encode_text("汉字🙂")
        self.assertEqual(len(tokens), len("汉字🙂".encode("utf-8")))
        self.assertTrue(all(t.end - t.start == 1 for t in tokens))
        self.assertEqual(self.codec.decode_ids([t.id for t in tokens]), "汉字🙂")

    def test_empty_file(self):
        (record,) = self.encode_file(EMPTY)
        self.assertEqual(record["byte_length"], 0)
        self.assertEqual(record["tokens"], [])
        self.assertEqual(self.codec.decode_ids([]), "")

    def test_invalid_utf8_input_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.bin"
            bad.write_bytes(b"abc\xffdef")
            proc = run_cli(
                "encode", "--artifacts", str(self.artifacts),
                "--input", str(bad), expect=2,
            )
            self.assertIn("UTF-8", proc.stderr)

    # --------------------------------------------------------------- batch

    def test_batch_jsonl_records_consistent_and_round_trip(self):
        records = [
            "banana 汉字🙂🙂",
            "café   cafÉ)",
            "中中！",
            "  ",          # pure whitespace record
            "",            # empty record
            "a a  a",
            "数据 数据",    # seen in training, unseen combination
        ]
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "records.jsonl"
            src.write_text(
                "".join(
                    json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in records
                ),
                encoding="utf-8",
            )
            enc = Path(tmp) / "enc.jsonl"
            run_cli(
                "encode", "--artifacts", str(self.artifacts),
                "--jsonl", str(src), "--out", str(enc),
            )
            lines = [
                json.loads(line)
                for line in enc.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(lines), len(records))
            # offsets tile each record and match the in-process codec exactly
            in_process = self.codec.encode_records(records)
            for record, text, expected in zip(lines, records, in_process):
                data = text.encode("utf-8")
                self.check_tokens_tile(record, data)
                self.assertEqual(
                    [t["id"] for t in record["tokens"]],
                    [t.id for t in expected],
                    "subprocess and in-process encodings must agree",
                )
            # decode the batch in a new process: every record round-trips
            dec = Path(tmp) / "dec.jsonl"
            run_cli(
                "decode", "--artifacts", str(self.artifacts),
                "--input", str(enc), "--out", str(dec),
            )
            got = [
                json.loads(line)["text"]
                for line in dec.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(got, records)

    def test_batching_never_changes_a_record(self):
        records = ["banana 汉字🙂🙂", "中中！", "a a  a"]
        together = self.codec.encode_records(records)
        for text, tokens in zip(records, together):
            self.assertEqual(self.codec.encode_text(text), tokens)

    def test_decode_rejects_broken_token_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            enc = Path(tmp) / "enc.jsonl"
            enc.write_text(
                json.dumps(
                    {
                        "record": 0,
                        "byte_length": 3,
                        "tokens": [
                            {"id": 97, "start": 0, "end": 1, "hex": "61"},
                            {"id": 97, "start": 1, "end": 2, "hex": "61"},
                            # gap: coverage stops at 2, byte_length is 3
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            proc = run_cli(
                "decode", "--artifacts", str(self.artifacts),
                "--input", str(enc), expect=2,
            )
            self.assertIn("cover", proc.stderr)


if __name__ == "__main__":
    unittest.main()
