"""Independent full-recount benchmark for the tiny corpus.

The reference implementation below deliberately does a *full* recount of all
adjacent pairs in every round — the exact opposite of the incremental
trainer.  It exists only to verify the trainer on the tiny corpus: after each
round we compare the chosen pair, the complete weighted frequency table, and
the post-replacement sequences.  The tiny corpus contains ``aaaaa`` and
``ababa`` specifically to exercise overlapping occurrences, invalidated
positions, and stale heap candidates.
"""
from __future__ import annotations

import random
import unittest

from bpe.corpus import read_corpus
from bpe.trainer import IncrementalBPETrainer, TrainParams

FIXTURE_TINY = "fixtures/tiny-corpus.jsonl"


# ------------------------------------------------------- reference (naive)


def split_runs(text: str) -> list[str]:
    runs = []
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        while j < n and text[j].isspace() == text[i].isspace():
            j += 1
        runs.append(text[i:j])
        i = j
    return runs


def reference_train(records, merge_steps, min_count):
    """Naive trainer: full recount of every adjacent pair in every round."""
    symbols = [bytes([i]) for i in range(256)]
    seg_weight: dict[bytes, int] = {}
    for text, count in records:
        for run in split_runs(text):
            data = run.encode("utf-8")
            seg_weight[data] = seg_weight.get(data, 0) + count
    words = [[weight, list(segment)] for segment, weight in sorted(seg_weight.items())]
    rounds = []
    for _ in range(merge_steps):
        counts: dict[tuple[int, int], int] = {}
        for weight, ids in words:
            for x, y in zip(ids, ids[1:]):
                counts[(x, y)] = counts.get((x, y), 0) + weight
        if not counts:
            break
        best_count = max(counts.values())
        best_pair = min(
            (pair for pair, c in counts.items() if c == best_count),
            key=lambda pair: (symbols[pair[0]], symbols[pair[1]]),
        )
        if best_count < min_count:
            break
        a, b = best_pair
        symbols.append(symbols[a] + symbols[b])
        new_id = len(symbols) - 1
        for _weight, ids in words:
            out = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids[:] = out
        post_counts: dict[tuple[int, int], int] = {}
        for weight, ids in words:
            for x, y in zip(ids, ids[1:]):
                post_counts[(x, y)] = post_counts.get((x, y), 0) + weight
        rounds.append(
            {
                "chosen": best_pair,
                "weighted_count": best_count,
                "new_id": new_id,
                "pair_counts": post_counts,
                "words": [(weight, list(ids)) for weight, ids in words],
            }
        )
    return symbols, rounds


# ------------------------------------------------------------ test helpers


def assert_positions_consistent(testcase, trainer):
    """pair_positions must hold exactly the live occurrences: nothing stale,
    nothing missing."""
    expected: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for w, word in enumerate(trainer.words):
        cur = word.head
        while cur != -1:
            nxt = word.next[cur]
            if nxt == -1:
                break
            expected.setdefault((word.sym[cur], word.sym[nxt]), set()).add((w, cur))
            cur = nxt
    testcase.assertEqual(expected, trainer.pair_positions)
    counts = {
        pair: sum(trainer.words[w].weight for w, _node in positions)
        for pair, positions in expected.items()
    }
    testcase.assertEqual(counts, trainer.pair_count)


class TinyRecountTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = read_corpus(FIXTURE_TINY)

    def train_observed(self, params):
        snapshots = []
        trainer = IncrementalBPETrainer(params, observer=snapshots.append)
        trainer.fit(self.records)
        return trainer, snapshots

    def test_each_round_matches_full_recount(self):
        params = TrainParams(merge_steps=24, min_weighted_pair_count=2)
        trainer, snapshots = self.train_observed(params)
        _symbols, ref_rounds = reference_train(
            self.records, params.merge_steps, params.min_weighted_pair_count
        )
        self.assertEqual(len(ref_rounds), len(snapshots))
        for ref, got in zip(ref_rounds, snapshots):
            rnd = ref["new_id"] - 256
            self.assertEqual(ref["chosen"], got["chosen"], f"round {rnd}: pair")
            self.assertEqual(
                ref["weighted_count"], got["weighted_count"], f"round {rnd}: count"
            )
            self.assertEqual(ref["new_id"], got["new_id"], f"round {rnd}: new id")
            # complete weighted frequency table after the replacement
            self.assertEqual(
                ref["pair_counts"], got["pair_counts"], f"round {rnd}: frequencies"
            )
            # full post-replacement sequences
            self.assertEqual(ref["words"], got["words"], f"round {rnd}: sequences")

    def test_expected_tiny_merges(self):
        """Hand-computed merge sequence for the tiny corpus (min count 2)."""
        params = TrainParams(merge_steps=24, min_weighted_pair_count=2)
        trainer, _snapshots = self.train_observed(params)
        got = [
            (
                trainer.symbols.bytes_of(m["left"]) + trainer.symbols.bytes_of(m["right"]),
                m["weighted_count"],
            )
            for m in trainer.merges
        ]
        self.assertEqual(
            got,
            [
                (b"aa", 8),      # aaaaa x2 contributes 4 pairs each
                (b"ab", 4),      # tie with (b, a)=4 broken by raw byte order
                (b"aba", 3),     # ababa x1 + aba x2
                (b"aaa", 2),     # tie (aa,aa) vs (aa,a) broken by raw byte order
                (b"aaaaa", 2),
            ],
        )
        self.assertEqual(trainer.stop_reason, "below_threshold")

    def test_overlap_and_stale_positions(self):
        """aaaaa: 'aa' occurs at 4 overlapping positions but only 2
        non-overlapping replacements happen; the consumed positions must
        vanish from the occurrence index."""
        params = TrainParams(merge_steps=1, min_weighted_pair_count=1)
        trainer, _snapshots = self.train_observed(params)
        merge = trainer.merges[0]
        self.assertEqual((merge["left"], merge["right"]), (97, 97))
        self.assertEqual(merge["weighted_count"], 8)   # counts overlap
        self.assertEqual(merge["occurrences_replaced"], 2)  # replaces non-overlap
        aaaaa_word = next(w for w in trainer.words if w.weight == 2)
        self.assertEqual(aaaaa_word.ids(), [256, 256, 97])
        # the merged pair is gone entirely: no stale position keeps counting
        self.assertNotIn((97, 97), trainer.pair_positions)
        self.assertNotIn((97, 97), trainer.pair_count)
        assert_positions_consistent(self, trainer)

    def test_ababa_overlap(self):
        """ababa -> [ab, ab, a]: the middle 'ab' occurrence at position 2
        survives while positions 1 and 3 (b...) never matched."""
        params = TrainParams(merge_steps=2, min_weighted_pair_count=1)
        trainer, _snapshots = self.train_observed(params)
        second = trainer.merges[1]
        self.assertEqual(
            (trainer.symbols.bytes_of(second["left"]),
             trainer.symbols.bytes_of(second["right"])),
            (b"a", b"b"),
        )
        self.assertEqual(second["occurrences_replaced"], 3)  # 2 in ababa + 1 in aba
        words = {w.weight: w.ids() for w in trainer.words}
        self.assertEqual(words[1], [257, 257, 97])  # ababa
        self.assertEqual(words[2], [257, 97])       # aba (weight 2)
        assert_positions_consistent(self, trainer)

    def test_positions_consistent_after_every_round(self):
        params = TrainParams(merge_steps=24, min_weighted_pair_count=1)
        trainer = IncrementalBPETrainer(
            params, observer=lambda _snap: assert_positions_consistent(self, trainer)
        )
        trainer.fit(self.records)
        self.assertEqual(trainer.stop_reason, "no_pairs")

    def test_stale_candidates_actually_evicted(self):
        params = TrainParams(merge_steps=24, min_weighted_pair_count=1)
        trainer, _snapshots = self.train_observed(params)
        # pair counts change under the heap, so stale entries must exist
        self.assertGreater(trainer.stats.candidates_evicted, 0)
        self.assertGreater(trainer.stats.adjacency_updates, 0)
        self.assertGreater(trainer.stats.positions_checked, 0)
        # incremental work never exceeds a per-round full recount
        self.assertLessEqual(
            trainer.stats.positions_checked,
            trainer.stats.full_rescan_positions_equivalent,
        )

    def test_stop_reasons(self):
        trainer, _ = self.train_observed(TrainParams(merge_steps=2, min_weighted_pair_count=1))
        self.assertEqual(trainer.stop_reason, "merge_steps_reached")
        self.assertEqual(len(trainer.merges), 2)

    def test_input_reorder_is_irrelevant(self):
        params = TrainParams(merge_steps=24, min_weighted_pair_count=1)
        baseline, _ = self.train_observed(params)
        for seed in range(5):
            shuffled = list(self.records)
            random.Random(seed).shuffle(shuffled)
            trainer = IncrementalBPETrainer(params).fit(shuffled)
            self.assertEqual(baseline.symbols.symbols, trainer.symbols.symbols)
            self.assertEqual(baseline.merges, trainer.merges)
            self.assertEqual(baseline.pair_count, trainer.pair_count)
            self.assertEqual(
                [w.ids() for w in baseline.words],
                [w.ids() for w in trainer.words],
            )


if __name__ == "__main__":
    unittest.main()
