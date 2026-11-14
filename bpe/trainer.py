"""Incremental byte-level BPE training.

A full corpus scan happens exactly once, at initialisation.  After that every
round only touches the adjacencies around the positions actually replaced:

* **Position identity** — each segment is a doubly linked list over parallel
  arrays; a node index is a stable position identity.  A merge splices out the
  right-hand node, so a stale occurrence entry can never be reached again by
  walking the list.
* **Occurrence tracking** — ``pair_positions`` maps every live adjacent pair
  to the exact set of ``(word, left_node)`` positions where it occurs, and
  ``pair_count`` keeps the corresponding weighted frequencies.  Both are
  updated eagerly around each splice: the disappearing left/self/right
  adjacencies are removed and the two new adjacencies are added, so vanished
  positions stop counting and no new adjacency is missed.
* **Best candidate** — a lazy max-heap keyed by ``(-count, left bytes, right
  bytes)`` implements the declared tie-break (lexicographic on the raw
  unsigned byte pairs).  Entries whose count has since changed are evicted on
  pop and counted as ``candidates_evicted``.

The counters ``positions_checked``, ``adjacency_updates`` and
``candidates_evicted`` — contrasted with ``full_rescan_positions_equivalent``
in the summary — demonstrate that the per-round work is genuinely incremental.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

from .core import SymbolTable, whitespace_run_spans

STOP_MERGE_STEPS_REACHED = "merge_steps_reached"
STOP_NO_PAIRS = "no_pairs"
STOP_BELOW_THRESHOLD = "below_threshold"
STOP_REASONS = {STOP_MERGE_STEPS_REACHED, STOP_NO_PAIRS, STOP_BELOW_THRESHOLD}


@dataclass(frozen=True)
class TrainParams:
    merge_steps: int
    min_weighted_pair_count: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.merge_steps, bool)
            or not isinstance(self.merge_steps, int)
            or self.merge_steps < 0
        ):
            raise ValueError("merge_steps must be a non-negative integer")
        if (
            isinstance(self.min_weighted_pair_count, bool)
            or not isinstance(self.min_weighted_pair_count, int)
            or self.min_weighted_pair_count < 1
        ):
            raise ValueError("min_weighted_pair_count must be an integer >= 1")


@dataclass
class TrainStats:
    # Positions examined by the one allowed full scan at initialisation.
    initial_scan_positions: int = 0
    # Nodes examined while locating occurrences of the selected pair (all
    # rounds after initialisation).  This is what a per-round full recount
    # would have to re-examine for every pair of every word.
    positions_checked: int = 0
    # Individual weighted pair-count modifications after initialisation.
    adjacency_updates: int = 0
    # Stale heap entries discarded while popping the best candidate.
    candidates_evicted: int = 0
    # What a naive per-round full recount would have examined: the sum, over
    # rounds, of the number of live adjacent pairs in the whole corpus.
    full_rescan_positions_equivalent: int = 0


class _Word:
    """One unique segment: a symbol sequence as a doubly linked list.

    Node indices are stable position identities.  A merge splices out the
    right node of the pair; the spliced node keeps its arrays but is
    unreachable from ``head``, so any occurrence entry naming it is dead.
    """

    __slots__ = ("weight", "sym", "prev", "next", "head", "size")

    def __init__(self, weight: int, data: bytes) -> None:
        n = len(data)
        self.weight = weight
        self.sym = list(data)
        self.prev = [i - 1 for i in range(n)]
        self.next = [i + 1 for i in range(n - 1)] + [-1]
        self.head = 0 if n else -1
        self.size = n

    def ids(self) -> list[int]:
        out: list[int] = []
        cur = self.head
        while cur != -1:
            out.append(self.sym[cur])
            cur = self.next[cur]
        return out


class IncrementalBPETrainer:
    """Train BPE merges with incremental pair/position maintenance."""

    def __init__(self, params: TrainParams, observer=None) -> None:
        self.params = params
        # observer(snapshot) is called after each applied merge; used by the
        # independent tiny-corpus recount benchmark.
        self.observer = observer
        self.symbols = SymbolTable()
        self.words: list[_Word] = []
        self.pair_count: dict[tuple[int, int], int] = {}
        self.pair_positions: dict[tuple[int, int], set[tuple[int, int]]] = {}
        self._heap: list[tuple[int, bytes, bytes, int, int]] = []
        self.stats = TrainStats()
        self.merges: list[dict] = []
        self.round_log: list[dict] = []
        self.stop_reason = STOP_MERGE_STEPS_REACHED
        self.training_records = 0
        self.initial_tokens = 0
        self._total_pairs = 0

    # ------------------------------------------------------------------ build

    def fit(self, records: list[tuple[str, int]]) -> "IncrementalBPETrainer":
        self.training_records = len(records)
        weights = self._segment_weights(records)
        # Sorting by segment bytes makes every structure independent of the
        # input record order; counts are weighted sums and the tie-break is on
        # raw bytes, so reordering the input cannot change the result.
        for segment in sorted(weights):
            self.words.append(_Word(weights[segment], segment))
        self.initial_tokens = sum(word.size for word in self.words)
        self._initial_scan()
        self._run()
        return self

    @staticmethod
    def _segment_weights(records: list[tuple[str, int]]) -> dict[bytes, int]:
        weights: dict[bytes, int] = {}
        for text, count in records:
            for i, j in whitespace_run_spans(text):
                segment = text[i:j].encode("utf-8")
                weights[segment] = weights.get(segment, 0) + count
        return weights

    def _initial_scan(self) -> None:
        """The single allowed full scan: seed counts, positions and heap."""
        for w, word in enumerate(self.words):
            cur = word.head
            while cur != -1:
                nxt = word.next[cur]
                if nxt == -1:
                    break
                self.stats.initial_scan_positions += 1
                self._note_pair(word.sym[cur], word.sym[nxt], w, cur, +1, push=False)
                cur = nxt
        self._total_pairs = sum(max(word.size - 1, 0) for word in self.words)
        for (a, b), count in self.pair_count.items():
            self._push(a, b, count)

    # ---------------------------------------------------------- maintenance

    def _push(self, a: int, b: int, count: int) -> None:
        heapq.heappush(
            self._heap,
            (-count, self.symbols.bytes_of(a), self.symbols.bytes_of(b), a, b),
        )

    def _note_pair(self, a: int, b: int, w: int, node: int, sign: int,
                   push: bool = True) -> None:
        """Add/remove one occurrence of pair ``(a, b)`` at ``(w, node)``."""
        pair = (a, b)
        count = self.pair_count.get(pair, 0) + sign * self.words[w].weight
        if count > 0:
            self.pair_count[pair] = count
        else:
            self.pair_count.pop(pair, None)
        if sign > 0:
            self.pair_positions.setdefault(pair, set()).add((w, node))
        else:
            positions = self.pair_positions.get(pair)
            if positions is not None:
                positions.discard((w, node))
                if not positions:
                    del self.pair_positions[pair]
        self.stats.adjacency_updates += 1
        if push and count > 0:
            self._push(a, b, count)

    def _pop_best(self) -> tuple[tuple[int, int], int] | None:
        """Best live candidate; stale heap entries are evicted and counted."""
        while self._heap:
            neg_count, _lb, _rb, a, b = heapq.heappop(self._heap)
            if self.pair_count.get((a, b), 0) == -neg_count:
                return (a, b), -neg_count
            self.stats.candidates_evicted += 1
        return None

    def _splice(self, word: _Word, w: int, i: int, j: int,
                a: int, b: int, new_id: int) -> None:
        """Replace ``(a, b)`` at nodes ``(i, j)`` with ``new_id`` in place."""
        p = word.prev[i]
        n = word.next[j]
        # Remove the disappearing adjacencies: left, self, right.
        self._note_pair(a, b, w, i, -1)
        if p != -1:
            self._note_pair(word.sym[p], a, w, p, -1)
        if n != -1:
            self._note_pair(b, word.sym[n], w, j, -1)
        # Splice node j out; node i keeps its identity and becomes new_id.
        word.sym[i] = new_id
        word.next[i] = n
        if n != -1:
            word.prev[n] = i
        word.size -= 1
        self._total_pairs -= 1
        # Add the two new adjacencies around the merged symbol.
        if p != -1:
            self._note_pair(word.sym[p], new_id, w, p, +1)
        if n != -1:
            self._note_pair(new_id, word.sym[n], w, i, +1)

    def _apply_merge(self, a: int, b: int, new_id: int) -> int:
        """Replace every occurrence of ``(a, b)`` non-overlapping, left to right.

        Only the words recorded in ``pair_positions[(a, b)]`` are visited.
        Overlapping occurrences are resolved by the walk itself: after a
        splice the walk resumes at the node after the merged symbol, so a
        position whose left node was consumed is never reconsidered (it is
        unreachable), and a still-live position is re-validated against the
        current symbols before being replaced.
        """
        positions = self.pair_positions.get((a, b))
        if not positions:
            return 0
        by_word: dict[int, set[int]] = {}
        for w, node in positions:
            by_word.setdefault(w, set()).add(node)
        replaced = 0
        for w, snapshot in by_word.items():
            word = self.words[w]
            cur = word.head
            while cur != -1:
                self.stats.positions_checked += 1
                nxt = word.next[cur]
                if (
                    cur in snapshot
                    and nxt != -1
                    and word.sym[cur] == a
                    and word.sym[nxt] == b
                ):
                    self._splice(word, w, cur, nxt, a, b, new_id)
                    replaced += 1
                    nxt = word.next[cur]
                cur = nxt
        return replaced

    # ------------------------------------------------------------------ round

    def _run(self) -> None:
        self.stop_reason = STOP_MERGE_STEPS_REACHED
        for round_index in range(self.params.merge_steps):
            stats_before = (
                self.stats.positions_checked,
                self.stats.adjacency_updates,
                self.stats.candidates_evicted,
            )
            best = self._pop_best()
            if best is None:
                self.stop_reason = STOP_NO_PAIRS
                break
            (a, b), count = best
            if count < self.params.min_weighted_pair_count:
                self.stop_reason = STOP_BELOW_THRESHOLD
                break
            self.stats.full_rescan_positions_equivalent += self._total_pairs
            new_id, _created = self.symbols.intern(
                self.symbols.bytes_of(a) + self.symbols.bytes_of(b)
            )
            replaced = self._apply_merge(a, b, new_id)
            merge = {
                "rank": round_index,
                "left": a,
                "right": b,
                "new_id": new_id,
                "weighted_count": count,
                "occurrences_replaced": replaced,
            }
            self.merges.append(merge)
            self.round_log.append(
                {
                    **merge,
                    "pair_hex": [
                        self.symbols.bytes_of(a).hex(),
                        self.symbols.bytes_of(b).hex(),
                    ],
                    "new_hex": self.symbols.bytes_of(new_id).hex(),
                    "positions_checked": self.stats.positions_checked - stats_before[0],
                    "adjacency_updates": self.stats.adjacency_updates - stats_before[1],
                    "candidates_evicted": self.stats.candidates_evicted - stats_before[2],
                }
            )
            if self.observer is not None:
                self.observer(
                    {
                        "round": round_index,
                        "chosen": (a, b),
                        "weighted_count": count,
                        "new_id": new_id,
                        "pair_counts": dict(self.pair_count),
                        "words": [(word.weight, word.ids()) for word in self.words],
                    }
                )

    # ---------------------------------------------------------------- summary

    def summary(self) -> dict:
        return {
            "format": "bpe-summary/1",
            "stop_reason": self.stop_reason,
            "rounds_completed": len(self.merges),
            "vocab_size": len(self.symbols),
            "num_merges": len(self.merges),
            "training_records": self.training_records,
            "unique_segments": len(self.words),
            "initial_tokens": self.initial_tokens,
            "final_tokens": sum(word.size for word in self.words),
            "initial_scan_positions": self.stats.initial_scan_positions,
            "positions_checked": self.stats.positions_checked,
            "adjacency_updates": self.stats.adjacency_updates,
            "candidates_evicted": self.stats.candidates_evicted,
            "full_rescan_positions_equivalent": self.stats.full_rescan_positions_equivalent,
            "rounds": self.round_log,
        }
