"""Joint event selection for one adjacent-frame pair via integer programming.

The model is a small binary integer program (solved with SciPy's HiGHS-backed
``milp``).  For a pair (prior frame P, next frame N) with candidate events C:

    variables   x_c in {0,1}  for every candidate c in C
    objective   minimise  sum_c cost(c) * x_c
    constraints for every prior object a:     sum_{c: parent(c)=a} x_c = 1
                for every next object  b:     sum_{c: b in children(c)} x_c = 1

The first constraint set says each prior object has exactly one outgoing event
(continue / divide / disappear); the second says each next object has exactly
one incoming event (be continued / be a daughter / be born).  Together they
enforce:

  * every observation is claimed by exactly one event (observation uniqueness);
  * every object has at most one parent (parent uniqueness) and a mother cannot
    both divide and continue -- its single outgoing event is one or the other;
  * births and disappearances are events in their own right, so the solver is
    never forced into a spurious link merely to "use up" an object.

Why integer programming rather than greedy nearest-neighbour: continuations and
divisions compete for the *same* next observations and for the same mothers.
Only a joint objective can arbitrate that competition.  The objective is a sum
of interpretable per-event costs from :mod:`celltrack.candidates`.

Near-equivalent alternatives
----------------------------
After the optimal event set is found, we repeatedly forbid it with a no-good cut
and re-solve, collecting every distinct event set whose rounded cost is within
``ambiguity_cost_margin`` of the optimum (bounded by ``max_alternatives`` and
``max_enumeration_nodes``).  When more than one such set exists the pair is
flagged ``undecided``; every alternative is retained with its cost and evidence.
A representative is chosen only for continuity of output, by a geometry-based
canonical signature -- never by label number or enumeration order -- and is
clearly marked as a representative of an undecided set, not the true lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import LinearConstraint, Bounds, milp
from scipy.sparse import csr_matrix

from .candidates import Candidate, PairCandidates
from .config import TrackConfig


@dataclass
class Solution:
    events: list[Candidate]
    cost: float
    cost_rounded: float
    status: str
    message: str = ""

    @property
    def event_keys(self) -> frozenset[tuple]:
        return frozenset(c.key for c in self.events)


@dataclass
class PairSolution:
    pair: PairCandidates
    chosen: Solution | None
    alternatives: list[Solution] = field(default_factory=list)
    undecided: bool = False
    ambiguity_reason: str = ""
    chosen_is_representative: bool = False
    node_count: int = 0
    enumeration_truncated: bool = False
    # Event keys that differ between near-equivalent alternatives.  An event
    # present in *every* alternative is not "undecided" even when the pair
    # overall is: only the contested links are.
    undecided_event_keys: frozenset = frozenset()


def _build_problem(pair: PairCandidates):
    """Return (c, A, lb, ub, integrality, bounds, index map) for the base model."""
    candidates = pair.candidates
    idx = {id(c): i for i, c in enumerate(candidates)}
    n = len(candidates)

    c = np.array([cand.cost for cand in candidates], dtype=float)
    integrality = np.ones(n, dtype=np.uint8)
    bounds = Bounds(np.zeros(n), np.ones(n))

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lb: list[float] = []
    ub: list[float] = []

    r = 0
    for a in pair.prev_objects:
        for cand in candidates:
            if cand.a_id == a.obs_id:
                rows.append(r); cols.append(idx[id(cand)]); vals.append(1.0)
        lb.append(1.0); ub.append(1.0); r += 1

    for b in pair.next_objects:
        for cand in candidates:
            if b.obs_id in cand.b_ids:
                rows.append(r); cols.append(idx[id(cand)]); vals.append(1.0)
        lb.append(1.0); ub.append(1.0); r += 1

    A = csr_matrix((vals, (rows, cols)), shape=(r, n))
    return c, A, np.array(lb), np.array(ub), integrality, bounds, candidates, idx


def _solve_milp(c, A, lb, ub, integrality, bounds, extra_rows=None) -> tuple[list[int], float, str, str]:
    """Solve the MILP, optionally with extra equality/inequality rows."""
    if A.shape[1] == 0:
        return [], 0.0, "ok", ""
    if extra_rows is not None:
        Ar, lbr, ubr = extra_rows
        from scipy.sparse import vstack
        A = vstack([A, Ar]).tocsr()
        lb = np.concatenate([lb, lbr])
        ub = np.concatenate([ub, ubr])
    res = milp(
        c,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(A, lb, ub),
        options={"presolve": True},
    )
    if not res.success:
        return [], float("inf"), res.status, res.message or "no feasible solution"
    x = np.asarray(res.x)
    selected = [i for i, v in enumerate(x) if v > 0.5]
    return selected, float(res.fun), "ok", ""


def solve_pair(pair: PairCandidates, cfg: TrackConfig) -> PairSolution:
    c, A, lb, ub, integrality, bounds, candidates, _idx = _build_problem(pair)

    # ---- empty-pair handling (all births or all disappearances) --------
    if not pair.prev_objects and not pair.next_objects:
        return PairSolution(pair=pair, chosen=Solution([], 0.0, 0.0, "ok"))

    selected_idx, best_cost, status, message = _solve_milp(
        c, A, lb, ub, integrality, bounds
    )
    node_count = 1
    if status != "ok":
        # The model is always feasible when each object has a birth/disappearance
        # candidate; infeasibility here would indicate an internal bug.
        return PairSolution(
            pair=pair,
            chosen=None,
            ambiguity_reason=f"solver failure: {message}",
        )

    best_rounded = round(best_cost, cfg.cost_decimals)
    chosen = Solution(
        events=[candidates[i] for i in selected_idx],
        cost=best_cost,
        cost_rounded=best_rounded,
        status="ok",
    )

    # ---- near-equivalent alternative enumeration -----------------------
    # Enumerate every distinct event set whose objective lies within
    # ``ambiguity_cost_margin`` of the optimum.  Each found set is excluded by a
    # no-good cut (sum_{c in S} x_c <= |S| - 1) and the MILP is re-solved.  For
    # small adjacent-frame problems this is exact up to the configured caps.
    alternatives: list[Solution] = [chosen]
    forbidden: list[frozenset[tuple]] = [chosen.event_keys]
    enumeration_truncated = False
    cap_hit = False

    # no-good rows accumulated across iterations
    extra_A_rows: list[np.ndarray] = []
    extra_lb: list[float] = []
    extra_ub: list[float] = []

    def _solve_with_cuts():
        if extra_A_rows:
            Ar = csr_matrix(np.vstack(extra_A_rows))
            lbr = np.array(extra_lb)
            ubr = np.array(extra_ub)
        else:
            Ar, lbr, ubr = None, None, None
        return _solve_milp(c, A, lb, ub, integrality, bounds,
                           extra_rows=(Ar, lbr, ubr))

    def _add_cut(event_keys: frozenset):
        row = np.zeros(len(candidates), dtype=float)
        for cand in event_keys:
            for i, cc in enumerate(candidates):
                if cc.key == cand:
                    row[i] = 1.0
                    break
            else:  # pragma: no cover - internal consistency guard
                raise RuntimeError("candidate key not found during no-good cut")
        extra_A_rows.append(row)
        extra_lb.append(-np.inf)
        extra_ub.append(float(len(event_keys)) - 1.0)

    while True:
        if len(alternatives) >= cfg.max_alternatives:
            cap_hit = True
            break
        if node_count >= cfg.max_enumeration_nodes:
            enumeration_truncated = True
            break
        # forbid all previously found sets
        while len(forbidden) > 0 and len(extra_A_rows) < len(forbidden):
            _add_cut(forbidden[len(extra_A_rows)])
        sel2, cost2, status2, _msg2 = _solve_with_cuts()
        node_count += 1
        if status2 != "ok":
            break
        rounded2 = round(cost2, cfg.cost_decimals)
        if rounded2 > best_rounded + cfg.ambiguity_cost_margin + 1e-12:
            break  # next cheapest lies outside the near-equivalence margin
        sol2 = Solution(
            events=[candidates[i] for i in sel2],
            cost=cost2,
            cost_rounded=rounded2,
            status="ok",
        )
        if sol2.event_keys in forbidden:  # pragma: no cover - cut guard
            enumeration_truncated = True
            break
        alternatives.append(sol2)
        forbidden.append(sol2.event_keys)

    # If the alternatives cap stopped us, confirm whether an additional
    # near-equivalent set actually exists, so truncation is reported truthfully.
    extra_equivalent_exists = False
    if cap_hit:
        while len(extra_A_rows) < len(forbidden):
            _add_cut(forbidden[len(extra_A_rows)])
        while node_count < cfg.max_enumeration_nodes:
            sel3, cost3, status3, _ = _solve_with_cuts()
            node_count += 1
            if status3 != "ok":
                break
            if round(cost3, cfg.cost_decimals) > best_rounded + cfg.ambiguity_cost_margin + 1e-12:
                break
            extra_equivalent_exists = True
            enumeration_truncated = True
            break

    undecided = len(alternatives) > 1 or extra_equivalent_exists
    reason = ""
    undecided_keys: frozenset = frozenset()
    if undecided:
        costs = sorted({a.cost_rounded for a in alternatives})
        if extra_equivalent_exists and len(alternatives) == 1:
            reason = (
                f"at least one more near-equivalent event set exists within "
                f"{cfg.ambiguity_cost_margin} cost units (not stored: cap reached); "
                f"representative cost {costs}."
            )
        else:
            reason = (
                f"{len(alternatives)} near-equivalent event sets within "
                f"{cfg.ambiguity_cost_margin} cost units; rounded costs {costs}."
            )
        # An event is contested iff it is not present in every alternative.
        key_sets = [set(a.event_keys) for a in alternatives]
        common = set.intersection(*key_sets) if key_sets else set()
        divergent = set()
        for ks in key_sets:
            divergent |= (ks - common)
        undecided_keys = frozenset(divergent)
        # With a truncated search we cannot compare against the unseen set, so
        # conservatively mark every event of the representative as contested.
        if extra_equivalent_exists and len(alternatives) == 1:
            undecided_keys = frozenset(alternatives[0].event_keys)
    if enumeration_truncated and undecided and len(alternatives) > 1:
        reason += " Alternative search hit a safety cap; more equivalents may exist."

    chosen_is_representative = False
    if undecided:
        # Deterministic, label-numbering-independent representative: minimum
        # canonical geometric signature across all near-optimal sets.  The
        # choice is stable under relabelling and under frame-listing order.
        def signature(sol: Solution):
            return tuple(sorted(ev.geom_signature for ev in sol.events))

        rep = min(alternatives, key=signature)
        chosen_is_representative = True
        chosen = rep

    return PairSolution(
        pair=pair,
        chosen=chosen,
        alternatives=alternatives,
        undecided=undecided,
        ambiguity_reason=reason,
        chosen_is_representative=chosen_is_representative,
        node_count=node_count,
        enumeration_truncated=enumeration_truncated,
        undecided_event_keys=undecided_keys,
    )
