"""Core geometry and continuity logic for periodic trajectory unwrapping."""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

# Absolute tolerance used when comparing distances in nm.  It is deliberately
# tiny: the fixtures are synthetic and all constraints have comfortable slack,
# so a loose tolerance would only hide genuine ambiguity.
DIST_TOL = 1e-9
# Upper guard for the rigorous candidate box.  A badly conditioned cell could
# otherwise make the box enormous.  Cells are validated before this is used.
MAX_CANDIDATE_BOX = 2_000_000
# We only need to know whether a within-frame problem is unique/ambiguous.
SOLUTION_LIMIT = 8


class AnalysisError(Exception):
    """Raised when the analysis cannot proceed."""


@dataclass(frozen=True)
class Cell:
    """A fixed triclinic cell whose **columns** are the lattice vectors."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=float)
        if m.shape != (3, 3):
            raise AnalysisError(f"cell must be 3x3 (fixed cell); got {m.shape}")
        if not np.all(np.isfinite(m)):
            raise AnalysisError("cell contains non-finite values")
        object.__setattr__(self, "matrix", m)

    @property
    def det(self) -> float:
        return float(np.linalg.det(self.matrix))

    @property
    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)

    def frac_to_cart(self, frac: np.ndarray) -> np.ndarray:
        """Convert row-vector fractional coordinates to Cartesian nm."""
        return np.asarray(frac, dtype=float) @ self.matrix.T

    def cart_to_frac(self, cart: np.ndarray) -> np.ndarray:
        """Convert row-vector Cartesian nm to fractional coordinates."""
        cart = np.asarray(cart, dtype=float)
        return np.linalg.solve(self.matrix, cart.T).T

    def shift_cart(self, lattice_offsets: np.ndarray) -> np.ndarray:
        """Return sum_k offset_k * a_k for integer offset row vectors."""
        offs = np.asarray(lattice_offsets, dtype=float)
        return offs @ self.matrix.T


@dataclass(frozen=True)
class AnalysisOptions:
    """Feasibility bounds; all lengths nm, times ps."""

    max_bond_nm: float
    max_com_speed_nm_per_ps: float
    max_gap_ps: float
    lag_times_ps: tuple[float, ...]
    lag_tolerance_ps: float = 1e-8
    dist_tol: float = DIST_TOL

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_bond_nm) or self.max_bond_nm <= 0:
            raise AnalysisError("max_bond_nm must be finite and > 0")
        if not np.isfinite(self.max_com_speed_nm_per_ps) or self.max_com_speed_nm_per_ps < 0:
            raise AnalysisError("max_com_speed_nm_per_ps must be finite and >= 0")
        if not np.isfinite(self.max_gap_ps) or self.max_gap_ps < 0:
            raise AnalysisError("max_gap_ps must be finite and >= 0")
        for lag in self.lag_times_ps:
            if not np.isfinite(lag) or lag < 0:
                raise AnalysisError("lag times must be finite and >= 0")


# ---------------------------------------------------------------------------
# Bounded lattice-image search
# ---------------------------------------------------------------------------


def _candidate_box(
    delta: np.ndarray, cell: Cell, radius: float, tol: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lo, hi, centre) for the rigorous integer search box.

    We need integer m with ||delta + H m|| <= radius.  Put
    centre = -H^{-1} delta.  Then delta + H m = H (m - centre), and for each
    component

        |m_k - centre_k| <= ||row_k(H^{-1})|| * ||H (m - centre)||
                         <= ||row_k(H^{-1})|| * radius.

    Every feasible m therefore lies in this finite box.  The box is then
    filtered with the exact Euclidean distance, so it adds no candidate and
    removes none.
    """

    centre = -cell.inverse @ delta
    # The feasibility test accepts ||delta + H m|| <= radius + tol, so bound
    # the fractional box with the inflated radius; this keeps the box a strict
    # superset of every candidate the exact filter can accept.
    eff_radius = radius + max(tol, 0.0)
    bounds = eff_radius * np.linalg.norm(cell.inverse, axis=1)
    pad = 1e-12
    lo = np.ceil(centre - bounds - pad).astype(np.int64)
    hi = np.floor(centre + bounds + pad).astype(np.int64)
    return lo, hi, centre


def lattice_candidates(
    delta: np.ndarray,
    cell: Cell,
    radius: float,
    tol: float = DIST_TOL,
    *,
    return_meta: bool = False,
) -> list[tuple[int, int, int]] | tuple[list[tuple[int, int, int]], dict[str, Any]]:
    """Enumerate all integer translations m with ||delta + H m|| <= radius.

    ``delta`` is a Cartesian displacement (nm).  The result is sorted
    lexicographically for determinism.
    """

    delta = np.asarray(delta, dtype=float)
    if delta.shape != (3,):
        raise AnalysisError(f"delta must have shape (3,), got {delta.shape}")
    lo, hi, centre = _candidate_box(delta, cell, radius, tol)
    spans = np.maximum(hi - lo + 1, 0)
    nbox = int(np.prod(spans.astype(object)))
    if nbox > MAX_CANDIDATE_BOX:
        raise AnalysisError(
            "lattice candidate box is unreasonably large "
            f"({nbox} points); cell conditioning or radius is invalid"
        )

    out: list[tuple[int, int, int]] = []
    for m in itertools.product(
        range(int(lo[0]), int(hi[0]) + 1),
        range(int(lo[1]), int(hi[1]) + 1),
        range(int(lo[2]), int(hi[2]) + 1),
    ):
        mt = np.asarray(m, dtype=float)
        if np.linalg.norm(delta + cell.matrix @ mt) <= radius + tol:
            out.append((int(m[0]), int(m[1]), int(m[2])))
    out.sort()
    if return_meta:
        meta = {
            "radius_nm": float(radius),
            "centre_frac": centre.tolist(),
            "box_lo": lo.tolist(),
            "box_hi": hi.tolist(),
            "box_size": nbox,
            "candidate_count": len(out),
        }
        return out, meta
    return out


# ---------------------------------------------------------------------------
# Within-frame molecule reconstruction
# ---------------------------------------------------------------------------


@dataclass
class FrameReconstruction:
    status: str
    offsets: np.ndarray | None = None
    bond_lengths_nm: list[float] | None = None
    solutions: list[np.ndarray] = field(default_factory=list)
    solution_count: int = 0
    truncated: bool = False
    edge_candidates: dict[str, list[tuple[int, int, int]]] = field(default_factory=dict)
    conflict: dict[str, Any] | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _edge_key(i: int, j: int) -> str:
    return f"{i}->{j}"


def _build_components(adj: dict[int, list[int]]) -> list[list[int]]:
    seen: set[int] = set()
    comps: list[list[int]] = []
    for start in sorted(adj):
        if start in seen:
            continue
        comp: list[int] = []
        q = deque([start])
        seen.add(start)
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(sorted(comp))
    return comps


def _explain_failure(
    raw: np.ndarray,
    bonds: Sequence[tuple[int, int]],
    parent: dict[int, int | None],
    order: list[int],
    cand: dict[tuple[int, int], list[tuple[int, int, int]]],
    max_bond_nm: float,
) -> dict[str, Any]:
    """Construct a concrete closure-conflict witness for diagnostics.

    The exhaustive DFS is what decides feasibility.  This helper simply tries
    the first candidate on every spanning-tree edge and reports the first
    non-tree edge that disagrees, which is the usual cycle-closure failure.
    """

    off: dict[int, tuple[int, int, int]] = {}
    for u in order:
        p = parent[u]
        if p is None:
            off[u] = (0, 0, 0)
            continue
        opts = cand.get((p, u), [])
        if not opts:
            return {
                "kind": "empty_candidate_set",
                "directed_edge": [p, u],
                "raw_delta_nm": (raw[u] - raw[p]).tolist(),
                "max_bond_nm": float(max_bond_nm),
            }
        off[u] = tuple(int(a + b) for a, b in zip(off[p], opts[0]))

    for i, j in bonds:
        required = tuple(int(a - b) for a, b in zip(off[j], off[i]))
        allowed = cand[(i, j)]
        if required not in allowed:
            return {
                "kind": "cycle_closure_conflict",
                "directed_edge": [i, j],
                "required_offset_j_minus_i": list(required),
                "allowed_offsets": [list(x) for x in allowed],
                "tree_offsets": {str(k): list(v) for k, v in off.items()},
                "note": (
                    "Every local bond has a feasible image, but no single "
                    "assignment satisfies all bonds of the closed cycle."
                ),
            }
    return {
        "kind": "tree_combinations_inconsistent",
        "tree_offsets": {str(k): list(v) for k, v in off.items()},
    }


def reconstruct_molecule_frame(
    raw: np.ndarray,
    bonds_local: Sequence[tuple[int, int]],
    cell: Cell,
    max_bond_nm: float,
    tol: float = DIST_TOL,
) -> FrameReconstruction:
    """Find consistent lattice images for one molecule in one frame.

    The root atom is fixed to its folded image (offset zero), which removes the
    irrelevant global lattice translation.  A spanning tree supplies trial
    offsets, but *every* bond, including non-tree cycle-closing bonds, is
    checked.  A spanning-tree walk that ignores inconsistent cycles is never
    accepted.
    """

    raw = np.asarray(raw, dtype=float)
    n = len(raw)
    if n == 0:
        return FrameReconstruction(status="invalid", message="molecule has no atoms")
    if not np.all(np.isfinite(raw)):
        return FrameReconstruction(status="invalid", message="non-finite atom coordinates")

    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    clean_bonds: list[tuple[int, int]] = []
    for bond in bonds_local:
        i, j = int(bond[0]), int(bond[1])
        if not (0 <= i < n and 0 <= j < n) or i == j:
            return FrameReconstruction(
                status="invalid", message=f"invalid bond ({i}, {j}) for {n} atoms"
            )
        key = (min(i, j), max(i, j))
        if key in {(min(a, b), max(a, b)) for a, b in clean_bonds}:
            continue
        clean_bonds.append((i, j))
        adj[i].append(j)
        adj[j].append(i)
    for i in adj:
        adj[i].sort()

    comps = _build_components(adj)
    if len(comps) != 1:
        return FrameReconstruction(
            status="disconnected_graph",
            message=(
                "bond graph is disconnected; relative lattice image between "
                f"components cannot be inferred (components={comps})"
            ),
        )

    # Candidate sets for both directions of each bond.
    cand: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for i, j in clean_bonds:
        cand[(i, j)] = lattice_candidates(raw[j] - raw[i], cell, max_bond_nm, tol)
        cand[(j, i)] = lattice_candidates(raw[i] - raw[j], cell, max_bond_nm, tol)
    edge_candidates = {
        _edge_key(i, j): [list(x) for x in cand[(i, j)]] for i, j in clean_bonds
    }
    edge_candidates.update(
        {_edge_key(j, i): [list(x) for x in cand[(j, i)]] for i, j in clean_bonds}
    )

    # Deterministic BFS spanning tree.
    root = 0
    parent: dict[int, int | None] = {root: None}
    order = [root]
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                order.append(v)
                q.append(v)
    if len(parent) != n:  # defensive; connectivity already checked
        return FrameReconstruction(
            status="disconnected_graph",
            message="bond graph traversal did not reach every atom",
            edge_candidates=edge_candidates,
        )

    solutions: list[np.ndarray] = []
    truncated = False
    assigned: dict[int, np.ndarray] = {}

    def rec(k: int) -> None:
        nonlocal truncated
        if len(solutions) >= SOLUTION_LIMIT:
            truncated = True
            return
        if k == n:
            solutions.append(
                np.asarray([assigned[i].copy() for i in range(n)], dtype=np.int64)
            )
            return
        u = order[k]
        p = parent[u]
        if p is None:
            assigned[u] = np.zeros(3, dtype=np.int64)
            rec(k + 1)
            del assigned[u]
            return
        for m in cand[(p, u)]:
            assigned[u] = assigned[p] + np.asarray(m, dtype=np.int64)
            ok = True
            for v in adj[u]:
                if v in assigned:
                    required = tuple(
                        int(a - b) for a, b in zip(assigned[u], assigned[v])
                    )
                    if required not in cand[(v, u)]:
                        ok = False
                        break
            if ok:
                rec(k + 1)
            del assigned[u]
            if len(solutions) >= SOLUTION_LIMIT:
                truncated = True
                return

    rec(0)

    if len(solutions) == 0:
        return FrameReconstruction(
            status="no_consistent_lattice_image",
            solution_count=0,
            edge_candidates=edge_candidates,
            conflict=_explain_failure(
                raw, clean_bonds, parent, order, cand, max_bond_nm
            ),
            message="no integer lattice assignment satisfies every bond",
        )

    # Choose a canonical solution only when it is unique.
    if len(solutions) > 1:
        return FrameReconstruction(
            status="ambiguous_within_frame",
            solution_count=len(solutions),
            truncated=truncated,
            solutions=solutions,
            edge_candidates=edge_candidates,
            message=(
                "multiple within-frame lattice assignments satisfy the bond graph "
                "(excluding the arbitrary global translation)"
            ),
        )

    off = solutions[0]
    lengths: list[float] = []
    for i, j in clean_bonds:
        yj = raw[j] + cell.matrix @ off[j].astype(float)
        yi = raw[i] + cell.matrix @ off[i].astype(float)
        lengths.append(float(np.linalg.norm(yj - yi)))
    return FrameReconstruction(
        status="ok",
        offsets=off,
        bond_lengths_nm=lengths,
        solutions=solutions,
        solution_count=1,
        edge_candidates=edge_candidates,
        message="unique lattice assignment up to the arbitrary global translation",
    )


# ---------------------------------------------------------------------------
# Whole-dataset analysis
# ---------------------------------------------------------------------------


def _com(masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return masses @ positions / float(masses.sum())


def _frac_diff_wrap(delta: np.ndarray) -> np.ndarray:
    """Nearest-integer wrap implied by a fractional step."""
    return -np.rint(delta).astype(np.int64)


def compute_msd(
    times: np.ndarray,
    com_unwrapped: np.ndarray,
    segment_ids: np.ndarray,
    lags: Iterable[float],
    lag_tol: float,
) -> dict[str, Any]:
    """MSD of already-unwrapped centres of mass, within segments only.

    A lag is reported as covered only if at least one pair exists inside one
    continuous segment with ``|dt - lag| <= lag_tol``.  Uncovered lags have
    ``msd_nm2 = None``; they are never filled with zero.
    """

    result: dict[str, Any] = {}
    nframes = len(times)
    for lag in lags:
        pairs: list[dict[str, Any]] = []
        sq: list[float] = []
        for i in range(nframes):
            if segment_ids[i] < 0:
                continue
            for j in range(i + 1, nframes):
                if segment_ids[j] != segment_ids[i]:
                    continue
                dt = float(times[j] - times[i])
                if abs(dt - lag) <= lag_tol:
                    d = com_unwrapped[j] - com_unwrapped[i]
                    val = float(d @ d)
                    pairs.append(
                        {
                            "from_frame": i,
                            "to_frame": j,
                            "dt_ps": dt,
                            "squared_displacement_nm2": val,
                        }
                    )
                    sq.append(val)
        key = _lag_key(lag)
        result[key] = {
            "lag_ps": float(lag),
            "covered": bool(sq),
            "pair_count": len(sq),
            "pair_indices": [[p["from_frame"], p["to_frame"]] for p in pairs],
            "pairs": pairs,
            "msd_nm2": float(np.mean(sq)) if sq else None,
        }
    return result


def _lag_key(lag: float) -> str:
    if float(lag).is_integer():
        return str(int(lag))
    return repr(float(lag))


def analyze_dataset(
    positions_nm: np.ndarray,
    time_ps: np.ndarray,
    cell: Cell,
    topology: dict[str, Any],
    options: AnalysisOptions,
) -> dict[str, Any]:
    """Reconstruct molecules and analyse continuity, translations and MSD.

    ``positions_nm`` is frame x atom x xyz, already folded; atom order follows
    ``topology["atoms"]``.  Bond indices are zero-based atom indices.
    """

    positions = np.asarray(positions_nm, dtype=float)
    times = np.asarray(time_ps, dtype=float)
    nframes, natoms, _ = positions.shape

    atoms = topology["atoms"]
    atom_ids = [str(a["atom_id"]) for a in atoms]
    atom_mol_ids = [str(a["molecule_id"]) for a in atoms]
    masses_all = np.asarray([float(a["mass"]) for a in atoms], dtype=float)

    molecule_order: list[str] = []
    mol_atoms: dict[str, list[int]] = {}
    for idx, mid in enumerate(atom_mol_ids):
        if mid not in mol_atoms:
            mol_atoms[mid] = []
            molecule_order.append(mid)
        mol_atoms[mid].append(idx)

    # Local (molecule-order) bond list per molecule.
    mol_bonds_global: dict[str, list[tuple[int, int]]] = {m: [] for m in molecule_order}
    for i, j in topology["bonds_zero_based"]:
        i, j = int(i), int(j)
        mid = atom_mol_ids[i]
        if atom_mol_ids[j] != mid:
            raise AnalysisError(
                f"bond ({i},{j}) crosses molecules {mid!r} and {atom_mol_ids[j]!r}"
            )
        mol_bonds_global[mid].append((i, j))

    # Time-step metadata used only for reporting/validation.
    diffs = np.diff(times) if nframes > 1 else np.asarray([])
    nominal = float(np.min(diffs[diffs > 0])) if np.any(diffs > 0) else None

    molecules_report: dict[str, Any] = {}
    atoms_total_offsets = np.zeros((nframes, natoms, 3), dtype=np.int64)
    atoms_valid = np.zeros((nframes, natoms), dtype=bool)
    positions_unwrapped = np.full_like(positions, np.nan)
    com_unwrapped_all = np.full((nframes, len(molecule_order), 3), np.nan)
    com_valid_all = np.zeros((nframes, len(molecule_order)), dtype=bool)
    segment_matrix = np.full((nframes, len(molecule_order)), -1, dtype=np.int64)
    cumulative_matrix = np.zeros((nframes, len(molecule_order), 3), dtype=np.int64)
    cumulative_valid = np.zeros((nframes, len(molecule_order)), dtype=bool)
    step_matrix = np.zeros((nframes, len(molecule_order), 3), dtype=np.int64)
    step_valid = np.zeros((nframes, len(molecule_order)), dtype=bool)
    within_ok_matrix = np.zeros((nframes, len(molecule_order)), dtype=bool)

    pooled_msd_pairs: dict[str, list[float]] = {
        _lag_key(lag): [] for lag in options.lag_times_ps
    }
    pooled_msd_pair_counts_by_mol: dict[str, dict[str, int]] = {
        _lag_key(lag): {} for lag in options.lag_times_ps
    }

    all_boundary_crossings: list[dict[str, Any]] = []

    for mol_index, mid in enumerate(molecule_order):
        atom_indices = mol_atoms[mid]
        local_global = {g: k for k, g in enumerate(atom_indices)}
        masses = masses_all[atom_indices]
        bonds_local = [
            (local_global[i], local_global[j]) for i, j in mol_bonds_global[mid]
        ]

        frame_entries: list[dict[str, Any]] = []
        # Within-frame reconstruction is independent per frame.
        for f in range(nframes):
            raw = positions[f, atom_indices]
            recon = reconstruct_molecule_frame(
                raw,
                bonds_local,
                cell,
                options.max_bond_nm,
                options.dist_tol,
            )
            entry: dict[str, Any] = {
                "frame_index": f,
                "time_ps": float(times[f]),
                "within_frame_status": recon.status,
                "solution_count": recon.solution_count,
                "solution_count_truncated": recon.truncated,
                "message": recon.message,
                "edge_candidates": recon.edge_candidates,
                "conflict": recon.conflict,
                "ok": recon.ok,
                "atom_lattice_offsets_within_frame": None,
                "reconstructed_atoms_nm": None,
                "bond_lengths_nm": None,
                "com_nm": None,
                "com_unwrapped_nm": None,
                "segment_id": None,
                "cumulative_lattice_translation": None,
                "step_lattice_translation": None,
                "transition_index": None,
            }
            if recon.ok:
                assert recon.offsets is not None
                y = raw + cell.shift_cart(recon.offsets)
                com = _com(masses, y)
                entry["ok"] = True
                entry["atom_lattice_offsets_within_frame"] = recon.offsets.tolist()
                entry["reconstructed_atoms_nm"] = y.tolist()
                entry["bond_lengths_nm"] = [float(x) for x in recon.bond_lengths_nm or []]
                entry["com_nm"] = com.tolist()
                within_ok_matrix[f, mol_index] = True
            else:
                within_ok_matrix[f, mol_index] = False
            frame_entries.append(entry)

        # ------------------------------------------------------------------
        # Continuity: unique lattice translation or break.
        # ------------------------------------------------------------------
        segments: list[dict[str, Any]] = []
        breaks: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        current_segment: dict[str, Any] | None = None
        next_segment_id = 0
        cumulative: np.ndarray | None = None

        def start_segment(f: int, reason: str | None) -> None:
            nonlocal current_segment, next_segment_id, cumulative
            current_segment = {
                "segment_id": next_segment_id,
                "frame_indices": [f],
                "start_frame": f,
                "start_time_ps": float(times[f]),
                "start_reason": reason,
            }
            next_segment_id += 1
            cumulative = np.zeros(3, dtype=np.int64)
            frame_entries[f]["segment_id"] = current_segment["segment_id"]
            frame_entries[f]["cumulative_lattice_translation"] = [0, 0, 0]
            frame_entries[f]["step_lattice_translation"] = None
            frame_entries[f]["transition_index"] = None

        def close_segment(end_f: int, end_reason: str, detail: dict[str, Any] | None = None) -> None:
            nonlocal current_segment, cumulative
            if current_segment is None:
                return
            current_segment["end_frame"] = end_f
            current_segment["end_time_ps"] = float(times[end_f])
            current_segment["end_reason"] = end_reason
            if detail is not None:
                current_segment["end_detail"] = detail
                breaks.append(
                    {
                        "segment_id": current_segment["segment_id"],
                        "after_frame": end_f,
                        "before_frame": end_f + 1,
                        "reason": end_reason,
                        "detail": detail,
                    }
                )
            segments.append(current_segment)
            current_segment = None
            cumulative = None

        for f in range(nframes):
            entry = frame_entries[f]
            if f == 0:
                if entry["ok"]:
                    start_segment(0, None)
                continue

            prev = frame_entries[f - 1]
            dt = float(times[f] - times[f - 1])
            step: dict[str, Any] = {
                "from_frame": f - 1,
                "to_frame": f,
                "dt_ps": dt,
                "nominal_interval_multiple": (
                    float(dt / nominal) if nominal not in (None, 0) else None
                ),
                "is_multi_interval": bool(
                    nominal not in (None, 0)
                    and abs(dt / nominal - round(dt / nominal)) <= 1e-9
                    and round(dt / nominal) >= 2
                ),
                "exceeds_max_gap": bool(dt > options.max_gap_ps + options.dist_tol),
                "displacement_radius_nm": None,
                "com_delta_previous_gauge_nm": None,
                "candidate_lattice_translations": None,
                "candidate_count": None,
                "chosen_lattice_translation": None,
                "unwrapped_com_displacement_nm": None,
                "decision": None,
                "reason": None,
                "detail": None,
            }

            if not prev["ok"]:
                step["decision"] = "break"
                step["reason"] = "previous_frame_within_frame_reconstruction_failed"
                step["detail"] = {"status": prev["within_frame_status"]}
                if entry["ok"]:
                    start_segment(
                        f,
                        "previous_frame_within_frame_reconstruction_failed",
                    )
                steps.append(step)
                frame_entries[f]["transition_index"] = len(steps) - 1
                continue

            if prev["segment_id"] is None:  # defensive: every ok frame starts a segment
                step["decision"] = "break"
                step["reason"] = "previous_frame_not_in_segment"
                if entry["ok"]:
                    start_segment(f, "previous_frame_not_in_segment")
                steps.append(step)
                frame_entries[f]["transition_index"] = len(steps) - 1
                continue

            if not entry["ok"]:
                step["decision"] = "break"
                step["reason"] = "current_frame_within_frame_reconstruction_failed"
                step["detail"] = {"status": entry["within_frame_status"]}
                close_segment(
                    f - 1,
                    "current_frame_within_frame_reconstruction_failed",
                    step["detail"],
                )
                steps.append(step)
                frame_entries[f]["transition_index"] = len(steps) - 1
                continue

            if dt > options.max_gap_ps + options.dist_tol:
                step["decision"] = "break"
                step["reason"] = "gap_exceeds_max_allowed"
                step["detail"] = {
                    "dt_ps": dt,
                    "max_gap_ps": options.max_gap_ps,
                    "excess_ps": dt - options.max_gap_ps,
                }
                close_segment(f - 1, "gap_exceeds_max_allowed", step["detail"])
                start_segment(f, "gap_exceeds_max_allowed")
                steps.append(step)
                frame_entries[f]["transition_index"] = len(steps) - 1
                continue

            # Feasible unique-translation test on centre-of-mass images.
            radius = options.max_com_speed_nm_per_ps * dt
            com_prev = np.asarray(prev["com_nm"], dtype=float)
            com_cur = np.asarray(entry["com_nm"], dtype=float)
            delta = com_cur - com_prev
            cands, meta = lattice_candidates(
                delta, cell, radius, options.dist_tol, return_meta=True
            )
            step["displacement_radius_nm"] = float(radius)
            step["com_delta_previous_gauge_nm"] = delta.tolist()
            step["candidate_lattice_translations"] = [list(x) for x in cands]
            step["candidate_count"] = len(cands)
            step["search_box"] = meta

            if len(cands) == 1:
                s = np.asarray(cands[0], dtype=np.int64)
                assert cumulative is not None
                cumulative = cumulative + s
                step["decision"] = "continue"
                step["reason"] = "unique_feasible_lattice_translation"
                step["chosen_lattice_translation"] = [int(x) for x in s]
                disp = delta + cell.shift_cart(s)
                step["unwrapped_com_displacement_nm"] = disp.tolist()
                frame_entries[f]["segment_id"] = frame_entries[f - 1]["segment_id"]
                frame_entries[f]["cumulative_lattice_translation"] = [
                    int(x) for x in cumulative
                ]
                frame_entries[f]["step_lattice_translation"] = [int(x) for x in s]
                frame_entries[f]["transition_index"] = len(steps)
                assert current_segment is not None
                current_segment["frame_indices"].append(f)
            else:
                if len(cands) == 0:
                    reason = "no_feasible_lattice_translation"
                    # Closest image distance, purely for the break explanation.
                    # The rigorous box can be empty, so guard with the integer
                    # point nearest the fractional centre plus its neighbours.
                    centre = np.asarray(meta["centre_frac"], dtype=float)
                    c0 = np.rint(centre).astype(np.int64)
                    guard = np.asarray(
                        list(
                            itertools.product(
                                range(int(c0[0]) - 1, int(c0[0]) + 2),
                                range(int(c0[1]) - 1, int(c0[1]) + 2),
                                range(int(c0[2]) - 1, int(c0[2]) + 2),
                            )
                        ),
                        dtype=float,
                    )
                    nearest = float(
                        np.min(np.linalg.norm(delta + cell.shift_cart(guard), axis=1))
                    )
                    step["detail"] = {
                        "displacement_radius_nm": radius,
                        "candidate_count": 0,
                        "nearest_translation_distance_nm": nearest,
                        "note": (
                            "No lattice image is within the allowed COM "
                            "displacement; the step is not continued."
                        ),
                    }
                else:
                    reason = "ambiguous_lattice_translation"
                    step["detail"] = {
                        "displacement_radius_nm": radius,
                        "candidate_count": len(cands),
                        "candidates": [list(x) for x in cands],
                        "note": (
                            "Multiple lattice images satisfy the maximum COM "
                            "displacement; the nearest image is NOT selected. "
                            "This step cannot be continued."
                        ),
                    }
                step["decision"] = "break"
                step["reason"] = reason
                close_segment(f - 1, reason, step["detail"])
                start_segment(f, reason)

            steps.append(step)
            frame_entries[f]["transition_index"] = len(steps) - 1

        if current_segment is not None:
            close_segment(nframes - 1, "end_of_trajectory", None)

        # ------------------------------------------------------------------
        # Materialise unwrapped coordinates and centre-of-mass trajectory.
        # ------------------------------------------------------------------
        com_unwrapped = np.full((nframes, 3), np.nan)
        for f, entry in enumerate(frame_entries):
            if not entry["ok"] or entry["segment_id"] is None:
                continue
            A = np.asarray(entry["cumulative_lattice_translation"], dtype=np.int64)
            mol_offsets = np.asarray(
                entry["atom_lattice_offsets_within_frame"], dtype=np.int64
            )
            total_offsets = mol_offsets + A[None, :]
            raw = positions[f, atom_indices]
            unwrapped = raw + cell.shift_cart(total_offsets)
            entry["atom_lattice_offsets_total"] = total_offsets.tolist()
            entry["unwrapped_atoms_nm"] = unwrapped.tolist()
            com_cart = _com(masses, unwrapped)
            com_unwrapped[f] = com_cart
            entry["com_unwrapped_nm"] = com_cart.tolist()

            atoms_total_offsets[f, atom_indices] = total_offsets
            atoms_valid[f, atom_indices] = True
            positions_unwrapped[f, atom_indices] = unwrapped
            com_unwrapped_all[f, mol_index] = com_cart
            com_valid_all[f, mol_index] = True
            segment_matrix[f, mol_index] = int(entry["segment_id"])
            cumulative_matrix[f, mol_index] = A
            cumulative_valid[f, mol_index] = True
            if entry["step_lattice_translation"] is not None:
                step_matrix[f, mol_index] = np.asarray(
                    entry["step_lattice_translation"], dtype=np.int64
                )
                step_valid[f, mol_index] = True

        # Boundary crossings in the folded representation.
        #
        # For a same-segment step the exact number of lattice images crossed by
        # an atom is the change in its absolute integer lattice offset, since
        # unwrapped motion is continuous there:  frac_delta = frac(unwrapped
        # delta) - offset_delta.  Across a break the cumulative gauge resets, so
        # only the folded jump is well defined; we label those as such instead
        # of inventing a crossing count.
        frac_all = cell.cart_to_frac(
            positions[:, atom_indices, :].reshape(-1, 3)
        ).reshape(nframes, len(atom_indices), 3)
        crossings: list[dict[str, Any]] = []
        for f in range(1, nframes):
            dfrac = frac_all[f] - frac_all[f - 1]
            same_segment = (
                frame_entries[f]["ok"]
                and frame_entries[f - 1]["ok"]
                and frame_entries[f]["segment_id"] is not None
                and frame_entries[f]["segment_id"]
                == frame_entries[f - 1]["segment_id"]
            )
            nearest_wrap = _frac_diff_wrap(dfrac)
            for local, a_index in enumerate(atom_indices):
                if same_segment:
                    total_now = np.asarray(
                        frame_entries[f]["atom_lattice_offsets_total"], dtype=np.int64
                    )
                    total_prev = np.asarray(
                        frame_entries[f - 1]["atom_lattice_offsets_total"],
                        dtype=np.int64,
                    )
                    crossed = total_now[local] - total_prev[local]
                    exact = True
                else:
                    crossed = nearest_wrap[local]
                    exact = False
                comps = [k for k in range(3) if crossed[k] != 0]
                if not comps:
                    continue
                crossings.append(
                    {
                        "molecule_id": mid,
                        "atom_index": int(a_index),
                        "atom_id": atom_ids[a_index],
                        "from_frame": f - 1,
                        "to_frame": f,
                        "components": comps,
                        "folded_fractional_delta": {
                            str(k): float(dfrac[local, k]) for k in range(3)
                        },
                        "crossed_lattice_images": [int(x) for x in crossed],
                        "count_is_exact": bool(exact),
                        "within_same_continuous_segment": bool(same_segment),
                        "crossing_basis": (
                            "change_in_absolute_lattice_offset"
                            if exact
                            else "nearest_wrap_of_folded_jump; true count "
                            "undetermined across a break"
                        ),
                    }
                )
        all_boundary_crossings.extend(crossings)

        # Fold-back / traceability check.
        fold_back_max_residual = 0.0
        fold_back_ok = True
        fold_back_checked = 0
        for f, entry in enumerate(frame_entries):
            if not entry["ok"] or entry["unwrapped_atoms_nm"] is None:
                continue
            total = np.asarray(entry["atom_lattice_offsets_total"], dtype=np.int64)
            raw = positions[f, atom_indices]
            unwrapped = np.asarray(entry["unwrapped_atoms_nm"], dtype=float)
            recomposed = raw + cell.shift_cart(total)
            residual = float(np.max(np.abs(recomposed - unwrapped)))
            frac_back = cell.cart_to_frac(unwrapped) - total
            frac_folded = cell.cart_to_frac(raw)
            fold_res = float(np.max(np.abs(frac_back - frac_folded)))
            fold_back_max_residual = max(fold_back_max_residual, residual, fold_res)
            fold_back_checked += 1
            if residual > 1e-9 or fold_res > 1e-9:
                fold_back_ok = False

        # MSD per molecule.
        msd = compute_msd(
            times,
            com_unwrapped,
            segment_matrix[:, mol_index],
            options.lag_times_ps,
            options.lag_tolerance_ps,
        )
        for lag_key, item in msd.items():
            for pair in item["pairs"]:
                pooled_msd_pairs[lag_key].append(pair["squared_displacement_nm2"])
            pooled_msd_pair_counts_by_mol[lag_key][mid] = item["pair_count"]

        # Segment summary with net displacements.
        seg_summary: list[dict[str, Any]] = []
        for seg in segments:
            fi = seg["frame_indices"]
            first, last = fi[0], fi[-1]
            net = None
            if frame_entries[first]["com_unwrapped_nm"] is not None and frame_entries[last][
                "com_unwrapped_nm"
            ] is not None:
                d = np.asarray(frame_entries[last]["com_unwrapped_nm"]) - np.asarray(
                    frame_entries[first]["com_unwrapped_nm"]
                )
                net = d.tolist()
            seg_summary.append(
                {
                    **seg,
                    "frame_count": len(fi),
                    "end_reason": seg.get("end_reason", "end_of_trajectory"),
                    "net_com_displacement_nm": net,
                }
            )
        # Replace segments list with enriched one for reporting.
        segments = seg_summary

        molecules_report[mid] = {
            "molecule_id": mid,
            "atom_indices": [int(x) for x in atom_indices],
            "atom_ids": [atom_ids[i] for i in atom_indices],
            "masses": [float(x) for x in masses],
            "total_mass": float(masses.sum()),
            "bond_indices_global": [
                [int(i), int(j)] for i, j in mol_bonds_global[mid]
            ],
            "bond_indices_local": [[int(i), int(j)] for i, j in bonds_local],
            "frames": frame_entries,
            "steps": steps,
            "segments": segments,
            "breaks": breaks,
            "boundary_crossings": crossings,
            "msd": msd,
            "feasible_all_frames_and_steps": bool(
                all(e["ok"] for e in frame_entries)
                and all(s["decision"] == "continue" for s in steps)
            ),
            "fold_back_check": {
                "ok": fold_back_ok,
                "frames_checked": fold_back_checked,
                "max_residual_nm": fold_back_max_residual,
            },
        }

    pooled_msd: dict[str, Any] = {}
    for lag in options.lag_times_ps:
        key = _lag_key(lag)
        vals = pooled_msd_pairs[key]
        pooled_msd[key] = {
            "lag_ps": float(lag),
            "covered": bool(vals),
            "pair_count": len(vals),
            "msd_nm2": float(np.mean(vals)) if vals else None,
            "pair_counts_by_molecule": pooled_msd_pair_counts_by_mol[key],
        }

    summary = {
        "n_frames": int(nframes),
        "n_atoms": int(natoms),
        "n_molecules": len(molecule_order),
        "molecules": molecule_order,
        "times_ps": [float(x) for x in times],
        "nominal_interval_ps": nominal,
        "max_gap_ps": options.max_gap_ps,
        "max_com_speed_nm_per_ps": options.max_com_speed_nm_per_ps,
        "max_bond_nm": options.max_bond_nm,
        "origin_gauge": (
            "Each continuous segment starts at an arbitrary absolute lattice "
            "image (the folded input does not determine it). All reported "
            "absolute coordinates are in that per-segment gauge; shapes, bond "
            "lengths and within-segment displacements are gauge-invariant."
        ),
        "all_frames_reconstructed": bool(np.all(within_ok_matrix)),
        "all_steps_continuous": bool(
            all(
                entry["decision"] == "continue"
                for mol in molecules_report.values()
                for entry in mol["steps"]
            )
            if nframes > 1
            else True
        ),
        "molecules_with_breaks": [
            mid
            for mid, mol in molecules_report.items()
            if len(mol["breaks"]) > 0
        ],
        "pooled_msd": pooled_msd,
        "boundary_crossing_count": len(all_boundary_crossings),
    }

    return {
        "summary": summary,
        "molecules": molecules_report,
        "boundary_crossings": all_boundary_crossings,
        "arrays": {
            "frame_index": np.arange(nframes, dtype=np.int64),
            "atom_index": np.arange(natoms, dtype=np.int64),
            "atom_ids": np.asarray(atom_ids),
            "atom_molecule_ids": np.asarray(atom_mol_ids),
            "molecule_ids": np.asarray(molecule_order),
            "time_ps": times,
            "positions_folded_nm": positions,
            "positions_unwrapped_nm": positions_unwrapped,
            "atom_unwrapped_valid": atoms_valid,
            "atom_lattice_offsets_total": atoms_total_offsets,
            "com_unwrapped_nm": com_unwrapped_all,
            "com_valid": com_valid_all,
            "segment_id": segment_matrix,
            "cumulative_lattice_translation": cumulative_matrix,
            "cumulative_translation_valid": cumulative_valid,
            "step_lattice_translation": step_matrix,
            "step_translation_valid": step_valid,
            "cell_columns_nm": cell.matrix,
        },
    }
