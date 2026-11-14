"""Input loading and validation for the unwrap command."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .core import AnalysisError, AnalysisOptions, Cell


class InputError(Exception):
    """Raised when an input file or its contents are invalid."""


@dataclass
class LoadedInputs:
    positions_nm: np.ndarray
    time_ps: np.ndarray
    cell: Cell
    topology: dict[str, Any]
    options: AnalysisOptions
    plan: dict[str, Any]
    checks: dict[str, Any]
    provenance: dict[str, Any]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        raise InputError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise InputError(f"{label} must be a JSON object: {path}")
    return data


def _require_key(data: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in data:
        raise InputError(f"{label} is missing required key {key!r}")
    return data[key]


def validate_topology(
    topology: dict[str, Any], natoms: int, *, allow_missing_masses: bool = False
) -> tuple[list[str], list[str], np.ndarray, list[tuple[int, int]]]:
    atoms = _require_key(topology, "atoms", "topology")
    bonds = _require_key(topology, "bonds_zero_based", "topology")
    if not isinstance(atoms, list) or not isinstance(bonds, list):
        raise InputError("topology 'atoms' and 'bonds_zero_based' must be lists")
    if len(atoms) != natoms:
        raise InputError(
            f"topology lists {len(atoms)} atoms but positions_nm has {natoms}"
        )

    atom_ids: list[str] = []
    molecule_ids: list[str] = []
    masses = np.zeros(len(atoms), dtype=float)
    for i, atom in enumerate(atoms):
        if not isinstance(atom, dict):
            raise InputError(f"topology atom {i} must be an object")
        for key in ("atom_id", "molecule_id", "mass"):
            if key not in atom:
                if key == "mass" and allow_missing_masses:
                    continue
                raise InputError(f"topology atom {i} is missing {key!r}")
        atom_id = str(atom.get("atom_id", f"atom{i}"))
        mol_id = str(atom.get("molecule_id", "molecule"))
        atom_ids.append(atom_id)
        molecule_ids.append(mol_id)
        if "mass" in atom:
            mass = atom["mass"]
            if isinstance(mass, bool) or not isinstance(mass, (int, float)):
                raise InputError(f"atom {i} mass must be a number; got {mass!r}")
            mass = float(mass)
            if not np.isfinite(mass) or mass <= 0:
                raise InputError(f"atom {i} mass must be finite and > 0; got {mass}")
            masses[i] = mass
        elif not allow_missing_masses:
            raise InputError(f"topology atom {i} is missing 'mass'")

    if len(set(atom_ids)) != len(atom_ids):
        raise InputError("topology atom_id values must be unique")

    clean_bonds: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for k, bond in enumerate(bonds):
        if not isinstance(bond, (list, tuple)) or len(bond) != 2:
            raise InputError(f"bond {k} must be a pair of atom indices")
        i, j = bond
        for v in (i, j):
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise InputError(f"bond {k} index {v!r} is not an integer")
        i, j = int(i), int(j)
        if not (0 <= i < natoms and 0 <= j < natoms):
            raise InputError(
                f"bond {k} references atom index out of range 0..{natoms - 1}: ({i}, {j})"
            )
        if i == j:
            raise InputError(f"bond {k} is a self-loop ({i}, {j})")
        if molecule_ids[i] != molecule_ids[j]:
            raise InputError(
                f"bond {k} ({i}, {j}) crosses molecules "
                f"{molecule_ids[i]!r} and {molecule_ids[j]!r}"
            )
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        clean_bonds.append((i, j))

    # Every atom must belong to a connected bond graph within its molecule.
    adjacency: dict[int, list[int]] = {i: [] for i in range(natoms)}
    for i, j in clean_bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)
    for mol in sorted(set(molecule_ids)):
        members = [i for i, m in enumerate(molecule_ids) if m == mol]
        if len(members) <= 1:
            continue
        start = members[0]
        seen_nodes = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if v not in seen_nodes:
                    seen_nodes.add(v)
                    stack.append(v)
        missing = sorted(set(members) - seen_nodes)
        if missing:
            raise InputError(
                f"molecule {mol!r} bond graph is disconnected; atoms {missing} "
                "have no path to the rest of the molecule, so their relative "
                "lattice image cannot be inferred"
            )

    return atom_ids, molecule_ids, masses, clean_bonds


def validate_plan(plan: dict[str, Any]) -> tuple[AnalysisOptions, dict[str, Any]]:
    required = [
        "max_bond_nm",
        "max_com_speed_nm_per_ps",
        "max_gap_ps",
        "lag_times_ps",
        "lag_tolerance_ps",
    ]
    for key in required:
        if key not in plan:
            raise InputError(f"plan is missing required key {key!r}")
    if plan.get("cell_vectors") != "columns":
        raise InputError(
            "plan['cell_vectors'] must be 'columns'; the cell layout is not negotiable"
        )
    if plan.get("mass_unit") != "relative atomic mass":
        raise InputError(
            "plan['mass_unit'] must be 'relative atomic mass'"
        )

    lags = _require_key(plan, "lag_times_ps", "plan")
    if not isinstance(lags, (list, tuple)) or not lags:
        raise InputError("plan['lag_times_ps'] must be a non-empty list")
    lag_tuple = tuple(float(x) for x in lags)
    if any(x < 0 for x in lag_tuple):
        raise InputError("lag times must be non-negative")

    try:
        options = AnalysisOptions(
            max_bond_nm=float(plan["max_bond_nm"]),
            max_com_speed_nm_per_ps=float(plan["max_com_speed_nm_per_ps"]),
            max_gap_ps=float(plan["max_gap_ps"]),
            lag_times_ps=lag_tuple,
            lag_tolerance_ps=float(plan["lag_tolerance_ps"]),
        )
    except (TypeError, ValueError) as exc:
        raise InputError(f"plan contains non-numeric bounds: {exc}") from exc
    except AnalysisError as exc:
        raise InputError(str(exc)) from exc

    tolerance = float(plan["lag_tolerance_ps"])
    if tolerance < 0:
        raise InputError("plan['lag_tolerance_ps'] must be >= 0")
    return options, {
        "cell_vectors": plan["cell_vectors"],
        "mass_unit": plan["mass_unit"],
        "continuity": plan.get("continuity"),
    }


def load_inputs(
    trajectory: str | Path,
    topology: str | Path,
    plan: str | Path,
    *,
    allow_missing_masses: bool = False,
    max_bond_nm_override: float | None = None,
) -> LoadedInputs:
    traj_path = Path(trajectory)
    topo_path = Path(topology)
    plan_path = Path(plan)

    if not traj_path.exists():
        raise InputError(f"trajectory not found: {traj_path}")
    for path, label in ((topo_path, "topology"), (plan_path, "plan")):
        if not path.exists():
            raise InputError(f"{label} not found: {path}")

    try:
        data = np.load(traj_path, allow_pickle=False)
    except (ValueError, OSError) as exc:
        raise InputError(f"trajectory is not a readable .npz: {traj_path}: {exc}") from exc

    required_arrays = ["positions_nm", "time_ps", "cell_columns_nm"]
    for name in required_arrays:
        if name not in data.files:
            raise InputError(f"trajectory is missing array {name!r}")
    positions = np.asarray(data["positions_nm"], dtype=float)
    times = np.asarray(data["time_ps"], dtype=float)
    cell_matrix = np.asarray(data["cell_columns_nm"], dtype=float)

    if positions.ndim != 3:
        raise InputError(
            f"positions_nm must be frame x atom x xyz; got shape {positions.shape}"
        )
    nframes, natoms, ncomp = positions.shape
    if ncomp != 3:
        raise InputError(f"positions_nm must have xyz=3; got {ncomp}")
    if natoms == 0 or nframes == 0:
        raise InputError("positions_nm must be non-empty")
    if not np.all(np.isfinite(positions)):
        raise InputError("positions_nm contains non-finite values")

    if times.ndim != 1 or len(times) != nframes:
        raise InputError(
            f"time_ps must have {nframes} entries; got shape {times.shape}"
        )
    if not np.all(np.isfinite(times)):
        raise InputError("time_ps contains non-finite values")
    if nframes > 1:
        diffs = np.diff(times)
        if np.any(diffs <= 0):
            raise InputError("time_ps must be strictly increasing")
    else:
        diffs = np.asarray([], dtype=float)

    if cell_matrix.shape != (3, 3):
        raise InputError(
            f"cell_columns_nm must be 3x3 with lattice vectors as columns; "
            f"got shape {cell_matrix.shape}"
        )
    if not np.all(np.isfinite(cell_matrix)):
        raise InputError("cell_columns_nm contains non-finite values")
    try:
        cell = Cell(cell_matrix)
    except AnalysisError as exc:
        raise InputError(str(exc)) from exc
    det = cell.det
    if abs(det) < 1e-12:
        raise InputError(f"cell is singular (det={det}); lattice images are not unique")
    singular_values = np.linalg.svd(cell_matrix, compute_uv=False)
    if singular_values[-1] / singular_values[0] < 1e-10:
        raise InputError("cell is numerically degenerate (condition number too large)")

    topology = _load_json(topo_path, "topology")
    plan = _load_json(plan_path, "plan")
    atom_ids, molecule_ids, masses, clean_bonds = validate_topology(
        topology, natoms, allow_missing_masses=allow_missing_masses
    )
    options, plan_meta = validate_plan(plan)

    # Bond bound precedence: explicit CLI override > topology-specific bound
    # (used by the standalone cycle fixture) > plan bound.
    bond_bound_source = "plan"
    bond_bound = options.max_bond_nm
    if "max_bond_nm" in topology:
        topo_bound = topology["max_bond_nm"]
        if isinstance(topo_bound, bool) or not isinstance(topo_bound, (int, float)):
            raise InputError("topology 'max_bond_nm' must be a number")
        topo_bound = float(topo_bound)
        if not np.isfinite(topo_bound) or topo_bound <= 0:
            raise InputError("topology 'max_bond_nm' must be finite and > 0")
        bond_bound = topo_bound
        bond_bound_source = "topology"
        plan_meta = dict(plan_meta)
        plan_meta["plan_max_bond_nm"] = options.max_bond_nm
    if max_bond_nm_override is not None:
        override = float(max_bond_nm_override)
        if not np.isfinite(override) or override <= 0:
            raise InputError("--max-bond-nm must be finite and > 0")
        bond_bound = override
        bond_bound_source = "command_line"
    if bond_bound != options.max_bond_nm:
        options = AnalysisOptions(
            max_bond_nm=bond_bound,
            max_com_speed_nm_per_ps=options.max_com_speed_nm_per_ps,
            max_gap_ps=options.max_gap_ps,
            lag_times_ps=options.lag_times_ps,
            lag_tolerance_ps=options.lag_tolerance_ps,
            dist_tol=options.dist_tol,
        )
    plan_meta = dict(plan_meta)
    plan_meta["bond_bound_source"] = bond_bound_source

    # Coordinate sanity: folded coordinates should lie in the home cell.
    frac = cell.cart_to_frac(positions.reshape(-1, 3)).reshape(nframes, natoms, 3)
    frac_lo = float(np.min(frac))
    frac_hi = float(np.max(frac))
    outside = bool(frac_lo < -1e-6 or frac_hi >= 1.0 + 1e-6)
    # Fix topology masses to the validated ones (may be zero if allowed).
    topology = dict(topology)
    topology["atoms"] = [
        {**atom, "mass": float(masses[i])} if "mass" in atom else atom
        for i, atom in enumerate(topology["atoms"])
    ]

    provenance = {
        "trajectory_path": str(traj_path.resolve()),
        "topology_path": str(topo_path.resolve()),
        "plan_path": str(plan_path.resolve()),
        "trajectory_sha256": _sha256(traj_path),
        "topology_sha256": _sha256(topo_path),
        "plan_sha256": _sha256(plan_path),
    }
    checks = {
        "n_frames": int(nframes),
        "n_atoms": int(natoms),
        "cell_shape": [3, 3],
        "cell_column_vectors_nm": cell.matrix.T.tolist(),
        "cell_determinant": float(det),
        "cell_condition_number": float(np.linalg.cond(cell_matrix)),
        "cell_is_fixed": True,
        "mass_positive_and_finite": bool(
            np.all(np.isfinite(masses) & (masses > 0))
        ) if not allow_missing_masses else "not_enforced",
        "total_mass": float(np.sum(masses)),
        "bonds_checked": len(clean_bonds),
        "atom_ids_unique": True,
        "bonds_within_range": True,
        "bonds_within_molecule": True,
        "molecules_connected": True,
        "time_strictly_increasing": True,
        "time_diffs_ps": [float(x) for x in diffs],
        "nominal_interval_ps": float(np.min(diffs)) if len(diffs) else None,
        "max_time_gap_ps": float(np.max(diffs)) if len(diffs) else None,
        "folded_coords_inside_home_cell": not outside,
        "folded_fractional_min": frac_lo,
        "folded_fractional_max": frac_hi,
        "plan_meta": plan_meta,
        "allow_missing_masses": allow_missing_masses,
    }
    if not allow_missing_masses and not checks["mass_positive_and_finite"]:
        raise InputError("atom masses must be finite and positive")
    if outside:
        raise InputError(
            "positions_nm are not folded into the home cell "
            f"(fractional range [{frac_lo}, {frac_hi}])"
        )

    # Preserve zero-based bonds as ints.
    topology["bonds_zero_based"] = [[int(i), int(j)] for i, j in clean_bonds]

    return LoadedInputs(
        positions_nm=positions,
        time_ps=times,
        cell=cell,
        topology=topology,
        options=options,
        plan=plan,
        checks=checks,
        provenance=provenance,
    )
