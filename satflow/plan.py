"""Loading and structural validation of the analysis plan (``plan.json``).

The plan supplies the spillover matrix ``S`` (rows = detectors, columns =
markers), the observation-band parameters and a small tree of closed
rectangular gates.  Gates may reference parent gates; the effective region
of every gate is the intersection (tightest bound per marker) of every gate
along its ancestor chain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .errors import StructureError

NUMERIC_KEYS = (
    "saturation_value",
    "absolute_tolerance",
    "feasibility_tolerance",
)


@dataclass(frozen=True)
class Gate:
    """One closed-rectangle gate after parent expansion."""

    gate_id: str
    parent: str | None
    # Effective closed bounds per marker: (lower, upper); None means unbounded.
    bounds: dict[str, tuple[float | None, float | None]]
    # Bounds declared directly on this gate (before inheritance), for reports.
    own_bounds: dict[str, tuple[float | None, float | None]]
    chain: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Plan:
    detectors: list[str]
    markers: list[str]
    spillover: "object"  # numpy.ndarray, shape (n_det, n_marker)
    saturation_value: float
    absolute_tolerance: float
    feasibility_tolerance: float
    gates: dict[str, Gate]
    raw: dict


def load_plan(path: str) -> Plan:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise StructureError(f"{path}: cannot read plan JSON: {exc}") from exc
    return parse_plan(raw, source=path)


def parse_plan(raw: dict, source: str = "plan") -> Plan:
    if not isinstance(raw, dict):
        raise StructureError(f"{source}: plan root must be a JSON object")

    detectors = _require_string_list(raw, "detectors", source)
    markers = _require_string_list(raw, "markers", source)
    # The FCS input contract is four channels; plan parsing itself also
    # supports smaller in-memory models so the joint logic can be checked on
    # hand-drawable low-dimensional examples.
    if not detectors or not markers:
        raise StructureError(f"{source}: detectors and markers must be non-empty")
    if len(detectors) != len(markers):
        raise StructureError(
            f"{source}: {len(detectors)} detectors but {len(markers)} markers; "
            "the square y = S x model needs equal counts"
        )
    if len(set(detectors)) != len(detectors):
        raise StructureError(f"{source}: duplicate detector names {detectors}")
    if len(set(markers)) != len(markers):
        raise StructureError(f"{source}: duplicate marker names {markers}")

    matrix_key = "spillover_rows_detectors_cols_markers"
    matrix = raw.get(matrix_key)
    n_det, n_marker = len(detectors), len(markers)
    if not isinstance(matrix, list) or len(matrix) != n_det:
        raise StructureError(f"{source}: {matrix_key} must be a {n_det}x{n_marker} array")
    rows = []
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != n_marker:
            raise StructureError(f"{source}: matrix row {i} must have {n_marker} numbers")
        try:
            values = [float(v) for v in row]
        except (TypeError, ValueError) as exc:
            raise StructureError(f"{source}: matrix row {i} contains a non-number") from exc
        rows.append(values)

    spillover = np.array(rows, dtype=float)
    if not np.isfinite(spillover).all():
        raise StructureError(f"{source}: spillover matrix contains NaN/Inf")
    if (spillover < 0).any():
        raise StructureError(
            f"{source}: spillover matrix has negative entries; y = S x with S >= 0 is required"
        )

    params = {}
    for key in NUMERIC_KEYS:
        value = raw.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not float(value) >= 0:
            raise StructureError(f"{source}: {key} must be a non-negative number")
        params[key] = float(value)
    if params["absolute_tolerance"] <= 0:
        raise StructureError(f"{source}: absolute_tolerance must be strictly positive")
    if params["saturation_value"] <= 0:
        raise StructureError(f"{source}: saturation_value must be strictly positive")

    gate_defs = raw.get("gates")
    if not isinstance(gate_defs, list) or not gate_defs:
        raise StructureError(f"{source}: gates must be a non-empty list")
    gates = _build_gates(gate_defs, markers, source)

    return Plan(
        detectors=detectors,
        markers=markers,
        spillover=spillover,
        saturation_value=params["saturation_value"],
        absolute_tolerance=params["absolute_tolerance"],
        feasibility_tolerance=params["feasibility_tolerance"],
        gates=gates,
        raw=raw,
    )


def _build_gates(gate_defs: list, markers: list[str], source: str) -> dict[str, Gate]:
    declared: dict[str, dict] = {}
    parents: dict[str, str | None] = {}
    for gate in gate_defs:
        if not isinstance(gate, dict):
            raise StructureError(f"{source}: every gate must be an object")
        gate_id = gate.get("id")
        if not isinstance(gate_id, str) or not gate_id:
            raise StructureError(f"{source}: gate with missing/invalid id: {gate!r}")
        if gate_id in declared:
            raise StructureError(f"{source}: duplicated gate id {gate_id!r}")
        parent = gate.get("parent")
        if parent is not None and not isinstance(parent, str):
            raise StructureError(f"{source}: gate {gate_id} parent must be a string or null")
        bounds_raw = gate.get("closed_bounds", {})
        if not isinstance(bounds_raw, dict) or not bounds_raw:
            raise StructureError(f"{source}: gate {gate_id} needs non-empty closed_bounds")
        own_bounds: dict[str, tuple[float | None, float | None]] = {}
        for marker, pair in bounds_raw.items():
            if marker not in markers:
                raise StructureError(
                    f"{source}: gate {gate_id} references unknown marker {marker!r}"
                )
            if not isinstance(pair, list) or len(pair) != 2:
                raise StructureError(
                    f"{source}: gate {gate_id} bound for {marker} must be [lower, upper]"
                )
            lower, upper = pair
            lower = _bound_number(lower, source, gate_id, marker, "lower")
            upper = _bound_number(upper, source, gate_id, marker, "upper")
            if lower is not None and upper is not None and lower > upper:
                raise StructureError(
                    f"{source}: gate {gate_id} bound for {marker} has lower {lower} > upper {upper}"
                )
            own_bounds[marker] = (lower, upper)
        declared[gate_id] = own_bounds
        parents[gate_id] = parent

    # Resolve parent references, detect cycles and unknown parents.
    expanded: dict[str, Gate] = {}
    for gate_id in declared:
        chain: list[str] = []
        cursor: str | None = gate_id
        seen: set[str] = set()
        while cursor is not None:
            if cursor in seen:
                raise StructureError(
                    f"{source}: parent cycle involving gate {cursor!r} (from {gate_id!r})"
                )
            seen.add(cursor)
            if cursor not in declared:
                raise StructureError(
                    f"{source}: gate {gate_id} references unknown parent {cursor!r}"
                )
            chain.append(cursor)
            cursor = parents[cursor]
        chain.reverse()  # root first, gate itself last

        effective: dict[str, tuple[float | None, float | None]] = {}
        for ancestor in chain:
            for marker, (lower, upper) in declared[ancestor].items():
                lo_prev, hi_prev = effective.get(marker, (None, None))
                lo = lower if lo_prev is None else (lower if lower is not None and lower > lo_prev else lo_prev)
                hi = upper if hi_prev is None else (upper if upper is not None and upper < hi_prev else hi_prev)
                if lo is not None and hi is not None and lo > hi:
                    raise StructureError(
                        f"{source}: gate {gate_id} is empty after parent inheritance "
                        f"(marker {marker}: [{lo}, {hi}])"
                    )
                effective[marker] = (lo, hi)
        expanded[gate_id] = Gate(
            gate_id=gate_id,
            parent=parents[gate_id],
            bounds=effective,
            own_bounds=declared[gate_id],
            chain=tuple(chain),
        )
    return expanded


def _bound_number(value, source, gate_id, marker, side):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructureError(
            f"{source}: gate {gate_id} {side} bound for {marker} must be a number or null"
        )
    number = float(value)
    if not np.isfinite(number):
        raise StructureError(f"{source}: gate {gate_id} {side} bound for {marker} is not finite")
    return number


def _require_string_list(raw: dict, key: str, source: str) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise StructureError(f"{source}: {key} must be a non-empty list of non-empty strings")
    return list(value)
