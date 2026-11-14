"""Static reliability model with shared common-cause events.

The model is a small deterministic Boolean circuit whose leaves are
independent primitive events.  Exactly two primitive events are driven by
unknown probabilities (shared "surge" / "heat" references); every other
primitive has a known fixed probability.

All probabilistic work in this package is exact marginalisation over the
2**P primitive configurations.  P is small by construction (the task is
limited to two unknown probabilities and a static graph).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

MAX_PRIMITIVES = 16  # 2**16 configurations; the fixtures use 8.


class ValidationError(ValueError):
    """Raised when the graph / events / plan / snapshots are inconsistent."""


# ---------------------------------------------------------------------------
# configuration loading
# ---------------------------------------------------------------------------

def parse_probability(text: str, context: str) -> float:
    raw = str(text).strip()
    if not raw:
        raise ValidationError(f"empty probability for {context}")
    try:
        value = float(Fraction(raw))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValidationError(f"cannot parse probability {text!r} for {context}") from exc
    if not (0.0 <= value <= 1.0):
        raise ValidationError(f"probability {value} for {context} is outside [0, 1]")
    return value


@dataclass(frozen=True)
class Event:
    event_id: str
    fixed_probability: float | None
    unknown_parameter: str | None
    column: int


@dataclass(frozen=True)
class Node:
    id: str
    op: str
    inputs: tuple[str, ...]
    k: int | None


@dataclass(frozen=True)
class PlanConfig:
    parameters: tuple[str, ...]
    bounds: dict[str, tuple[float, float]]
    support_loglik_drop: float
    repair_node: str
    repair_forced_failure: bool

    @property
    def lower(self) -> np.ndarray:
        return np.array([self.bounds[p][0] for p in self.parameters], dtype=float)

    @property
    def upper(self) -> np.ndarray:
        return np.array([self.bounds[p][1] for p in self.parameters], dtype=float)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_plan(path: str | Path) -> PlanConfig:
    raw = load_json(path)
    params_raw = raw.get("parameters")
    if not isinstance(params_raw, dict):
        raise ValidationError("plan.parameters must be an object")
    parameters = tuple(params_raw.keys())
    if len(parameters) != 2:
        raise ValidationError(
            f"exactly two unknown probabilities are in scope, got {parameters}"
        )
    bounds: dict[str, tuple[float, float]] = {}
    for name, pair in params_raw.items():
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValidationError(f"bounds for {name} must be [low, high]")
        lo, hi = float(pair[0]), float(pair[1])
        if not (0.0 <= lo < hi <= 1.0):
            raise ValidationError(f"invalid bounds for {name}: {pair!r}")
        bounds[name] = (lo, hi)
    try:
        drop = float(raw["support_loglik_drop"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError("plan.support_loglik_drop must be numeric") from exc
    if drop < 0:
        raise ValidationError("support_loglik_drop must be non-negative")
    repair = raw.get("one_repair", {})
    node = repair.get("node")
    if not isinstance(node, str) or not node:
        raise ValidationError("plan.one_repair.node must be a node id")
    forced = bool(repair.get("forced_failure_state", False))
    return PlanConfig(parameters, bounds, drop, node, forced)


def load_events(path: str | Path) -> list[Event]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"event_id", "fixed_probability", "unknown_parameter"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValidationError(f"events CSV must contain columns {sorted(required)}")
    events: list[Event] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        name = (row.get("event_id") or "").strip()
        if not name:
            raise ValidationError(f"events row {i + 2}: empty event_id")
        if name in seen:
            raise ValidationError(f"duplicate event_id {name!r}")
        seen.add(name)
        fixed_raw = (row.get("fixed_probability") or "").strip()
        unknown_raw = (row.get("unknown_parameter") or "").strip()
        if bool(fixed_raw) == bool(unknown_raw):
            raise ValidationError(
                f"event {name!r} must have exactly one of fixed_probability / "
                "unknown_parameter"
            )
        fixed = parse_probability(fixed_raw, f"event {name}") if fixed_raw else None
        events.append(Event(name, fixed, unknown_raw or None, len(events)))
    return events


def load_snapshots(path: str | Path, observables: Sequence[str]) -> tuple[list[dict[str, int | None]], list[str]]:
    """Load and validate snapshot rows.

    ``NA`` (and an empty field) means unobserved and is represented as
    ``None``.  It is never converted to 0.
    """

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        rows = list(reader)
    allowed = {"cycle_id", *observables}
    unknown_cols = [c for c in header if c not in allowed]
    if unknown_cols:
        raise ValidationError(f"snapshot has unexpected columns: {unknown_cols}")
    missing = [c for c in observables if c not in header]
    if missing:
        raise ValidationError(f"snapshot is missing observable columns: {missing}")
    parsed: list[dict[str, int | None]] = []
    cycle_ids: list[str] = []
    for i, row in enumerate(rows):
        cycle_id = (row.get("cycle_id") or f"row-{i + 2}").strip()
        cycle_ids.append(cycle_id)
        observations: dict[str, int | None] = {}
        for name in observables:
            raw = (row.get(name) or "").strip()
            if raw == "" or raw.upper() == "NA":
                observations[name] = None
                continue
            if raw not in {"0", "1"}:
                raise ValidationError(
                    f"row {i + 2} ({cycle_id}), column {name}: expected 0/1/NA, got {raw!r}"
                )
            observations[name] = int(raw)
        parsed.append(observations)
    if not parsed:
        raise ValidationError("snapshot contains no cycles")
    return parsed, cycle_ids


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class ReliabilityModel:
    def __init__(
        self,
        system: Mapping[str, Any],
        events: Sequence[Event],
        plan: PlanConfig,
    ) -> None:
        self.system = dict(system)
        self.events = tuple(events)
        self.plan = plan
        self.parameters = plan.parameters

        self.observables = tuple(system.get("observables", ()))
        self.root = system.get("root")
        raw_nodes = system.get("nodes")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise ValidationError("system.nodes must be a non-empty list")
        if not isinstance(self.root, str):
            raise ValidationError("system.root must be a node id")

        self.primitives = tuple(event.event_id for event in self.events)
        primitive_set = set(self.primitives)
        self.nodes: dict[str, Node] = {}
        for raw in raw_nodes:
            if not isinstance(raw, Mapping):
                raise ValidationError("each node must be an object")
            node_id = raw.get("id")
            if not isinstance(node_id, str) or not node_id:
                raise ValidationError("each node needs a non-empty string id")
            if node_id in self.nodes:
                raise ValidationError(f"duplicate node id {node_id!r}")
            if node_id in primitive_set:
                raise ValidationError(
                    f"node id {node_id!r} collides with a primitive event id"
                )
            op = raw.get("op")
            if op not in {"or", "and", "kofn"}:
                raise ValidationError(f"node {node_id!r}: unsupported op {op!r}")
            inputs_raw = raw.get("inputs")
            if not isinstance(inputs_raw, list) or not inputs_raw:
                raise ValidationError(f"node {node_id!r}: inputs must be a non-empty list")
            inputs = tuple(inputs_raw)
            if any(not isinstance(x, str) for x in inputs):
                raise ValidationError(f"node {node_id!r}: inputs must be strings")
            if len(set(inputs)) != len(inputs):
                duplicates = sorted({x for x in inputs if inputs.count(x) > 1})
                raise ValidationError(
                    f"node {node_id!r}: duplicate input(s) {duplicates} in the same gate"
                )
            k: int | None = None
            if op == "kofn":
                k_raw = raw.get("k")
                if not isinstance(k_raw, int) or isinstance(k_raw, bool):
                    raise ValidationError(f"node {node_id!r}: kofn needs integer k")
                if not (1 <= k_raw <= len(inputs)):
                    raise ValidationError(
                        f"node {node_id!r}: k={k_raw} must be in 1..{len(inputs)}"
                    )
                k = k_raw
            self.nodes[node_id] = Node(node_id, op, inputs, k)

        if self.root not in self.nodes:
            raise ValidationError(f"root {self.root!r} is not a node")
        if not self.observables:
            raise ValidationError("system.observables must be non-empty")
        if len(set(self.observables)) != len(self.observables):
            raise ValidationError("system.observables contains duplicates")
        known = set(self.nodes) | primitive_set
        unknown_obs = [o for o in self.observables if o not in known]
        if unknown_obs:
            raise ValidationError(f"observables are not graph nodes: {unknown_obs}")

        self._event_by_id = {e.event_id: e for e in self.events}
        self._validate_unknowns()
        self._validate_inputs(primitive_set)
        self.topo_order = self._topological_order()
        self._enum_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._value_cache: dict[frozenset[str], dict[str, np.ndarray]] = {}

    # -- validation --------------------------------------------------------
    def _validate_unknowns(self) -> None:
        plan_params = set(self.parameters)
        event_params = [e.unknown_parameter for e in self.events if e.unknown_parameter]
        unknown_events: dict[str, list[str]] = {p: [] for p in self.parameters}
        for event in self.events:
            if event.unknown_parameter is None:
                continue
            param = event.unknown_parameter
            if param not in plan_params:
                raise ValidationError(
                    f"event {event.event_id!r} references unknown parameter {param!r} "
                    "not declared in plan"
                )
            unknown_events[param].append(event.event_id)
        for param, names in unknown_events.items():
            if len(names) != 1:
                raise ValidationError(
                    f"parameter {param!r} must be bound by exactly one primitive event, "
                    f"got {names}"
                )
        used = set(event_params)
        if used != plan_params:
            raise ValidationError(
                f"plan parameters {sorted(plan_params)} do not match event-bound "
                f"parameters {sorted(used)}"
            )
        self.parameter_event_columns = tuple(
            self._event_by_id[unknown_events[p][0]].column for p in self.parameters
        )

    def _validate_inputs(self, primitive_set: set[str]) -> None:
        for node in self.nodes.values():
            for inp in node.inputs:
                if inp not in self.nodes and inp not in primitive_set:
                    raise ValidationError(
                        f"node {node.id!r}: input {inp!r} is neither a node nor a "
                        "primitive event"
                    )
        if not self.plan.repair_node:
            raise ValidationError("repair node is empty")
        if self.plan.repair_node not in self.nodes:
            raise ValidationError(
                f"repair node {self.plan.repair_node!r} is not in the graph"
            )
        if self.plan.repair_forced_failure:
            raise ValidationError(
                "only the plan's forced-healthy repair (forced_failure_state=false) "
                "is implemented"
            )

    def _topological_order(self) -> tuple[str, ...]:
        state: dict[str, int] = {}  # 0/absent = new, 1 = visiting, 2 = done
        order: list[str] = []
        primitive_set = set(self.primitives)

        def visit(name: str, stack: tuple[str, ...]) -> None:
            if name in primitive_set:
                return
            if name not in self.nodes:
                raise ValidationError(f"unknown node {name!r}")
            st = state.get(name, 0)
            if st == 1:
                cycle = " -> ".join((*stack, name))
                raise ValidationError(f"cycle detected in graph: {cycle}")
            if st == 2:
                return
            state[name] = 1
            node = self.nodes[name]
            for inp in node.inputs:
                visit(inp, (*stack, name))
            state[name] = 2
            order.append(name)

        for node_id in self.nodes:
            visit(node_id, ())
        return tuple(order)

    # -- enumeration -------------------------------------------------------
    def _enumerate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (configs, fixed_weights, group_code).

        ``configs`` is (2**P, P) boolean.  ``fixed_weights`` is the product of
        fixed-event probabilities for each configuration (unknown events are
        excluded).  ``group_code`` encodes the two unknown-event bits with the
        first plan parameter as the high bit.
        """

        if self._enum_cache is not None:
            return self._enum_cache
        n_primitives = len(self.events)
        if n_primitives > MAX_PRIMITIVES:
            raise ValidationError(
                f"{n_primitives} primitive events exceed the exact-enumeration "
                f"limit of {MAX_PRIMITIVES}"
            )
        total = 1 << n_primitives
        indices = np.arange(total, dtype=np.uint64)
        configs = np.empty((total, n_primitives), dtype=bool)
        for col in range(n_primitives):
            shift = np.uint64(n_primitives - 1 - col)
            configs[:, col] = ((indices >> shift) & np.uint64(1)).astype(bool)
        fixed_weights = np.ones(total, dtype=float)
        for event in self.events:
            if event.fixed_probability is None:
                continue
            p = event.fixed_probability
            fixed_weights *= np.where(configs[:, event.column], p, 1.0 - p)
        group = np.zeros(total, dtype=np.uint8)
        for param_index, col in enumerate(self.parameter_event_columns):
            shift = len(self.parameters) - 1 - param_index
            group |= configs[:, col].astype(np.uint8) << np.uint8(shift)
        self._enum_cache = (configs, fixed_weights, group)
        return self._enum_cache

    @property
    def configs(self) -> np.ndarray:
        return self._enumerate()[0]

    @property
    def fixed_weights(self) -> np.ndarray:
        return self._enumerate()[1]

    @property
    def group_codes(self) -> np.ndarray:
        return self._enumerate()[2]

    def basis(self, theta: Sequence[float]) -> np.ndarray:
        """Probability of each (unknown-bit) group, in group-code order."""

        values = np.asarray(theta, dtype=float).reshape(-1)
        if values.size != len(self.parameters):
            raise ValueError("theta has wrong length")
        if len(values) == 2:
            p0, p1 = float(values[0]), float(values[1])
            return np.array(
                [(1 - p0) * (1 - p1), (1 - p0) * p1, p0 * (1 - p1), p0 * p1],
                dtype=float,
            )
        # Fallback for a future >2 parameter extension (not in scope).
        out = np.ones(1, dtype=float)
        for p in values:
            out = np.concatenate([out * (1.0 - p), out * p])
        return out

    def basis_flat(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Basis matrix (4, N) for arrays of the two parameter values."""

        p0 = np.asarray(first, dtype=float)
        p1 = np.asarray(second, dtype=float)
        return np.stack(
            [
                (1 - p0) * (1 - p1),
                (1 - p0) * p1,
                p0 * (1 - p1),
                p0 * p1,
            ],
            axis=0,
        )

    # -- circuit evaluation ------------------------------------------------
    def evaluate_nodes(
        self, clamped_healthy: Iterable[str] = ()
    ) -> dict[str, np.ndarray]:
        """Evaluate every primitive and node for all primitive configurations."""

        clamped = frozenset(clamped_healthy)
        cached = self._value_cache.get(clamped)
        if cached is not None:
            return cached
        configs = self.configs
        values: dict[str, np.ndarray] = {
            event.event_id: configs[:, event.column] for event in self.events
        }
        for node_id in self.topo_order:
            node = self.nodes[node_id]
            if node_id in clamped:
                values[node_id] = np.zeros(configs.shape[0], dtype=bool)
                continue
            inputs = [values[name] for name in node.inputs]
            if node.op == "or":
                value = np.logical_or.reduce(inputs)
            elif node.op == "and":
                value = np.logical_and.reduce(inputs)
            else:
                total = np.zeros(configs.shape[0], dtype=np.int16)
                for arr in inputs:
                    total += arr.astype(np.int16)
                value = total >= int(node.k)
            values[node_id] = value
        self._value_cache[clamped] = values
        return values

    def risk_coefficients(self, repaired: bool = False) -> np.ndarray:
        """Coefficients b[g] with P(system fails) = sum_g b[g] * basis_g(theta).

        ``repaired`` clamps the plan's node healthy and cuts its input effects;
        shared events continue to act on every other node.
        """

        clamped = (self.plan.repair_node,) if repaired else ()
        values = self.evaluate_nodes(clamped_healthy=clamped)
        system_fails = values[self.root].astype(bool)
        groups = self.group_codes
        weights = self.fixed_weights
        coefficients = np.zeros(2 ** len(self.parameters), dtype=float)
        for code in range(coefficients.size):
            coefficients[code] = float(weights[(groups == code) & system_fails].sum())
        return coefficients

    def validate_graph(self) -> list[str]:
        self._validate_inputs(set(self.primitives))
        self._topological_order()
        return [
            f"graph is a DAG with {len(self.nodes)} nodes and "
            f"{len(self.events)} primitive events"
        ]


def build_model(
    system_path: str | Path,
    events_path: str | Path,
    plan_path: str | Path,
) -> ReliabilityModel:
    plan = load_plan(plan_path)
    events = load_events(events_path)
    system = load_json(system_path)
    return ReliabilityModel(system, events, plan)
