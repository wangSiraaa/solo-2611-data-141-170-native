"""Shared pytest helpers.

The helpers here deliberately re-derive everything from first principles
(plain NumPy arithmetic on the stored constraints) instead of calling back
into the package's LP code, so the tests are independent witnesses rather
than tautologies.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from satflow.plan import parse_plan
from satflow.analysis import analyze_events
from satflow.solver import build_constraints

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")


@pytest.fixture(scope="session")
def fixture_plan():
    with open(os.path.join(FIXTURE_DIR, "plan.json"), encoding="utf-8") as handle:
        return parse_plan(json.load(handle), source="fixtures/plan.json")


@pytest.fixture(scope="session")
def fixture_data():
    import fcsparser

    _, frame = fcsparser.parse(os.path.join(FIXTURE_DIR, "events.fcs"), reformat_meta=False)
    return frame[["FL1", "FL2", "FL3", "FL4"]].to_numpy(dtype=np.float32)


@pytest.fixture(scope="session")
def fixture_results(fixture_plan, fixture_data):
    return analyze_events(fixture_data, fixture_plan)


def make_plan_dict(
    detectors,
    markers,
    spillover,
    gates,
    *,
    saturation_value=100.0,
    absolute_tolerance=0.5,
    feasibility_tolerance=1e-7,
):
    return {
        "detectors": detectors,
        "markers": markers,
        "spillover_rows_detectors_cols_markers": [list(map(float, row)) for row in spillover],
        "observation_model": "detector_vector = S @ nonnegative_marker_vector",
        "saturation_value": float(saturation_value),
        "absolute_tolerance": float(absolute_tolerance),
        "feasibility_tolerance": float(feasibility_tolerance),
        "saturation_rule": "observed >= saturation means lower bound only",
        "normal_rule": "observed +/- absolute_tolerance",
        "gates": gates,
    }


def plan_from(detectors, markers, spillover, gates, **kwargs):
    return parse_plan(make_plan_dict(detectors, markers, spillover, gates, **kwargs))


def write_minimal_fcs(path, data, channel_names, *, version="FCS3.1", datatype="F",
                      byteord="1,2,3,4", mode="L", par=None, tot=None, pn_names=None):
    """Write a minimal but spec-shaped FCS 3.1 list-mode float32 file."""
    assert data.dtype == np.float32
    n_par = par if par is not None else data.shape[1]
    n_events = tot if tot is not None else data.shape[0]
    names = pn_names if pn_names is not None else channel_names

    pairs = [
        ("$BEGINANALYSIS", "0"),
        ("$ENDANALYSIS", "0"),
        ("$BEGINSTEXT", "0"),
        ("$ENDSTEXT", "0"),
        ("$BYTEORD", byteord),
        ("$DATATYPE", datatype),
        ("$MODE", mode),
        ("$NEXTDATA", "0"),
        ("$PAR", str(n_par)),
        ("$TOT", str(n_events)),
    ]
    for i, name in enumerate(names, start=1):
        pairs.extend([
            (f"$P{i}B", "32"),
            (f"$P{i}E", "0,0"),
            (f"$P{i}N", name),
            (f"$P{i}R", "100"),
        ])
    # Offset keywords always carry 8-character values, so TEXT length is fixed
    # before the real offsets are known.
    pairs.append(("$BEGINDATA", "00000000"))
    pairs.append(("$ENDDATA", "00000000"))
    text = "|" + "|".join(f"{k}|{v}" for k, v in pairs) + "|"
    begin_data = 256 + len(text.encode("ascii"))
    end_data = begin_data + data.nbytes - 1
    pairs[-2] = ("$BEGINDATA", f"{begin_data:08d}")
    pairs[-1] = ("$ENDDATA", f"{end_data:08d}")
    text = "|" + "|".join(f"{k}|{v}" for k, v in pairs) + "|"
    text_bytes = text.encode("ascii")
    assert 256 + len(text_bytes) == begin_data

    header = (
        f"{version}    "
        f"{'256':>8}"
        f"{256 + len(text_bytes) - 1:>8}"
        f"{begin_data:>8}"
        f"{end_data:>8}"
        f"{'0':>8}{'0':>8}"
    ).encode("ascii").ljust(256, b" ")
    assert len(header) == 256

    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(text_bytes)
        handle.write(data.tobytes())
    return path


def independent_feasible(point, observation, plan, eps=1e-7):
    """Re-check a marker vector against the observation model using NumPy only.

    Mirrors the documented rules: normal rows are two-sided bands, saturated
    rows are lower bounds, markers are non-negative.
    """
    x = np.asarray(point, dtype=float)
    if (x < -eps).any():
        return False
    detectors = np.asarray(plan.spillover) @ x
    for i, reading in enumerate(observation):
        y = float(reading)
        value = float(detectors[i])
        if y >= plan.saturation_value:
            if value < plan.saturation_value - plan.absolute_tolerance - eps:
                return False
        else:
            lo = y - plan.absolute_tolerance
            hi = y + plan.absolute_tolerance
            if value < lo - eps or value > hi + eps:
                return False
    return True


def independent_in_gate(point, gate, markers, eps=1e-7):
    x = np.asarray(point, dtype=float)
    for marker, (lo, hi) in gate.bounds.items():
        value = x[markers.index(marker)]
        if lo is not None and value < lo - eps:
            return False
        if hi is not None and value > hi + eps:
            return False
    return True


def independent_out_gate(point, gate, markers, eps=1e-7):
    """Strictly outside the closed gate: some bound violated by at least eps."""
    x = np.asarray(point, dtype=float)
    for marker, (lo, hi) in gate.bounds.items():
        value = x[markers.index(marker)]
        if lo is not None and value < lo - eps:
            return True
        if hi is not None and value > hi + eps:
            return True
    return False


def event_constraints(observation, plan):
    return build_constraints(observation, plan)
