"""Structural validation: illegal FCS files and illegal plans must be refused."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import write_minimal_fcs
from satflow.errors import StructureError
from satflow.fcsio import load_fcs
from satflow.plan import parse_plan

DETECTORS = ["FL1", "FL2", "FL3", "FL4"]


def _data():
    return np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)


def test_valid_minimal_fcs_loads(tmp_path):
    path = write_minimal_fcs(tmp_path / "ok.fcs", _data(), DETECTORS)
    data, names, text = load_fcs(str(path), DETECTORS)
    assert names == DETECTORS
    assert data.shape == (2, 4) and data.dtype == np.float32
    np.testing.assert_array_equal(data, _data())


def test_channel_order_matched_by_name_not_position(tmp_path):
    # Shuffle the on-disk channel order; load_fcs must reorder to plan order.
    shuffled = ["FL4", "FL2", "FL1", "FL3"]
    original = _data()
    shuffled_data = original[:, [3, 1, 0, 2]].copy()
    path = write_minimal_fcs(tmp_path / "shuffle.fcs", shuffled_data, shuffled, pn_names=shuffled)
    data, names, _ = load_fcs(str(path), DETECTORS)
    assert names == DETECTORS
    # Columns are returned in plan order regardless of on-disk position.
    np.testing.assert_allclose(data[:, 0], original[:, 0])
    np.testing.assert_allclose(data[:, 3], original[:, 3])


def test_missing_detector_channel_refused(tmp_path):
    path = write_minimal_fcs(
        tmp_path / "missing.fcs", _data(), DETECTORS, pn_names=["FL1", "FL2", "FL3", "XX"]
    )
    with pytest.raises(StructureError, match="not found among FCS channels"):
        load_fcs(str(path), DETECTORS)


def test_wrong_version_refused(tmp_path):
    path = write_minimal_fcs(tmp_path / "v30.fcs", _data(), DETECTORS, version="FCS3.0")
    with pytest.raises(StructureError, match="unsupported FCS version"):
        load_fcs(str(path), DETECTORS)


def test_wrong_datatype_refused(tmp_path):
    path = write_minimal_fcs(tmp_path / "i32.fcs", _data(), DETECTORS, datatype="I")
    with pytest.raises(StructureError, match=r"\$DATATYPE"):
        load_fcs(str(path), DETECTORS)


def test_wrong_byteord_refused(tmp_path):
    path = write_minimal_fcs(tmp_path / "be.fcs", _data(), DETECTORS, byteord="4,3,2,1")
    with pytest.raises(StructureError, match=r"\$BYTEORD"):
        load_fcs(str(path), DETECTORS)


def test_wrong_mode_refused(tmp_path):
    # fcsparser rejects histogram mode itself; the file must still surface as
    # a structural refusal rather than a raw traceback.
    path = write_minimal_fcs(tmp_path / "histo.fcs", _data(), DETECTORS, mode="H")
    with pytest.raises(StructureError):
        load_fcs(str(path), DETECTORS)


def test_tot_mismatch_refused(tmp_path):
    # fcsparser reshapes using $TOT and fails first; either way the file is a
    # StructureError rather than a crash or silent truncation.
    path = write_minimal_fcs(tmp_path / "tot.fcs", _data(), DETECTORS, tot=99)
    with pytest.raises(StructureError):
        load_fcs(str(path), DETECTORS)


def test_nonfinite_data_refused(tmp_path):
    bad = _data()
    bad[1, 2] = np.float32(np.nan)
    path = write_minimal_fcs(tmp_path / "nan.fcs", bad, DETECTORS)
    with pytest.raises(StructureError, match="NaN/Inf"):
        load_fcs(str(path), DETECTORS)


def test_negative_spillover_refused():
    with pytest.raises(StructureError, match="negative entries"):
        parse_plan({
            "detectors": ["D1", "D2"], "markers": ["M1", "M2"],
            "spillover_rows_detectors_cols_markers": [[1.0, -0.1], [0.0, 1.0]],
            "saturation_value": 100.0, "absolute_tolerance": 0.5, "feasibility_tolerance": 1e-7,
            "gates": [{"id": "g", "parent": None, "closed_bounds": {"M1": [0, None]}}],
        })


def test_matrix_shape_must_match_names():
    with pytest.raises(StructureError):
        parse_plan({
            "detectors": ["D1", "D2"], "markers": ["M1", "M2"],
            "spillover_rows_detectors_cols_markers": [[1.0, 0.0]],
            "saturation_value": 100.0, "absolute_tolerance": 0.5, "feasibility_tolerance": 1e-7,
            "gates": [{"id": "g", "parent": None, "closed_bounds": {"M1": [0, None]}}],
        })


def test_unknown_gate_marker_refused():
    with pytest.raises(StructureError, match="unknown marker"):
        parse_plan({
            "detectors": ["D1"], "markers": ["M1"],
            "spillover_rows_detectors_cols_markers": [[1.0]],
            "saturation_value": 100.0, "absolute_tolerance": 0.5, "feasibility_tolerance": 1e-7,
            "gates": [{"id": "g", "parent": None, "closed_bounds": {"ZZ": [0, None]}}],
        })


def test_bad_json_path_refused(tmp_path):
    missing = tmp_path / "nope.json"
    from satflow.plan import load_plan

    with pytest.raises(StructureError, match="cannot read plan JSON"):
        load_plan(str(missing))
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(StructureError, match="cannot read plan JSON"):
        load_plan(str(broken))


def test_cli_returns_exit_code_2_on_structure_error(tmp_path):
    from satflow.cli import main

    bad_fcs = write_minimal_fcs(tmp_path / "bad.fcs", _data(), DETECTORS, version="FCS3.0")
    plan = {
        "detectors": DETECTORS, "markers": ["A", "B", "C", "D"],
        "spillover_rows_detectors_cols_markers": np.eye(4).tolist(),
        "saturation_value": 100.0, "absolute_tolerance": 0.5, "feasibility_tolerance": 1e-7,
        "gates": [{"id": "g", "parent": None, "closed_bounds": {"A": [0, None]}}],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    code = main(["analyze", "--fcs", str(bad_fcs), "--plan", str(plan_path),
                 "--outdir", str(tmp_path / "out")])
    assert code == 2
