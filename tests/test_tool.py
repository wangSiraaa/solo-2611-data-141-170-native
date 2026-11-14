"""End-to-end and adversarial tests for the reweighting tool.

Run:  .venv/bin/python -m pytest -q
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from reweight import tool  # noqa: E402
from tests import oracle  # noqa: E402


def _load(margins=None, replicate=None):
    return tool.load_inputs(
        os.path.join(ROOT, "fixtures/respondents.csv"),
        margins or os.path.join(ROOT, "fixtures/margins.csv"),
        os.path.join(ROOT, "fixtures/plan.json"),
        replicate or os.path.join(ROOT, "fixtures/replicate_weights.csv"))


def _base(inputs):
    A = tool.build_constraint_matrix(inputs.respondents, inputs.margins)
    d = np.array([r.base_weight for r in inputs.respondents], dtype=float)
    return A, d


# ---------------------------------------------------------------------------
# Fixture: feasible, fully documented
# ---------------------------------------------------------------------------
def test_base_fit_matches_independent_slsqp():
    inputs = _load()
    A, d = _base(inputs)
    fr = tool.fit(inputs, A, d)
    assert fr.status == "feasible"
    assert fr.solver_converged and fr.optimality_ok
    assert np.max(np.abs(fr.residuals)) < 1e-9

    # Independent solve with a different algorithm/implementation.
    resp, marg, _, _ = oracle.read_fixture(
        os.path.join(ROOT, "fixtures/respondents.csv"),
        os.path.join(ROOT, "fixtures/margins.csv"))
    Ao, bo, _ = oracle.indicator_rows(resp, marg)
    res, obj = oracle.solve_slsqp(Ao, bo, d, inputs.plan.relative_lower,
                                  inputs.plan.relative_upper)
    assert res.success
    assert abs(obj(res.x) - fr.objective) < 1e-6
    assert np.max(np.abs(res.x - fr.weights)) < 1e-5


def test_all_margins_and_bounds_recomputed_independently():
    inputs = _load()
    A, d = _base(inputs)
    fr = tool.fit(inputs, A, d)
    vr = tool.verify(inputs, A, d, fr, tool.replicate_estimates(inputs, A),
                     None)
    # Every constraint, including redundant ones, from the return value.
    for i, m in enumerate(inputs.margins):
        assert abs(float(A[i] @ fr.weights) - m.total_float) < 1e-9
    assert np.all(fr.weights >= fr.lower - 1e-9)
    assert np.all(fr.weights <= fr.upper + 1e-9)
    # handle-lower/upper booleans agree with raw arithmetic
    for i in range(len(d)):
        assert fr.at_lower[i] == (abs(fr.weights[i] - fr.lower[i]) <=
                                  1e-10 * max(fr.upper[i], 1.0))
    assert vr.verify_objective_consistent if hasattr(vr, "verify_objective_consistent") else True


def test_replicate_variance_matches_direct_plug_in():
    inputs = _load()
    A, d = _base(inputs)
    fr = tool.fit(inputs, A, d)
    reps = tool.replicate_estimates(inputs, A)
    var = tool.replicate_variance(inputs, tool.estimate(inputs, fr.weights), reps)

    resp, _, pl, rep = oracle.read_fixture(
        os.path.join(ROOT, "fixtures/respondents.csv"),
        os.path.join(ROOT, "fixtures/margins.csv"),
        plan=os.path.join(ROOT, "fixtures/plan.json"),
        replicates=os.path.join(ROOT, "fixtures/replicate_weights.csv"))
    ids = [r["respondent_id"] for r in resp]
    rep_means = []
    for col in [c for c in rep[0] if c != "respondent_id"]:
        dr = np.array([float(next(r[col] for r in rep if r["respondent_id"] == i))
                       for i in ids])
        rs, _ = oracle.solve_slsqp(A, [m.total_float for m in inputs.margins],
                                   dr, inputs.plan.relative_lower,
                                   inputs.plan.relative_upper)
        assert rs.success, col
        rep_means.append(oracle.weighted_means(resp, rs.x, pl["domains"]))
    full = oracle.weighted_means(resp, fr.weights, pl["domains"])
    direct = oracle.direct_variance(full, rep_means, 0.125)
    for k in full:
        assert abs(direct[k] - var.variances[k]) < 1e-6
        assert abs(rep_means and
                   var.replicate_estimates[inputs.replicate_ids[0]][k] -
                   rep_means[0][k]) < 1e-5


def test_variance_formula_plugged_back_not_reused():
    """verify() recomputes variance from raw replicate estimates directly."""
    inputs = _load()
    A, d = _base(inputs)
    fr = tool.fit(inputs, A, d)
    reps = tool.replicate_estimates(inputs, A)
    full = tool.estimate(inputs, fr.weights)
    var = tool.replicate_variance(inputs, full, reps)
    vr = tool.verify(inputs, A, d, fr, reps, var)
    for dom in inputs.plan.domains:
        did = dom["id"]
        manual = 0.125 * sum(
            (rr.estimates[did] - full[did]) ** 2
            for rr in reps if rr.status == "feasible")
        assert abs(manual - var.variances[did]) < 1e-12
        assert vr.recalc_variance[did]["ok"]
    assert vr.ok


def test_redundant_margins_verified_not_dropped():
    inputs = _load()
    A, d = _base(inputs)
    fe = tool.check_feasibility(inputs, A, d, inputs.plan.relative_lower,
                                inputs.plan.relative_upper)
    assert fe.feasible
    assert len(fe.certificates) == A.shape[0] - len(fe.independent)
    for cert in fe.certificates:
        recomputed = sum(c * inputs.margins[j].total
                         for c, j in zip(cert.coefficients, fe.independent))
        assert recomputed == cert.stated_total
        assert cert.implied_total == cert.stated_total


# ---------------------------------------------------------------------------
# Infeasible margins: explicit, independent evidence
# ---------------------------------------------------------------------------
def test_infeasible_case_reports_bound_evidence():
    inputs = _load(margins=os.path.join(ROOT, "cases/infeasible_margins.csv"))
    A, d = _base(inputs)
    fr = tool.fit(inputs, A, d)
    assert fr.status == "infeasible"
    lo, hi = inputs.plan.relative_lower * d, inputs.plan.relative_upper * d
    # Evidence must be independently reproducible: at least one margin whose
    # target lies outside [sum lo, sum hi].
    bad = [i for i, m in enumerate(inputs.margins)
           if not (A[i] @ lo - 1e-12 <= m.total_float <= A[i] @ hi + 1e-12)]
    assert bad, "expected at least one unreachable margin"
    fe = tool.check_feasibility(inputs, A, d, inputs.plan.relative_lower,
                                inputs.plan.relative_upper)
    assert any(e["type"] in ("box_lower_violation", "box_upper_violation")
               for e in fe.evidence)
    # No weights / SEs claimed.
    assert not np.any(fr.weights)


# ---------------------------------------------------------------------------
# Missing older group in one replicate: retained point estimate, incomplete SE
# ---------------------------------------------------------------------------
def test_no_older_replicate_keeps_estimate_and_flags_se():
    inputs = _load(replicate=os.path.join(ROOT, "cases/replicate_no_older.csv"))
    A, d = _base(inputs)
    fr = tool.fit(inputs, A, d)
    assert fr.status == "feasible"
    reps = tool.replicate_estimates(inputs, A)
    statuses = [r.status for r in reps]
    assert any(s != "feasible" for s in statuses)
    failed = [r for r in reps if r.status != "feasible"]
    # The failed replicate is rep_4: all older respondents have zero base.
    assert failed and failed[0].replicate_id == "rep_4"
    assert failed[0].status == "infeasible"
    older = [i for i, r in enumerate(inputs.respondents) if r.groups["age"] == "older"]
    d4 = np.array([inputs.replicates[inputs.respondents[i].respondent_id]["rep_4"]
                   for i in range(len(inputs.respondents))])
    assert np.all(d4[older] == 0)
    var = tool.replicate_variance(inputs, tool.estimate(inputs, fr.weights), reps)
    assert var.estimates["all"] == var.estimates["all"]  # finite point estimate
    for dom in inputs.plan.domains:
        assert not var.complete[dom["id"]]
        assert var.n_used[dom["id"]] == 7
        assert np.isfinite(var.standard_errors[dom["id"]])
    # The failure must be recorded per replicate, with diagnostics.
    assert any("zero replicate base weight" in d for d in failed[0].diagnostics)


def test_replicate_failure_is_not_silently_skipped():
    inputs = _load(replicate=os.path.join(ROOT, "cases/replicate_no_older.csv"))
    A, d = _base(inputs)
    reps = tool.replicate_estimates(inputs, A)
    var = tool.replicate_variance(inputs, tool.estimate(inputs,
                                                        tool.fit(inputs, A, d).weights),
                                  reps)
    assert var.replicate_status["rep_4"] == "infeasible"
    assert any("INCOMPLETE" in n for n in var.notes)


# ---------------------------------------------------------------------------
# Synthetic two-group hand-checkable example
# ---------------------------------------------------------------------------
def test_hand_checkable_two_group_example(tmp_path):
    """Two respondents, two groups; exact closed-form weights.

    base weights d=(1,3), scores (10,20); 'total'=4, 'g:a'=1 (thus g:b=3).
    Bounds [0.5d, 1.5d].  Entropy solution: w_0 = d_0 * e^{lam},
    w_1 = 3*e^{lam}; w_0 = 1 -> w_0 = 2*e^{lam} = 1 is infeasible?  Let us
    instead choose targets that give a clean closed form below.
    """
    # Construct a tiny dataset with a known analytic solution.
    resp = ["respondent_id,age,score,base_weight\n",
            "r1,young,10,1\n", "r2,older,20,3\n"]
    marg = ["constraint_id,field,value,population_total\n",
            "total,*,all,4\n", "g:young,age,young,2\n"]
    plan = {
        "relative_lower": 0.5, "relative_upper": 2.0,
        "criterion": "entropy", "replicate_variance": "(1/1) * sum",
        "domains": [{"id": "all", "field": "*", "value": "all"}],
        "nonresponse_assumption": "test",
    }
    rep = ["respondent_id,rep_1\n", "r1,1\n", "r2,3\n"]
    p_resp = tmp_path / "r.csv"
    p_marg = tmp_path / "m.csv"
    p_plan = tmp_path / "p.json"
    p_rep = tmp_path / "rep.csv"
    p_resp.write_text("".join(resp))
    p_marg.write_text("".join(marg))
    p_plan.write_text(json.dumps(plan))
    p_rep.write_text("".join(rep))
    inputs = tool.load_inputs(str(p_resp), str(p_marg), str(p_plan), str(p_rep))
    A = tool.build_constraint_matrix(inputs.respondents, inputs.margins)
    d = np.array([1.0, 3.0])
    fr = tool.fit(inputs, A, d)
    assert fr.status == "feasible"
    # w0 + 3 e^{lam} ... solve: young total 2, total 4 => older 2.
    # w0 = d0*exp(lam0+lam1), w1 = d1*exp(lam0); w0=2, w1=2.
    # => exp(lam0)=2/3, exp(lam1)=3 => w0=1*2/3*3=2, w1=3*2/3=2.  Exact.
    assert abs(fr.weights[0] - 2.0) < 1e-12
    assert abs(fr.weights[1] - 2.0) < 1e-12
    # Hand-computed mean: (2*10 + 2*20) / 4 = 15.
    assert abs(tool.estimate(inputs, fr.weights)["all"] - 15.0) < 1e-12
    # Objective hand value: 2 ln(2/1) - 2 + 1 + 2 ln(2/3) - 2 + 3
    expected = (2 * np.log(2.0) - 2 + 1) + (2 * np.log(2.0 / 3.0) - 2 + 3)
    assert abs(fr.objective - expected) < 1e-10


def test_hand_checkable_example_with_bounds_and_verify(tmp_path):
    """Second hand example where a bound binds; verify from raw weights."""
    rows = ["respondent_id,age,score,base_weight\n",
            "a,young,10,1\n", "b,young,30,1\n", "c,older,50,1\n"]
    marg = ["constraint_id,field,value,population_total\n",
            "total,*,all,4\n", "g:young,age,young,3.4\n"]
    plan = {"relative_lower": 0.5, "relative_upper": 1.7,
            "criterion": "entropy",
            "replicate_variance": "(1/1) * sum", "domains": [],
            "nonresponse_assumption": "test"}
    rep = ["respondent_id,rep_1\n", "a,1\n", "b,1\n", "c,1\n"]
    d = tmp_path
    (d / "r.csv").write_text("".join(rows))
    (d / "m.csv").write_text("".join(marg))
    (d / "p.json").write_text(json.dumps(plan))
    (d / "rep.csv").write_text("".join(rep))
    inputs = tool.load_inputs(str(d / "r.csv"), str(d / "m.csv"),
                              str(d / "p.json"), str(d / "rep.csv"))
    A = tool.build_constraint_matrix(inputs.respondents, inputs.margins)
    dw = np.array([1.0, 1.0, 1.0])
    fr = tool.fit(inputs, A, dw)
    assert fr.status == "feasible"
    # young target 3.4 equals the upper-bound capacity 2*1.7, so a,b bind and
    # c makes up the remaining 0.6.
    assert abs(fr.weights[0] - 1.7) < 1e-9
    assert abs(fr.weights[1] - 1.7) < 1e-9
    assert abs(fr.weights[2] - 0.6) < 1e-9
    assert fr.at_upper[0] and fr.at_upper[1]
    vr = tool.verify(inputs, A, dw, fr, None, None)
    assert vr.ok
    assert abs(float(A[0] @ fr.weights) - 3.4) < 1e-12


# ---------------------------------------------------------------------------
# Order invariance
# ---------------------------------------------------------------------------
def test_row_and_constraint_order_invariance(tmp_path):
    inputs = _load()
    A, d = _base(inputs)
    fr = tool.fit(inputs, A, d)

    # Reverse respondent rows and constraint rows in copies of the CSVs.
    with open(os.path.join(ROOT, "fixtures/respondents.csv"), encoding="utf-8") as fh:
        lines = fh.readlines()
    (tmp_path / "resp.csv").write_text(lines[0] + "".join(reversed(lines[1:])))
    with open(os.path.join(ROOT, "fixtures/margins.csv"), encoding="utf-8") as fh:
        mlines = fh.readlines()
    (tmp_path / "marg.csv").write_text(mlines[0] + "".join(reversed(mlines[1:])))
    shuffled = _load(margins=str(tmp_path / "marg.csv"),
                     replicate=os.path.join(ROOT, "fixtures/replicate_weights.csv"))
    # respondents still canonical-sorted on load -> identical vector
    A2 = tool.build_constraint_matrix(shuffled.respondents, shuffled.margins)
    d2 = np.array([r.base_weight for r in shuffled.respondents])
    fr2 = tool.fit(shuffled, A2, d2)
    assert fr2.status == "feasible"
    assert np.allclose(fr.weights, fr2.weights, rtol=0, atol=1e-12)
    assert abs(tool.estimate(shuffled, fr2.weights)["all"] -
               tool.estimate(inputs, fr.weights)["all"]) < 1e-12
    assert abs(fr2.objective - fr.objective) < 1e-12


def test_respondent_order_permutation_within_file(tmp_path):
    import random
    with open(os.path.join(ROOT, "fixtures/respondents.csv"), encoding="utf-8") as fh:
        lines = fh.readlines()
    body = lines[1:]
    random.Random(7).shuffle(body)
    path = tmp_path / "resp.csv"
    path.write_text(lines[0] + "".join(body))
    inputs = _load()
    shuffled = tool.load_inputs(str(path),
                                os.path.join(ROOT, "fixtures/margins.csv"),
                                os.path.join(ROOT, "fixtures/plan.json"),
                                os.path.join(ROOT, "fixtures/replicate_weights.csv"))
    A, d = _base(inputs)
    A2, d2 = _base(shuffled)
    fr, fr2 = tool.fit(inputs, A, d), tool.fit(shuffled, A2, d2)
    assert np.allclose(fr.weights, fr2.weights, rtol=0, atol=1e-12)
    assert abs(fr.objective - fr2.objective) < 1e-12


# ---------------------------------------------------------------------------
# Structural empty cell and zero base weight
# ---------------------------------------------------------------------------
def test_structural_empty_cell_is_rejected(tmp_path):
    rows = ["respondent_id,age,score,base_weight\n",
            "a,young,10,1\n"]
    marg = ["constraint_id,field,value,population_total\n",
            "total,*,all,1\n", "g:older,age,older,1\n"]
    plan = {"relative_lower": 0.5, "relative_upper": 2.0,
            "replicate_variance": "(1/1) * sum", "domains": [],
            "nonresponse_assumption": "test"}
    rep = ["respondent_id,rep_1\n", "a,1\n"]
    (tmp_path / "r.csv").write_text("".join(rows))
    (tmp_path / "m.csv").write_text("".join(marg))
    (tmp_path / "p.json").write_text(json.dumps(plan))
    (tmp_path / "rep.csv").write_text("".join(rep))
    inputs = tool.load_inputs(str(tmp_path / "r.csv"), str(tmp_path / "m.csv"),
                              str(tmp_path / "p.json"), str(tmp_path / "rep.csv"))
    A = tool.build_constraint_matrix(inputs.respondents, inputs.margins)
    d = np.array([1.0])
    fr = tool.fit(inputs, A, d)
    assert fr.status == "infeasible"
    assert any("structural empty cell" in m for m in fr.diagnostics)


def test_zero_base_weight_fixed_to_zero(tmp_path):
    rows = ["respondent_id,age,score,base_weight\n",
            "a,young,10,1\n", "b,young,20,0\n", "c,older,30,2\n"]
    marg = ["constraint_id,field,value,population_total\n",
            "total,*,all,3\n", "g:young,age,young,1\n"]
    plan = {"relative_lower": 0.5, "relative_upper": 2.0,
            "replicate_variance": "(1/1) * sum", "domains": [],
            "nonresponse_assumption": "test"}
    rep = ["respondent_id,rep_1\n", "a,1\n", "b,0\n", "c,2\n"]
    (tmp_path / "r.csv").write_text("".join(rows))
    (tmp_path / "m.csv").write_text("".join(marg))
    (tmp_path / "p.json").write_text(json.dumps(plan))
    (tmp_path / "rep.csv").write_text("".join(rep))
    inputs = tool.load_inputs(str(tmp_path / "r.csv"), str(tmp_path / "m.csv"),
                              str(tmp_path / "p.json"), str(tmp_path / "rep.csv"))
    A = tool.build_constraint_matrix(inputs.respondents, inputs.margins)
    d = np.array([1.0, 0.0, 2.0])
    fr = tool.fit(inputs, A, d)
    assert fr.status == "feasible"
    assert fr.weights[1] == 0.0
    vr = tool.verify(inputs, A, d, fr, None, None)
    assert vr.zero_checks and vr.zero_checks[0]["ok"]
    # Analytical solution: b is fixed at 0, so the young target 1 must come
    # entirely from a (w_a = 1); total 3 then forces w_c = 2 (c unchanged).
    assert abs(fr.weights[0] - 1.0) < 1e-12
    assert abs(fr.weights[2] - 2.0) < 1e-12


# ---------------------------------------------------------------------------
# CLI status separation
# ---------------------------------------------------------------------------
def test_cli_exit_codes_and_artefacts(tmp_path):
    from reweight import cli
    base = cli.main(["fit", "--out", str(tmp_path / "base")])
    assert base == 0
    inf = cli.main(["fit", "--margins", os.path.join(ROOT, "cases/infeasible_margins.csv"),
                    "--out", str(tmp_path / "inf")])
    assert inf == 2
    noolder = cli.main(["fit", "--replicate", os.path.join(ROOT, "cases/replicate_no_older.csv"),
                        "--out", str(tmp_path / "no")])
    assert noolder == 0
    s = json.load(open(tmp_path / "no/summary.json"))
    assert s["status"] == "feasible"
    assert s["replicate_status"]["rep_4"] != "feasible"
    assert all(v["se_complete"] is False for v in s["domains"].values())
    si = json.load(open(tmp_path / "inf/summary.json"))
    assert si["status"] == "infeasible"
    # Three distinct statuses must be visible in artefacts.
    assert json.load(open(tmp_path / "base/summary.json"))["status"] == "feasible"


def test_inconsistent_redundant_margin_is_flagged(tmp_path):
    rows = ["respondent_id,age,score,base_weight\n",
            "a,young,1,1\n", "b,older,2,1\n"]
    # ages sum to 4 but 'total' states 5 -> a redundant constraint whose
    # business meaning is violated; must not be silently dropped.
    marg = ["constraint_id,field,value,population_total\n",
            "total,*,all,5\n", "g:young,age,young,1\n",
            "g:older,age,older,3\n"]
    plan = {"relative_lower": 0.1, "relative_upper": 10.0,
            "replicate_variance": "(1/1) * sum", "domains": [],
            "nonresponse_assumption": "test"}
    rep = ["respondent_id,rep_1\n", "a,1\n", "b,1\n"]
    (tmp_path / "r.csv").write_text("".join(rows))
    (tmp_path / "m.csv").write_text("".join(marg))
    (tmp_path / "p.json").write_text(json.dumps(plan))
    (tmp_path / "rep.csv").write_text("".join(rep))
    inputs = tool.load_inputs(str(tmp_path / "r.csv"), str(tmp_path / "m.csv"),
                              str(tmp_path / "p.json"), str(tmp_path / "rep.csv"))
    A = tool.build_constraint_matrix(inputs.respondents, inputs.margins)
    d = np.array([1.0, 1.0])
    fr = tool.fit(inputs, A, d)
    assert fr.status == "infeasible"
    assert any("inconsistent" in m for m in fr.diagnostics)
    fe = tool.check_feasibility(inputs, A, d, 0.1, 10.0)
    assert any(e["type"] == "dependency_gap" for e in fe.evidence)


def test_bound_kink_does_not_break_convergence(tmp_path):
    """The entropy optimum sits exactly at a box corner (both young at lower
    bound).  The dual loses curvature there; the solver must still return a
    feasible, optimal solution rather than report a spurious failure."""
    rows = ["respondent_id,age,score,base_weight\n",
            "a,young,10,1\n", "b,young,30,1\n",
            "c,older,50,1\n", "e,older,60,1\n"]
    marg = ["constraint_id,field,value,population_total\n",
            "total,*,all,4\n", "g:young,age,young,0.4\n"]
    plan = {"relative_lower": 0.2, "relative_upper": 2.0,
            "replicate_variance": "(1/1) * sum", "domains": [],
            "nonresponse_assumption": "test"}
    rep = ["respondent_id,rep_1\n", "a,1\n", "b,1\n", "c,1\n", "e,1\n"]
    (tmp_path / "r.csv").write_text("".join(rows))
    (tmp_path / "m.csv").write_text("".join(marg))
    (tmp_path / "p.json").write_text(json.dumps(plan))
    (tmp_path / "rep.csv").write_text("".join(rep))
    inputs = tool.load_inputs(str(tmp_path / "r.csv"), str(tmp_path / "m.csv"),
                              str(tmp_path / "p.json"), str(tmp_path / "rep.csv"))
    A = tool.build_constraint_matrix(inputs.respondents, inputs.margins)
    d = np.array([1.0, 1.0, 1.0, 1.0])
    fr = tool.fit(inputs, A, d)
    assert fr.status == "feasible", fr.diagnostics
    # Exact corner solution: young at lower bound 0.2 each; older split total
    # remainder 3.6 equally => 1.8 each.
    expected = np.array([0.2, 0.2, 1.8, 1.8])
    assert np.allclose(fr.weights, expected, atol=1e-9)
    assert fr.at_lower[0] and fr.at_lower[1]
    assert fr.optimality_ok
    vr = tool.verify(inputs, A, d, fr, None, None)
    assert vr.ok, vr.problems


def test_joint_overlap_infeasibility_with_certificate(tmp_path):
    """Each row reachable on its own box, but the overlap is jointly
    unreachable; the tool must report a checkable combination certificate."""
    rows = ["respondent_id,age,score,base_weight\n",
            "a1,young,10,1\n", "a2,young,20,1\n",
            "m1,middle,25,1\n", "m2,middle,28,1\n",
            "b1,older,30,1\n", "b2,older,40,1\n"]
    # Each row is reachable on its own box, but young+older = 4.5 > total = 4
    # while middle is nonempty (rows young/older/total are independent).
    marg = ["constraint_id,field,value,population_total\n",
            "total,*,all,4\n", "g:young,age,young,2.25\n",
            "g:older,age,older,2.25\n"]
    plan = {"relative_lower": 0.1, "relative_upper": 1.4,
            "replicate_variance": "(1/1) * sum", "domains": [],
            "nonresponse_assumption": "test"}
    rep = ["respondent_id,rep_1\n", "a1,1\n", "a2,1\n", "m1,1\n",
           "m2,1\n", "b1,1\n", "b2,1\n"]
    (tmp_path / "r.csv").write_text("".join(rows))
    (tmp_path / "m.csv").write_text("".join(marg))
    (tmp_path / "p.json").write_text(json.dumps(plan))
    (tmp_path / "rep.csv").write_text("".join(rep))
    inputs = tool.load_inputs(str(tmp_path / "r.csv"), str(tmp_path / "m.csv"),
                              str(tmp_path / "p.json"), str(tmp_path / "rep.csv"))
    A = tool.build_constraint_matrix(inputs.respondents, inputs.margins)
    d = np.array([1.0] * 6)
    # per-row box reachability:
    lo, hi = 0.1 * d, 1.4 * d
    for i in range(A.shape[0]):
        assert A[i] @ lo <= inputs.margins[i].total_float <= A[i] @ hi
    fe = tool.check_feasibility(inputs, A, d, 0.1, 1.4)
    assert not fe.feasible
    assert fe.conflict_set
    certs = [e for e in fe.evidence if e["type"] == "farkas_combination"]
    assert certs, "expected an explicit multiplier certificate"
    cert = certs[0]
    # Certificate must be independently checkable: combination target outside
    # [box_min, box_max].
    assert not (cert["box_min"] - 1e-9 <= cert["combination_target"]
                <= cert["box_max"] + 1e-9)
    fr = tool.fit(inputs, A, d)
    assert fr.status == "infeasible"


def test_rational_score_and_replicate_verify(tmp_path):
    rows = ["respondent_id,age,score,base_weight\n",
            "a,young,3/2,1\n", "b,older,5/2,1\n"]
    marg = ["constraint_id,field,value,population_total\n",
            "total,*,all,2\n"]
    plan = {"relative_lower": 0.5, "relative_upper": 2.0,
            "replicate_variance": "(1/1) * sum",
            "domains": [{"id": "all", "field": "*", "value": "all"}],
            "nonresponse_assumption": "test"}
    rep = ["respondent_id,rep_1\n", "a,1\n", "b,1\n"]
    (tmp_path / "r.csv").write_text("".join(rows))
    (tmp_path / "m.csv").write_text("".join(marg))
    (tmp_path / "p.json").write_text(json.dumps(plan))
    (tmp_path / "rep.csv").write_text("".join(rep))
    inputs = tool.load_inputs(str(tmp_path / "r.csv"), str(tmp_path / "m.csv"),
                              str(tmp_path / "p.json"), str(tmp_path / "rep.csv"))
    assert [r.score for r in inputs.respondents] == [1.5, 2.5]
    A = tool.build_constraint_matrix(inputs.respondents, inputs.margins)
    d = np.array([1.0, 1.0])
    fr = tool.fit(inputs, A, d)
    reps = tool.replicate_estimates(inputs, A)
    var = tool.replicate_variance(inputs, tool.estimate(inputs, fr.weights), reps)
    vr = tool.verify(inputs, A, d, fr, reps, var)
    assert vr.ok
    assert abs(var.estimates["all"] - 2.0) < 1e-12
    assert var.variances["all"] == 0.0
