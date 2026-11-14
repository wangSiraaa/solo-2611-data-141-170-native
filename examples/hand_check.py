"""Two hand-checkable examples (2-group / 3-person) with exact analytic checks.

Run:  .venv/bin/python examples/hand_check.py

Example 1 (two groups)
    d = (1, 3), scores = (10, 20)
    margins: total = 4, young = 2  (so older = 2)
    bounds: [0.5 d, 2 d]
    Entropy solution: w = (2, 2).
    Mean over 'all' = (2*10 + 2*20) / 4 = 15.
    Objective = (2 ln 2 - 2 + 1) + (2 ln(2/3) - 2 + 3).
    Closed form: exp(lam0) = 2/3, exp(lam1) = 3.

Example 2 (bound binds)
    three respondents, base weight 1 each, scores (10, 30, 50)
    margins: total = 4, young = 3.4
    bounds: [0.5, 1.7]
    The two young respondents are capped at 1.7, so w = (1.7, 1.7, 0.6).
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reweight import tool  # noqa: E402


def _run(rows, marg, plan, d):
    tmp = tempfile.mkdtemp()
    open(os.path.join(tmp, "r.csv"), "w").write(rows)
    open(os.path.join(tmp, "m.csv"), "w").write(marg)
    open(os.path.join(tmp, "p.json"), "w").write(json.dumps(plan))
    ids = [r.split(",")[0] for r in rows.strip().splitlines()[1:]]
    open(os.path.join(tmp, "rep.csv"), "w").write(
        "respondent_id,rep_1\n" + "".join(f"{i},1\n" for i in ids))
    inputs = tool.load_inputs(os.path.join(tmp, "r.csv"),
                              os.path.join(tmp, "m.csv"),
                              os.path.join(tmp, "p.json"),
                              os.path.join(tmp, "rep.csv"))
    A = tool.build_constraint_matrix(inputs.respondents, inputs.margins)
    return inputs, tool.fit(inputs, A, np.asarray(d, float))


def example_one():
    rows = ("respondent_id,age,score,base_weight\n"
            "r1,young,10,1\nr2,older,20,3\n")
    marg = ("constraint_id,field,value,population_total\n"
            "total,*,all,4\ng:young,age,young,2\n")
    plan = {"relative_lower": 0.5, "relative_upper": 2.0,
            "replicate_variance": "(1/1) * sum",
            "domains": [{"id": "all", "field": "*", "value": "all"}],
            "nonresponse_assumption": "hand-check example"}
    inputs, fr = _run(rows, marg, plan, [1.0, 3.0])
    w = fr.weights
    expected_w = np.array([2.0, 2.0])
    expected_mean = (2 * 10 + 2 * 20) / 4
    expected_obj = (2 * math.log(2) - 2 + 1) + (2 * math.log(2 / 3) - 2 + 3)
    mean = tool.estimate(inputs, w)["all"]
    print("Example 1 (two groups)")
    print(f"  weights  = {np.round(w, 10)}   expected {expected_w}")
    print(f"  mean     = {mean:.12f}   hand {expected_mean}")
    print(f"  objective= {fr.objective:.12f}   hand {expected_obj:.12f}")
    assert np.allclose(w, expected_w, atol=1e-10)
    assert abs(mean - expected_mean) < 1e-12
    assert abs(fr.objective - expected_obj) < 1e-10
    print("  PASS\n")


def example_two():
    rows = ("respondent_id,age,score,base_weight\n"
            "a,young,10,1\nb,young,30,1\nc,older,50,1\n")
    marg = ("constraint_id,field,value,population_total\n"
            "total,*,all,4\ng:young,age,young,3.4\n")
    plan = {"relative_lower": 0.5, "relative_upper": 1.7,
            "replicate_variance": "(1/1) * sum", "domains": [],
            "nonresponse_assumption": "hand-check example"}
    inputs, fr = _run(rows, marg, plan, [1.0, 1.0, 1.0])
    w = fr.weights
    expected_w = np.array([1.7, 1.7, 0.6])
    print("Example 2 (upper bound binds)")
    print(f"  weights  = {np.round(w, 10)}   expected {expected_w}")
    assert np.allclose(w, expected_w, atol=1e-9)
    assert fr.at_upper[0] and fr.at_upper[1]
    A = tool.build_constraint_matrix(inputs.respondents, inputs.margins)
    vr = tool.verify(inputs, A, np.array([1.0, 1.0, 1.0]), fr, None, None)
    assert vr.ok
    print("  PASS\n")


if __name__ == "__main__":
    example_one()
    example_two()
    print("All hand checks passed.")
