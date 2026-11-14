"""Independent oracle used by the tests.

Deliberately does NOT import the solver or its verification helpers.  Every
quantity here is computed from raw numbers with plain arithmetic / a separate
optimiser so that the tests check the tool rather than restate it.
"""
from __future__ import annotations

import csv
import json
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize


def read_fixture(respondents="fixtures/respondents.csv",
                 margins="fixtures/margins.csv",
                 plan="fixtures/plan.json",
                 replicates="fixtures/replicate_weights.csv"):
    with open(respondents, newline="", encoding="utf-8-sig") as fh:
        resp = list(csv.DictReader(fh))
    with open(margins, newline="", encoding="utf-8-sig") as fh:
        marg = list(csv.DictReader(fh))
    with open(plan, encoding="utf-8") as fh:
        pl = json.load(fh)
    with open(replicates, newline="", encoding="utf-8-sig") as fh:
        rep = list(csv.DictReader(fh))
    return resp, marg, pl, rep


def indicator_rows(resp, marg):
    A, b, ids = [], [], []
    for m in marg:
        A.append([1.0 if (m["field"] == "*" or r[m["field"]] == m["value"])
                  else 0.0 for r in resp])
        b.append(float(Fraction(m["population_total"])))
        ids.append(m["constraint_id"])
    return np.array(A), np.array(b), ids


def independent_subset(A: np.ndarray) -> list[int]:
    """Float-rank greedy reduction, independent of the tool's exact rank."""
    chosen: list[int] = []
    for i in range(A.shape[0]):
        trial = chosen + [i]
        if np.linalg.matrix_rank(A[trial]) == len(trial):
            chosen = trial
    return chosen


def solve_slsqp(A, b, d, lower, upper):
    """Independent primal solve: SLSQP on the same entropy objective.

    Uses a LinProg-computed feasible start and a float-rank independent
    equality subset so SLSQP is not confused by redundant equalities.
    """
    from scipy.optimize import linprog
    n = len(d)
    lo = lower * d
    hi = upper * d
    d0 = np.array(d, float)

    def obj(w):
        pos = d0 > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sum(w[pos] * np.log(np.where(w[pos] > 0, w[pos], 1) /
                                          d0[pos]) - w[pos] + d0[pos])

    def jac(w):
        pos = d0 > 0
        g = np.zeros(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            g[pos] = np.log(np.where(w[pos] > 0, w[pos], 1) / d0[pos])
        return g

    idx = independent_subset(A)
    b_arr = np.asarray(b, dtype=float)
    Ai, bi = A[idx], b_arr[idx]
    start = linprog(np.zeros(n), A_eq=Ai, b_eq=bi, bounds=list(zip(lo, hi)),
                    method="highs")
    assert start.status == 0, start.message
    x0 = np.clip(start.x, lo, hi)
    cons = [{"type": "eq", "fun": lambda w, r=row, t=t: float(np.dot(r, w) - t),
             "jac": lambda w, r=row: r} for row, t in zip(Ai, bi)]
    bounds = list(zip(lo, hi))
    res = minimize(obj, x0, jac=jac, constraints=cons, method="SLSQP",
                   bounds=bounds, options={"ftol": 1e-14, "maxiter": 2000})
    return res, obj


def weighted_means(resp, w, domains):
    out = {}
    for dom in domains:
        mask = np.array([dom["field"] == "*" or r[dom["field"]] == dom["value"]
                         for r in resp])
        sc = np.array([float(r["score"]) for r in resp])[mask]
        ww = np.asarray(w, float)[mask]
        out[dom["id"]] = float(ww @ sc / ww.sum())
    return out


def direct_variance(full_means, replicate_means, coefficient):
    return {k: coefficient * sum((rep[k] - full_means[k]) ** 2
                                 for rep in replicate_means)
            for k in full_means}
