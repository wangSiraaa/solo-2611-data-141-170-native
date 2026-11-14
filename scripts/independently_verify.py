"""独立核对脚本：只从落盘工件（不调用设计器内部对象）重建并核对。

核对内容：
1. run_table.csv 满足块名额、锁定运行、块内重复，且不含任何禁用组合；
2. 按 factors/model 声明 *独立重算* 设计矩阵，与 design_matrix.csv 逐元素比对；
3. 独立计算块调整后秩，确认全部非截距目标效应可估计；
4. randomized_runs.csv 与 run_table.csv 是同一组块内多重集（只换执行顺序）。

用法：``PYTHONPATH=src .venv/bin/python scripts/independently_verify.py outputs``
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.insert(0, os.path.join(ROOT, "src"))

from expdesign.factors import FactorSet  # noqa: E402
from expdesign.model import INTERCEPT_NAME, Model  # noqa: E402
from expdesign.plan import matrix_rank  # noqa: E402

EPS = 1e-9


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(outdir: str) -> int:
    factors = FactorSet.from_json_file(os.path.join(ROOT, "fixtures", "factors.json"))
    with open(os.path.join(ROOT, "fixtures", "model.json"), encoding="utf-8") as fh:
        model = Model.from_dict(factors, json.load(fh))
    with open(os.path.join(ROOT, "fixtures", "constraints.json"),
              encoding="utf-8") as fh:
        constraints = json.load(fh)
    with open(os.path.join(ROOT, "fixtures", "plan.json"), encoding="utf-8") as fh:
        plan = json.load(fh)

    rows = read_csv(os.path.join(outdir, "run_table.csv"))
    failures: list[str] = []

    # --- 1. 块名额 ---------------------------------------------------------
    quotas = {b["id"]: b["runs"] for b in plan["blocks"]}
    for bid, quota in quotas.items():
        n = sum(1 for r in rows if r["block"] == bid)
        if n != quota:
            failures.append(f"块 {bid} 实际 {n} 行 != 名额 {quota}")

    # --- 2. 锁定运行原样保留 ----------------------------------------------
    locked = read_csv(os.path.join(ROOT, "fixtures", "locked_runs.csv"))
    for lr in locked:
        match = [r for r in rows if r["run_id"] == lr["run_id"]]
        if not match:
            failures.append(f"缺少锁定运行 {lr['run_id']}")
            continue
        r = match[0]
        if r["block"] != lr["block"] or any(
            r[f.name] != lr[f.name] for f in factors.factors
        ):
            failures.append(f"锁定运行 {lr['run_id']} 被移动或修改")

    # --- 3. 禁用组合不得出现；块内重复 ------------------------------------
    forbidden = constraints["forbidden_all"]
    by_block: dict[str, list[tuple]] = {}
    for r in rows:
        combo = tuple(
            _coerce(f, r[f.name]) for f in factors.factors
        )
        by_block.setdefault(r["block"], []).append(combo)
        for rule in forbidden:
            if all(_coerce(factors.get(k), r[k]) == _coerce(
                    factors.get(k), v) for k, v in rule.items()):
                failures.append(f"出现禁用组合: {dict(zip(factors.names, combo))}")
    for bid, combos in by_block.items():
        if len(set(combos)) == len(combos):
            failures.append(f"块 {bid} 缺少块内重复")

    # --- 4. 独立重建设计矩阵并与工件逐元素比对 ----------------------------
    combos_flat = [
        tuple(_coerce(f, r[f.name]) for f in factors.factors) for r in rows
    ]
    rebuilt = model.treatment_matrix(combos_flat)
    block_ids = [b["id"] for b in plan["blocks"]]
    blocks = [r["block"] for r in rows]
    rebuilt_full = model.full_matrix(rebuilt, blocks, block_ids)

    dm = read_csv(os.path.join(outdir, "design_matrix.csv"))
    artifact = np.array([
        [float(r[name]) for name in model.full_column_names(block_ids)]
        for r in dm
    ])
    if artifact.shape != rebuilt_full.shape:
        failures.append(f"设计矩阵形状不一致 {artifact.shape} vs {rebuilt_full.shape}")
    elif not np.allclose(artifact, rebuilt_full, atol=1e-9):
        failures.append("独立重建矩阵与 design_matrix.csv 存在元素差异")

    # --- 5. 块调整后可估计性 ----------------------------------------------
    z = np.zeros((len(rows), len(block_ids)))
    for i, bid in enumerate(blocks):
        z[i, block_ids.index(bid)] = 1.0
    counts = z.sum(axis=0)
    q = np.eye(len(rows)) - (z / counts) @ z.T
    target = [i for i, n in enumerate(model.treatment_names) if n != INTERCEPT_NAME]
    qx = q @ rebuilt[:, target]
    rank = matrix_rank(qx, tol=1e-10 * max(qx.shape))
    if rank != len(target):
        failures.append(
            f"块调整后目标秩 {rank} != 目标列数 {len(target)}，效应不可估计")

    # --- 6. 随机顺序保持多重集 --------------------------------------------
    rand_rows = read_csv(os.path.join(outdir, "randomized_runs.csv"))
    for bid in block_ids:
        a = sorted(tuple(r[f.name] for f in factors.factors)
                   for r in rows if r["block"] == bid)
        b = sorted(tuple(r[f.name] for f in factors.factors)
                   for r in rand_rows if r["block"] == bid)
        if a != b:
            failures.append(f"块 {bid} 随机顺序改变了组合多重集")

    print(f"独立核对目录: {outdir}")
    print(f"运行行数={len(rows)}  目标列数={len(target)}  调整后秩={rank}")
    if failures:
        print("核对失败:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("全部独立核对通过：名额/锁定/禁用/重复/矩阵重建/可估计/随机顺序")
    return 0


def _coerce(factor, text):
    if factor.is_numeric:
        return float(text)
    return text


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "outputs"))
