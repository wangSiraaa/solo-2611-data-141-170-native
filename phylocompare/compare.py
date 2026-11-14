"""compare 命令的编排：校验、全数据拟合、共享 bootstrap 比较与报告数据。

支持频率是 bootstrap 重采样下“该候选最优（或近乎并列）”的经验频率，
不是贝叶斯后验概率，报告中明确标注。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bootstrap import BootstrapPlan, make_bootstrap_plan, resampled_patterns
from .errors import CompareError
from .fasta import Alignment, read_fasta
from .model import SitePatterns, compress_sites
from .newick import parse_newick
from .optim import (
    BOUNDARY_TOL,
    DEFAULT_BOUNDS,
    FTOL,
    GTOL,
    MAX_ITER,
    STARTS,
    FitResult,
    fit_branch_lengths,
)
from .tree import UnrootedTree

MAX_TAXA = 5
MAX_CANDIDATES = 4
# 两个候选的对数似然差小于该值视为“近乎并列”，两边都记为支持。
NEAR_TIE_LL = 0.5


def load_inputs(
    fasta_path: str | Path,
    newick_path: str | Path,
) -> tuple[Alignment, list[UnrootedTree]]:
    """读取并交叉校验 FASTA 与候选 Newick（等长、唯一叶名、相同叶集合等）。"""
    alignment = read_fasta(fasta_path)
    if len(alignment.names) != MAX_TAXA:
        raise CompareError(
            f"题目固定 {MAX_TAXA} 个分类单元，FASTA 实际有 {len(alignment.names)} 条序列"
        )
    roots = parse_newick(Path(newick_path).read_text(encoding="utf-8"), str(newick_path))
    if not 1 <= len(roots) <= MAX_CANDIDATES:
        raise CompareError(
            f"候选树数量应在 1..{MAX_CANDIDATES} 之间，实际为 {len(roots)}"
        )
    trees = [
        UnrootedTree.from_newick(root, alignment.names) for root in roots
    ]
    return alignment, trees


def site_pattern_summary(alignment: Alignment, patterns: SitePatterns) -> dict:
    """位点模式统计：模式数、含 N 的模式、恒定/可变位点等。"""
    n_n_sites = int(np.any(np.vstack([
        alignment.sequences[name] for name in alignment.names
    ]) == 4, axis=0).sum())
    stacked = np.vstack([
        alignment.sequences[name] for name in alignment.names
    ])
    variable = 0
    for col in range(alignment.n_sites):
        known = stacked[:, col]
        known = known[known != 4]
        if np.unique(known).size > 1:
            variable += 1
    return {
        "n_sites": alignment.n_sites,
        "n_taxa": len(alignment.names),
        "taxa": list(alignment.names),
        "distinct_patterns": int(patterns.patterns.shape[0]),
        "sites_with_unknown_N": n_n_sites,
        "variable_sites_ignoring_N": variable,
        "constant_sites_ignoring_N": alignment.n_sites - variable,
        "pattern_count_top10": sorted(
            (
                {
                    "pattern": "".join(
                        "N" if b == 4 else "ACGT"[b] for b in pattern
                    ),
                    "count": int(count),
                }
                for pattern, count in zip(patterns.patterns, patterns.counts)
            ),
            key=lambda item: (-item["count"], item["pattern"]),
        )[:10],
    }


def _fit_payload(tree: UnrootedTree, result: FitResult) -> dict:
    return {
        "topology": [list(edge.key) for edge in tree.edges],
        "optimization_ok": result.ok,
        "log_likelihood": None if not result.ok else result.loglik,
        "fitted_branch_lengths": (
            None
            if not result.ok
            else {
                "|".join(edge.key): value
                for edge, value in zip(tree.edges, result.lengths)
            }
        ),
        "boundary_parameters": result.boundary_edges,
        "failure_reason": result.failure_reason,
        "multi_start_evidence": [asdict(record) for record in result.starts],
    }


def run_comparison(
    fasta_path: str | Path,
    newick_path: str | Path,
    n_bootstrap: int,
    seed: int,
    bounds: tuple[float, float] = DEFAULT_BOUNDS,
) -> dict:
    """执行完整比较，返回可直接 JSON 序列化的报告字典。"""
    alignment, trees = load_inputs(fasta_path, newick_path)

    # 重复拓扑在固定模型下只会得到完全相同的似然；显式报错而不是默默重复计数。
    topologies = [tree.topology_key for tree in trees]
    if len(set(topologies)) != len(topologies):
        raise CompareError("候选树中存在重复拓扑，请去除后再比较")

    full_patterns = compress_sites(alignment)
    run_parameters = {
        "model": "JC69",
        "root_prior": "uniform over ACGT",
        "unknown_character": "N; emission likelihood 1 for every latent base",
        "branch_bounds": list(bounds),
        "multi_start": [
            "newick_initial_clipped",
            *[f"uniform_{value:g}" for value in STARTS],
        ],
        "optimizer": "scipy.optimize.minimize L-BFGS-B",
        "convergence": {"gtol": GTOL, "ftol": FTOL, "maxiter": MAX_ITER},
        "boundary_tolerance": BOUNDARY_TOL,
        "bootstrap_replicates": n_bootstrap,
        "bootstrap_seed": seed,
        "near_tie_loglik_threshold": NEAR_TIE_LL,
        "likelihood_engine": "log-space Felsenstein pruning with logsumexp (no underflow scaling)",
        "support_frequency_note": (
            "bootstrap 支持频率是重采样下的经验最优频率，"
            "不是贝叶斯后验概率，不做后验概率声明"
        ),
        "numpy_version": np.__version__,
    }

    # ---- 全数据拟合 -----------------------------------------------------
    candidate_results: list[dict] = []
    fits: list[FitResult | None] = []
    for idx, tree in enumerate(trees):
        result = fit_branch_lengths(tree, full_patterns, bounds)
        fits.append(result if result.ok else None)
        payload = _fit_payload(tree, result)
        payload["candidate_id"] = f"C{idx + 1}"
        candidate_results.append(payload)

    valid = [i for i, fit in enumerate(fits) if fit is not None]
    if not valid:
        raise CompareError("全部候选拟合失败，无法进行比较；各候选的失败证据已在结果中")
    best_full = max(valid, key=lambda i: fits[i].loglik)  # type: ignore[union-attr]
    for i in valid:
        candidate_results[i]["delta_loglik_vs_best"] = (
            fits[i].loglik - fits[best_full].loglik  # type: ignore[union-attr]
        )
        candidate_results[i]["near_tied_best_full_data"] = (
            abs(candidate_results[i]["delta_loglik_vs_best"]) <= NEAR_TIE_LL
        )
    for i, payload in enumerate(candidate_results):
        if i not in valid:
            payload["delta_loglik_vs_best"] = None
            payload["near_tied_best_full_data"] = None
    ranking = [
        candidate_results[i]["candidate_id"]
        for i in sorted(valid, key=lambda i: -fits[i].loglik)  # type: ignore[union-attr]
    ]

    # ---- 共享 bootstrap -------------------------------------------------
    plan: BootstrapPlan = make_bootstrap_plan(alignment, n_bootstrap, seed)
    replicate_records: list[dict] = []
    win_counts = {f"C{i + 1}": 0 for i in range(len(trees))}
    near_tie_counts = {f"C{i + 1}": 0 for i in range(len(trees))}
    fail_counts = {f"C{i + 1}": 0 for i in range(len(trees))}
    pairwise_wins = {
        f"C{i + 1}": {f"C{j + 1}": 0 for j in range(len(trees)) if j != i}
        for i in range(len(trees))
    }
    delta_accum: dict[tuple[str, str], list[float]] = {}

    for rep in range(n_bootstrap):
        columns = plan.columns[rep]
        rep_patterns = resampled_patterns(alignment, columns)
        rep_lls: dict[str, float] = {}
        rep_record: dict = {
            "replicate": rep,
            "resample_columns_sha256_prefix": hashlib.sha256(
                columns.astype(np.int64).tobytes()
            ).hexdigest()[:16],
            "distinct_patterns": int(rep_patterns.patterns.shape[0]),
            "candidates": {},
        }
        for idx, tree in enumerate(trees):
            cid = f"C{idx + 1}"
            result = fit_branch_lengths(tree, rep_patterns, bounds)
            if not result.ok:
                fail_counts[cid] += 1
                rep_record["candidates"][cid] = {
                    "optimization_ok": False,
                    "failure_reason": result.failure_reason,
                    "multi_start_evidence": [asdict(r) for r in result.starts],
                }
                continue
            rep_lls[cid] = result.loglik
            rep_record["candidates"][cid] = {
                "optimization_ok": True,
                "log_likelihood": result.loglik,
                "fitted_branch_lengths": {
                    "|".join(edge.key): value
                    for edge, value in zip(tree.edges, result.lengths)
                },
                "boundary_parameters": result.boundary_edges,
            }

        if rep_lls:
            top_ll = max(rep_lls.values())
            winners = [cid for cid, ll in rep_lls.items() if ll == top_ll]
            near = [
                cid for cid, ll in rep_lls.items() if top_ll - ll <= NEAR_TIE_LL
            ]
            for cid in winners:
                win_counts[cid] += 1
            for cid in near:
                near_tie_counts[cid] += 1
            rep_record["strict_winners"] = winners
            rep_record["near_tied_leaders"] = near
            ids = sorted(rep_lls)
            for a_pos, cid_a in enumerate(ids):
                for cid_b in ids[a_pos + 1:]:
                    delta = rep_lls[cid_a] - rep_lls[cid_b]
                    delta_accum.setdefault((cid_a, cid_b), []).append(delta)
                    if delta > 0:
                        pairwise_wins[cid_a][cid_b] += 1
                    elif delta < 0:
                        pairwise_wins[cid_b][cid_a] += 1
        else:
            rep_record["strict_winners"] = []
            rep_record["near_tied_leaders"] = []
        replicate_records.append(rep_record)

    evaluated = n_bootstrap
    pairwise_summary = []
    ids = [f"C{i + 1}" for i in range(len(trees))]
    for pos, cid_a in enumerate(ids):
        for cid_b in ids[pos + 1:]:
            deltas = delta_accum.get((cid_a, cid_b), [])
            if deltas:
                arr = np.asarray(deltas)
                pairwise_summary.append({
                    "pair": [cid_a, cid_b],
                    "delta_mean": float(arr.mean()),
                    "delta_sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "n_evaluated": int(arr.size),
                    "wins": {
                        cid_a: pairwise_wins[cid_a][cid_b],
                        cid_b: pairwise_wins[cid_b][cid_a],
                    },
                })

    support = {
        cid: {
            "strict_win_frequency": win_counts[cid] / evaluated,
            "best_or_near_tied_frequency": near_tie_counts[cid] / evaluated,
            "fit_failures": fail_counts[cid],
        }
        for cid in ids
    }

    return {
        "input": {
            "fasta": str(fasta_path),
            "candidate_newick": str(newick_path),
        },
        "run_parameters": run_parameters,
        "site_patterns": site_pattern_summary(alignment, full_patterns),
        "bootstrap_resample_fingerprint": plan.fingerprint(),
        "candidates": candidate_results,
        "full_data_ranking": ranking,
        "pairwise_bootstrap": pairwise_summary,
        "bootstrap_support_frequencies": support,
        "bootstrap_replicates": replicate_records,
    }


def _to_native(obj):
    """递归把 numpy 标量/数组转成原生 Python 类型，保证 JSON 可序列化。"""
    if isinstance(obj, dict):
        return {key: _to_native(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_report(report: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_to_native(report), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return out_path
