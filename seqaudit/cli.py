"""命令行入口：audit / verify。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import run_audit
from .io import (
    InputError,
    load_lot,
    load_observations,
    load_plan,
    sha256_file,
)
from .tiny import enumerate_tiny, load_tiny_spec
from .verify import verify_report


def _inputs_meta(plan_path: str, lot_path: str, obs_path: str) -> dict:
    return {
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "lot": {"path": str(lot_path), "sha256": sha256_file(lot_path)},
        "observations": {"path": str(obs_path), "sha256": sha256_file(obs_path)},
    }


def _write_json(obj: dict, out: str | None) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f"[写出] {p}")
    else:
        print(text)


def _print_audit_summary(report: dict) -> None:
    elig = report["eligibility"]
    print("=== 抽样资格审核 ===")
    for c in elig["checks"]:
        mark = "ok  " if c["ok"] else "FAIL"
        print(f"  [{mark}] {c['name']}: {c['detail']}")
    if not elig["ok"]:
        print("\n资格失败：保留以上诊断，不出具有效审计证书。")
        return
    audit = report["audit"]
    print("\n=== 顺序审计查看点（证据阈值 = {}）===".format(audit["threshold"]))
    print(
        f"  {'抽号':>4} {'类型':<16} {'累计':>4} {'缺陷':>4} {'剩余':>4} "
        f"{'证据(精确)':>24} {'证据≈':>12} {'允许接受':>6}"
    )
    for l in audit["looks"]:
        ev_f = l["evidence_float"]
        print(
            f"  {l['draw']:>4} {l['look_type']:<16} {l['observed']:>4} "
            f"{l['defects']:>4} {l['remaining']:>4} "
            f"{l['evidence_exact']:>24} "
            f"{'inf' if ev_f is None else format(ev_f, '.6g'):>12} "
            f"{'是' if l['accept_allowed'] else '否':>6}"
        )
    print(f"\n证据首次达阈位置（任意抽）: {audit['first_evidence_crossing_draw']}")
    print(f"首次允许接受的查看点: {audit['first_accept_draw']}")
    print(f"结论: {audit['decision']} — {audit['decision_reason']}")
    if "full_inspection" in audit:
        fi = audit["full_inspection"]
        print(
            f"全量检查: 真实缺陷总数={fi['true_total_defects']}，"
            f"结论与真实状态一致={fi['decision_matches_truth']}"
        )


def cmd_audit(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    lot_ids = load_lot(args.lot)
    obs_rows = load_observations(args.observations)
    meta = _inputs_meta(args.plan, args.lot, args.observations)
    report, code = run_audit(plan, lot_ids, obs_rows, meta)
    _print_audit_summary(report)
    _write_json(report, args.out)
    return code


def _print_verify_record(record: dict) -> None:
    print("=== 验证记录 ===")
    for c in record["checks"]:
        mark = "ok  " if c["ok"] else "FAIL"
        print(f"  [{mark}] {c['name']}: {c['detail']}")
    print(f"\n总体: {record['overall']}")


def cmd_verify(args: argparse.Namespace) -> int:
    if args.tiny:
        n, d0, alpha = load_tiny_spec(args.tiny)
        record = enumerate_tiny(n, d0, alpha)
        null_rows = record["null_configurations"]
        print("=== tiny 独立枚举 ===")
        print(
            f"N={n}, 零假设 D>={d0}, alpha={record['design']['alpha']}, "
            f"阈值={record['design']['evidence_threshold']}"
        )
        for r in null_rows:
            print(
                f"  D={r['defects_total']} 配置{tuple(r['defect_positions'])}: "
                f"曾接受 {r['ever_accepted']}/{r['permutations']} = "
                f"{r['probability_exact']} ≈ {r['probability_float']:.4f}"
            )
        print(
            f"零假设上最大曾接受概率 = {record['max_false_accept_probability_exact']} "
            f"≈ {record['max_false_accept_probability_float']:.4f} "
            f"<= alpha({record['alpha_float']:.4f})? "
            f"{'是' if record['risk_bound_respected'] else '否'}"
        )
        print(
            "合格配置全量检查必接受: "
            f"{'是' if record['all_good_configurations_accepted_by_full_inspection'] else '否'}"
        )
        _write_json(record, args.out)
        return 0 if (
            record["risk_bound_respected"]
            and record["all_good_configurations_accepted_by_full_inspection"]
        ) else 1

    plan = load_plan(args.plan)
    lot_ids = load_lot(args.lot)
    obs_rows = load_observations(args.observations)
    meta = _inputs_meta(args.plan, args.lot, args.observations)
    try:
        report = json.loads(Path(args.certificate).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"无法读取证书: {exc}") from exc
    record = verify_report(plan, lot_ids, obs_rows, report, meta)
    _print_verify_record(record)
    _write_json(record, args.out)
    return 0 if record["overall"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="seqaudit",
        description="有限总体无放回抽查的风险受控顺序审计（audit/verify）",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("audit", help="顺序审计，出具证书或资格诊断")
    pa.add_argument("--plan", required=True)
    pa.add_argument("--lot", required=True)
    pa.add_argument("--observations", required=True)
    pa.add_argument("--out", default=None, help="证书/诊断报告 JSON 输出路径")
    pa.set_defaults(func=cmd_audit)

    pv = sub.add_parser("verify", help="独立重算并核对证书，或对 tiny 做枚举验证")
    pv.add_argument("--plan")
    pv.add_argument("--lot")
    pv.add_argument("--observations")
    pv.add_argument("--certificate")
    pv.add_argument("--tiny", default=None, help="tiny 规格 JSON（枚举验证模式）")
    pv.add_argument("--out", default=None, help="验证记录 JSON 输出路径")
    pv.set_defaults(func=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify" and not args.tiny:
        missing = [
            f"--{k}"
            for k in ("plan", "lot", "observations", "certificate")
            if getattr(args, k) is None
        ]
        if missing:
            print(f"verify 缺少参数: {', '.join(missing)}（或使用 --tiny）", file=sys.stderr)
            return 2
    try:
        return args.func(args)
    except InputError as exc:
        print(f"输入错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
