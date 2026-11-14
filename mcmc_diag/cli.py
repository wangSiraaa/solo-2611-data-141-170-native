"""命令行入口：analyze（诊断 + 报告）与 inspect（链恢复记录）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .recovery import recover_chains
from .report import run_diagnostics, write_summary_json, write_summary_md, make_figures, STATUS_LABELS_ZH


def _load_extra_entries(add_fragment: str | None, fragment_path_override: str | None) -> list[dict]:
    """加载 cases/add_fragment.json 描述的附加片段条目。

    fragment_path_override 用于冲突变体：add_fragment.json 的 conflict_variant
    说明为“change only path to ../cases/c1-conflict.csv”，即仅替换 path。
    """
    if not add_fragment:
        return []
    with open(add_fragment, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    entry = dict(spec["add_to_manifest"])
    if fragment_path_override:
        entry["path"] = fragment_path_override
    return [entry]


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", default="fixtures/manifest.json", help="manifest.json 路径")
    p.add_argument("--add-fragment", default=None, help="附加片段说明 JSON（如 cases/add_fragment.json）")
    p.add_argument(
        "--fragment-path",
        default=None,
        help="覆盖附加片段的 path（冲突变体: ../cases/c1-conflict.csv）",
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    extra = _load_extra_entries(args.add_fragment, args.fragment_path)
    recovered = recover_chains(args.manifest, extra_entries=extra)
    results = run_diagnostics(recovered)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_summary_json(recovered, results, out_dir)
    md_path = write_summary_md(recovered, results, out_dir)
    figures = make_figures(recovered, results, out_dir)

    print(f"有效链: {recovered.valid_chain_ids}  共同保留长度 L={recovered.common_length}")
    for c in recovered.chains:
        if c.status == "blocked":
            print(f"  [阻断] {c.chain_id}: {'; '.join(c.problems)}")
        elif c.n_discarded_tail:
            print(f"  [截断] {c.chain_id}: 保留 {c.common_length}，丢弃尾部 {c.n_discarded_tail}")
    print(f"{'参数':<18}{'状态':<34}{'未通过项'}")
    for p, r in results.items():
        v = r["verdict"]
        print(f"{p:<18}{v['status']:<34}{','.join(v['failed_checks']) or '-'}")
    print(f"\n报告: {json_path}, {md_path}, 图 {len(figures)} 张 -> {out_dir}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    extra = _load_extra_entries(args.add_fragment, args.fragment_path)
    recovered = recover_chains(args.manifest, extra_entries=extra)
    record = recovered.to_record()
    if args.json:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0
    print(f"共同保留长度 L={record['common_length']}  有效链: {record['valid_chain_ids']}")
    for c in record["chains"]:
        if args.chain and c["chain_id"] != args.chain:
            continue
        print(f"\n链 {c['chain_id']}: 状态={c['status']} warmup_end={c['warmup_end']} thin={c['thin']}")
        print(f"  读入 {c['n_rows_loaded']} 行, 去重 {c['n_duplicates_removed']}, "
              f"去 warmup {c['n_warmup_removed']}, 保留 {c['n_retained_after_warmup']}"
              f" (iteration {c['retained_first_iteration']}..{c['retained_last_iteration']})")
        if c["common_length_used"] is not None:
            print(f"  多链诊断使用最早 {c['common_length_used']} 个, 丢弃尾部 {c['n_discarded_tail']} 个")
        for f in c["fragments"]:
            print(f"  片段 {f['path']}: 声明[{f['declared_start_iteration']},{f['declared_end_iteration']}] "
                  f"实际[{f['first_iteration']},{f['last_iteration']}] 行数={f['n_rows']} "
                  f"去重={f['duplicates_removed']} 状态={f['status']}")
            for prob in f["problems"]:
                print(f"    - {prob}")
        for prob in c["problems"]:
            print(f"  ! {prob}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mcmc_diag", description="分片 MCMC 样本链恢复与估计可信度诊断")
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="恢复链、计算诊断并生成报告")
    _add_common_args(p_an)
    p_an.add_argument("--out", default="report", help="报告输出目录")
    p_an.set_defaults(func=cmd_analyze)

    p_in = sub.add_parser("inspect", help="只输出链恢复记录")
    _add_common_args(p_in)
    p_in.add_argument("--chain", default=None, help="只看某条链")
    p_in.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_in.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
