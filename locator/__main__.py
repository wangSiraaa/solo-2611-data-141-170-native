"""命令行入口。

用法:
  python -m locator locate --fixtures fixtures --case good --out outputs/good
  python -m locator locate --stations S.csv --arrivals A.csv --plan P.json --out DIR
  python -m locator run-all --fixtures fixtures --out outputs
  python -m locator selftest --out outputs/selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import io as lio
from .engine import HUBER_C, run_location
from .report import write_all, write_combined_report_md
from .selftest import run_selftest


def _load_case(args) -> tuple[str, lio.Plan, dict, list, Path]:
    if args.case:
        fixtures = Path(args.fixtures)
        s_path = fixtures / f"stations_{args.case}.csv"
        a_path = fixtures / f"arrivals_{args.case}.csv"
        p_path = Path(args.plan) if args.plan else fixtures / "plan.json"
        case = args.case
    else:
        if not (args.stations and args.arrivals and args.plan):
            raise lio.InputError(
                "需指定 --case(配合 --fixtures)或同时指定 "
                "--stations/--arrivals/--plan")
        s_path, a_path, p_path = (Path(args.stations), Path(args.arrivals),
                                  Path(args.plan))
        case = args.name or a_path.stem
    for p in (s_path, a_path, p_path):
        if not p.exists():
            raise lio.InputError(f"文件不存在: {p}")
    plan = lio.load_plan(p_path)
    stations = lio.load_stations(s_path)
    arrivals = lio.load_arrivals(a_path, stations, plan.velocities)
    out_dir = Path(args.out) if args.out else Path("outputs") / case
    return case, plan, stations, arrivals, out_dir


def _cmd_locate(args) -> int:
    case, plan, stations, arrivals, out_dir = _load_case(args)
    result = run_location(stations, arrivals, plan, c=args.huber_c)
    write_all(out_dir, case, result)
    best = result["best"]
    print(f"[{case}] 最优解: x={best['x_km']:.4f} km, y={best['y_km']:.4f} km, "
          f"depth={best['depth_km']:.4f} km, t0={best['origin_time_s']:.4f} s, "
          f"代价={best['cost']:.4f}")
    print(f"[{case}] 不同极小点 {result['n_distinct_minima']} 个, "
          f"支持区域体素 {result['support']['n_voxels']} 个, "
          f"连通块 {len(result['support']['components'])} 个"
          + (f", 触及边界: {', '.join(result['support']['touches_boundary'])}"
             if result["support"]["touches_boundary"] else ""))
    print(f"[{case}] 输出目录: {out_dir}")
    return 0


def _cmd_run_all(args) -> int:
    fixtures = Path(args.fixtures)
    plan = lio.load_plan(fixtures / "plan.json")
    import json
    cases = json.loads((fixtures / "plan.json").read_text(encoding="utf-8"))["cases"]
    out_root = Path(args.out)
    results = []
    for case in cases:
        stations = lio.load_stations(fixtures / f"stations_{case}.csv")
        arrivals = lio.load_arrivals(fixtures / f"arrivals_{case}.csv",
                                     stations, plan.velocities)
        result = run_location(stations, arrivals, plan, c=args.huber_c)
        write_all(out_root / case, case, result)
        results.append((case, result))
        best = result["best"]
        print(f"[{case}] 最优解: ({best['x_km']:.3f}, {best['y_km']:.3f}, "
              f"{best['depth_km']:.3f}) km, t0={best['origin_time_s']:.3f} s, "
              f"极小点 {result['n_distinct_minima']} 个, "
              f"支持区域 {result['support']['n_voxels']} 体素")
    write_combined_report_md(out_root / "report.md", results)
    print(f"总报告: {out_root / 'report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="locate",
        description="单事件到时定位:固定常速模型 + 稳健联合反演 + 网格支持区域")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--fixtures", default="fixtures", help="输入目录(配合 --case)")
        p.add_argument("--case", help="案例名,文件取 stations_<case>.csv 等")
        p.add_argument("--stations", help="台站 CSV(显式指定)")
        p.add_argument("--arrivals", help="到时 CSV(显式指定)")
        p.add_argument("--plan", help="plan JSON(显式指定)")
        p.add_argument("--name", help="显式模式下的案例名")
        p.add_argument("--out", help="输出目录")
        p.add_argument("--huber-c", type=float, default=HUBER_C,
                       help="Huber 稳健尺度(单位:标准差,默认 %(default)s)")

    common(sub.add_parser("locate", help="定位单个案例"))
    common(sub.add_parser("run-all", help="运行 plan.json 中列出的全部案例并写总报告"))
    sp = sub.add_parser("selftest", help="解析无噪声自检 + 独立残差重算 + 负例校验")
    sp.add_argument("--out", default="outputs/selftest", help="输出目录")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "selftest":
            return run_selftest(Path(args.out))
        if args.cmd == "run-all":
            return _cmd_run_all(args)
        return _cmd_locate(args)
    except lio.InputError as exc:
        print(f"输入错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
