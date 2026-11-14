"""命令行入口：``python -m fluxaudit audit <mesh.vtu> [选项]``。

成功审计退出码 0；网格因重复单元/零体积/非流形被拒绝退出码 2；
用法或 IO 错误退出码 1。拒绝时只输出 diagnostics.json，不产出成功产物。
"""

from __future__ import annotations

import argparse
import json
import sys

from .audit import AuditConfig, AuditRejected, run_audit
from .report import write_all, write_rejection


def _load_plan(plan_path: str | None) -> dict:
    if plan_path is None:
        return {}
    with open(plan_path, encoding="utf-8") as fh:
        return json.load(fh)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fluxaudit",
        description="一阶四面体网格的逐材料进出通量审计（不修复、不解 PDE）。",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit", help="审计单个 VTU 网格并输出 CSV/JSON 证据")
    p.add_argument("mesh", help="VTU UnstructuredGrid 文件路径")
    p.add_argument("--plan", help="plan.json：单位、场名、容差等默认值")
    p.add_argument("--out", default="audit_out", help="输出目录（默认 audit_out）")
    p.add_argument("--length-unit", help="坐标长度单位，覆盖 plan.json")
    p.add_argument("--field-unit", help="通量密度单位，覆盖 plan.json")
    p.add_argument("--field-name", help="节点向量场名称，覆盖 plan.json")
    p.add_argument("--material-name", help="单元材料字段名，覆盖 plan.json")
    p.add_argument("--tolerance", type=float, help="数值容差，覆盖 plan.json")
    return parser


def _config_from(args: argparse.Namespace) -> AuditConfig:
    plan = _load_plan(args.plan)
    return AuditConfig(
        length_unit=args.length_unit or plan.get("length_unit", "m"),
        field_unit=args.field_unit or plan.get(
            "field_unit", "unit m^-2 s^-1"),
        field_name=args.field_name or plan.get("field", "flux"),
        material_name=args.material_name or plan.get(
            "material_field", "material_id"),
        tolerance=(args.tolerance if args.tolerance is not None
                   else float(plan.get("numeric_tolerance", 1e-10))),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "audit":  # pragma: no cover - argparse 已保证
        return 1
    config = _config_from(args)
    try:
        result = run_audit(args.mesh, config)
    except AuditRejected as exc:
        diag_path = write_rejection(exc, args.out)
        print(f"AUDIT REJECTED: {args.mesh}", file=sys.stderr)
        for err in exc.diagnostics["errors"]:
            print(f"  - {err}", file=sys.stderr)
        print(f"diagnostics written to {diag_path}", file=sys.stderr)
        return 2

    paths = write_all(result, args.out)
    d = result.diagnostics
    g = d["global_checks"]
    print(f"AUDIT {d['status'].upper()}: {args.mesh}")
    print(f"  cells={d['n_cells']} points={d['n_points']} "
          f"materials={d['materials']}")
    print(f"  faces: boundary={g['n_faces_by_kind']['boundary']} "
          f"interface={g['n_faces_by_kind']['interface']} "
          f"interior={g['n_faces_by_kind']['interior']}")
    for s in result.material_summaries:
        print(f"  material {s.material_id}: inflow={s.inflow_flux:.6g} "
              f"outflow={s.outflow_flux:.6g} net={s.face_flux_net:.6g} "
              f"div_integral={s.divergence_integral:.6g} "
              f"residual={s.residual:.3e} "
              f"[{config.length_unit}, flux in "
              f"({config.field_unit})*{config.length_unit}^2]")
    print(f"  global face flux={g['global_face_flux']:.12g} "
          f"div integral={g['global_divergence_integral']:.12g} "
          f"residual={g['global_residual']:.3e}")
    print(f"  interface max cancellation residual="
          f"{g['max_interface_flux_cancellation_residual']:.3e}; "
          f"checks_passed={g['passed']}")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
