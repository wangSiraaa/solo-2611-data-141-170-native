"""命令行入口。

用法::

    python -m sounding --config fixtures/config.json \\
        --beams fixtures/beams.csv \\
        --navigation fixtures/navigation.csv \\
        --tide fixtures/tide.csv \\
        --sound-speed fixtures/sound_speed.csv \\
        --output-dir output

退出码：0 表示整批处理完成（允许存在 failed/undetermined 波束，
原因写入输出）；输入文件本身无法读取/解析时返回 2。
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .pipeline import STATUS_FAILED, STATUS_UNDETERMINED, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="单航段多波束测深归位（水平分层声速 + 姿态四元数 + 潮位）"
    )
    parser.add_argument("--config", required=True, help="config.json 路径")
    parser.add_argument("--beams", required=True, help="beams.csv 路径")
    parser.add_argument("--navigation", required=True, help="navigation.csv 路径")
    parser.add_argument("--tide", required=True, help="tide.csv 路径")
    parser.add_argument(
        "--sound-speed", required=True, help="sound_speed.csv 路径"
    )
    parser.add_argument(
        "--output-dir", default="output", help="输出目录（默认 output）"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        result = run_pipeline(
            beams_path=args.beams,
            nav_path=args.navigation,
            tide_path=args.tide,
            speed_path=args.sound_speed,
            config=config,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError) as exc:
        print(f"[fatal] 输入无法处理: {exc}", file=sys.stderr)
        return 2

    print(f"汇总: {result['summary_path']}")
    print(f"全量证据: {result['detail_path']}")
    stats = result["stats"]
    print(
        "统计: total={total} ok={ok} undetermined={undetermined} "
        "failed={failed}".format(**stats)
    )
    for r in result["results"]:
        if r.status == "ok":
            n, e, d = r.sounding_point_ned_m
            print(
                f"  [ok]            {r.beam_id:>4} t={r.time_ms:>6.0f} ms "
                f"NED=({n:10.4f}, {e:10.4f}, {d:10.4f}) "
                f"depth={r.end_depth_below_surface_m:9.4f} m "
                f"segments={len(r.segments)}"
            )
        else:
            tag = "undetermined" if r.status == STATUS_UNDETERMINED else "failed"
            print(
                f"  [{tag:<14}] {r.beam_id:>4} t={r.time_ms:>6.0f} ms "
                f"reasons={r.failure_reasons}"
            )

    # 存在失败波束不视为程序错误：证据已按波束落盘
    if stats[STATUS_FAILED] or stats[STATUS_UNDETERMINED]:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
