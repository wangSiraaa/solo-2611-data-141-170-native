"""命令行入口：python run_joint_reg.py <observations.csv> <anchor.json> <out_dir>"""

from __future__ import annotations

import sys

from jointreg.pipeline import run_pipeline
from jointreg.writer import write_outputs


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    result = run_pipeline(argv[1], argv[2])
    paths = write_outputs(result, argv[3])
    s = result.summary
    print(f"案例输出目录: {argv[3]}")
    print(
        f"站数={s['n_stations']} 标靶={s['n_targets']} "
        f"拟合观测={s['n_fit_observations']} holdout={s['n_holdout_observations']}"
    )
    print("拟合残差 RMS (m):", s["fit_residual_rms_m"])
    print("holdout 一致性 RMS (m):", s["holdout_rms_m"])
    print("被稳健降权观测数:", s["n_downweighted_observations"])
    for name, p in sorted(paths.items()):
        print(f"  {name}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
