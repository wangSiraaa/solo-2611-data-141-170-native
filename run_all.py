"""便捷入口：依次构建/核验/出图三个场景（dem、tiny_flat、enclosed）。

    python run_all.py
等价于手工执行 hydro build / verify / report 三组子命令。
"""

from __future__ import annotations

import sys

from hydro.build import run_build
from hydro.report import render_map, text_arrow_map
from hydro.verify import run_verify

JOBS = [
    ("fixtures/dem.asc", "fixtures/outlets.csv", "outputs/dem"),
    ("fixtures/tiny_flat.asc", None, "outputs/tiny"),
    ("cases/enclosed.asc", None, "outputs/enclosed"),
]


def main() -> int:
    for dem, outlets, outdir in JOBS:
        run_build(dem, outlets, outdir)
        run_verify(dem, outlets, outdir)
        render_map(outdir)
        text_arrow_map(outdir)
    print("\n全部场景 build + verify + map 完成，产物见 outputs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
