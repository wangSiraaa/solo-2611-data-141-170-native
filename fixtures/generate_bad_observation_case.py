"""生成 fixtures/cases/bad-observation.csv。

在主数据基础上对 B 站 T06 注入一个 0.30 m 的粗差（约 30 sigma，sigma=0.01 m），
其余记录（含主数据自带的 C/T05 错误）保持不变，用于演示稳健联合调整
如何同时处理多处错配离群而不删除任何观测。

用法：python fixtures/generate_bad_observation_case.py
"""

from __future__ import annotations

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "observations.csv")
DST = os.path.join(HERE, "cases", "bad-observation.csv")

GROSS_ERROR_M = 0.30
TARGET_STATION, TARGET_ID = "B", "T06"


def main() -> None:
    with open(SRC, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, data = rows[0], rows[1:]
    injected = False
    for r in data:
        if r[0] == TARGET_STATION and r[1] == TARGET_ID:
            r[2] = f"{float(r[2]) + GROSS_ERROR_M:.10f}"  # x 方向粗差
            injected = True
    assert injected, "未找到注入目标 B/T06"
    with open(DST, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(data)
    print(f"已写出 {DST}（在 {TARGET_STATION}/{TARGET_ID} 的 x 注入 {GROSS_ERROR_M} m 粗差）")


if __name__ == "__main__":
    main()
