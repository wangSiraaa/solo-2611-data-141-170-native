"""独立复算工具：在少数给定频率上直接构建设计矩阵核对加权残差。

不经过候选搜索流程，仅用 NumPy 正规方程在指定频率上求解 WLS，
并与 period_analysis.model.weighted_fit 的结果逐项比对。

示例：
  .venv/bin/python scripts/recompute_check.py fixtures/spread.csv \
      --freqs 0.36621 0.5 1.35883 --harmonics 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from period_analysis.io_utils import load_lightcurve          # noqa: E402
from period_analysis.model import design_matrix, weighted_fit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path")
    ap.add_argument("--freqs", nargs="+", type=float, required=True)
    ap.add_argument("--harmonics", type=int, default=2, choices=(1, 2))
    args = ap.parse_args()

    lc = load_lightcurve(args.csv_path)
    if not lc.clean:
        print(f"输入存在 {len(lc.issues)} 条问题，拒绝复算：", file=sys.stderr)
        for i in lc.issues:
            print(f"  行{i.csv_row} {i.observation_id} {i.column}: {i.kind}", file=sys.stderr)
        return 2

    t, y, s = lc.time, lc.brightness, lc.sigma
    t0 = float(np.mean(t))
    w = 1.0 / s**2
    ok = True
    print(f"{'freq':>10} | {'chi2_independent':>16} | {'chi2_package':>14} | "
          f"{'max|beta diff|':>14} | {'max|resid diff|':>15}")
    for f in args.freqs:
        X = design_matrix(t, f, args.harmonics, t0)
        Xw = X * w[:, None]
        XtWX = Xw.T @ X
        beta_indep = np.linalg.solve(XtWX, Xw.T @ y)
        resid_indep = y - X @ beta_indep
        chi2_indep = float(resid_indep @ (w * resid_indep))

        fit = weighted_fit(t, y, s, f, args.harmonics, t0)
        d_beta = float(np.max(np.abs(beta_indep - fit.coefs)))
        d_res = float(np.max(np.abs(resid_indep - fit.residual)))
        match = np.isclose(chi2_indep, fit.chi2, rtol=1e-10) and d_beta < 1e-10
        ok &= match
        print(f"{f:10.6f} | {chi2_indep:16.8f} | {fit.chi2:14.8f} | "
              f"{d_beta:14.2e} | {d_res:15.2e}  {'OK' if match else 'MISMATCH'}")

        # 协方差口径核对：(X'WX)^-1 对角元即各参数方差
        cov_indep = np.linalg.inv(XtWX)
        assert np.allclose(cov_indep, fit.cov, rtol=1e-10)
    print("\n协方差矩阵 (X'WX)^-1 与包内拟合口径一致。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
