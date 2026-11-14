"""运行全部演示案例并打印人可读对比。

案例：
  main            四站正常闭环（数据中同时含 1 个错配离群与共线站 D）
  bad-observation 额外注入第二个粗差，验证稳健降权
  collinear       同主数据，重点展示共线站自由度报告（复用 main 输入）
  disconnected    E 站与 A 无共有标靶路径，6 自由度不可识别
  exact-small     两站四点精确小例，残差应到机器精度量级
"""

from __future__ import annotations

import os

from jointreg.pipeline import run_pipeline
from jointreg.writer import write_outputs

ROOT = os.path.dirname(os.path.abspath(__file__))
ANCHOR = os.path.join(ROOT, "fixtures", "anchor.json")

CASES = [
    ("main", "fixtures/observations.csv"),
    ("bad-observation", "fixtures/cases/bad-observation.csv"),
    ("collinear", "fixtures/observations.csv"),
    ("disconnected", "fixtures/cases/disconnected.csv"),
    ("exact-small", "fixtures/cases/exact-small.csv"),
]


def _fmt(v, nd=5):
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> None:
    for name, rel in CASES:
        csv_path = os.path.join(ROOT, rel)
        out_dir = os.path.join(ROOT, "outputs", name)
        result = run_pipeline(csv_path, ANCHOR)
        write_outputs(result, out_dir)
        s = result.summary

        print("=" * 78)
        print(f"案例 {name}（{rel}）-> {out_dir}")
        print(
            f"站={s['n_stations']} 标靶={s['n_targets']} "
            f"拟合观测={s['n_fit_observations']} holdout={s['n_holdout_observations']}"
        )
        print("  连接: 锚分量=%s 断开站=%s" % (
            result.graph.anchor_component, result.graph.disconnected_stations
        ))
        for st in result.station_rows:
            print(
                f"  站 {st['station_id']}: {st['support_verdict']:11s} "
                f"可识别={st['identifiable']}  {st['status']}"
            )
        print("  拟合残差 RMS(m): init=%s  joint_huber=%s  plain_lsq=%s" % (
            _fmt(s["fit_residual_rms_m"]["initialization"]),
            _fmt(s["fit_residual_rms_m"]["joint_huber"]),
            _fmt(s["fit_residual_rms_m"]["plain_lsq"]),
        ))
        print("  内点拟合 RMS(m): init=%s  joint_huber=%s  plain_lsq=%s" % (
            _fmt(s["inlier_fit_residual_rms_m"]["initialization"]),
            _fmt(s["inlier_fit_residual_rms_m"]["joint_huber"]),
            _fmt(s["inlier_fit_residual_rms_m"]["plain_lsq"]),
        ))
        print("  holdout RMS(m):  init=%s  joint_huber=%s  plain_lsq=%s" % (
            _fmt(s["holdout_rms_m"]["initialization"]),
            _fmt(s["holdout_rms_m"]["joint_huber"]),
            _fmt(s["holdout_rms_m"]["plain_lsq"]),
        ))
        # 如实显示：联合调整相对初始化，内点拟合与 holdout 各自是改善还是恶化。
        # 总 RMS 口径含保留离群点，其"恶化"是稳健性的正确表现，不在这里当拟合退步。
        ir0, ir1 = (s["inlier_fit_residual_rms_m"]["initialization"],
                    s["inlier_fit_residual_rms_m"]["joint_huber"])
        h0, h1 = s["holdout_rms_m"]["initialization"], s["holdout_rms_m"]["joint_huber"]
        if ir0 is not None and h0 is not None:
            idir = "改善" if ir1 <= ir0 else "恶化"
            hdir = "改善" if h1 <= h0 else "恶化"
            print(f"  联合调整 vs 初始化（公平口径）: 内点拟合{idir}（{ir0:.4f}->{ir1:.4f}），"
                  f"holdout 一致性{hdir}（{h0:.4f}->{h1:.4f}）；恶化也如实显示")
        print("  闭环 RMS(m) [内点 RMS]:")
        for c in result.closure_summary:
            extra = (
                f" [{_fmt(c.get('inlier_closure_rms_m'))}]"
                if c.get("inlier_closure_rms_m") is not None else ""
            )
            print(
                f"    {c['solution']:14s} {c['station_i']}-{c['station_j']}: "
                f"{_fmt(c['closure_rms_m'])}{extra}"
            )
        flagged = [
            r for r in result.residual_rows
            if r["solution"] == "joint_huber" and r["action"] == "downweighted_huber"
        ]
        if flagged:
            print("  稳健降权观测（保留未删除）:")
            for r in flagged:
                print(
                    f"    {r['station_id']}/{r['target_id']} "
                    f"残差={r['residual_norm_m']:.4f} m "
                    f"({r['max_abs_normalized_sigma']:.1f}σ) 权重={r['robust_weight']:.3f}"
                )
        if result.ident is not None:
            print(
                "  联合雅可比 SVD: 零空间方向=%d 弱方向=%d（共 %d 参数）"
                % (
                    len(result.ident["null_directions"]),
                    len(result.ident["weak_directions"]),
                    result.ident["n_params"],
                )
            )
        print()


if __name__ == "__main__":
    main()
