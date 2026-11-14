"""独立验证脚本：不调用求解器内部状态，只从输入 CSV 与输出 JSON 重算核对。

检查项：
  1. 用报告的 R/t 独立代入 world=R@local+t，重算每条拟合观测残差，与输出残差逐分量一致；
  2. R^T R = I、det(R)=+1；锚站 R=I、t=0；
  3. 打乱输入记录行序后重跑：站身份与结果不变（身份只由 station_id/target_id 决定）；
  4. 三至四个手工点验证变换方向（含已知 90° 旋转 + 平移）；
  5. 反射防护：构造镜面对应点，Kabsch 必须返回 det=+1 并标注 reflection_corrected；
  6. exact-small 案例残差达到机器精度量级。
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from jointreg.geometry import SO3_exp, weighted_kabsch
from jointreg.io import load_observations
from jointreg.adjust import JointModel
from jointreg.pipeline import run_pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))
ANCHOR = os.path.join(ROOT, "fixtures", "anchor.json")
TOL = 1e-9


def check(cond: bool, msg: str) -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        raise AssertionError(msg)


def independent_residual_check(case: str, csv_path: str) -> None:
    print(f"[{case}] 独立代入重算残差 / 正交性 / 行列式 / 锚基准")
    out = os.path.join(ROOT, "outputs", case, "report.json")
    report = json.load(open(out, encoding="utf-8"))
    poses = report["station_poses"]
    residual_lookup = {
        (r["station_id"], r["target_id"]): r
        for r in report["observation_residuals"]
        if r["solution"] == "joint_huber"
    }
    target_lookup = {t["target_id"]: t for t in report["targets"] if not t["holdout_only"]}

    for o in load_observations(csv_path):
        if o.holdout:
            continue  # holdout 不参与拟合，无拟合残差行
        pose = poses[o.station]
        R = np.array(pose["R"])
        t = np.array(pose["t_m"])
        # 独立代入
        w = R @ o.local + t
        X = np.array([
            target_lookup[o.target]["x_m"],
            target_lookup[o.target]["y_m"],
            target_lookup[o.target]["z_m"],
        ])
        e = w - X
        row = residual_lookup[(o.station, o.target)]
        check(
            np.allclose(e, [row["resx_m"], row["resy_m"], row["resz_m"]], atol=1e-10),
            f"{o.station}/{o.target} 独立重算残差与报告一致",
        )

    for s, pose in poses.items():
        R = np.array(pose["R"])
        check(np.allclose(R.T @ R, np.eye(3), atol=TOL), f"站 {s}: R^T R = I")
        check(abs(np.linalg.det(R) - 1.0) < TOL, f"站 {s}: det(R) = +1（无反射混入）")
    anchor = report["summary"]["anchor_station"]
    check(
        np.allclose(poses[anchor]["R"], np.eye(3), atol=TOL)
        and np.allclose(poses[anchor]["t_m"], 0.0, atol=TOL),
        f"锚站 {anchor}: R=I, t=0（世界基准固定）",
    )


def row_order_invariance() -> None:
    print("[main] 行序不变性：打乱 CSV 记录顺序后重跑")
    rng = np.random.default_rng(42)
    base = load_observations(os.path.join(ROOT, "fixtures", "observations.csv"))
    order = rng.permutation(len(base))
    import csv as _csv

    with tempfile.TemporaryDirectory() as td:
        shuffled = os.path.join(td, "shuffled.csv")
        with open(shuffled, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(
                ["station_id", "target_id", "x_m", "y_m", "z_m", "sigma_m", "holdout"]
            )
            for i in order:
                o = base[i]
                w.writerow([
                    o.station, o.target, *o.local.tolist(), o.sigma,
                    "true" if o.holdout else "false",
                ])
        r1 = run_pipeline(
            os.path.join(ROOT, "fixtures", "observations.csv"), ANCHOR
        )
        r2 = run_pipeline(shuffled, ANCHOR)
        # 可识别量必须逐元素相同；共线站绕直线轴的旋转角本身不可识别，
        # 其代表姿态允许不同，改在下方按可识别自由度核对。
        status = {st["station_id"]: st for st in r1.station_rows}
        for s in r1.ds.stations:
            if status[s]["identifiable"]:
                check(np.allclose(r1.R[s], r2.R[s], atol=1e-12), f"站 {s}: R 与行序无关")
            else:
                # R 在直线方向上的作用必须一致（两个可识别旋转角）
                line = np.array(status[s]["line_axis_local"])
                check(
                    np.allclose(r1.R[s] @ line, r2.R[s] @ line, atol=1e-10),
                    f"站 {s}: 共线轴经 R 的映射与行序无关（不可识别的是绕轴转角）",
                )
            check(np.allclose(r1.t[s], r2.t[s], atol=1e-12), f"站 {s}: t 与行序无关")
        for tg in r1.X:
            check(np.allclose(r1.X[tg], r2.X[tg], atol=1e-12), f"标靶 {tg}: 坐标与行序无关")
        # 共线站的拟合观测残差（物理可检验量）也必须与行序无关
        for o in r1.ds.fit_observations():
            if status[o.station]["identifiable"]:
                continue
            e1 = r1.R[o.station] @ o.local + r1.t[o.station] - r1.X[o.target]
            e2 = r2.R[o.station] @ o.local + r2.t[o.station] - r2.X[o.target]
            check(
                np.allclose(e1, e2, atol=1e-10),
                f"站 {o.station}/{o.target}: 可检验残差与行序无关",
            )


def hand_points_direction() -> None:
    print("[手工点] 变换方向验证 world = R @ local + t（3+1 个点）")
    # 已知变换：绕 z 轴 +90°，平移 (10, 20, 30)
    angle = np.pi / 2
    R_true = SO3_exp(np.array([0.0, 0.0, angle]))
    t_true = np.array([10.0, 20.0, 30.0])
    local = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    world = (R_true @ local.T).T + t_true
    kab = weighted_kabsch(local, world)
    check(np.allclose(kab.R, R_true, atol=TOL), "Kabsch 恢复 R=Rz(90°)")
    check(np.allclose(kab.t, t_true, atol=TOL), "Kabsch 恢复 t=(10,20,30)")
    # 方向语义：local x 轴应映射到 world y 方向
    mapped_x = kab.R @ np.array([1.0, 0.0, 0.0])
    check(
        np.allclose(mapped_x, [0.0, 1.0, 0.0], atol=TOL),
        "local +x 经 R 映射到 world +y（约定方向正确，非 world->local）",
    )
    # 单点前向代入核对
    check(
        np.allclose(kab.R @ local[1] + kab.t, world[1], atol=TOL),
        "R@local+t 前向代入与构造世界点一致",
    )


def reflection_guard() -> None:
    print("[反射防护] 镜面对应点不得返回 det=-1 的旋转")
    rng = np.random.default_rng(7)
    local = rng.normal(size=(6, 3))
    mirror = np.diag([1.0, 1.0, -1.0])  # 反射
    world = (mirror @ local.T).T + np.array([1.0, 2.0, 3.0])
    kab = weighted_kabsch(local, world)
    check(np.linalg.det(kab.R) > 0, "反射场景下仍强制 det(R)=+1")
    check(kab.reflection_corrected, "明确标注 reflection_corrected=True")
    check(kab.rotational_rank == 3, "秩信息仍为 3（不因反射处理误报退化）")


def exact_small_case() -> None:
    print("[exact-small] 精确小例残差应为机器精度量级")
    r = run_pipeline(
        os.path.join(ROOT, "fixtures", "cases", "exact-small.csv"), ANCHOR
    )
    rms = r.summary["fit_residual_rms_m"]["joint_huber"]
    check(rms < 1e-8, f"joint_huber 拟合 RMS={rms:.2e} < 1e-8 m")
    # B 站应完整可识别
    b = next(st for st in r.station_rows if st["station_id"] == "B")
    check(b["identifiable"] and b["support_verdict"] == "full", "B 站完整可识别")


def disconnected_case() -> None:
    print("[disconnected] E 站与 A 不连通：状态与临时锚处理")
    report = json.load(
        open(os.path.join(ROOT, "outputs", "disconnected", "report.json"), encoding="utf-8")
    )
    e = next(st for st in report["stations"] if st["station_id"] == "E")
    check(not e["connected_to_anchor"], "E 站标注为未连接锚站")
    check(not e["identifiable"], "E 站相对 A 的 6 自由度标注为不可识别")
    check("E" in report["graph"]["disconnected_stations"], "图分析断开列表含 E")


def analytic_jacobian_check() -> None:
    print("[雅可比] 解析雅可比 vs 中心有限差分（含非零旋转向量点）")
    r = run_pipeline(
        os.path.join(ROOT, "fixtures", "observations.csv"), ANCHOR
    )
    fit = r.ds.fit_observations()
    model = JointModel(
        fit, r.ds.anchor_station, r.R_init, r.t_init,
        {tg: r.X_init[tg] for tg in r.X_init},
    )

    def fd_jac(x):
        eps = 1e-7
        Jn = np.zeros((model.n_obs, model.layout.n_params))
        for i in range(model.layout.n_params):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            Jn[:, i] = (model.residuals(xp) - model.residuals(xm)) / (2 * eps)
        return Jn

    rng = np.random.default_rng(0)
    for label, x in [
        ("初始点", model.x),
        ("扰动点", model.x + rng.normal(scale=0.08, size=model.layout.n_params)),
    ]:
        rel = np.max(np.abs(model.jacobian(x) - fd_jac(x))) / np.max(np.abs(fd_jac(x)))
        check(rel < 1e-6, f"{label}：解析/数值雅可比相对误差 {rel:.2e} < 1e-6")


def main() -> None:
    independent_residual_check("main", "fixtures/observations.csv")
    independent_residual_check("bad-observation", "fixtures/cases/bad-observation.csv")
    independent_residual_check("disconnected", "fixtures/cases/disconnected.csv")
    independent_residual_check("exact-small", "fixtures/cases/exact-small.csv")
    row_order_invariance()
    analytic_jacobian_check()
    hand_points_direction()
    reflection_guard()
    exact_small_case()
    disconnected_case()
    print("\n全部独立验证通过。")


if __name__ == "__main__":
    main()
