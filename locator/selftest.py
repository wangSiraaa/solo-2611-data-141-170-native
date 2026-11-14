"""解析自检:无噪声小例核对源位置/发震时刻,并独立代入每条到时重算残差。

- 用固定模型手工生成无噪声到时,跑完整定位流程,核对恢复参数;
- 用纯 Python(math 模块,不经 numpy)从 solution.json 出发独立重算每条
  pick 的预测到时与标准化残差,与 residuals.csv 对比;
- 加一条误拾取验证稳健性;
- 负例:非法输入必须被校验拒绝。
"""
from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from . import io as lio
from .engine import run_location
from .report import write_all

# 无噪声真值(盒内任意取的非对称点,避免对称性干扰唯一性检验)
TRUE_X, TRUE_Y, TRUE_Z, TRUE_T0 = 2.5, -3.0, 7.0, 0.8
VP, VS = 6.0, 3.5

# 不规则分布的 9 个台站(避免对称几何)
STATIONS = [
    ("t0", 14.2, 1.3), ("t1", 9.8, 11.5), ("t2", -1.7, 14.6),
    ("t3", -12.4, 8.2), ("t4", -14.8, -2.6), ("t5", -9.1, -11.9),
    ("t6", 1.2, -14.4), ("t7", 11.7, -9.3), ("t8", 4.4, 5.2),
]

PLAN_DICT = {
    "velocity_km_s": {"P": VP, "S": VS},
    "position_bounds_km": {"x": [-10, 10], "y": [-10, 10], "depth": [0, 15]},
    "origin_time_bounds_seconds": [-5, 5],
    "profile_grid_step_km": 1.0,
    "profile_score_delta": 4.0,
}

POS_TOL_KM = 1e-3
T0_TOL_S = 1e-3
RECOMP_TOL = 1e-9


def _synthetic_arrivals(outlier: bool = False) -> list[tuple]:
    rows = []
    for i, (sid, sx, sy) in enumerate(STATIONS):
        d = math.sqrt((TRUE_X - sx) ** 2 + (TRUE_Y - sy) ** 2 + TRUE_Z ** 2)
        for phase, v, sig in (("P", VP, 0.02), ("S", VS, 0.03)):
            t = TRUE_T0 + d / v
            if outlier and sid == "t3" and phase == "P":
                t += 0.8                      # 人为误拾取
            rows.append((f"st-{i}-{phase}", sid, phase, t, sig))
    return rows


def _write_inputs(tmp: Path, arrivals) -> tuple[Path, Path, Path]:
    s_path = tmp / "stations.csv"
    with open(s_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["station_id", "x_km", "y_km", "depth_km"])
        for sid, sx, sy in STATIONS:
            w.writerow([sid, sx, sy, 0.0])
    a_path = tmp / "arrivals.csv"
    with open(a_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pick_id", "station_id", "phase", "time_seconds", "sigma_seconds"])
        for row in arrivals:
            w.writerow(row)
    p_path = tmp / "plan.json"
    p_path.write_text(json.dumps(PLAN_DICT))
    return s_path, a_path, p_path


def _load(s_path, a_path, p_path):
    plan = lio.load_plan(p_path)
    stations = lio.load_stations(s_path)
    arrivals = lio.load_arrivals(a_path, stations, plan.velocities)
    return plan, stations, arrivals


def _independent_residual_check(out_dir: Path) -> tuple[int, float]:
    """纯 Python 独立重算:从 solution.json 读解,逐条代入到时模型。"""
    sol = json.loads((out_dir / "solution.json").read_text(encoding="utf-8"))
    best = sol["best_solution"]
    vel = sol["model"]["velocities_km_s"]
    sta = {}
    with open(out_dir / "stations.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            sta[row["station_id"]] = (float(row["x_km"]), float(row["y_km"]),
                                      float(row["depth_km"]))
    max_diff = 0.0
    n = 0
    with open(out_dir / "residuals.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            sx, sy, sz = sta[row["station_id"]]
            d = math.sqrt((best["x_km"] - sx) ** 2 + (best["y_km"] - sy) ** 2
                          + (best["depth_km"] - sz) ** 2)
            pred = best["origin_time_s"] + d / vel[row["phase"]]
            std_resid = (float(row["observed_s"]) - pred) / float(row["sigma_s"])
            max_diff = max(max_diff,
                           abs(pred - float(row["predicted_s"])),
                           abs(std_resid - float(row["standardized_residual"])))
            n += 1
    return n, max_diff


def _expect_input_error(name: str, fn, checks: list, failures: list) -> None:
    try:
        fn()
    except lio.InputError:
        checks.append(f"PASS 负例拒绝: {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"FAIL 负例 {name}: 抛出了非 InputError: {exc!r}")
    else:
        failures.append(f"FAIL 负例 {name}: 未被拒绝")


def _negative_tests(checks: list, failures: list, tmp: Path) -> None:
    s_path, a_path, p_path = _write_inputs(tmp, _synthetic_arrivals())

    def run_with(arrivals_text=None, stations_text=None, plan_dict=None):
        sp, ap, pp = s_path, a_path, p_path
        if stations_text is not None:
            sp = tmp / "neg_stations.csv"
            sp.write_text(stations_text)
        if arrivals_text is not None:
            ap = tmp / "neg_arrivals.csv"
            ap.write_text(arrivals_text)
        if plan_dict is not None:
            pp = tmp / "neg_plan.json"
            pp.write_text(json.dumps(plan_dict))
        pl = lio.load_plan(pp)
        st = lio.load_stations(sp)
        lio.load_arrivals(ap, st, pl.velocities)

    good_rows = list(csv.reader(open(a_path)))

    def mutated(mutate) -> str:
        rows = [r[:] for r in good_rows]
        mutate(rows)
        return "\n".join(",".join(r) for r in rows) + "\n"

    _expect_input_error("重复 pick_id", lambda: run_with(
        arrivals_text=mutated(lambda r: r.__setitem__(2, r[1][:]))), checks, failures)
    _expect_input_error("同台站同相位重复 pick", lambda: run_with(
        arrivals_text=mutated(lambda r: r.append(
            ["st-x-P2", r[1][1], r[1][2], r[1][3], r[1][4]]))), checks, failures)
    _expect_input_error("引用不存在的台站", lambda: run_with(
        arrivals_text=mutated(lambda r: r[1].__setitem__(1, "ghost"))), checks, failures)
    _expect_input_error("未知相位", lambda: run_with(
        arrivals_text=mutated(lambda r: r[1].__setitem__(2, "R"))), checks, failures)
    _expect_input_error("标准差为 0", lambda: run_with(
        arrivals_text=mutated(lambda r: r[1].__setitem__(4, "0"))), checks, failures)
    _expect_input_error("标准差为负", lambda: run_with(
        arrivals_text=mutated(lambda r: r[1].__setitem__(4, "-0.1"))), checks, failures)
    _expect_input_error("到时非有限值", lambda: run_with(
        arrivals_text=mutated(lambda r: r[1].__setitem__(3, "nan"))), checks, failures)
    _expect_input_error("台站重复定义", lambda: run_with(
        stations_text="station_id,x_km,y_km,depth_km\nt0,1,2,0\nt0,3,4,0\n"),
        checks, failures)
    bad_plan = dict(PLAN_DICT)
    bad_plan["velocity_km_s"] = {"P": 6.0, "S": 6.0}
    _expect_input_error("不同相位共用同一波速", lambda: run_with(plan_dict=bad_plan),
                        checks, failures)
    bad_plan2 = dict(PLAN_DICT)
    bad_plan2["velocity_km_s"] = {"P": -6.0, "S": 3.5}
    _expect_input_error("波速非正", lambda: run_with(plan_dict=bad_plan2),
                        checks, failures)


def run_selftest(out_dir: Path) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks: list[str] = []
    failures: list[str] = []

    with tempfile.TemporaryDirectory(dir=out_dir) as td:
        tmp = Path(td)
        # ---- 1. 无噪声解析例:完整流程 ----
        s_path, a_path, p_path = _write_inputs(tmp, _synthetic_arrivals())
        plan, stations, arrivals = _load(s_path, a_path, p_path)
        result = run_location(stations, arrivals, plan)
        case_dir = out_dir / "noise_free"
        write_all(case_dir, "selftest_noise_free", result)
        # 输入副本供独立重算读取
        (case_dir / "stations.csv").write_text(s_path.read_text())
        (case_dir / "arrivals.csv").write_text(a_path.read_text())

        best = result["best"]
        pos_err = math.sqrt((best["x_km"] - TRUE_X) ** 2
                            + (best["y_km"] - TRUE_Y) ** 2
                            + (best["depth_km"] - TRUE_Z) ** 2)
        t0_err = abs(best["origin_time_s"] - TRUE_T0)
        if pos_err < POS_TOL_KM and t0_err < T0_TOL_S:
            checks.append(f"PASS 无噪声例: 位置误差 {pos_err:.2e} km, "
                          f"t0 误差 {t0_err:.2e} s")
        else:
            failures.append(f"FAIL 无噪声例: 位置误差 {pos_err:.3e} km, "
                            f"t0 误差 {t0_err:.3e} s")
        if result["n_distinct_minima"] == 1:
            checks.append("PASS 无噪声例: 多初值搜索只发现 1 个极小点")
        else:
            failures.append(f"FAIL 无噪声例: 出现 {result['n_distinct_minima']} 个极小点")

        # 网格最小点应落在真值 1 个体素内
        g = result["grid"]
        idx = np.unravel_index(int(g["cost"].argmin()), g["cost"].shape)
        gx = float(g["x_km"][idx[0]])
        gy = float(g["y_km"][idx[1]])
        gz = float(g["depth_km"][idx[2]])
        step = plan.profile_grid_step_km
        if (abs(gx - TRUE_X) <= step and abs(gy - TRUE_Y) <= step
                and abs(gz - TRUE_Z) <= step):
            checks.append(f"PASS 无噪声例: 网格最小点 ({gx:.0f},{gy:.0f},{gz:.0f}) "
                          "在真值 1 个体素内")
        else:
            failures.append(f"FAIL 无噪声例: 网格最小点 ({gx},{gy},{gz}) 偏离真值")

        # ---- 2. 独立重算每条到时的残差 ----
        n_pick, max_diff = _independent_residual_check(case_dir)
        if max_diff <= RECOMP_TOL:
            checks.append(f"PASS 独立重算: {n_pick} 条 pick 最大偏差 {max_diff:.2e}")
        else:
            failures.append(f"FAIL 独立重算: 最大偏差 {max_diff:.3e} > {RECOMP_TOL:.0e}")

        # ---- 3. 稳健性:一条 +0.8 s 误拾取 ----
        s2, a2, p2 = _write_inputs(tmp, _synthetic_arrivals(outlier=True))
        plan2, stations2, arrivals2 = _load(s2, a2, p2)
        result2 = run_location(stations2, arrivals2, plan2)
        b2 = result2["best"]
        pos_err2 = math.sqrt((b2["x_km"] - TRUE_X) ** 2 + (b2["y_km"] - TRUE_Y) ** 2
                             + (b2["depth_km"] - TRUE_Z) ** 2)
        dw = [r for r in result2["residuals"] if r["downweighted"]]
        if pos_err2 < 0.3 and dw:
            checks.append(f"PASS 稳健性: 1 条误拾取下位置误差 {pos_err2:.3f} km, "
                          f"{len(dw)} 条记录被降权")
        else:
            failures.append(f"FAIL 稳健性: 位置误差 {pos_err2:.3f} km, "
                            f"降权记录 {len(dw)} 条")

        # ---- 4. 负例校验 ----
        _negative_tests(checks, failures, tmp)

    lines = ["# selftest 结果", ""]
    lines += [f"- {c}" for c in checks]
    if failures:
        lines += ["", "## 失败"] + [f"- {f}" for f in failures]
    lines.append("")
    summary = {"n_pass": len(checks), "n_fail": len(failures),
               "checks": checks, "failures": failures}
    (out_dir / "selftest_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "selftest_report.md").write_text("\n".join(lines), encoding="utf-8")
    for c in checks:
        print(c)
    for f in failures:
        print(f)
    print(f"selftest: {len(checks)} 通过, {len(failures)} 失败")
    return 1 if failures else 0
