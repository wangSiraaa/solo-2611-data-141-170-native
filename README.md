# 单事件到时材料

本目录仅含题目输入，不含待实现程序。全部材料为本地合成，无真实个人或仪器数据。Python 3.11；若使用列出的库，可执行 `python -m pip install -r requirements.txt`。无需 Docker、账号或网络数据下载。

每组 stations/arrivals 是同一个合成单事件的独立实验，不联合拟合三组。单位 km、秒，depth 向下为正；台站位于 depth=0。波速固定、射线直线、均匀介质；无需波形拾取或事件关联。

weak 为近一维台阵且只有 P 相，空间几何可造成多解。outlier 有少量误拾取。profile_score_delta 只定义报告的近优代价区域，不是统计置信水平；若使用稳健损失，说明其尺度和剖面比较口径。

## 实现与用法

定位程序在 `locator/`(Python 3.11 + NumPy/SciPy),命令为 `locate`:

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m locator run-all --fixtures fixtures --out outputs   # 三组全跑 + 总报告
.venv/bin/python -m locator locate --case good --out outputs/good       # 单组(或 ./locate --case good)
.venv/bin/python -m locator selftest --out outputs/selftest             # 解析自检
```

`selftest` 用无噪声合成例核对源位置/发震时刻(误差 < 1e-3),并用纯标准库独立代入每条到时重算残差,另含稳健性与非法输入负例检查。每组输出 `solution.json`、`residuals.csv`、`profile.csv`、`depth_profile.csv`、`report.md`;总报告在 `outputs/report.md`。
