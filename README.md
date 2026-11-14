# 单事件到时材料

本目录仅含题目输入，不含待实现程序。全部材料为本地合成，无真实个人或仪器数据。Python 3.11；若使用列出的库，可执行 `python -m pip install -r requirements.txt`。无需 Docker、账号或网络数据下载。

每组 stations/arrivals 是同一个合成单事件的独立实验，不联合拟合三组。单位 km、秒，depth 向下为正；台站位于 depth=0。波速固定、射线直线、均匀介质；无需波形拾取或事件关联。

weak 为近一维台阵且只有 P 相，空间几何可造成多解。outlier 有少量误拾取。profile_score_delta 只定义报告的近优代价区域，不是统计置信水平；若使用稳健损失，说明其尺度和剖面比较口径。
