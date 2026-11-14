# 周期轨迹材料

本目录仅含题目输入，不含待实现程序。全部材料为本地合成，无真实个人或仪器数据。Python 3.11；若使用列出的库，可执行 `python -m pip install -r requirements.txt`。无需 Docker、账号或网络数据下载。

NPZ 的 positions_nm 维度为 frame×atom×xyz，晶胞矩阵的列为三个晶格向量，r=Hf；输入原子坐标已折回 f∈[0,1)。时间为 ps，长度 nm。topology 的 bonds 是 atom 数组索引，每个分子有固定质量和键图。

plan 给出每步最大质心速度和最大允许缺帧间隔，这些是可行性界限，不是精确速度模型。cycle-case 是单独的负向材料，用 cycle-topology 中 3.6 nm 键界；各局部短键可看似正常但闭环无法作为有限分子一致展开。它不和主轨迹拼接。
