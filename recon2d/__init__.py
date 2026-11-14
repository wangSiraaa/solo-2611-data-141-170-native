"""二维直线射线层析重建工具（规则网格、非负 NNLS、邻格平滑正则）。

模块划分：
- geometry:    射线-网格求交、裁切、逐格路径长度、稀疏前向算子
- dataio:      rays.csv / grid.json 读取与输入诊断
- regularize:  邻格差分矩阵与无支持格先验选择矩阵
- solve:       误差加权非负最小二乘求解与正则扫描
- diagnostics: 秩 / 近零空间 / 稳定性 / 有限角度探测
- cli:         命令行入口与结果落盘
"""

__version__ = "1.0.0"

# 题目约束：只处理 12x12 以下网格、百条射线
MAX_GRID = 12
MAX_RAYS = 100
