"""离线小DEM水文分析包：填洼、D8流向、汇流量、出口流域与守恒核验。

只依赖 NumPy；地图报告由 hydro.report 额外使用 matplotlib。
所有输入输出均为本地 ESRI ASCII Grid（或等价的带地理元数据数组），
不访问网络，不做投影变换、点云地面估计或镶嵌。
"""

__all__ = [
    "gridio",
    "fill",
    "flow",
    "accumulation",
    "basins",
    "verify",
    "report",
]
