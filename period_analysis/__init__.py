"""不规则光变曲线周期分析包。

模块划分：
- model:       加权谐波设计矩阵与 WLS 拟合
- io_utils:    CSV 读取与重复/非有限/非正误差定位
- periodogram: 频率网格、chi2 周期图、局部细化、候选去重
- window:      采样窗、谱窗、每日别名族
- holdout:     整夜留出与 P/2P 近等价识别
- report:      JSON/CSV/Markdown 交付物
- cli:         本地 analyze 命令
"""

__all__ = ["model", "io_utils", "periodogram", "window", "holdout", "report", "cli"]
