"""分片 MCMC 样本的链恢复与估计可信度离线诊断。

模块划分：
- recovery:    按 manifest 恢复链（去重 / 冲突 / 缺段 / 元信息检查、warmup 去除、共同长度截断）
- diagnostics: 秩正态化 split/folded R-hat、bulk/tail ESS、均值 MCSE（核心自行实现）
- interpret:   多链一致性与单链有效信息量联合判读规则
- report:      JSON / Markdown / 静态图报告
- cli:         analyze 与 inspect 命令
"""

__version__ = "0.1.0"
