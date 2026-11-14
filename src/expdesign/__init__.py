"""实验运行设计工具包。

模块划分见各文件 docstring；常用入口：

* :func:`expdesign.design.load_bundle` / :func:`expdesign.design.run_design`
  端到端装配与搜索；
* :class:`expdesign.errors.ConfigError` / :class:`InfeasibleError` /
  :class:`SearchFailure` 三类严格区分的失败。
"""

from .errors import ConfigError, InfeasibleError, SearchFailure

__all__ = ["ConfigError", "InfeasibleError", "SearchFailure"]
__version__ = "1.0.0"
