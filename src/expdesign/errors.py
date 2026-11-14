"""异常类型与退出码。

设计上把三种失败严格分开：

* :class:`ConfigError` —— 输入自相矛盾或出现未知模型项，属于"先拒绝"的配置错误；
* :class:`InfeasibleError` —— 可用数学证据（秩/枚举/计数）证明不存在可行设计；
* :class:`SearchFailure` —— 预算与结构上可能可行，但有限搜索未找到方案，
  不能当作不可行证明。
"""

from __future__ import annotations


class ExpDesignError(Exception):
    """本包所有自定义异常的基类。"""


class ConfigError(ExpDesignError):
    """输入声明非法或互相矛盾（未知项、坏水平、锁定运行冲突等）。"""


class InfeasibleError(ExpDesignError):
    """存在结构性证据表明无可行设计。

    ``reason`` 取 :data:`REASONS` 中的键，``evidence`` 存放可复核的证据字典，
    例如秩、零空间向量、枚举计数等。
    """

    REASON_STRUCTURAL_ALIAS = "structural_alias"
    REASON_BUDGET = "budget_insufficient"
    REASON_LOCKED = "locked_runs_conflict"

    def __init__(self, reason: str, message: str, evidence: dict | None = None):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.evidence = evidence or {}


class SearchFailure(ExpDesignError):
    """预算/结构未被证伪，但启发式搜索在限定预算内未找到可行方案。"""


#: CLI 退出码约定。
EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_INFEASIBLE = 3
EXIT_SEARCH_FAILURE = 4
