"""有代价约束的可读分流规则学习工具（精确求解，标准库实现）。

公开入口：
- :func:`fit`    ：从标记案例学习有序规则、例外清单与探针歧义报告
- :func:`apply`  ：从保存的可检查规则 JSON 重新分流
- :func:`verify` ：独立重放核验 + 自制穷举基准
"""

from .api import fit, apply, verify

__all__ = ["fit", "apply", "verify"]
