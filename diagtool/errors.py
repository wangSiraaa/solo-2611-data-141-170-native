"""工具内统一的错误类型。"""


class DiagError(Exception):
    """诊断工具基础错误。"""


class UnsupportedSyntaxError(DiagError):
    """遇到工具明确不支持的 SQL 语法。未知语法一律拒绝而非猜测。"""


class PlanError(DiagError):
    """plan.json / allowed_changes.json 声明与查询或数据库不一致。"""
