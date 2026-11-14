"""输入校验失败等用户可见错误。"""


class CompareError(ValueError):
    """所有可预期的输入/拟合错误的基类，消息面向最终用户。"""
