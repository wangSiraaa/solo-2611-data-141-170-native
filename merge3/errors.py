"""合并工具的自定义异常。"""


class Merge3Error(Exception):
    """所有可预期的合并错误基类，CLI 统一捕获。"""


class InputChangedError(Merge3Error):
    """三份输入的摘要与计划/裁决绑定时的摘要不一致。"""


class DecisionError(Merge3Error):
    """裁决文件本身不合法（未知冲突、非法选项、重复裁决等）。"""


class UnresolvedConflictsError(Merge3Error):
    """仍有冲突未解决，或某个裁决引出了新的约束冲突。

    residual 字段携带仍需裁决的冲突清单（含可选项），绝不发布半成品。
    """

    def __init__(self, message, residual):
        super().__init__(message)
        self.residual = residual
