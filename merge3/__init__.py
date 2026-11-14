"""merge3 — 固定两表（customers / orders）的一次离线三方合并工具。

三个入口：
    python -m merge3 plan    识别 base/left/right 的真实变更，产出计划
    python -m merge3 resolve 套用用户裁决文件，全量复查约束，要么全成要么全拒
    python -m merge3 export  导出完整 SQLite 与逐变更来源，并重新打开校验

只依赖 Python 3.11 标准库。
"""

__version__ = "1.0.0"
