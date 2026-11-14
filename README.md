# 饱和 FCS 材料

本目录仅含题目输入，不含待实现程序。全部材料为本地合成，无真实个人或仪器数据。Python 3.11；若使用列出的库，可执行 `python -m pip install -r requirements.txt`。无需 Docker、账号或网络数据下载。

events.fcs 是 FCS 3.1、little-endian float32、list mode、四通道。通道由 $PnN 匹配，不按数组位置猜测。标记真值非负，plan 的矩阵行是 detector、列是 marker，y=Sx；不要转置。

检测上限是100，达到上限的通道只有下界99.5，不能仍添加100.5的上界。其余观测采用±0.5区间，负读数允许存储但可能与非负物理模型不相容。门是闭区间约束，null 表示无界，子门继承父门。群比例分母为全部可行事件，避免另引入不确定父门分母；不可行事件单独报告。
