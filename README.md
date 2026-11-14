# 四面体通量材料

本目录仅含题目输入，不含待实现程序。全部材料为本地合成，无真实个人或仪器数据。Python 3.11；若使用列出的库，可执行 `python -m pip install -r requirements.txt`。无需 Docker、账号或网络数据下载。

mesh.vtu 为真实 VTK UnstructuredGrid：线性 tetra、节点 flux 向量、单元 material_id。节点下标是拓扑身份；不以坐标接近擅自合并节点。不同的顶点排列本身不等于坏几何，外向由该面的对侧顶点确定。

材料间公共面需分别以各材料外向记录，同材料内部面不是其边界。三个 negative 文件分别用于重复单元、零体积、非流形诊断；每个单独运行。坐标单位 m，场为给定线性节点场，不需要解 PDE。
