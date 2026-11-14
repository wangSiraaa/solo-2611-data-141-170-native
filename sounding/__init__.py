"""单航段测深归位工具。

坐标约定（见 fixtures/config.json 与 README）：

* 船体系 body-FRD：x 向前(forward)、y 向右(starboard)、z 向下(down)；
* 世界系 world-NED：north / east / down，向下为正，相对固定基准面；
* 姿态四元数 (w, x, y, z) 表示 body-FRD -> world-NED 的旋转；
* 潮位 tide_m 向上为正，瞬时水面的 NED down 坐标为 ``-tide_m``；
* 声速层深度自瞬时水面向下计，层内声速分段常数。

模块划分：

* :mod:`sounding.config`        -- 配置加载与校验
* :mod:`sounding.io_data`       -- CSV 输入读取
* :mod:`sounding.interpolation` -- 时间包围、线性插值、单位四元数短弧插值
* :mod:`sounding.geometry`      -- 姿态旋转、杆臂、波束初始方向
* :mod:`sounding.raytrace`      -- 水平分层 Snell 射线追踪
* :mod:`sounding.pipeline`      -- 单波束归位与整批流水线、结果落盘
"""

from .config import Config, load_config
from .pipeline import run_pipeline

__all__ = ["Config", "load_config", "run_pipeline"]
__version__ = "1.0.0"
