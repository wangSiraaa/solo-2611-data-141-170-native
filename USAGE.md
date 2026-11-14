# 单航段测深归位工具

把多波束声呐的波束往返时间归位为世界 NED 测点。水平分层常速声速剖面、
body-FRD → world-NED 姿态四元数、固定杆臂、潮位与时间插值均在工具内处理；
**不**估计声速剖面、不重建海底曲面、不做完整运动补偿或实时导航。

## 坐标与符号约定

| 项目 | 约定 |
| --- | --- |
| 船体系 | FRD：x 向前、y 向右、z 向下 |
| 世界系 | NED：north/east/down，down 为正，相对固定基准面 |
| 姿态 | 单位四元数 `(w,x,y,z)`，body-FRD → world-NED；`q` 与 `-q` 同一姿态 |
| 潮位 | `tide_m` 向上为正；瞬时水面 NED down = `-tide_m` |
| 声速层 | 深度自**瞬时水面**向下计，层内分段常数、层内无梯度 |
| 波束方向（船体系） | `[0, sin(angle), cos(angle)]`，angle 为弧度 |
| 单程时间 | 往返时间 / 2；发射期间换能器静止，路径可逆 |

## 处理流程

1. **时间支持**：对每个波束时刻分别定位导航位置、姿态、潮位。
   - 位置、潮位线性插值；姿态用单位四元数**短弧 slerp**
     （先把右端四元数翻到与左端同一半球，杜绝线性混成零四元数）。
   - 精确命中采样时刻直接取该记录；只有被两条记录包围且间隔不超过
     `max_interpolation_gap_ms` 才插值；缺口超门槛或超出时间范围分别报告
     `gap_too_large` / `out_of_range`，**不回退最近邻**。
2. **姿态几何（射线之前）**：四元数转旋转矩阵，杆臂随姿态旋转得到换能器
   NED 位置；波束初始方向在船体系生成后旋到世界系。世界系 down 分量
   非正的波束判为非向下波束。
3. **瞬时水面与分层深度**：水面 down = `-tide`，换能器相对水面深度
   `z_xducer - z_surface`；声速层深度以该水面为零。
4. **Snell 逐层追踪**：层内直线，`sin(θ)/c = p` 全程守恒，水平方位不变；
   射线可在层内结束或跨多层。达到临界传播（`sin θ ≥ 1`）或剩余时间超出
   已知声速深度时返回**未确定**状态并保留已走路径段，不虚构速度。
5. **测点坐标**：`N/E = 换能器 N/E + 各段水平偏移累加`，
   `down = 水面 down + 终止深度`。

## 安装与运行

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m sounding \
    --config fixtures/config.json \
    --beams fixtures/beams.csv \
    --navigation fixtures/navigation.csv \
    --tide fixtures/tide.csv \
    --sound-speed fixtures/sound_speed.csv \
    --output-dir output
```

输出：

- `output/soundings.csv`：每波束一行的汇总（状态、NED、水面下深度、段数、原因）；
- `output/soundings_detail.json`：全量支持证据——三类插值的包围下标/时间/
  间隔/比例、四元数、船体与世界波束方向、换能器位置、水面与起点深度、
  Snell 常量、逐段（层号、角度、进出深度、各类长度、段时间、水平分量）
  以及完整失败原因。

## 独立复核

复核脚本不导入本包、不重新归位，只读 JSON 输出，按段长度/声速重算总时间、
核对界面 Snell 折射与完整坐标链：

```sh
.venv/bin/python verify_soundings.py output/soundings_detail.json
```

## 测试

```sh
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ -q
```

覆盖：常速竖直/斜束解析核对、分层折射与时间守恒、临界角与超剖面未确定、
倾斜姿态、四元数整体取反等价、时间缺口/越界、零/负时间、非向下波束、
非法层界、非单位四元数与配置校验。

## 失败/未确定原因码

| 原因码 | 含义 |
| --- | --- |
| `negative_two_way_time` / `zero_two_way_time` | 往返时间为负 / 为零 |
| `position_unavailable:*` / `attitude_unavailable:*` / `tide_unavailable:*` | 时间包围越界、缺口超门槛或姿态非法 |
| `non_downward_beam` | 波束世界方向 down 分量 ≤ 0 |
| `transducer_above_surface` | 换能器在瞬时水面之上 |
| `sound_speed_layers_invalid` | 层界/声速非法（厚度非正、层间不连续等） |
| `start_below_known_profile_undetermined` | 起点已超出已知声速深度 |
| `critical_propagation_undetermined` | 界面处达到临界传播（全反射/掠射） |
| `beyond_known_sound_speed_undetermined` | 剩余时间无更深层声速可用 |
