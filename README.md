# 扫频录音的房间脉冲响应与可靠衰减测量

本目录是题目输入材料，未包含答案实现。所有业务与观测数据均为本地合成数据。

## 文件

- `fixtures/sweep.wav`
- `fixtures/recording.wav`
- `fixtures/reference_delay.wav`
- `fixtures/plan.json`
- `cases/noisy_recording.wav`

以题面规定的字段语义与边界为准。cases/ 中的文件只替换说明中指明的输入，其他基础输入保持不变。小材料用于调试与核验，不能通过写死文件名、行号或样例答案完成任务。

运行环境：Python 3.11。可用 `python -m pip install -r requirements.txt` 安装普通依赖；不需要 Docker。

## 交付实现

本仓库实现了离线测量工具 `sweeproom`（Python 3.11 + NumPy/SciPy），只处理题面所述
单一情形（16 kHz、PCM16、单声道、一次已知对数扫频、不接麦克风）。

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m sweeproom analyze --out output
.venv/bin/python -m sweeproom analyze --recording cases/noisy_recording.wav --out output/noisy
.venv/bin/python -m sweeproom verify  --out output/verify
```

方法、阈值与判定规则见 `docs/METHOD.md`；数值产物写入 `output/`（已忽略，不入库）。

