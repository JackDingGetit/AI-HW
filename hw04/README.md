# hw04 - 语音识别与处理实验

## 📁 目录结构

```
hw04/
├── README.md                 # 项目说明文档
├── 4月18日.mp4               # 音频文件
├── jianying.md               # 剪映相关文档
├── text_gen.md               # 文本生成相关文档
└── work3/                    # 语音识别实验目录
    ├── asr_report.md         # ASR方案对比分析报告
    ├── asr_vosk_realtime.py  # 实时语音识别代码
    ├── experiment_record.md  # 实验记录文档
    ├── requirements.txt      # Python依赖文件
    ├── models/               # Vosk模型目录
    │   └── vosk-model-small-cn-0.22/  # 中文语音识别模型
    └── __pycache__/          # Python编译缓存
```

## 📋 任务说明

### Task 1: 语音识别方案调研
- **文件**: `work3/asr_report.md`
- **内容**: 对比分析 OpenAI Whisper、Vosk、FunASR 三种开源语音识别方案

### Task 2: 麦克风实时识别实现
- **文件**: `work3/asr_vosk_realtime.py`
- **内容**: 基于 Vosk 的麦克风实时语音识别程序

### Task 3: 实验记录
- **文件**: `work3/experiment_record.md`
- **内容**: 测试样例、延迟观察、错误率分析等实验记录

## 🚀 运行方式

### 环境要求
- Python 3.8+
- Windows/macOS/Linux

### 安装依赖
```bash
cd work3
pip install -r requirements.txt
```

### 运行实时语音识别
```bash
cd work3
python asr_vosk_realtime.py
```

## 📝 提交物清单

| 文件 | 说明 |
|------|------|
| `work3/asr_report.md` | ASR方案对比分析报告 |
| `work3/asr_vosk_realtime.py` | 可复现代码 |
| `work3/requirements.txt` | 依赖说明 |
| `work3/experiment_record.md` | 实验记录 |

## 🛠️ 技术栈

- **语音识别**: Vosk (vosk-model-small-cn-0.22)
- **音频处理**: PyAudio
- **开发语言**: Python 3.x
