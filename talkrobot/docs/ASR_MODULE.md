# ASR Module 说明文档

## 概述

ASRModule（Automatic Speech Recognition Module）是语音识别模块，负责将音频数据转换为文本。该模块基于 FunASR 框架，使用 SenseVoice 模型进行高精度的语音识别。

## 模块信息

- **文件路径**: `talkrobot/modules/asr_module.py`
- **依赖库**: `funasr`, `numpy`, `loguru`
- **默认模型**: `iic/SenseVoiceSmall`

## 类定义

### `ASRModule`

语音识别模块的主类。

## 构造方法

### `__init__(model_name: str, device: str = "cuda")`

初始化 ASR 模块，加载指定的语音识别模型。

**参数**:
- `model_name` (str): 模型名称，推荐使用 `iic/SenseVoiceSmall`
- `device` (str, 可选): 运行设备，可选 `"cuda"` 或 `"cpu"`，默认 `"cuda"`

**示例**:
```python
from talkrobot.modules.asr_module import ASRModule

# 使用 GPU
asr = ASRModule(model_name="iic/SenseVoiceSmall", device="cuda")

# 使用 CPU
asr = ASRModule(model_name="iic/SenseVoiceSmall", device="cpu")
```

**注意事项**:
- 首次运行会自动下载模型文件（约 200MB）
- 使用 GPU 可显著提升识别速度
- 模型加载通常需要 5-10 秒

## 公共方法

### `transcribe(audio_data: np.ndarray) -> str`

将音频数据转换为文本。

**参数**:
- `audio_data` (np.ndarray): 音频数据，numpy 数组格式
  - 采样率应为 16000 Hz
  - 数据类型为 float32
  - 取值范围 [-1.0, 1.0]
  - 可以是一维或二维数组（自动展平）

**返回值**:
- `str`: 识别出的文本内容
  - 成功时返回识别的文本
  - 失败或音频过短时返回空字符串 `""`

**示例**:
```python
import numpy as np
import soundfile as sf

# 从文件读取音频
audio_data, sample_rate = sf.read("test.wav")

# 执行识别
text = asr.transcribe(audio_data)
print(f"识别结果: {text}")
```

**异常处理**:
- 方法内部捕获所有异常，不会抛出
- 遇到错误时记录日志并返回空字符串
- 音频长度少于 0.1 秒（1600 个采样点）时自动跳过

## 配置参数

在 `talkrobot/config.py` 中可配置：

```python
class Config:
    # ASR 配置
    ASR_MODEL = "iic/SenseVoiceSmall"  # 模型名称
    ASR_DEVICE = "cuda"                 # 运行设备
```

## 识别特性

### 支持的功能

1. **多语言识别**: 自动检测语言（中文、英文等）
2. **逆文本规范化（ITN）**: 自动将数字、日期等转换为规范格式
   - 例如："二零二六年三月三日" → "2026年3月3日"
3. **标点符号**: 自动添加标点符号
4. **噪音抑制**: 对环境噪音有一定的鲁棒性

### 音频要求

- **采样率**: 16000 Hz（推荐）
- **声道数**: 单声道（mono）
- **位深度**: 16-bit 或 32-bit float
- **最短时长**: 0.1 秒（1600 个采样点）
- **推荐时长**: 0.5 - 30 秒

## 性能指标

### 识别速度（参考）

| 设备 | 音频时长 | 处理时间 | RTF* |
|------|---------|----------|------|
| NVIDIA RTX 3090 | 5 秒 | ~0.3 秒 | 0.06 |
| NVIDIA GTX 1080 Ti | 5 秒 | ~0.8 秒 | 0.16 |
| Intel i7-10700K (CPU) | 5 秒 | ~3.5 秒 | 0.70 |

_*RTF (Real-Time Factor): 处理时长 / 音频时长，越小越好_

### 识别准确率

- **安静环境**: >95%
- **一般噪音**: >90%
- **强噪音**: >80%

## 使用示例

### 基本使用

```python
from talkrobot.modules.asr_module import ASRModule
from talkrobot.config import Config
import numpy as np

# 初始化模块
asr = ASRModule(
    model_name=Config.ASR_MODEL,
    device=Config.ASR_DEVICE
)

# 创建测试音频（实际使用时从麦克风或文件读取）
audio = np.random.randn(16000 * 2).astype(np.float32) * 0.1

# 执行识别
text = asr.transcribe(audio)
print(f"识别结果: {text}")
```

### 从麦克风录音并识别

```python
import sounddevice as sd
from talkrobot.modules.asr_module import ASRModule

# 初始化 ASR
asr = ASRModule(model_name="iic/SenseVoiceSmall", device="cuda")

# 录音参数
duration = 5  # 秒
sample_rate = 16000

print("开始录音...")
audio_data = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype=np.float32
)
sd.wait()
print("录音结束")

# 识别
text = asr.transcribe(audio_data)
print(f"识别结果: {text}")
```

### 批量处理音频文件

```python
import soundfile as sf
import os
from talkrobot.modules.asr_module import ASRModule

# 初始化 ASR
asr = ASRModule(model_name="iic/SenseVoiceSmall", device="cuda")

# 批量处理
audio_dir = "audio_files/"
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):
        filepath = os.path.join(audio_dir, filename)
        audio_data, sr = sf.read(filepath)
        
        # 重采样到 16kHz（如果需要）
        if sr != 16000:
            from scipy import signal
            audio_data = signal.resample(
                audio_data,
                int(len(audio_data) * 16000 / sr)
            )
        
        text = asr.transcribe(audio_data)
        print(f"{filename}: {text}")
```

### 错误处理

```python
from talkrobot.modules.asr_module import ASRModule
import numpy as np

asr = ASRModule(model_name="iic/SenseVoiceSmall", device="cuda")

# 音频数据验证
audio_data = np.random.randn(16000 * 2).astype(np.float32)

# 检查音频长度
if len(audio_data) < 1600:
    print("音频过短，无法识别")
else:
    text = asr.transcribe(audio_data)
    if text:
        print(f"识别成功: {text}")
    else:
        print("识别失败或无有效语音")
```

## 常见问题

### Q1: 为什么返回空字符串？

**可能原因**:
1. 音频时长过短（< 0.1 秒）
2. 音频中没有检测到人声
3. 音量太小或音频质量差
4. 模型加载失败或识别异常

**解决方法**:
- 确保音频时长至少 0.5 秒
- 检查音频质量和音量
- 查看日志输出的错误信息

### Q2: 识别速度慢怎么办？

**优化建议**:
1. 使用 GPU 加速（`device="cuda"`）
2. 减少单次识别的音频长度
3. 确保 CUDA 和 PyTorch 正确安装
4. 关闭其他占用 GPU 的程序

### Q3: 如何提高识别准确率？

**建议**:
1. 使用质量好的麦克风
2. 在安静环境下录音
3. 说话清晰、语速适中
4. 确保音量适中（不要太小或太大）
5. 音频采样率使用 16000 Hz

### Q4: 支持哪些语言？

- 主要支持：中文、英文
- 其他语言需要测试确认
- 模型会自动检测语言

### Q5: 如何离线使用？

1. 首次联网运行，模型会自动下载
2. 模型缓存在 `~/.cache/modelscope/` 目录
3. 后续可离线使用（需要 `disable_update=True`）

## 日志输出

模块使用 `loguru` 记录日志：

```
[INFO] 正在加载ASR模型: iic/SenseVoiceSmall
[INFO] ASR模型加载完成
[INFO] 识别结果: 你好世界
[WARNING] 音频过短,跳过识别
[ERROR] ASR识别出错: CUDA out of memory
```

## 相关资源

- **FunASR 文档**: https://github.com/alibaba-damo-academy/FunASR
- **SenseVoice 模型**: https://www.modelscope.cn/models/iic/SenseVoiceSmall
- **配置文件**: `talkrobot/config.py`
- **测试代码**: `talkrobot/tests/test_asr.py`

## 更新日志

- **v1.0** (2026-03-03): 初始版本，支持基础语音识别功能
