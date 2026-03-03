# AudioRecorder 说明文档

## 概述

AudioRecorder 是音频录制核心模块，负责从麦克风采集音频并触发 ASR 识别。该模块支持两种录音模式，并集成了 Silero VAD（Voice Activity Detection）用于自动检测语音段。

## 模块信息

- **文件路径**: `talkrobot/core/audio_recorder.py`
- **依赖库**: `sounddevice`, `pynput`, `numpy`, `silero-vad`, `torch`, `loguru`
- **主要功能**: 音频采集、语音活动检测、实时状态显示

## 支持的录音模式

### 1. Push 模式（按键录音）

- 按住 **Q** 键开始录音
- 松开 **Q** 键结束录音
- 适用场景：精确控制录音时机，避免误触发

### 2. Continuous 模式（持续监听）

- 系统持续监听环境音频
- 基于 Silero VAD 自动检测语音活动
- 检测到语音后自动开始录音
- 静默超过设定时长后自动结束
- 适用场景：免提对话、自然交互

## 类定义

### `AudioRecorder`

音频录制器主类。

## 构造方法

### `__init__(...)`

初始化音频录制器。

**完整签名**:
```python
def __init__(
    self,
    sample_rate: int = 16000,
    channels: int = 1,
    listen_mode: str = "push",
    vad_check_interval: float = 0.25,
    silence_duration: float = 1.5,
    min_speech_duration: float = 0.3,
)
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `sample_rate` | int | 16000 | 音频采样率（Hz），推荐 16000 |
| `channels` | int | 1 | 声道数，1=单声道，2=立体声 |
| `listen_mode` | str | "push" | 录音模式："push" 或 "continuous" |
| `vad_check_interval` | float | 0.25 | VAD 检测间隔（秒），continuous 模式有效 |
| `silence_duration` | float | 1.5 | 静默多久后视为说话结束（秒） |
| `min_speech_duration` | float | 0.3 | 最短语音时长（秒），过短丢弃 |

**示例**:
```python
from talkrobot.core.audio_recorder import AudioRecorder

# Push 模式
recorder = AudioRecorder(
    sample_rate=16000,
    channels=1,
    listen_mode="push"
)

# Continuous 模式（自定义参数）
recorder = AudioRecorder(
    sample_rate=16000,
    channels=1,
    listen_mode="continuous",
    vad_check_interval=0.2,      # 更频繁的检测
    silence_duration=2.0,         # 更长的等待时间
    min_speech_duration=0.5       # 过滤更短的语音
)
```

## 公共方法

### `start(on_audio_complete: Callable)`

启动录音器，开始监听音频输入。

**参数**:
- `on_audio_complete` (Callable): 录音完成时的回调函数
  - 签名: `def callback(audio_data: np.ndarray) -> None`
  - `audio_data`: numpy 数组，形状为 `(N, 1)` 或 `(N,)`，类型为 `float32`

**行为**:
- 打开音频输入流
- 根据模式启动相应的监听机制
- 阻塞当前线程直到用户退出（Ctrl+C）

**示例**:
```python
import numpy as np
from talkrobot.core.audio_recorder import AudioRecorder

def handle_audio(audio_data: np.ndarray):
    """处理录制的音频"""
    print(f"收到音频: {len(audio_data)} 个采样点")
    duration = len(audio_data) / 16000
    print(f"时长: {duration:.2f} 秒")
    # 这里可以调用 ASR 模块进行识别

recorder = AudioRecorder(listen_mode="push")
recorder.start(on_audio_complete=handle_audio)
```

### `notify_process_done()`

通知录音器音频处理已完成，可以继续采集下一段语音。

**用途**:
- Continuous 模式下，防止在处理上一段音频时采集新的音频
- 通常在音频处理完成后由 ConversationManager 调用

**示例**:
```python
def handle_audio_async(audio_data: np.ndarray):
    """异步处理音频"""
    import threading
    
    def process():
        # 模拟处理
        time.sleep(2)
        # 处理完成后通知
        recorder.notify_process_done()
    
    thread = threading.Thread(target=process)
    thread.start()

recorder = AudioRecorder(listen_mode="continuous")
# recorder.start(...) 会自动处理
```

## 公共属性

### `is_tts_playing` (bool)

标志 TTS 是否正在播放。

**用途**:
- 用于 Continuous 模式下屏蔽 TTS 自身的语音
- 设置为 `True` 时，录音器会跳过音频采集
- TTS 播放完成后应设置回 `False`

**示例**:
```python
recorder = AudioRecorder(listen_mode="continuous")

# TTS 开始播放前
recorder.is_tts_playing = True

# 播放 TTS...
tts.synthesize(text, play_audio=True)

# TTS 播放完成后
recorder.is_tts_playing = False
```

## 私有方法（供参考）

### `_init_vad()`

加载 Silero VAD 模型（仅 Continuous 模式）。

### `_vad_monitor_loop()`

VAD 监控线程主循环，持续检测语音活动。

### `_print_vad_status(rms, has_speech)`

在终端原地刷新 VAD 状态行（实时显示）。

### `_clear_vad_status()`

清除状态行，避免与正常输出混淆。

### `audio_callback(indata, frames, time, status)`

sounddevice 音频流回调函数。

### `on_press(key)` / `on_release(key)`

键盘事件处理（Push 模式）。

## 工作流程

### Push 模式流程

```
1. 用户按下 Q 键
   ↓
2. is_recording = True，开始采集音频
   ↓
3. audio_callback 将音频块写入 audio_frames
   ↓
4. 用户松开 Q 键
   ↓
5. is_recording = False，停止采集
   ↓
6. 合并所有音频块
   ↓
7. 调用 on_audio_complete(audio_data)
```

### Continuous 模式流程

```
1. 音频流持续运行，audio_callback 写入缓冲区
   ↓
2. VAD 监控线程每 vad_check_interval 秒检测一次
   ↓
3. 检测到语音 (has_speech=True)
   ↓
4. is_recording = True，开始积累音频
   ↓
5. 持续检测，更新 _last_voice_time
   ↓
6. 静默超过 silence_duration 秒
   ↓
7. 检查语音时长 >= min_speech_duration
   ↓
8. is_recording = False，合并音频
   ↓
9. _processing = True，调用 on_audio_complete(audio_data)
   ↓
10. 等待 notify_process_done()，_processing = False
```

## VAD 状态栏

Continuous 模式下会实时显示 VAD 状态：

```
🎙️  监听中    |░░░░░░░░░░░░░░░░░░░░| RMS:0.0012
🔴 录音中 [2.3s] 🗣️ |████████░░░░░░░░░░░░| RMS:0.0423
⏳ 处理中    |░░░░░░░░░░░░░░░░░░░░| RMS:0.0000
🔊 播放中    |░░░░░░░░░░░░░░░░░░░░| RMS:0.0000
```

**状态说明**:
- **🎙️ 监听中**: 等待语音输入
- **🔴 录音中**: 检测到语音，正在录音
- **⏳ 处理中**: 音频处理中（ASR/LLM/Memory）
- **🔊 播放中**: TTS 播放中
- **🗣️**: 当前帧检测到语音
- **音量条**: 实时音量可视化（基于 RMS）
- **RMS**: Root Mean Square 音量值

## 配置参数

在 `talkrobot/config.py` 中可配置：

```python
class Config:
    # 音频配置
    SAMPLE_RATE = 16000
    CHANNELS = 1
    
    # 监听模式
    DEFAULT_LISTEN_MODE = "push"  # 或 "continuous"
    
    # Continuous 模式 VAD 配置
    VAD_CHECK_INTERVAL = 0.25      # 检测间隔（秒）
    VAD_SILENCE_DURATION = 1.5     # 静默阈值（秒）
    VAD_MIN_SPEECH_DURATION = 0.3  # 最短语音（秒）
```

## 使用示例

### 示例 1: 基础 Push 模式

```python
from talkrobot.core.audio_recorder import AudioRecorder

def on_audio(audio_data):
    print(f"录到 {len(audio_data)} 个采样点")

recorder = AudioRecorder(
    sample_rate=16000,
    listen_mode="push"
)

print("按住 Q 键说话，松开结束")
print("按 Ctrl+C 退出")

try:
    recorder.start(on_audio_complete=on_audio)
except KeyboardInterrupt:
    print("退出")
```

### 示例 2: Continuous 模式 + ASR

```python
from talkrobot.core.audio_recorder import AudioRecorder
from talkrobot.modules.asr_module import ASRModule

# 初始化 ASR
asr = ASRModule(model_name="iic/SenseVoiceSmall", device="cuda")

def on_audio(audio_data):
    """录音完成后进行识别"""
    text = asr.transcribe(audio_data)
    if text:
        print(f"识别结果: {text}")
    else:
        print("识别失败")

# 初始化录音器
recorder = AudioRecorder(
    sample_rate=16000,
    listen_mode="continuous",
    silence_duration=2.0,  # 2秒静默后结束
    min_speech_duration=0.5  # 至少0.5秒
)

print("直接说话即可，系统自动检测")
print("停顿2秒后自动结束")

try:
    recorder.start(on_audio_complete=on_audio)
except KeyboardInterrupt:
    print("退出")
```

### 示例 3: 完整对话系统

```python
from talkrobot.core.audio_recorder import AudioRecorder
from talkrobot.modules.asr_module import ASRModule
from talkrobot.modules.tts_module import TTSModule
from talkrobot.modules.llm_module import LLMModule
import threading

# 初始化模块
asr = ASRModule(model_name="iic/SenseVoiceSmall", device="cuda")
tts = TTSModule(lang_code='z', voice='zf_xiaoyi')
llm = LLMModule(api_key="...", model="qwen-plus")
recorder = AudioRecorder(listen_mode="continuous")

def process_audio(audio_data):
    """完整对话流程"""
    # 1. ASR
    user_text = asr.transcribe(audio_data)
    if not user_text:
        return
    print(f"用户: {user_text}")
    
    # 2. LLM
    response = llm.generate_response(user_text, context=[])
    print(f"机器人: {response}")
    
    # 3. TTS（屏蔽麦克风）
    recorder.is_tts_playing = True
    try:
        tts.synthesize(response, play_audio=True)
    finally:
        recorder.is_tts_playing = False
    
    # 4. 通知完成
    recorder.notify_process_done()

def async_process(audio_data):
    """异步处理避免阻塞录音"""
    thread = threading.Thread(target=process_audio, args=(audio_data,))
    thread.daemon = True
    thread.start()

try:
    recorder.start(on_audio_complete=async_process)
except KeyboardInterrupt:
    print("退出")
```

### 示例 4: 保存录音到文件

```python
import soundfile as sf
from talkrobot.core.audio_recorder import AudioRecorder

recording_count = 0

def save_audio(audio_data):
    """保存录音到文件"""
    global recording_count
    recording_count += 1
    filename = f"recording_{recording_count}.wav"
    sf.write(filename, audio_data, 16000)
    print(f"已保存: {filename}")

recorder = AudioRecorder(listen_mode="push")
print("按 Q 键录音，每次录音自动保存")

try:
    recorder.start(on_audio_complete=save_audio)
except KeyboardInterrupt:
    print(f"共录制 {recording_count} 段音频")
```

### 示例 5: 实时音量监控

```python
import numpy as np
from talkrobot.core.audio_recorder import AudioRecorder

def analyze_audio(audio_data):
    """分析音频特征"""
    # 计算 RMS
    rms = float(np.sqrt(np.mean(audio_data ** 2)))
    
    # 计算峰值
    peak = float(np.max(np.abs(audio_data)))
    
    # 计算时长
    duration = len(audio_data) / 16000
    
    print(f"时长: {duration:.2f}s, RMS: {rms:.4f}, 峰值: {peak:.4f}")

recorder = AudioRecorder(listen_mode="push")
recorder.start(on_audio_complete=analyze_audio)
```

## 高级特性

### 1. TTS 播放期间屏蔽麦克风

防止 TTS 语音被误识别为用户输入：

```python
# ConversationManager 中的实现
if self.audio_recorder:
    self.audio_recorder.is_tts_playing = True
try:
    self.tts.synthesize(response, play_audio=True)
finally:
    self.audio_recorder.is_tts_playing = False
```

### 2. 防止并发处理

使用 `_processing` 标志避免同时处理多段音频：

```python
# 设置处理中标志
recorder._processing = True

# 处理音频...
process_audio(audio_data)

# 处理完成后通知
recorder.notify_process_done()  # 内部会设置 _processing = False
```

### 3. 实时状态栏

Continuous 模式下自动显示：
- 使用 `\r` 回到行首实现原地刷新
- 通过 `_clear_vad_status()` 避免与普通输出混淆
- 在重要事件（录音开始/结束）时清除状态栏

### 4. VAD 参数调优

根据使用场景调整参数：

**快速响应**（适合对话）:
```python
AudioRecorder(
    vad_check_interval=0.1,    # 更快检测
    silence_duration=0.8,      # 更快结束
    min_speech_duration=0.2    # 允许更短语音
)
```

**稳定识别**（适合长句）:
```python
AudioRecorder(
    vad_check_interval=0.3,    # 降低 CPU 占用
    silence_duration=2.5,      # 更多思考时间
    min_speech_duration=0.5    # 过滤干扰
)
```

## 性能指标

### CPU 占用

| 模式 | 检测间隔 | CPU 占用 | 说明 |
|------|---------|----------|------|
| Push | N/A | < 1% | 几乎无占用 |
| Continuous | 0.25s | 3-5% | VAD 检测 |
| Continuous | 0.1s | 8-12% | 更频繁检测 |

### 延迟分析

| 阶段 | 延迟 | 说明 |
|------|------|------|
| 语音开始检测 | 0.1-0.3s | 取决于 vad_check_interval |
| 语音结束检测 | 1.5-2.5s | silence_duration + check_interval |
| 音频回调延迟 | < 0.05s | sounddevice 固有延迟 |

### 内存占用

- **基础**: 50-100 MB（VAD 模型）
- **录音中**: +10 MB / 10秒音频

## 常见问题

### Q1: Push 模式按 Q 键无反应？

**可能原因**:
1. 终端焦点不在程序窗口
2. 权限问题（Linux 需要 sudo 或加入 input 组）
3. pynput 版本问题

**解决方法**:
```bash
# Linux 权限设置
sudo usermod -a -G input $USER
# 重新登录生效

# 检查 pynput 版本
pip show pynput
```

### Q2: Continuous 模式一直不触发录音？

**可能原因**:
1. 麦克风权限未授予
2. 麦克风音量太小
3. VAD 模型加载失败
4. 环境噪音过大

**解决方法**:
```bash
# 测试麦克风
python -c "import sounddevice as sd; print(sd.query_devices())"

# 检查 VAD 日志
# 查看 talkrobot/logs/*.log
```

### Q3: 录音结束太早或太晚？

**原因**: `silence_duration` 参数不合适

**解决方法**:
```python
# 如果结束太早，延长静默等待时间
recorder = AudioRecorder(
    silence_duration=2.5,  # 增加到 2.5 秒
    listen_mode="continuous"
)

# 如果结束太晚，缩短等待时间
recorder = AudioRecorder(
    silence_duration=1.0,  # 减少到 1.0 秒
    listen_mode="continuous"
)
```

### Q4: 为什么有时候会漏掉开头的语音？

**原因**: VAD 检测间隔导致错过开头

**解决方法**:
```python
# 减小检测间隔
recorder = AudioRecorder(
    vad_check_interval=0.1,  # 从 0.25 减少到 0.1
    listen_mode="continuous"
)
```

注意：更小的间隔会增加 CPU 占用。

### Q5: TTS 播放时还在录音是怎么回事？

**原因**: 未正确设置 `is_tts_playing` 标志

**解决方法**:
```python
# 确保使用 try/finally
recorder.is_tts_playing = True
try:
    tts.synthesize(text, play_audio=True)
finally:
    recorder.is_tts_playing = False  # 确保一定被执行
```

### Q6: 如何调试 VAD 检测问题？

**方法**: 启用 DEBUG 模式

```bash
python -m talkrobot.main --debug --listen-mode continuous
```

会输出详细的 VAD 检测日志：
```
DEBUG | VAD检测: 处理 4 个音频块
DEBUG | VAD结果: has_speech=True, rms=0.0234, is_recording=True
DEBUG | 录音中，累计静默: 0.50s
```

## 错误处理

### 麦克风不可用

```python
try:
    recorder.start(on_audio_complete=handle_audio)
except OSError as e:
    print(f"麦克风错误: {e}")
    print("请检查麦克风连接和权限")
```

### VAD 模型加载失败

```python
try:
    recorder = AudioRecorder(listen_mode="continuous")
except ImportError:
    print("未安装 silero-vad")
    print("运行: pip install silero-vad")
except Exception as e:
    print(f"VAD 初始化失败: {e}")
```

## 依赖安装

```bash
# 核心依赖
pip install sounddevice numpy pynput loguru

# Continuous 模式依赖
pip install torch silero-vad

# 可选：音频文件支持
pip install soundfile
```

## 相关文档

- **Configuration**: `talkrobot/config.py`
- **Conversation Manager**: `talkrobot/core/conversation_manager.py`
- **ASR Module**: `talkrobot/docs/ASR_MODULE.md`
- **Testing**: `talkrobot/docs/TESTING.md`

## 更新日志

- **v1.2** (2026-03-03): 添加 Debug 日志输出，增强问题诊断能力
- **v1.1** (2026-03-03): 修复 TTS 播放期间状态栏问题
- **v1.0** (2026-03-01): 初始版本，支持 Push 和 Continuous 两种模式
