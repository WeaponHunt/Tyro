# 测试文档

## ASR 模块测试

### 测试文件位置
- **测试代码**: `talkrobot/tests/test_asr.py`
- **模块文档**: `talkrobot/docs/ASR_MODULE.md`

### 快速运行测试

#### 方式 1: 完整测试（推荐）

运行所有 9 个测试用例，全面验证 ASR 模块功能：

```bash
cd /home/acir/Tyro
python -m talkrobot.tests.test_asr
```

测试内容包括：
1. ✅ 模块初始化
2. ✅ 基础语音识别
3. ✅ 不同长度音频
4. ✅ 不同音量音频
5. ✅ 不同数组形状
6. ✅ 空音频和边界情况
7. ✅ 性能基准测试
8. ✅ 真实音频文件识别
9. ✅ 并发识别测试

**预计耗时**: 30-60 秒（取决于 GPU/CPU 性能）

#### 方式 2: 快速测试

仅测试基础功能，适合快速验证：

```bash
cd /home/acir/Tyro
python -m talkrobot.tests.test_asr quick
```

**预计耗时**: 5-10 秒

### 测试输出示例

```
🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 
ASR 模块完整测试套件
🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 

============================================================
测试1: ASR 模块初始化
============================================================
✅ 模块初始化成功
   加载时间: 6.32 秒

============================================================
测试2: 基础语音识别（模拟音频）
============================================================
   音频时长: 2 秒
   音频采样点: 32000
   正在识别...
   处理时间: 0.234 秒
   RTF: 0.117
ℹ️  识别结果为空（模拟音频通常无法识别出内容）

...（其他测试）...

============================================================
🎉 ASR 模块测试完成
============================================================
```

### 使用真实音频测试

如果想测试真实语音识别效果：

1. 准备一个 WAV 音频文件（包含人声）
2. 将文件放到 `example/` 目录
3. 运行完整测试，第 8 项测试会自动识别该文件

```bash
# 示例
cp my_voice.wav example/
python -m talkrobot.tests.test_asr
```

### 自定义测试

你也可以在 Python 中直接导入使用：

```python
from talkrobot.tests.test_asr import test_initialization, test_basic_transcription

# 初始化模块
asr = test_initialization()

# 运行单个测试
test_basic_transcription(asr)
```

### 性能建议

- **GPU 加速**: 确保在 `talkrobot/config.py` 中设置 `ASR_DEVICE = "cuda"`
- **内存要求**: 至少 4GB RAM（GPU 2GB VRAM）
- **首次运行**: 需要下载模型（约 200MB），请确保网络通畅

### 故障排除

#### 问题 1: 模块初始化失败

```
❌ 模块初始化失败: No module named 'funasr'
```

**解决方法**:
```bash
pip install funasr
```

#### 问题 2: CUDA 相关错误

```
❌ CUDA out of memory
```

**解决方法**:
- 修改 `config.py` 使用 CPU: `ASR_DEVICE = "cpu"`
- 或释放 GPU 内存

#### 问题 3: 无法找到测试文件

```
ModuleNotFoundError: No module named 'talkrobot.tests'
```

**解决方法**:
确保在项目根目录运行，或添加到 Python 路径：
```bash
cd /home/acir/Tyro
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m talkrobot.tests.test_asr
```

### 查看详细日志

测试过程中的详细日志会保存到：
```
talkrobot/logs/robot_*.log
```

可以查看完整的调试信息。

## 其他模块测试

项目中还有其他模块的测试：

```bash
# AudioRecorder 测试
python -m talkrobot.tests.test_audio_recorder

# TTS 测试
python -m talkrobot.tests.test_tts

# LLM 测试
python -m talkrobot.tests.test_llm

# Memory 测试
python -m talkrobot.tests.test_memory
```

---

## AudioRecorder 模块测试

### 测试文件位置
- **测试代码**: `talkrobot/tests/test_audio_recorder.py`
- **模块文档**: `talkrobot/docs/AUDIO_RECORDER.md`

### 快速运行测试

#### 方式 1: 完整测试（推荐，包含交互式测试）

运行所有 11 个测试用例：

```bash
cd /home/acir/Tyro
python -m talkrobot.tests.test_audio_recorder
```

测试内容包括：
1. ✅ Push 模式初始化
2. ✅ Continuous 模式初始化（含 VAD 加载）
3. ✅ 参数配置验证
4. ✅ 音频回调模拟
5. ✅ TTS 播放标志测试
6. ✅ 音频处理标志测试
7. ✅ VAD 状态栏显示
8. ✅ 真实麦克风测试（Push 模式，交互式）
9. ✅ 真实麦克风测试（Continuous 模式，交互式）
10. ✅ 并发回调测试
11. ✅ 内存管理测试

**预计耗时**: 20-40 秒（不含交互式测试）

#### 方式 2: 快速测试

仅测试基础功能，跳过交互式测试：

```bash
cd /home/acir/Tyro
python -m talkrobot.tests.test_audio_recorder quick
```

**预计耗时**: 5-10 秒

### 测试输出示例

```
🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  
AudioRecorder 模块完整测试套件
🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  🎙️  

============================================================
测试1: Push 模式初始化
============================================================
✅ Push 模式初始化成功
   采样率: 16000 Hz
   声道数: 1
   监听模式: push

============================================================
测试2: Continuous 模式初始化（含 VAD 模型加载）
============================================================
✅ Continuous 模式初始化成功
   采样率: 16000 Hz
   初始化耗时: 3.21s
   VAD 模型: 已加载

...（其他测试）...
```

### 交互式测试说明

测试 8 和 9 需要真实麦克风：

**测试 8 - Push 模式**:
- 按住 Q 键说话
- 松开 Q 键结束
- 测试录音功能

**测试 9 - Continuous 模式**:
- 直接说话即可
- 系统自动检测语音开始/结束
- 测试 VAD 自动检测

可以选择跳过这些测试（输入 `n`）。

### 依赖要求

**基础依赖**（Push 模式）:
```bash
pip install sounddevice numpy pynput loguru
```

**完整依赖**（Continuous 模式）:
```bash
pip install sounddevice numpy pynput loguru torch silero-vad
```

**可选依赖**（内存测试）:
```bash
pip install psutil
```

### 故障排除

#### 问题 1: 麦克风不可用

```
❌ 测试失败: [Errno -9996] Invalid input device
```

**解决方法**:
```bash
# 查看可用音频设备
python -c "import sounddevice; print(sounddevice.query_devices())"

# 检查麦克风权限（Linux）
ls -l /dev/snd/
```

#### 问题 2: VAD 模型加载失败

```
⚠️  需要安装依赖: No module named 'torch'
```

**解决方法**:
```bash
pip install torch silero-vad
```

#### 问题 3: pynput 权限问题（Linux）

```
❌ 测试失败: You must be root
```

**解决方法**:
```bash
sudo usermod -a -G input $USER
# 重新登录后生效
```

### 性能基准

完整测试会验证以下性能指标：

- **初始化时间**: < 5 秒（含 VAD 加载）
- **音频回调延迟**: < 50ms
- **并发处理能力**: 支持多线程回调
- **内存占用增长**: < 50MB（1000 次回调）

## 贡献测试用例

欢迎贡献更多测试用例！请确保：
1. 测试代码有清晰的文档字符串
2. 包含适当的错误处理
3. 输出易于理解的测试结果
4. 遵循项目的代码风格
