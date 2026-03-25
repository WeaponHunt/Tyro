# TalkRobot - 智能对话机器人

基于ASR、TTS、LLM和Memory构建的模块化对话机器人系统。

## 功能特性

- 🎤 **语音输入**: 支持按键模式（按住Q键录音）和持续监听模式（自动检测语音）
- ⌨️ **文本输入**: 支持 `--no-asr` 模式，直接在终端键盘输入对话
- 🔊 **语音输出**: 自动将回复转为语音播放（可通过 `--no-tts` 关闭）
- 🧠 **智能对话**: 基于Qwen大模型的自然对话
- 💾 **长期记忆**: 自动记录对话历史并智能检索
- 🪟 **滑动窗口记忆**: 可配置最近 n 轮对话上下文，与检索记忆一起提供给大模型
- 📝 **手动记忆**: 支持手动为指定用户添加记忆
- 👥 **多用户隔离**: 不同用户的记忆数据独立存储
- 🧑‍🤝‍🧑 **人脸驱动交互对象**: 可按当前识别人脸自动切换交互用户

## 项目结构

```
talkrobot/
├── config.py              # 配置管理
├── main.py                # 主程序入口
├── modules/               # 功能模块
│   ├── asr_module.py     # 语音识别
│   ├── tts_module.py     # 语音合成
│   ├── llm_module.py     # 大语言模型
│   └── memory_module.py  # 记忆管理
├── core/                  # 核心组件
│   ├── audio_recorder.py      # 音频录制
│   └── conversation_manager.py # 对话管理
└── tests/                 # 测试文件
    ├── test_asr.py
    ├── test_tts.py
    ├── test_llm.py
    └── test_memory.py
```

## 安装依赖

```bash
pip install funasr kokoro easy_tts_server openai mem0 sounddevice pynput loguru numpy soundfile silero-vad
```

## 使用方法

### 1. 启动完整系统

```bash
cd /home/acir/Tyro

# 使用默认用户启动（默认按键模式）
python -m talkrobot.main

# 指定用户启动
python -m talkrobot.main --user ljc
# 或
python -m talkrobot.main chat --user ljc

# 关闭TTS语音播放（仅显示文字回复）
python -m talkrobot.main --user ljc --no-tts
python -m talkrobot.main chat --user ljc --no-tts

# 使用 easy_tts_server 作为 TTS 后端（中文）
python -m talkrobot.main chat --tts-provider easy_tts_server --language zh

# 使用 Kokoro 作为 TTS 后端（英文）
python -m talkrobot.main chat --tts-provider kokoro --language en

# 控制大模型用英文回复
python -m talkrobot.main chat --language en

# 控制大模型用中文回复
python -m talkrobot.main chat --language zh

# 使用持续监听模式（无需按键，直接说话即可）
python -m talkrobot.main --user ljc --listen-mode continuous

# 使用按键模式（按住Q键说话，默认行为）
python -m talkrobot.main --user ljc --listen-mode push

# 使用 no-asr 模式（禁用语音识别，改为终端输入）
python -m talkrobot.main --user ljc --no-asr

# 启用流式回复（边生成边播报）
python -m talkrobot.main --user ljc --streaming

# 设置滑动窗口轮数（让模型额外看到最近 5 轮对话）
python -m talkrobot.main --user ljc --history-rounds 5

# 启用人脸识别，按当前识别人脸自动热切换交互对象
python -m talkrobot.main chat --enable-face

# 指定人脸识别摄像头
python -m talkrobot.main chat --enable-face --face-camera-index 0

# 在 continuous 非响应阶段，见到熟人主动问好
python -m talkrobot.main chat --enable-face --say-hallo --listen-mode continuous
```

### 2. 手动添加记忆

```bash
# 单条添加
python -m talkrobot.main add-memory --user ljc --content "我喜欢吃苹果"

# 交互式批量添加（不传 --content 进入交互模式，输入 q 退出）
python -m talkrobot.main add-memory --user ljc
```

### 3. 测试单个模块

```bash
# 测试ASR
python -m talkrobot.tests.test_asr

# 测试TTS
python -m talkrobot.tests.test_tts

# 测试LLM
python -m talkrobot.tests.test_llm

# 测试Memory
python -m talkrobot.tests.test_memory
```

## 配置说明

在 `config.py` 中修改以下配置:

- `ASR_DEVICE`: ASR运行设备 (cuda/cpu)
- `TTS_VOICE`: TTS音色选择
- `TTS_PROVIDER`: TTS后端选择 (`kokoro` / `easy_tts_server`)
- `LANGUAGE`: 统一语言开关 (`zh` / `en`，同时作用于TTS和LLM)
- `LLM_API_KEY`: 阿里云API密钥
- `SYSTEM_PROMPT`: 机器人人设
- `PERSONA_PROFILE_PATH`: 用户人格配置文件路径（默认 `talkrobot/persona_profiles.json`）
- `GLOBAL_SYSTEM_PROMPT`: 全局提示词（会拼接在用户人格 prompt 后）
- `ENABLE_PERSONA_AUTO_UPDATE`: 是否启用后台人格自动更新（默认开启）
- `DEFAULT_LISTEN_MODE`: 默认监听模式 ("push" / "continuous")
- `SLIDING_WINDOW_ROUNDS`: 滑动窗口历史轮数（0 表示关闭）
- `VAD_CHECK_INTERVAL`: VAD 检测间隔（秒，默认 0.25）
- `VAD_SILENCE_DURATION`: 静默多久判定说话结束（秒）
- `VAD_MIN_SPEECH_DURATION`: 最短语音时长，过短的丢弃（秒）

> 启动参数 `--streaming` 可启用流式回复生成。

### 按用户配置人格 Prompt

系统支持按用户名配置不同的人格 prompt。默认配置文件为 `talkrobot/persona_profiles.json`，示例：

```json
{
    "default": {
        "system_prompt": "你是一个友好、简洁的中文助手。"
    },
    "ljc": {
        "system_prompt": "语气更轻松口语化，回答简短直接。"
    }
}
```

生效优先级：

1. 当前用户专属配置（如 `"ljc"`）
2. `default.system_prompt`
3. `Config.SYSTEM_PROMPT`（最终兜底）

最终发送给大模型的 system prompt 按以下顺序拼接：

1. 用户人格 prompt（含 default/Config 回退）
2. `Config.GLOBAL_SYSTEM_PROMPT`
3. 表情指令 prompt（若启用表情模块）

> 对于新增用户或未手动配置的用户，系统会自动使用默认人格 prompt。

### 后台自动更新人格（LangGraph）

系统支持在单轮对话中后台触发人格更新 Agent（不阻塞当前回复生成）：

- 分支1：判断是否需要更新人格 prompt
- 分支2：生成候选新版人格 prompt

上述两个分支由 LangGraph 并行执行；若判定通过，会将更新结果写回 `talkrobot/persona_profiles.json`，并在后续轮次生效。

若环境未安装 LangGraph，则自动跳过该能力，不影响原有对话功能。

可通过启动参数临时关闭：

```bash
python -m talkrobot.main chat --disable-persona-auto-update
```

> 启动参数 `--enable-face` 启用人脸识别后，会根据当前识别对象自动切换用户；
> 若该对象不存在长期记忆数据，则自动回退为“仅滑动窗口短期记忆”模式。
> 若当前帧未检测到有效人脸，也会按“陌生人用户”处理。
> 启动参数 `--say-hallo` 开启后，仅在 continuous 的非响应阶段检测到熟人时主动问好（称呼基于该用户记忆检索）。

## 操作说明

### 按键模式 (`--listen-mode push`，默认)

1. 启动程序后,等待所有模块初始化完成
2. 按住 `Q` 键开始说话
3. 松开 `Q` 键结束录音
4. 系统自动识别、生成回复并播放语音
5. 按 `Ctrl+C` 退出程序

### 持续监听模式 (`--listen-mode continuous`)

1. 启动程序后,等待所有模块初始化完成
2. 先说“你好”进入响应模式（未唤醒时不会回复）
3. 进入响应模式后，系统通过 Silero VAD 自动检测语音并回复
4. 停顿超过设定时间（默认1.5秒）后视为说话结束
5. 说“再见”可退出响应模式，退出后将忽略后续语音
6. **机器人说话时会自动屏蔽麦克风**，避免机器人听到自己的回复
7. 按 `Ctrl+C` 退出程序
8. s键打断说话
9. w键唤醒/睡眠
> **提示**: 持续监听模式使用 [Silero VAD](https://github.com/snakers4/silero-vad) 进行语音检测，相关参数可在 `config.py` 中调整。

### 终端输入模式 (`--no-asr`)

1. 启动程序后,等待模块初始化完成
2. 在终端提示符 `你:` 后直接输入内容并回车
3. 系统基于输入文本生成回复（可选TTS播放）
4. 输入 `q` / `quit` / `exit` 退出程序

## 模块说明

### ASRModule (语音识别)
- 使用 FunASR SenseVoice 模型
- 支持中文识别
- 支持GPU加速

### TTSModule (语音合成)
- 支持 Kokoro / easy_tts_server 双后端
- 支持中文与英文
- 支持多种音色
- 实时流式播放
- 支持字符串与生成器输入

### LLMModule (大语言模型)
- 使用阿里Qwen模型
- 支持上下文记忆
- 可自定义人设

### MemoryModule (记忆管理)
- 使用 Mem0 向量数据库
- 自动存储对话历史
- 智能检索相关记忆
- 支持手动添加记忆（单条 / 交互式批量）
- 多用户独立数据库，路径: `mem_db/<用户名>/`

## 开发说明

每个模块都是独立的,可以单独测试和改进:

1. 修改模块代码
2. 运行对应的测试文件
3. 确认功能正常后集成到主程序

## 注意事项

- 首次运行会下载模型文件,需要较长时间
- 建议使用GPU运行ASR模块,CPU较慢
- TTS播放时会阻塞,等待播放完成
- Memory模块会在本地创建索引文件

## 故障排查

1. **录音无反应**: 检查麦克风权限
2. **识别不准确**: 确保环境安静,说话清晰
3. **播放无声音**: 检查系统音量和音频设备
4. **内存不足**: 减少缓存或使用CPU模式