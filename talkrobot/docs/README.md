# Tyro TalkRobot 文档中心

欢迎查看 Tyro TalkRobot 的技术文档。

## 📚 文档列表

### 模块文档

- **[Code Structure](CODE_STRUCTURE.md)** - 当前代码结构与扩展点说明
  - 入口、Core、Modules、Agent 分层
  - Tool、MCP、Skill 扩展方式
  - 日志、测试与清理原则

- **[Memory Architecture](MEMORY_ARCHITECTURE.md)** - 可替换长期记忆架构
  - `MemoryModule` facade 与 `MemoryBackend` 接口
  - Mem0 与本地 JSON backend
  - provider 配置与测试方式

- **[MCP Integration](MCP_INTEGRATION.md)** - 外部 MCP server 接入说明
  - JSON 配置与环境变量
  - Python 构造接口
  - 当前实验 MCP tools

- **[ASR Module](ASR_MODULE.md)** - 语音识别模块完整文档
  - 接口说明
  - 使用示例
  - 性能指标
  - 常见问题

- **[AudioRecorder](AUDIO_RECORDER.md)** - 音频录制模块完整文档
  - Push 和 Continuous 两种模式
  - VAD 自动检测
  - 实时状态显示
  - 配置参数详解

### 测试文档

- **[Testing Guide](TESTING.md)** - 测试使用指南
  - ASR 模块测试
  - AudioRecorder 模块测试
  - 快速测试
  - 完整测试套件
  - 故障排除

## 🚀 快速开始

### 查看模块文档
```bash
# ASR 模块
cat talkrobot/docs/ASR_MODULE.md

# AudioRecorder 模块
cat talkrobot/docs/AUDIO_RECORDER.md
```

### 运行测试
```bash
# ASR 测试
python -m talkrobot.tests.test_asr

# AudioRecorder 测试
python -m talkrobot.tests.test_audio_recorder
```

## 📖 文档结构

```
talkrobot/docs/
├── README.md           # 本文件
├── CODE_STRUCTURE.md   # 代码结构说明
├── MEMORY_ARCHITECTURE.md # Memory 架构说明
├── MCP_INTEGRATION.md  # MCP 接入说明
├── ASR_MODULE.md       # ASR 模块文档
├── AUDIO_RECORDER.md   # AudioRecorder 模块文档
└── TESTING.md          # 测试指南
```

## 🔗 相关资源

- **项目主 README**: `../README.md`
- **配置文件**: `../config.py`
- **测试目录**: `../tests/`
- **模块目录**: `../modules/`

## 📝 待完善文档

以下模块文档待补充：
- [x] ~~ASR Module 文档~~
- [x] ~~AudioRecorder 文档~~
- [ ] TTS Module 文档
- [ ] LLM Module 文档
- [ ] Memory Module 文档
- [ ] Expression Module 文档
- [ ] Conversation Manager 文档

## 💡 贡献文档

欢迎贡献文档！请确保：
1. 使用 Markdown 格式
2. 包含清晰的示例代码
3. 提供完整的 API 说明
4. 包含常见问题解答

## 📧 反馈

如有文档问题或建议，请提交 Issue 或 Pull Request。
