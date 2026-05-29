# Memory 架构

TalkRobot 的长期记忆现在分为两层：

- `MemoryModule`: 对外 facade，保持原有调用接口不变。
- `MemoryBackend`: 可替换 backend 接口，负责真实存储和检索。

## 对外接口

运行代码继续依赖 `talkrobot.modules.memory.memory_module.MemoryModule`：

```python
memory = MemoryModule(config=config, user_id=user_id, provider="simple")
memory.add_memory("用户稳定信息: 我喜欢打篮球", async_mode=False)
context = memory.search_memory("篮球", limit=3)
items = memory.get_all_memories()
memory.shutdown()
```

兼容方法：

- `add_memory(text, async_mode=True)`
- `add_user_memory_if_stable(user_text, async_mode=True)`
- `search_memory(query, limit=3)`
- `get_all_memories()`
- `shutdown()`

## Backend 接口

新增 backend 时实现 `talkrobot.modules.memory.base.MemoryBackend`：

```python
class MyMemoryBackend(MemoryBackend):
    def add(self, text: str) -> None: ...
    def search(self, query: str, limit: int = 3) -> list[MemoryRecord]: ...
    def get_all(self) -> list[MemoryRecord]: ...
    def shutdown(self) -> None: ...
```

然后在 `talkrobot/modules/memory/factory.py` 的 `create_memory_backend()` 中注册 provider 名称。

## 已内置实现

- `mem0`: 默认实现，适配原有 Mem0 + Chroma 配置。
- `simple`: 本地 JSON 实现，无外部服务和 API key，适合测试、离线开发和验证替换链路。

## 配置

通过环境变量选择 backend：

```bash
TALKROBOT_MEMORY_PROVIDER=mem0
TALKROBOT_MEMORY_PROVIDER=simple
```

`simple` backend 默认存储在：

```text
talkrobot/mem_db/simple/
```

也可以设置：

```bash
TALKROBOT_SIMPLE_MEMORY_DB_PATH=/path/to/simple-memory
```

## 测试

本地 JSON backend 不依赖外部模型或 API key：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n robot_sys python -m pytest talkrobot/tests/test_memory_abstraction.py
```

这个测试覆盖：

- facade 调用 `simple` backend。
- 添加、搜索、列出、关闭。
- 重启后从 JSON 文件恢复。
- 稳定记忆过滤逻辑与 backend 解耦。
