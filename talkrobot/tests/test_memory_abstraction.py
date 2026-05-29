from talkrobot.modules.memory.memory_module import MemoryModule


def test_simple_memory_backend_searches_and_persists(tmp_path):
    config = {"path": str(tmp_path)}
    user_id = "user_test"

    memory = MemoryModule(config=config, user_id=user_id, provider="simple")
    memory.add_memory("用户稳定信息: 我喜欢打篮球", async_mode=False)
    memory.add_memory("用户稳定信息: 我在科技公司工作", async_mode=False)

    context = memory.search_memory("篮球", limit=2)
    all_memories = memory.get_all_memories()
    memory.shutdown()

    assert "我喜欢打篮球" in context
    assert len(all_memories) == 2

    reloaded = MemoryModule(config=config, user_id=user_id, provider="simple")
    try:
        assert "科技公司" in reloaded.search_memory("科技公司", limit=1)
    finally:
        reloaded.shutdown()


def test_stable_memory_filter_is_backend_independent(tmp_path):
    memory = MemoryModule(config={"path": str(tmp_path)}, user_id="user_test", provider="simple")
    try:
        assert memory.add_user_memory_if_stable("我叫小明", async_mode=False) is True
        assert memory.add_user_memory_if_stable("帮我写一首诗", async_mode=False) is False
        assert "小明" in memory.search_memory("小明", limit=3)
    finally:
        memory.shutdown()
