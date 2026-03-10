"""
Memory模块测试
"""
import time

from talkrobot.config import Config
from talkrobot.modules.memory_module import MemoryModule

def test_memory():
    """测试Memory模块"""
    test_start = time.perf_counter()

    print("="*50)
    print("测试 Memory 模块")
    print("="*50)

    test_user = "test_user"

    init_start = time.perf_counter()
    memory = MemoryModule(
        config=Config.get_memory_config(test_user),
        user_id=Config.get_user_id(test_user)
    )
    init_elapsed = time.perf_counter() - init_start
    print(f"初始化耗时: {init_elapsed:.3f}s")
    
    # 测试添加记忆
    print("\n1. 测试添加记忆")
    test_memories = [
        "我叫小明,今年25岁",
        "我喜欢打篮球和游泳",
        "我在一家科技公司工作"
    ]

    add_start = time.perf_counter()
    for mem in test_memories:
        single_add_start = time.perf_counter()
        print(f"添加: {mem}")
        memory.add_memory(mem)
        single_add_elapsed = time.perf_counter() - single_add_start
        print(f"  添加耗时: {single_add_elapsed:.3f}s")
    add_elapsed = time.perf_counter() - add_start
    print(f"批量添加总耗时: {add_elapsed:.3f}s")
    
    # 测试检索记忆
    print("\n2. 测试检索记忆")
    query = "告诉我关于我的信息"
    print(f"查询: {query}")

    search_start = time.perf_counter()
    context = memory.search_memory(query)
    search_elapsed = time.perf_counter() - search_start

    print(f"检索结果:\n{context}")
    print(f"检索耗时: {search_elapsed:.3f}s")
    print(f"总耗时: {time.perf_counter() - test_start:.3f}s")
    
    print("\n✅ Memory模块测试完成")

if __name__ == "__main__":
    test_memory()