"""
Memory模块测试
"""
from talkrobot.config import Config
from talkrobot.modules.memory_module import MemoryModule

def test_memory():
    """测试Memory模块"""
    print("="*50)
    print("测试 Memory 模块")
    print("="*50)
    
    test_user = "test_user"
    
    memory = MemoryModule(
        config=Config.get_memory_config(test_user),
        user_id=Config.get_user_id(test_user)
    )
    
    # 测试添加记忆
    print("\n1. 测试添加记忆")
    test_memories = [
        "我叫小明,今年25岁",
        "我喜欢打篮球和游泳",
        "我在一家科技公司工作"
    ]
    
    for mem in test_memories:
        print(f"添加: {mem}")
        memory.add_memory(mem)
    
    # 测试检索记忆
    print("\n2. 测试检索记忆")
    query = "告诉我关于我的信息"
    print(f"查询: {query}")
    context = memory.search_memory(query)
    print(f"检索结果:\n{context}")
    
    print("\n✅ Memory模块测试完成")

if __name__ == "__main__":
    test_memory()