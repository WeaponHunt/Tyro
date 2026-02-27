import os
from mem0 import Memory

# 1. 定义 Qwen 的配置
config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "my_memories",
            "path": "./mem0_db"  # 关键：指定数据保存的本地目录
        }
    },
    "llm": {
        "provider": "openai", # 阿里兼容 OpenAI 协议，所以 provider 选 openai
        "config": {
            "model": "qwen-plus", # 使用你代码里的模型名
            "api_key": "sk-9d42be35fbba4ef8ab8d217c2a613869",
            "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", # 关键：指向阿里服务器
            "max_tokens": 1500,
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-v2", # 注意：阿里向量模型建议用这个
            "api_key": "sk-9d42be35fbba4ef8ab8d217c2a613869",
            "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
        }
    }
}

# 2. 初始化 Memory
m = Memory.from_config(config)

# 3. 测试存储记忆
# m.add("我喜欢看恐怖电影", user_id="user_01")
# m.add("我也喜欢看喜剧电影", user_id="user_01")


# 4. 测试检索
result = m.search("我喜欢看什么类型的电影？", user_id="user_01")
print(result)  # 应该能检索到 "我喜欢看恐怖电影" 和 "我也喜欢看喜剧电影"