"""
LLM模块测试
"""
from talkrobot.config import Config
from talkrobot.modules.llm_module import LLMModule

def test_llm():
    """测试LLM模块"""
    print("="*50)
    print("测试 LLM 模块")
    print("="*50)
    
    llm = LLMModule(
        api_key=Config.LLM_API_KEY,
        base_url=Config.LLM_BASE_URL,
        model=Config.LLM_MODEL,
        system_prompt=Config.SYSTEM_PROMPT
    )
    
    test_input = "你好,请介绍一下你自己"
    print(f"\n用户输入: {test_input}")
    print("正在生成回复...")
    
    response = llm.generate_response(test_input)
    print(f"AI回复: {response}")
    print("\n✅ LLM模块测试完成")

if __name__ == "__main__":
    test_llm()