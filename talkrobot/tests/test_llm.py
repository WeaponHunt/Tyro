"""
LLM模块测试
"""
import time
from typing import Optional

from talkrobot.config import Config
from talkrobot.modules.llm.llm_module import LLMModule

def _run_single_round(llm: LLMModule, user_input: str) -> float:
    """执行单轮测试并返回生成耗时(秒)"""
    print(f"\n用户输入: {user_input}")
    print("正在生成回复...")

    infer_start = time.perf_counter()
    response = llm.generate_response(user_input)
    infer_elapsed = time.perf_counter() - infer_start

    print(f"AI回复: {response}")
    print(f"LLM生成耗时: {infer_elapsed:.3f}s")
    return infer_elapsed


def run_llm_test(user_input: Optional[str] = None, interactive: bool = False):
    """测试LLM模块

    Args:
        user_input: 指定单轮输入，不传则使用默认测试输入
        interactive: 是否开启交互模式（多轮输入，输入 q/quit/exit 结束）
    """
    test_start = time.perf_counter()

    print("="*50)
    print("测试 LLM 模块")
    print("="*50)

    init_start = time.perf_counter()
    llm = LLMModule(
        api_key=Config.LLM_API_KEY,
        base_url=Config.LLM_BASE_URL,
        model=Config.LLM_MODEL,
        system_prompt=Config.SYSTEM_PROMPT
    )
    init_elapsed = time.perf_counter() - init_start
    print(f"初始化耗时: {init_elapsed:.3f}s")

    if interactive:
        print("\n进入交互模式，输入 q / quit / exit 结束。")
        round_count = 0
        total_infer_elapsed = 0.0

        while True:
            current_input = input("\n请输入测试内容: ").strip()
            if current_input.lower() in {"q", "quit", "exit"}:
                break
            if not current_input:
                print("输入为空，请重新输入。")
                continue

            round_count += 1
            total_infer_elapsed += _run_single_round(llm, current_input)

        if round_count == 0:
            print("未执行任何测试轮次。")
        else:
            print(f"\n交互测试轮次: {round_count}")
            print(f"平均响应耗时: {total_infer_elapsed / round_count:.3f}s")
    else:
        test_input = user_input or "你好,请介绍一下你自己"
        _run_single_round(llm, test_input)

    print(f"总耗时: {time.perf_counter() - test_start:.3f}s")
    print("\n✅ LLM模块测试完成")


def test_llm():
    """pytest入口: 执行单轮默认测试"""
    run_llm_test()

if __name__ == "__main__":
    run_llm_test(interactive=True)