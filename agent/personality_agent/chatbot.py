import argparse
import json
from datetime import datetime
from typing import Any

from langchain.chat_models import init_chat_model
from memory.memory_update import load_user_memory, save_user_memory, update_memory


llm = init_chat_model(
    "qwen-turbo",
    model_provider="openai",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key="sk-9d42be35fbba4ef8ab8d217c2a613869",
    temperature=0,
)


def generate_reply(user_input: str, user_profile: dict[str, Any]) -> str:
    prompt = (
        "你是一个简洁、友好的中文助手。"
        f"\n当前用户画像（可能不完整）: {json.dumps(user_profile, ensure_ascii=False)}"
        f"\n用户输入: {user_input}"
        "\n请直接给出自然的对话回复。"
    )
    return llm.invoke(prompt).content


def run_turn(user_name: str, user_input: str) -> dict[str, Any]:
    memory_result = update_memory(
        user_name=user_name,
        user_input=user_input,
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    profile = memory_result.get("profile", {})
    events = memory_result.get("events", [])
    reply = generate_reply(user_input, profile)

    latest_events = events[-5:] if isinstance(events, list) else []
    final_output = (
        f"助手回复：\n{reply}\n\n"
        f"当前用户画像：\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
        f"最近事件记忆（最多5条）：\n{json.dumps(latest_events, ensure_ascii=False, indent=2)}"
    )

    return {
        "assistant_reply": reply,
        "user_profile": profile,
        "user_events": events,
        "extracted_traits": memory_result.get("extracted_traits", {}),
        "extracted_events": memory_result.get("extracted_events", []),
        "final_output": final_output,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对话 Agent（复用 memory_update 记忆模块）")
    parser.add_argument("--user_name", required=True, help="当前会话用户名")
    args = parser.parse_args()

    user_name = args.user_name.strip()
    if not user_name:
        raise ValueError("user_name 不能为空")

    profile, events = load_user_memory(user_name)
    print(f"对话 Agent 已启动。当前用户: {user_name}，输入 quit 退出。")
    if profile or events:
        print(f"已加载历史记忆：画像字段 {len(profile)} 个，事件 {len(events)} 条。")

    try:
        while True:
            user_text = input("\n你: ").strip()
            if user_text.lower() in {"quit", "exit", "q"}:
                print("已退出。")
                break
            if not user_text:
                continue

            result = run_turn(user_name, user_text)
            profile = result.get("user_profile", profile)
            events = result.get("user_events", events)
            print("\n" + result.get("final_output", ""))
    except (KeyboardInterrupt, EOFError):
        print("\n检测到程序关闭，正在保存画像...")
    finally:
        save_user_memory(user_name, profile, events)
        print("用户记忆（画像+事件）已保存。")