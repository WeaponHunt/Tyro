import json
import re
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model


llm = init_chat_model(
	"qwen-turbo",
	model_provider="openai",
	openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
	openai_api_key="sk-9d42be35fbba4ef8ab8d217c2a613869",
	temperature=0,
)

PROFILE_STORE_PATH = Path(__file__).resolve().parent.parent / "user_profiles.json"


class MemoryState(TypedDict, total=False):
	user_name: str
	user_input: str
	current_time: str
	user_profile: dict[str, Any]
	user_events: list[dict[str, Any]]
	raw_event_extract: str
	raw_trait_extract: str
	extracted_events: list[dict[str, Any]]
	extracted_traits: dict[str, Any]
	merged_profile: dict[str, Any]
	merged_events: list[dict[str, Any]]


def _parse_json_object(text: str) -> dict[str, Any]:
	text = (text or "").strip()
	try:
		data = json.loads(text)
		return data if isinstance(data, dict) else {}
	except json.JSONDecodeError:
		pass

	match = re.search(r"\{[\s\S]*\}", text)
	if not match:
		return {}
	try:
		data = json.loads(match.group(0))
		return data if isinstance(data, dict) else {}
	except json.JSONDecodeError:
		return {}


def _parse_json_array(text: str) -> list[dict[str, Any]]:
	text = (text or "").strip()
	try:
		data = json.loads(text)
		return data if isinstance(data, list) else []
	except json.JSONDecodeError:
		pass

	match = re.search(r"\[[\s\S]*\]", text)
	if not match:
		return []
	try:
		data = json.loads(match.group(0))
		return data if isinstance(data, list) else []
	except json.JSONDecodeError:
		return []


def _infer_time_from_text(text: str, now: datetime) -> tuple[str | None, str]:
	normalized = (text or "").strip()
	if not normalized:
		return None, "unknown"

	if "昨天" in normalized or "昨晚" in normalized:
		return (now - timedelta(days=1)).strftime("%Y-%m-%d"), "past"
	if "前天" in normalized:
		return (now - timedelta(days=2)).strftime("%Y-%m-%d"), "past"
	if "今天" in normalized or "今晚" in normalized:
		return now.strftime("%Y-%m-%d"), "present"
	if "明天" in normalized:
		return (now + timedelta(days=1)).strftime("%Y-%m-%d"), "future"
	if "后天" in normalized:
		return (now + timedelta(days=2)).strftime("%Y-%m-%d"), "future"
	if "下周" in normalized:
		return (now + timedelta(days=7 - now.weekday())).strftime("%Y-%m-%d"), "future"
	if "上周" in normalized:
		return (now - timedelta(days=now.weekday() + 7)).strftime("%Y-%m-%d"), "past"
	if any(word in normalized for word in ["计划", "打算", "准备", "将要"]):
		return None, "future"
	if any(word in normalized for word in ["已经", "刚刚", "之前", "曾经"]):
		return None, "past"
	if any(word in normalized for word in ["正在", "现在", "目前"]):
		return now.strftime("%Y-%m-%d"), "present"

	return None, "unknown"


def _normalize_event(item: dict[str, Any], user_input: str, now: datetime) -> dict[str, Any] | None:
	event = str(item.get("event", "")).strip()
	if not event:
		return None

	event_time = item.get("event_time")
	if isinstance(event_time, str):
		event_time = event_time.strip() or None
	else:
		event_time = None

	time_relation = str(item.get("time_relation", "unknown")).strip().lower()
	if time_relation not in {"past", "present", "future", "unknown"}:
		time_relation = "unknown"

	source_text = str(item.get("source_text", "")).strip() or user_input
	confidence = item.get("confidence", 5)
	if not isinstance(confidence, (int, float)):
		confidence = 5

	inferred_time, inferred_relation = _infer_time_from_text(source_text or event, now)
	if not event_time:
		event_time = inferred_time
	if time_relation == "unknown":
		time_relation = inferred_relation

	return {
		"event": event,
		"event_time": event_time,
		"time_relation": time_relation,
		"recorded_at": now.strftime("%Y-%m-%d %H:%M:%S"),
		"confidence": int(confidence),
		"source_text": source_text,
	}


def load_user_memory(user_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	if not PROFILE_STORE_PATH.exists():
		return {}, []

	try:
		all_profiles = json.loads(PROFILE_STORE_PATH.read_text(encoding="utf-8"))
	except (json.JSONDecodeError, OSError):
		return {}, []

	user_data = all_profiles.get(user_name, {}) if isinstance(all_profiles, dict) else {}
	if not isinstance(user_data, dict):
		return {}, []

	if "profile" not in user_data and "events" not in user_data:
		return user_data, []

	profile = user_data.get("profile", {})
	events = user_data.get("events", [])
	return (profile if isinstance(profile, dict) else {}), (events if isinstance(events, list) else [])


def save_user_memory(user_name: str, profile: dict[str, Any], events: list[dict[str, Any]]) -> None:
	all_profiles: dict[str, Any] = {}
	if PROFILE_STORE_PATH.exists():
		try:
			data = json.loads(PROFILE_STORE_PATH.read_text(encoding="utf-8"))
			if isinstance(data, dict):
				all_profiles = data
		except (json.JSONDecodeError, OSError):
			all_profiles = {}

	all_profiles[user_name] = {"profile": profile, "events": events}
	PROFILE_STORE_PATH.write_text(json.dumps(all_profiles, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_events(state: MemoryState):
	prompt = (
		"你是信息抽取助手。请从用户输入中提取‘事件记忆’，仅输出 JSON 数组，不要输出其它文本。"
		f"\n当前时间: {state['current_time']}"
		"\n必须结合当前时间与用户描述中的相对时间（如昨天、明天、下周）判断事件发生时间。"
		"\n每个事件结构: "
		"{\"event\":字符串,\"event_time\":YYYY-MM-DD或null,\"time_relation\":past|present|future|unknown,\"confidence\":0-10整数,\"source_text\":原句片段}"
		f"\n用户输入: {state['user_input']}"
	)
	msg = llm.invoke(prompt)
	return {"raw_event_extract": msg.content}


def extract_traits(state: MemoryState):
	prompt = (
		"你是信息抽取助手。请从用户输入中提取用户特质，严格输出 JSON 对象，不要输出其它文本。"
		"\n只保留有明确信号的字段，可选字段: personality, interests, preferences, communication_style, values, goals"
		"\n其中 personality 建议为关键词数组，如 [\"外向\",\"务实\"]。"
		f"\n用户输入: {state['user_input']}"
	)
	msg = llm.invoke(prompt)
	return {"raw_trait_extract": msg.content}


def _merge_traits(old_profile: dict[str, Any], new_traits: dict[str, Any]) -> dict[str, Any]:
	merged = dict(old_profile)
	for key, value in new_traits.items():
		if value is None:
			continue
		if isinstance(value, str) and not value.strip():
			continue
		if key in {"personality", "interests", "preferences", "values", "goals"}:
			old_list = merged.get(key, [])
			if not isinstance(old_list, list):
				old_list = [old_list]
			new_list = value if isinstance(value, list) else [value]
			merged[key] = old_list + new_list
			continue
		merged[key] = value
	return merged


def _merge_events(old_events: list[dict[str, Any]], new_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
	merged = list(old_events)
	exists = {
		(str(e.get("event", "")).strip(), str(e.get("event_time", "")).strip())
		for e in merged
		if isinstance(e, dict)
	}
	for e in new_events:
		key = (str(e.get("event", "")).strip(), str(e.get("event_time", "")).strip())
		if key[0] and key not in exists:
			merged.append(e)
			exists.add(key)
	return merged


def aggregate_memory(state: MemoryState):
	traits = _parse_json_object(state.get("raw_trait_extract", ""))
	raw_events = _parse_json_array(state.get("raw_event_extract", ""))

	now = datetime.strptime(state["current_time"], "%Y-%m-%d %H:%M:%S")
	normalized_events = []
	for item in raw_events:
		if isinstance(item, dict):
			normalized = _normalize_event(item, state["user_input"], now)
			if normalized:
				normalized_events.append(normalized)

	merged_profile = _merge_traits(state.get("user_profile", {}), traits)
	merged_events = _merge_events(state.get("user_events", []), normalized_events)

	return {
		"extracted_traits": traits,
		"extracted_events": normalized_events,
		"merged_profile": merged_profile,
		"merged_events": merged_events,
	}


def build_memory_updater():
	graph = StateGraph(MemoryState)
	graph.add_node("extract_events", extract_events)
	graph.add_node("extract_traits", extract_traits)
	graph.add_node("aggregate_memory", aggregate_memory)

	graph.add_edge(START, "extract_events")
	graph.add_edge(START, "extract_traits")
	graph.add_edge("extract_events", "aggregate_memory")
	graph.add_edge("extract_traits", "aggregate_memory")
	graph.add_edge("aggregate_memory", END)
	return graph.compile()


def update_memory(user_name: str, user_input: str, current_time: str | None = None) -> dict[str, Any]:
	profile, events = load_user_memory(user_name)
	now_text = current_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

	updater = build_memory_updater()
	result = updater.invoke(
		{
			"user_name": user_name,
			"user_input": user_input,
			"current_time": now_text,
			"user_profile": profile,
			"user_events": events,
		}
	)

	merged_profile = result.get("merged_profile", profile)
	merged_events = result.get("merged_events", events)
	save_user_memory(user_name, merged_profile, merged_events)

	return {
		"user_name": user_name,
		"current_time": now_text,
		"extracted_traits": result.get("extracted_traits", {}),
		"extracted_events": result.get("extracted_events", []),
		"profile": merged_profile,
		"events": merged_events,
	}


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="记忆更新模块测试")
	parser.add_argument("--user_name", default="demo_user", help="测试用户名，默认 demo_user")
	args = parser.parse_args()

	user_name = args.user_name.strip() or "demo_user"
	print(f"记忆更新测试已启动。当前用户: {user_name}，输入 quit 退出。")

	try:
		while True:
			user_text = input("\n请输入测试内容: ").strip()
			if user_text.lower() in {"quit", "exit", "q"}:
				print("已退出。")
				break
			if not user_text:
				continue

			result = update_memory(user_name, user_text)
			preview = {
				"extracted_traits": result.get("extracted_traits", {}),
				"extracted_events": result.get("extracted_events", []),
				"profile": result.get("profile", {}),
				"latest_events": result.get("events", [])[-5:],
			}
			print("\n记忆更新结果:")
			print(json.dumps(preview, ensure_ascii=False, indent=2))
	except (KeyboardInterrupt, EOFError):
		print("\n测试结束。")
