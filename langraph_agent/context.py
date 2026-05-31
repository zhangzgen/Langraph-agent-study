from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from langraph_agent.config import config
from langraph_agent.models import AgentState
from langraph_agent.prompt import build_summary_prompt_messages

TOOL_RESULT_PLACEHOLDER = (
    "[工具执行结果已超过 KV cache 过期时间，已替换为占位符以压缩上下文。]"
)


def extract_total_tokens(message: BaseMessage) -> int | None:
    """从模型响应消息中提取 LangChain 标准化后的 total_tokens。"""
    usage_metadata = getattr(message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        total_tokens = usage_metadata.get("total_tokens")
        if isinstance(total_tokens, int):
            return total_tokens

    token_usage = getattr(message, "response_metadata", {}).get("token_usage")
    if isinstance(token_usage, dict):
        total_tokens = token_usage.get("total_tokens")
        if isinstance(total_tokens, int):
            return total_tokens

    return None


def should_compact_context(
    state: AgentState,
    *,
    token_threshold: int = config.COMPACT_TOKEN_THRESHOLD,
    recent_messages_to_keep: int = config.RECENT_MESSAGES_TO_KEEP,
) -> bool:
    """判断当前对话是否应该在本轮最终回答后压缩。"""
    messages = state.get("messages", [])
    if len(messages) <= recent_messages_to_keep:
        return False

    last_message = messages[-1] if messages else None
    if not isinstance(last_message, AIMessage):
        return False
    if last_message.tool_calls:
        return False

    total_tokens = state.get("last_total_tokens")
    return isinstance(total_tokens, int) and total_tokens >= token_threshold


def should_expire_tool_results(
    state: AgentState,
    *,
    ttl_seconds: int = config.KV_CACHE_TTL_SECONDS,
    now: float | None = None,
) -> bool:
    """判断是否应在下一次模型调用前把工具结果替换为占位符。"""
    messages = state.get("messages", [])
    if not messages or not any(isinstance(message, AIMessage) for message in messages):
        return False
    if not any(_tool_result_needs_placeholder(message) for message in messages):
        return False

    last_model_or_turn_at = state.get("last_model_or_turn_at")
    if not isinstance(last_model_or_turn_at, int | float):
        return False
    if ttl_seconds < 0:
        return False

    current_time = now if now is not None else time.time()
    return current_time - float(last_model_or_turn_at) >= ttl_seconds


def build_tool_result_placeholder_messages(
    messages: list[BaseMessage],
    *,
    placeholder: str = TOOL_RESULT_PLACEHOLDER,
) -> list[BaseMessage]:
    """构造用于替换全部 ToolMessage 内容的 messages 状态更新。"""
    rewritten_messages = [
        message.model_copy(update={"content": placeholder})
        if isinstance(message, ToolMessage)
        else message
        for message in messages
    ]
    return [
        RemoveMessage(id=REMOVE_ALL_MESSAGES, content=""),
        *rewritten_messages,
    ]


def count_tool_results_to_placeholder(messages: list[BaseMessage]) -> int:
    """统计仍包含真实结果、需要替换的工具消息数量。"""
    return sum(1 for message in messages if _tool_result_needs_placeholder(message))


def build_compacted_messages(
    messages: list[BaseMessage],
    *,
    recent_messages_to_keep: int = config.RECENT_MESSAGES_TO_KEEP,
) -> list[BaseMessage]:
    """构造用于替换 messages 状态的删除指令和最近消息窗口。"""
    recent_messages = select_recent_messages(messages, recent_messages_to_keep)
    return [
        RemoveMessage(id=REMOVE_ALL_MESSAGES, content=""),
        *recent_messages,
    ]


def select_recent_messages(
    messages: list[BaseMessage],
    max_messages: int = config.RECENT_MESSAGES_TO_KEEP,
) -> list[BaseMessage]:
    """选择最近消息，并避免从孤立 ToolMessage 开始。"""
    if len(messages) <= max_messages:
        return list(messages)

    start = max(0, len(messages) - max_messages)
    while start > 0 and isinstance(messages[start], ToolMessage):
        start -= 1
    return list(messages[start:])


def messages_to_text(messages: list[BaseMessage]) -> str:
    """把历史消息转成适合摘要模型阅读的纯文本。"""
    lines = []
    for index, message in enumerate(messages, start=1):
        role = _message_role(message)
        content = _message_content_to_text(message.content)
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            content = f"{content}\n工具调用: {tool_calls}"
        lines.append(f"{index}. [{role}] {content}")
    return "\n\n".join(lines)


def build_summary_prompt(
    state: AgentState,
    *,
    recent_messages_to_keep: int = config.RECENT_MESSAGES_TO_KEEP,
) -> list[BaseMessage]:
    """构造会话压缩摘要提示词。"""
    messages = state.get("messages", [])
    recent_messages = select_recent_messages(messages, recent_messages_to_keep)
    summarize_count = max(0, len(messages) - len(recent_messages))
    messages_for_summary = messages[:summarize_count]
    existing_summary = state.get("session_summary") or "无"

    return build_summary_prompt_messages(
        existing_summary=existing_summary,
        messages_for_summary=messages_to_text(messages_for_summary),
    )


def _message_role(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, ToolMessage):
        return f"tool:{message.name or message.tool_call_id}"
    return message.type


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return repr(content)


def _tool_result_needs_placeholder(message: BaseMessage) -> bool:
    return (
        isinstance(message, ToolMessage)
        and message.content != TOOL_RESULT_PLACEHOLDER
    )
