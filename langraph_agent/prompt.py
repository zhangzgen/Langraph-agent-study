from __future__ import annotations

from functools import cache

from langchain_core.messages import BaseMessage
from langsmith import Client

from langraph_agent.config import config


@cache
def pull_prompt(prompt_id: str):
    """从 LangSmith 拉取指定 prompt，并在当前进程内缓存结果。"""
    return Client().pull_prompt(prompt_id)


def build_react_prompt_messages(
    *,
    skill_catalog: str,
    session_summary: str | None,
) -> list[BaseMessage]:
    """渲染主 ReAct Agent 的远程 system prompt 消息。"""
    session_summary_block = (
        f"\n\n当前会话摘要:\n{session_summary}" if session_summary else ""
    )
    prompt = pull_prompt(config.REACT_PROMPT_ID)
    messages = prompt.invoke(
        {
            "skill_catalog": skill_catalog,
            "session_summary_block": session_summary_block,
        }
    ).to_messages()
    return [message for message in messages if message.content]


def build_summary_prompt_messages(
    *,
    existing_summary: str,
    messages_for_summary: str,
) -> list[BaseMessage]:
    """渲染会话压缩摘要使用的远程 prompt 消息。"""
    prompt = pull_prompt(config.SUMMARY_PROMPT_ID)
    return prompt.invoke(
        {
            "existing_summary": existing_summary,
            "messages_for_summary": messages_for_summary,
        }
    ).to_messages()
