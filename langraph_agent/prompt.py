from __future__ import annotations

from functools import cache
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

from langraph_agent.config import config


@cache
def load_prompt(prompt_id: str | None, local_filename: str) -> Any:
    """加载远程优先、支持本地回退的聊天提示词模板。

    Description:
        当配置了 LangSmith prompt 标识时优先拉取远程模板；远程模板不可用时，
        从项目根目录的 prompts 文件夹读取对应文本模板，并缓存加载结果。
    Args:
        prompt_id (str | None): LangSmith 中的 prompt 标识；为空时跳过远程加载。
        local_filename (str): prompts 文件夹中的本地模板文件名。
    Returns:
        Any: 提供 invoke 方法、可渲染为聊天消息的提示词模板实例。
    """
    if prompt_id:
        try:
            return Client().pull_prompt(prompt_id)
        except Exception:
            pass

    template = (config.PROJECT_ROOT / "prompts" / local_filename).read_text(
        encoding="utf-8"
    )
    return ChatPromptTemplate.from_messages([("system", template)])


def build_react_prompt_messages(
    *,
    skill_catalog: str,
    session_summary: str | None,
    plan_document: str | None = None,
) -> list[BaseMessage]:
    """渲染主 ReAct Agent 的 system prompt 消息。

    Description:
        加载主执行 Agent 的远程优先提示词模板，并按需附加会话摘要和已审核
        通过的执行计划，使执行阶段能沿用计划阶段产物。
    Args:
        skill_catalog (str): 当前可用 Skill 目录文本。
        session_summary (str | None): 历史会话压缩摘要；为空时不注入。
        plan_document (str | None): 用户审核通过的计划书；为空时不注入。
    Returns:
        list[BaseMessage]: 可直接拼接到对话历史前的 prompt 消息列表。
    """
    session_summary_block = (
        f"\n\n当前会话摘要:\n{session_summary}" if session_summary else ""
    )
    prompt = load_prompt(config.REACT_PROMPT_ID, "react.txt")
    messages = prompt.invoke(
        {
            "skill_catalog": skill_catalog,
            "session_summary_block": session_summary_block,
        }
    ).to_messages()
    filtered_messages = [message for message in messages if message.content]
    if plan_document:
        filtered_messages.append(
            SystemMessage(
                content=(
                    "以下计划书已由用户审核通过，执行 Agent 必须把它作为当前任务的"
                    f"实施依据，并在需要时继续使用工具完成任务。\n\n{plan_document}"
                )
            )
        )
    return filtered_messages


def build_plan_prompt_messages(
    *,
    skill_catalog: str,
    session_summary: str | None,
) -> list[BaseMessage]:
    """构建 plan 模式前置 ReAct 循环的 system prompt。

    Description:
        加载计划阶段的远程优先提示词模板，要求模型先通过只读文件工具和
        ask_human 澄清需求，最后输出给执行 Agent 使用的计划书。
    Args:
        skill_catalog (str): 当前可用 Skill 目录文本，用于辅助模型判断可用能力。
        session_summary (str | None): 历史会话压缩摘要；为空时不注入。
    Returns:
        list[BaseMessage]: 计划阶段模型调用所需的 system prompt 消息列表。
    """
    session_summary_block = (
        f"\n\n当前会话摘要:\n{session_summary}" if session_summary else ""
    )
    prompt = load_prompt(config.PLAN_PROMPT_ID, "plan.txt")
    return prompt.invoke(
        {
            "skill_catalog": skill_catalog,
            "session_summary_block": session_summary_block,
        }
    ).to_messages()


def build_summary_prompt_messages(
    *,
    existing_summary: str,
    messages_for_summary: str,
) -> list[BaseMessage]:
    """渲染会话压缩摘要使用的 system prompt 消息。

    Description:
        加载摘要阶段的远程优先提示词模板，并注入已有摘要与待压缩的消息文本。
    Args:
        existing_summary (str): 现有会话摘要；没有历史摘要时由调用方传入占位文本。
        messages_for_summary (str): 本轮需要压缩到摘要中的历史消息文本。
    Returns:
        list[BaseMessage]: 摘要模型调用所需的提示词消息列表。
    """
    prompt = load_prompt(config.SUMMARY_PROMPT_ID, "summary.txt")
    return prompt.invoke(
        {
            "existing_summary": existing_summary,
            "messages_for_summary": messages_for_summary,
        }
    ).to_messages()
