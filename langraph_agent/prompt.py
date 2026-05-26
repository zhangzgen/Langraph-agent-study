from __future__ import annotations

from functools import cache

from langchain_core.messages import BaseMessage, SystemMessage
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
    plan_document: str | None = None,
) -> list[BaseMessage]:
    """渲染主 ReAct Agent 的远程 system prompt 消息。

    Description:
        从 LangSmith 拉取主执行 Agent 的 system prompt，并按需附加会话摘要和
        已审核通过的执行计划，使执行阶段能沿用计划阶段产物。
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
    prompt = pull_prompt(config.REACT_PROMPT_ID)
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
        生成计划阶段专用的本地 system prompt，要求模型先通过只读文件工具和
        ask_human 澄清需求，最后输出给执行 Agent 使用的计划书。
    Args:
        skill_catalog (str): 当前可用 Skill 目录文本，用于辅助模型判断可用能力。
        session_summary (str | None): 历史会话压缩摘要；为空时不注入。
    Returns:
        list[BaseMessage]: 计划阶段模型调用所需的 system prompt 消息列表。
    """
    summary_block = f"\n\n当前会话摘要:\n{session_summary}" if session_summary else ""
    return [
        SystemMessage(
            content=(
                "你是计划阶段的 ReAct Agent，只负责在执行前澄清需求并产出计划书。"
                "你可以调用只读文件工具理解项目，也可以调用 ask_human 向用户提问。"
                "ask_human 支持选择题和说明题；当任务目标、边界、验收标准或风险点"
                "不明确时，应优先提问。"
                "\n\n调用 ask_human 时只能使用以下两种 JSON 格式："
                '\n1. 选择题: {"choose_list":{"问题一":["选项 A","选项 B"],'
                '"问题二":["选项 C","选项 D"]}}'
                '\n2. 说明题: {"question":"需要用户补充说明的问题文本"}'
                "choose_list 必须是字典，键是问题文本，值是字符串选项列表，"
                "可以在一次调用中提供一个或多个问题；"
                "不要同时传 choose_list 和 question。"
                "\n\n当信息足够时，不要调用工具，直接输出计划书。计划书是给后续"
                "执行 Agent 看的，不是面向用户的说明文。计划书必须具体写明任务目标、"
                "相关文件、实施步骤、关键实现细节、验证方式、注意事项。"
                "禁止建议把少于 5 行的代码片段抽取为函数，保持核心逻辑高内聚。"
                "\n\n输出计划书时使用标题“执行计划书”，并只输出计划书正文。"
                f"\n\n可用 Skill:\n{skill_catalog}"
                f"{summary_block}"
            )
        )
    ]


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
