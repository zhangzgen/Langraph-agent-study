from __future__ import annotations

from functools import cache

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
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
    active_task_context: str | None = None,
) -> list[BaseMessage]:
    """渲染主 ReAct Agent 的远程 system prompt 消息，并按需追加当前计划任务。"""
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
    rendered_messages = [message for message in messages if message.content]
    if active_task_context:
        rendered_messages.append(SystemMessage(content=active_task_context))
    return rendered_messages


def build_planner_prompt_messages(
    *,
    conversation_text: str,
    session_summary: str | None,
    existing_tasks_json: str,
    plan_feedback: str | None,
) -> list[BaseMessage]:
    """构造 planner 节点使用的消息，要求模型通过专属工具写入任务列表和计划书。"""
    feedback_block = f"\n用户反馈：\n{plan_feedback}" if plan_feedback else ""
    summary_block = f"\n会话摘要：\n{session_summary}" if session_summary else ""
    return [
        SystemMessage(
            content=(
                "你是计划规划器。请基于用户目标生成可执行任务列表，并基于完整任务元数据"
                "组织一份给用户审核的自然语言计划书。你必须调用 create_plan 或 revise_plan "
                "工具提交最终计划，不要用普通文本或 JSON 直接回答。"
                "tasks 参数中的每个任务必须包含 id、title、description、status、result。"
                "新规划或重新规划时，未执行任务的 status 必须是 pending。"
                "如需查看已有计划，可以先调用 get_current_plan 或 get_plan_tasks。"
            )
        ),
        HumanMessage(
            content=(
                f"当前对话：\n{conversation_text}{summary_block}\n\n"
                f"已有任务 JSON：\n{existing_tasks_json}{feedback_block}"
            )
        ),
    ]


def build_replanner_prompt_messages(
    *,
    task_json: str,
    conversation_text: str,
) -> list[BaseMessage]:
    """构造 replanner 节点使用的消息，要求模型通过专属工具更新任务终态。"""
    return [
        SystemMessage(
            content=(
                "你是任务状态复核器。请根据当前任务和最新执行对话判断任务是否达成。"
                "你必须调用 complete_task、fail_task 或 skip_task 其中一个工具提交判断，"
                "不要用普通文本或 JSON 直接回答。如需查看计划状态，可以先调用"
                " get_current_plan、get_plan_tasks 或 get_current_task。"
            )
        ),
        HumanMessage(
            content=f"当前任务 JSON：\n{task_json}\n\n最新执行对话：\n{conversation_text}"
        ),
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
