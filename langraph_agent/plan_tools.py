from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


@tool
def create_plan(tasks: list[dict[str, Any]], plan_document: str) -> str:
    """创建新的全局计划任务列表和面向用户审核的计划书。"""
    return "计划创建请求已记录。"


@tool
def revise_plan(tasks: list[dict[str, Any]], plan_document: str) -> str:
    """根据用户反馈修订全局计划任务列表和面向用户审核的计划书。"""
    return "计划修订请求已记录。"


@tool
def get_current_plan() -> str:
    """读取当前计划书内容，供 planner 或 replanner 在决策前参考。"""
    return "当前计划书将由图节点根据状态返回。"


@tool
def get_plan_tasks() -> str:
    """读取当前计划任务列表，供 planner 或 replanner 在决策前参考。"""
    return "当前计划任务列表将由图节点根据状态返回。"


@tool
def get_current_task() -> str:
    """读取当前正在执行的任务，供 replanner 复核任务结果前参考。"""
    return "当前任务将由图节点根据状态返回。"


@tool
def complete_task(task_id: str, result: str) -> str:
    """将指定任务标记为已完成，并记录任务执行结果摘要。"""
    return "任务完成请求已记录。"


@tool
def fail_task(task_id: str, reason: str) -> str:
    """将指定任务标记为失败，并记录失败原因。"""
    return "任务失败请求已记录。"


@tool
def skip_task(task_id: str, reason: str) -> str:
    """将指定任务标记为跳过，并记录跳过原因。"""
    return "任务跳过请求已记录。"


PLANNER_TOOLS = [
    create_plan,
    revise_plan,
    get_current_plan,
    get_plan_tasks,
]

REPLANNER_TOOLS = [
    complete_task,
    fail_task,
    skip_task,
    get_current_plan,
    get_plan_tasks,
    get_current_task,
]

