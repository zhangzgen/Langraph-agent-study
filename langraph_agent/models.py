from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


ApprovalStatus = Literal[
    "auto_approved",    # 自动批准：低风险工具调用直接通过
    "review_required",  # 需人工审核：中风险工具需用户确认
    "approved",         # 用户批准：人工审核通过
    "rejected",         # 用户拒绝：人工审核不通过
    "executed",         # 已执行：工具已成功执行
    "failed",           # 执行失败：工具执行出错
]

TaskStatus = Literal[
    "pending",      # 待办：计划已生成，等待执行
    "in_progress",  # 执行中：Executor 正在处理该任务
    "completed",    # 已完成：Replanner 确认已达成
    "failed",       # 失败：多次重试后放弃
    "skipped",      # 跳过：因为前提条件改变或用户修改计划
]

PlanApprovalStatus = Literal[
    "pending_review",  # 等待用户审核计划书
    "approved",        # 用户批准了全局计划
    "rejected",        # 用户拒绝，要求重新规划（带有反馈）
]


class ToolApproval(TypedDict):
    tool_call_id: str          # 工具调用的唯一标识符
    tool_name: str             # 被调用的工具名称
    args: dict[str, Any]       # 工具调用的参数
    status: ApprovalStatus     # 当前审批状态
    reason: str | None         # 审批原因或备注（如自动批准的理由或拒绝的原因）


class PlanTask(TypedDict):
    id: str                    # 任务唯一标识符，用于跨节点追踪当前执行任务
    title: str                 # 任务标题，用于计划书和调试输出展示
    description: str           # 任务目标和执行要求，供 Executor 聚焦处理
    status: TaskStatus         # 当前任务状态，由 planner 初始化、replanner 更新
    result: NotRequired[str]   # 任务执行结果摘要，由 replanner 在任务结束后写入


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 对话消息历史（自动追加新消息）
    session_summary: NotRequired[str]                     # 会话摘要：长对话的压缩总结
    last_total_tokens: NotRequired[int]                   # 上次统计的总 token 数量
    context_compaction: NotRequired[dict[str, Any]]       # 上下文压缩配置或结果
    pending_approvals: NotRequired[list[ToolApproval]]    # 待审批的工具调用列表
    approved_tool_calls: NotRequired[list[dict[str, Any]]]  # 已批准的工具调用记录
    rejected_tool_calls: NotRequired[list[dict[str, Any]]]  # 已拒绝的工具调用记录
    tool_audit_log: NotRequired[list[ToolApproval]]       # 工具调用审计日志
    plan_tasks: NotRequired[list[PlanTask]]                # 当前全局任务列表，保存每个任务的元数据、状态、结果
    plan_document: NotRequired[str]                        # planner 基于所有任务信息组织出的自然语言计划书
    plan_approval_status: NotRequired[PlanApprovalStatus]  # 当前计划审核状态
    plan_feedback: NotRequired[str]                        # 用户拒绝计划时给出的修改反馈
    current_task_id: NotRequired[str | None]               # 当前正在执行的任务 ID


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    path: Path
