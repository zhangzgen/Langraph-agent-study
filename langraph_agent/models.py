from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


ApprovalStatus = Literal[
    "auto_approved",
    "review_required",
    "approved",
    "rejected",
    "executed",
    "failed",
]


class ToolApproval(TypedDict):
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    status: ApprovalStatus
    reason: str | None


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_summary: NotRequired[str]
    last_total_tokens: NotRequired[int]
    context_compaction: NotRequired[dict[str, Any]]
    pending_approvals: NotRequired[list[ToolApproval]]
    approved_tool_calls: NotRequired[list[dict[str, Any]]]
    rejected_tool_calls: NotRequired[list[dict[str, Any]]]
    tool_audit_log: NotRequired[list[ToolApproval]]


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    path: Path
