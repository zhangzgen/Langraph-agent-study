from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import interrupt

from langraph_agent.config import config
from langraph_agent.models import AgentState, ApprovalStatus, ToolApproval
from langraph_agent.tools import TOOLS

# 这些工具默认允许直接执行。注意“工具名白名单”不是唯一条件：
# 如果参数里包含敏感路径，仍会升级为人工审核。
AUTO_APPROVED_TOOLS = {
    "calculator",
    "current_time",
    "file_search",
    "list_directory",
    "list_skills",
    "load_skill",
    "read_file",
    "web_extract",
    "web_search",
}

# 这些工具具有修改文件系统或执行本地命令的能力，必须让 LangGraph
# interrupt 暂停图执行，等人类确认后才能继续。
REVIEW_REQUIRED_TOOLS = {
    "bash",
    "copy_file",
    "file_delete",
    "move_file",
    "write_file",
}
SENSITIVE_PATH_PARTS = {
    ".env",
    ".git",
    ".venv",
    "__pycache__",
}
SENSITIVE_PATH_SUFFIXES = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".pyc",
)


TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def classify_tool_calls_node(
    state: AgentState,
) -> dict[str, list[dict[str, Any]] | list[ToolApproval]]:
    """将模型请求的工具调用分成自动执行和等待人工审核两类。"""
    tool_calls = _get_last_tool_calls(state)
    if not tool_calls:
        return {
            "pending_approvals": [],
            "approved_tool_calls": [],
            "rejected_tool_calls": [],
        }

    approved_calls = []
    review_required_calls = []
    audit_entries = []
    for tool_call in tool_calls:
        if should_auto_approve_tool_call(tool_call):
            approved_calls.append(tool_call)
            audit_entries.append(_approval_entry(tool_call, "auto_approved", None))
        else:
            review_required_calls.append(tool_call)
            audit_entries.append(
                _approval_entry(tool_call, "review_required", _review_reason(tool_call))
            )

    return {
        "pending_approvals": [
            _approval_entry(tool_call, "review_required", _review_reason(tool_call))
            for tool_call in review_required_calls
        ],
        "approved_tool_calls": approved_calls,
        "rejected_tool_calls": [],
        "tool_audit_log": _append_audit_log(state, audit_entries),
    }


def approval_gate_node(
    state: AgentState,
) -> dict[str, list[dict[str, Any]] | list[ToolApproval]]:
    """对 pending_approvals 执行细粒度人工审批，并更新批准/拒绝列表。"""
    pending_approvals = state.get("pending_approvals", [])
    if not pending_approvals:
        return {}

    review_required_calls = [_tool_call_from_approval(item) for item in pending_approvals]
    # interrupt 会把图暂停在当前节点。CLI 收到 __interrupt__ 后展示审批信息，
    # 再用 Command(resume=...) 把人类决策传回这里，继续从这一行往下执行。
    decision = interrupt(build_tool_approval_request(review_required_calls))
    approved_call_ids = normalize_approved_call_ids(decision, review_required_calls)

    approved_calls = list(state.get("approved_tool_calls", []))
    rejected_calls = []
    audit_entries = []
    for tool_call in review_required_calls:
        if tool_call.get("id") in approved_call_ids:
            approved_calls.append(tool_call)
            audit_entries.append(_approval_entry(tool_call, "approved", "用户批准执行。"))
        else:
            rejected_calls.append(tool_call)
            audit_entries.append(_approval_entry(tool_call, "rejected", "用户拒绝执行。"))

    return {
        "pending_approvals": [],
        "approved_tool_calls": approved_calls,
        "rejected_tool_calls": rejected_calls,
        "tool_audit_log": _append_audit_log(state, audit_entries),
    }


def execute_tools_node(state: AgentState) -> dict[str, list[ToolMessage] | list[ToolApproval]]:
    """执行已批准的工具调用，并为被拒绝的调用生成 ToolMessage。"""
    messages = []
    audit_entries = []
    for tool_call in state.get("approved_tool_calls", []):
        message = _execute_tool_call(tool_call)
        messages.append(message)
        status = "failed" if message.content.startswith("工具执行失败:") else "executed"
        audit_entries.append(_approval_entry(tool_call, status, None))

    for tool_call in state.get("rejected_tool_calls", []):
        messages.append(_rejected_tool_message(tool_call))

    return {
        "messages": messages,
        "pending_approvals": [],
        "approved_tool_calls": [],
        "rejected_tool_calls": [],
        "tool_audit_log": _append_audit_log(state, audit_entries),
    }


def should_auto_approve_tool_call(tool_call: dict[str, Any]) -> bool:
    """判断一个工具调用是否可以跳过人工审核直接执行。"""
    tool_name = tool_call.get("name")
    if tool_name not in AUTO_APPROVED_TOOLS:
        return False
    return not _has_sensitive_path_argument(tool_call.get("args") or {})


def build_tool_approval_request(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """构造传给 LangGraph interrupt 的结构化人工审核请求。"""
    return {
        "type": "tool_approval",
        "message": "以下工具调用不在自动执行白名单中，需要人工确认。",
        "tool_calls": [
            {
                "id": tool_call.get("id"),
                "name": tool_call.get("name"),
                "args": tool_call.get("args") or {},
                "reason": _review_reason(tool_call),
            }
            for tool_call in tool_calls
        ],
    }


def normalize_approved_call_ids(
    decision: Any,
    review_required_calls: list[dict[str, Any]],
) -> set[str]:
    """把人工审核返回值标准化为被批准执行的 tool_call id 集合。"""
    all_call_ids = {
        tool_call["id"]
        for tool_call in review_required_calls
        if tool_call.get("id")
    }
    if decision is True:
        return all_call_ids
    if decision is False or decision is None:
        return set()
    if not isinstance(decision, dict):
        return set()

    if decision.get("approved") is True:
        return all_call_ids
    if decision.get("approved") is False:
        return set()

    approved_call_ids = decision.get("approved_call_ids")
    if isinstance(approved_call_ids, list):
        return {
            call_id
            for call_id in approved_call_ids
            if isinstance(call_id, str) and call_id in all_call_ids
        }
    return set()


def has_tool_calls(state: AgentState) -> bool:
    """判断当前状态最后一条 AIMessage 是否包含 tool_calls。"""
    return bool(_get_last_tool_calls(state))


def has_pending_approvals(state: AgentState) -> bool:
    """判断当前状态中是否存在待人工审核的工具调用。"""
    return bool(state.get("pending_approvals"))


def _get_last_tool_calls(state: AgentState) -> list[dict[str, Any]]:
    """从状态最后一条 AIMessage 中取出模型请求的 tool_calls。"""
    messages = state["messages"]
    if not messages:
        return []
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        return []
    return list(last_message.tool_calls or [])


def _execute_tool_call(tool_call: dict[str, Any]) -> ToolMessage:
    """执行单个工具调用，并把结果包装为 LangChain ToolMessage。"""
    tool_name = tool_call.get("name") or ""
    tool_call_id = tool_call.get("id") or tool_name
    tool = TOOLS_BY_NAME.get(tool_name)
    if tool is None:
        content = f"未知工具，未执行: {tool_name}"
    else:
        content = _invoke_tool(tool, tool_call.get("args") or {})
    return ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)


def _invoke_tool(tool: BaseTool, args: dict[str, Any]) -> str:
    """调用 LangChain tool，并把异常转成可返回给模型的文本。"""
    try:
        if tool.name == "bash":
            # bash 工具内部仍有自己的确认逻辑。图层已经通过 interrupt
            # 完成人工审批时，用上下文标记避免同一次调用重复询问。
            from langraph_agent.tools.shell import graph_approved_execution

            with graph_approved_execution():
                result = tool.invoke(args)
        else:
            result = tool.invoke(args)
    except Exception as exc:
        return f"工具执行失败: {type(exc).__name__}: {exc}"
    return _stringify_tool_result(result)


def _stringify_tool_result(result: Any) -> str:
    """将工具结果转成字符串，并限制长度避免塞爆模型上下文。"""
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, ensure_ascii=False, indent=2)
        except TypeError:
            text = str(result)
    if len(text) <= config.OUTPUT_LIMIT:
        return text
    return text[: config.OUTPUT_LIMIT] + "\n...[tool output truncated]"


def _rejected_tool_message(tool_call: dict[str, Any]) -> ToolMessage:
    """为被人工拒绝的工具调用生成对应 ToolMessage。"""
    tool_name = tool_call.get("name") or ""
    tool_call_id = tool_call.get("id") or tool_name
    return ToolMessage(
        content=f"用户未批准执行工具 {tool_name}，因此该工具调用没有执行。",
        tool_call_id=tool_call_id,
        name=tool_name,
    )


def _approval_entry(
    tool_call: dict[str, Any],
    status: ApprovalStatus,
    reason: str | None,
) -> ToolApproval:
    """把 LangChain tool_call 转成可持久记录的审批事件。"""
    tool_name = tool_call.get("name") or ""
    return {
        "tool_call_id": tool_call.get("id") or tool_name,
        "tool_name": tool_name,
        "args": tool_call.get("args") or {},
        "status": status,
        "reason": reason,
    }


def _tool_call_from_approval(approval: ToolApproval) -> dict[str, Any]:
    """把审批状态对象还原为执行器使用的 tool_call 结构。"""
    return {
        "id": approval["tool_call_id"],
        "name": approval["tool_name"],
        "args": approval["args"],
    }


def _append_audit_log(
    state: AgentState,
    entries: list[ToolApproval],
) -> list[ToolApproval]:
    """在普通 list state 字段上显式追加审计事件。"""
    if not entries:
        return state.get("tool_audit_log", [])
    return [*state.get("tool_audit_log", []), *entries]


def _has_sensitive_path_argument(args: dict[str, Any]) -> bool:
    """检查工具参数中是否包含敏感路径。"""
    return any(
        _is_sensitive_path(value)
        for key, value in args.items()
        if key.endswith("path") or key in {"path", "file_path", "dir_path"}
    )


def _is_sensitive_path(value: Any) -> bool:
    """判断单个路径字符串是否命中项目内敏感文件或目录规则。"""
    if not isinstance(value, str):
        return False
    path = PurePosixPath(value)
    if path.name.startswith(".env"):
        return True
    if path.suffix in SENSITIVE_PATH_SUFFIXES:
        return True
    return any(part in SENSITIVE_PATH_PARTS for part in path.parts)


def _review_reason(tool_call: dict[str, Any]) -> str:
    """返回某个工具调用需要人工审核的原因说明。"""
    tool_name = tool_call.get("name")
    args = tool_call.get("args") or {}
    if _has_sensitive_path_argument(args):
        return "路径包含敏感文件或目录，需要人工确认。"
    if tool_name in REVIEW_REQUIRED_TOOLS:
        return "该工具可能修改文件系统或执行本地命令，需要人工确认。"
    return "该工具不在自动执行白名单中。"
