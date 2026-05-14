from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langraph_agent.checkpoints import resolve_checkpoint_db_path, sqlite_checkpointer
from langraph_agent.config import config
from langraph_agent.context import (
    build_compacted_messages,
    build_summary_prompt,
    extract_total_tokens,
    messages_to_text,
    should_compact_context,
)
from langraph_agent.llm import build_llm
from langraph_agent.models import AgentState, PlanTask, TaskStatus
from langraph_agent.plan_tools import PLANNER_TOOLS, REPLANNER_TOOLS
from langraph_agent.prompt import (
    build_planner_prompt_messages,
    build_react_prompt_messages,
    build_replanner_prompt_messages,
)
from langraph_agent.skills.registry import discover_skills, format_skill_catalog
from langraph_agent.tool_guard import (
    approval_gate_node,
    classify_tool_calls_node,
    execute_tools_node,
    has_pending_approvals,
    has_tool_calls,
)


VALID_TASK_STATUSES = {
    "pending",
    "in_progress",
    "completed",
    "failed",
    "skipped",
}
PLAN_READ_ONLY_TOOL_NAMES = {
    "get_current_plan",
    "get_plan_tasks",
    "get_current_task",
}
PLANNER_WRITE_TOOL_NAMES = {"create_plan", "revise_plan"}
REPLANNER_STATUS_BY_TOOL: dict[str, TaskStatus] = {
    "complete_task": "completed",
    "fail_task": "failed",
    "skip_task": "skipped",
}


def build_graph(
    with_memory: bool = False,
    checkpointer: BaseCheckpointSaver | None = None,
    compact_token_threshold: int = config.COMPACT_TOKEN_THRESHOLD,
    recent_messages_to_keep: int = config.RECENT_MESSAGES_TO_KEEP,
):
    """构建 LangGraph ReAct/Plan 状态机，并按需启用 checkpointer。"""
    llm = build_llm()
    summary_llm = build_llm(bind_tools=False)
    planner_llm = _bind_tools_if_supported(build_llm(bind_tools=False), PLANNER_TOOLS)
    replanner_llm = _bind_tools_if_supported(build_llm(bind_tools=False), REPLANNER_TOOLS)
    skill_catalog = format_skill_catalog(discover_skills())

    def call_llm(
        state: AgentState,
        config: RunnableConfig,
    ) -> dict[str, object]:
        """调用模型，让模型在普通 ReAct 或当前计划任务上下文中决定下一步。"""
        prompt_messages = build_react_prompt_messages(
            skill_catalog=skill_catalog,
            session_summary=state.get("session_summary"),
            active_task_context=(
                _build_active_task_context(state)
                if _is_plan_mode_enabled(config)
                else None
            ),
        )
        # 这里是 ReAct 中的“Reason/Act 决策”阶段：
        # 模型读取用户问题和历史消息，决定直接回答，还是返回 tool_calls。
        response = llm.invoke([*prompt_messages, *state["messages"]])
        update: dict[str, object] = {"messages": [response]}
        total_tokens = extract_total_tokens(response)
        if total_tokens is not None:
            update["last_total_tokens"] = total_tokens
        return update

    def planner_node(state: AgentState) -> dict[str, object]:
        """调用 planner 专属工具生成或修订任务列表，并写入用户可读计划书。"""
        tool_call = _invoke_until_plan_tool_call(
            planner_llm,
            build_planner_prompt_messages(
                conversation_text=messages_to_text(state.get("messages", [])),
                session_summary=state.get("session_summary"),
                existing_tasks_json=json.dumps(
                    state.get("plan_tasks", []),
                    ensure_ascii=False,
                    indent=2,
                ),
                plan_feedback=(
                    state.get("plan_feedback")
                    if state.get("plan_approval_status") == "rejected"
                    else None
                ),
            ),
            state,
            write_tool_names=PLANNER_WRITE_TOOL_NAMES,
        )
        args = tool_call.get("args") or {}
        tasks = _normalize_plan_tasks(args.get("tasks", []))
        return {
            "plan_tasks": tasks,
            "plan_document": str(args.get("plan_document") or ""),
            "plan_approval_status": "pending_review",
            "plan_feedback": "",
            "current_task_id": None,
        }

    def plan_approval_node(state: AgentState) -> dict[str, object]:
        """暂停状态机等待用户审核计划书，并根据审核结果进入执行或重新规划。"""
        decision = interrupt(build_plan_approval_request(state))
        if _is_plan_approved(decision):
            tasks, current_task_id = _activate_next_pending_task(
                state.get("plan_tasks", [])
            )
            update: dict[str, object] = {
                "plan_tasks": tasks,
                "plan_approval_status": "approved",
                "plan_feedback": "",
                "current_task_id": current_task_id,
            }
            if current_task_id is None:
                update["messages"] = [AIMessage(content="计划已批准，但没有待执行任务。")]
            return update

        return {
            "plan_approval_status": "rejected",
            "plan_feedback": _extract_plan_feedback(decision),
            "current_task_id": None,
        }

    def replanner_node(state: AgentState) -> dict[str, object]:
        """调用 replanner 专属工具复核当前任务，并更新任务终态。"""
        current_task = _find_current_task(state)
        if current_task is None:
            return {"current_task_id": None}

        tool_call = _invoke_until_plan_tool_call(
            replanner_llm,
            build_replanner_prompt_messages(
                task_json=json.dumps(current_task, ensure_ascii=False, indent=2),
                conversation_text=messages_to_text(state.get("messages", [])),
            ),
            state,
            write_tool_names=set(REPLANNER_STATUS_BY_TOOL),
        )
        tool_name = str(tool_call.get("name") or "")
        args = tool_call.get("args") or {}
        status = REPLANNER_STATUS_BY_TOOL[tool_name]
        task_id = str(args.get("task_id") or current_task["id"])
        result = str(args.get("result") or args.get("reason") or "")

        tasks = _update_task_status(
            state.get("plan_tasks", []),
            task_id,
            status,
            result,
        )
        tasks, next_task_id = _activate_next_pending_task(tasks)
        update: dict[str, object] = {
            "plan_tasks": tasks,
            "current_task_id": next_task_id,
        }
        if next_task_id is None:
            update["messages"] = [AIMessage(content=_build_plan_final_summary(tasks))]
        return update

    def summarize_and_compact(state: AgentState) -> dict[str, object]:
        """把旧消息压缩进摘要，并物理裁剪 messages 状态。"""
        prompt = build_summary_prompt(
            state,
            recent_messages_to_keep=recent_messages_to_keep,
        )
        response = summary_llm.invoke(prompt)
        compacted_messages = build_compacted_messages(
            state["messages"],
            recent_messages_to_keep=recent_messages_to_keep,
        )
        kept_message_count = len(compacted_messages) - 1
        previous_message_count = len(state["messages"])
        return {
            "session_summary": str(response.content),
            "messages": compacted_messages,
            "context_compaction": {
                "previous_message_count": previous_message_count,
                "kept_message_count": kept_message_count,
                "removed_message_count": previous_message_count - kept_message_count,
                "last_total_tokens": state.get("last_total_tokens"),
                "token_threshold": compact_token_threshold,
            },
        }

    react_executor = _build_react_executor_graph(call_llm=call_llm)

    # 主图只负责编排计划模式和子图执行；ReAct 工具循环封装在 react_executor 子图内。
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("plan_approval", plan_approval_node)
    builder.add_node("react_executor", react_executor)
    builder.add_node("replanner", replanner_node)
    builder.add_node("summarize_and_compact", summarize_and_compact)

    builder.add_conditional_edges(
        START,
        _route_from_start,
        {"planner": "planner", "react_executor": "react_executor"},
    )
    builder.add_edge("planner", "plan_approval")
    builder.add_conditional_edges(
        "plan_approval",
        _route_after_plan_approval,
        {"planner": "planner", "react_executor": "react_executor", END: END},
    )
    builder.add_conditional_edges(
        "react_executor",
        lambda state, config: _route_after_react_executor(
            state,
            config,
            compact_token_threshold=compact_token_threshold,
            recent_messages_to_keep=recent_messages_to_keep,
        ),
        {"replanner": "replanner", "summarize_and_compact": "summarize_and_compact", END: END},
    )
    builder.add_conditional_edges(
        "replanner",
        _route_after_replanner,
        {"react_executor": "react_executor", END: END},
    )
    builder.add_edge("summarize_and_compact", END)

    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)

    if not with_memory:
        return builder.compile()

    # checkpointer 是 LangGraph 的“会话记忆”入口。
    # 同一个 thread_id 下，每次 invoke/stream 只需要传入新增消息；
    # LangGraph 会从 checkpointer 取出旧 State，再把新消息追加进去。
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def _build_react_executor_graph(*, call_llm):
    """构建可嵌套的 ReAct 子图，封装模型决策、工具审批和工具执行循环。"""
    builder = StateGraph(AgentState)
    builder.add_node("llm", call_llm)
    builder.add_node("classify_tool_calls", classify_tool_calls_node)
    builder.add_node("approval_gate", approval_gate_node)
    builder.add_node("execute_tools", execute_tools_node)

    builder.add_edge(START, "llm")
    builder.add_conditional_edges(
        "llm",
        _route_after_react_llm,
        {
            "classify_tool_calls": "classify_tool_calls",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "classify_tool_calls",
        _route_after_classify,
        {"approval_gate": "approval_gate", "execute_tools": "execute_tools"},
    )
    builder.add_edge("approval_gate", "execute_tools")
    builder.add_edge("execute_tools", "llm")
    return builder.compile()


def _route_from_start(state: AgentState, config: RunnableConfig) -> str:
    """根据 configurable.use_plan_mode 决定进入计划模式还是原始 ReAct 子图。"""
    return "planner" if _is_plan_mode_enabled(config) else "react_executor"


def _route_after_react_llm(state: AgentState) -> str:
    """根据 ReAct 子图内模型输出决定进入工具状态机或结束子图。"""
    if has_tool_calls(state):
        return "classify_tool_calls"
    return END


def _route_after_react_executor(
    state: AgentState,
    config: RunnableConfig,
    *,
    compact_token_threshold: int,
    recent_messages_to_keep: int,
) -> str:
    """根据计划模式、任务状态和上下文长度决定 ReAct 子图后的主图走向。"""
    if _should_replan_after_react_executor(state, config):
        return "replanner"
    if should_compact_context(
        state,
        token_threshold=compact_token_threshold,
        recent_messages_to_keep=recent_messages_to_keep,
    ):
        return "summarize_and_compact"
    return END


def _route_after_plan_approval(state: AgentState) -> str:
    """根据计划审批状态决定重新规划、开始执行或直接结束。"""
    if state.get("plan_approval_status") == "rejected":
        return "planner"
    if state.get("plan_approval_status") == "approved" and state.get("current_task_id"):
        return "react_executor"
    return END


def _route_after_replanner(state: AgentState) -> str:
    """根据 replanner 是否选出下一项任务决定继续执行或结束计划。"""
    return "react_executor" if state.get("current_task_id") else END


def _route_after_classify(state: AgentState) -> str:
    """根据分类结果决定先人工审批还是直接执行工具。"""
    return "approval_gate" if has_pending_approvals(state) else "execute_tools"


def _is_plan_mode_enabled(runnable_config: RunnableConfig | None) -> bool:
    """读取 LangGraph configurable.use_plan_mode，判断本次运行是否启用计划模式。"""
    if not runnable_config:
        return False
    configurable = runnable_config.get("configurable") or {}
    return bool(configurable.get("use_plan_mode"))


def _should_replan_after_react_executor(
    state: AgentState,
    runnable_config: RunnableConfig | None,
) -> bool:
    """判断 ReAct 子图输出是否属于计划任务结果，是否需要交给 replanner 更新状态。"""
    return (
        _is_plan_mode_enabled(runnable_config)
        and state.get("plan_approval_status") == "approved"
        and bool(state.get("current_task_id"))
    )


def _bind_tools_if_supported(llm: Any, tools: list[Any]) -> Any:
    """为支持 bind_tools 的模型绑定专属计划工具，测试替身不支持时保持原对象。"""
    bind_tools = getattr(llm, "bind_tools", None)
    if callable(bind_tools):
        return bind_tools(tools)
    return llm


def _invoke_until_plan_tool_call(
    llm: Any,
    prompt_messages: list[BaseMessage],
    state: AgentState,
    *,
    write_tool_names: set[str],
) -> dict[str, Any]:
    """调用 planner/replanner 模型，处理只读工具，并返回第一个写状态工具调用。"""
    messages = list(prompt_messages)
    for _ in range(4):
        response = llm.invoke(messages)
        tool_calls = list(getattr(response, "tool_calls", None) or [])
        for tool_call in tool_calls:
            if tool_call.get("name") in write_tool_names:
                return tool_call

        read_results = _build_plan_read_tool_messages(tool_calls, state)
        if not read_results:
            break
        messages.extend([response, *read_results])

    raise RuntimeError("planner/replanner 未返回有效的计划专属工具调用。")


def _build_plan_read_tool_messages(
    tool_calls: list[dict[str, Any]],
    state: AgentState,
) -> list[ToolMessage]:
    """执行计划专属只读工具调用，并返回可继续喂给模型的 ToolMessage。"""
    messages = []
    for tool_call in tool_calls:
        tool_name = tool_call.get("name") or ""
        if tool_name not in PLAN_READ_ONLY_TOOL_NAMES:
            continue
        messages.append(
            ToolMessage(
                content=_plan_read_tool_content(tool_name, state),
                tool_call_id=tool_call.get("id") or tool_name,
                name=tool_name,
            )
        )
    return messages


def _plan_read_tool_content(tool_name: str, state: AgentState) -> str:
    """根据当前状态返回计划只读工具的文本结果。"""
    if tool_name == "get_current_plan":
        return state.get("plan_document", "") or "当前没有计划书。"
    if tool_name == "get_current_task":
        task = _find_current_task(state)
        return json.dumps(task or {}, ensure_ascii=False, indent=2)
    return json.dumps(state.get("plan_tasks", []), ensure_ascii=False, indent=2)


def build_plan_approval_request(state: AgentState) -> dict[str, Any]:
    """构造传给 LangGraph interrupt 的计划审核请求。"""
    return {
        "type": "plan_approval",
        "message": "请审核以下计划书。批准后将按任务顺序逐项执行；拒绝时请给出修改反馈。",
        "plan_document": state.get("plan_document", ""),
        "tasks": state.get("plan_tasks", []),
    }


def _is_plan_approved(decision: Any) -> bool:
    """把计划审核返回值标准化为是否批准计划。"""
    if decision is True:
        return True
    if not isinstance(decision, dict):
        return False
    return bool(decision.get("approved"))


def _extract_plan_feedback(decision: Any) -> str:
    """从计划审核返回值中提取用户反馈文本。"""
    if isinstance(decision, dict) and isinstance(decision.get("feedback"), str):
        return decision["feedback"].strip()
    return "用户拒绝了当前计划，请重新规划。"


def _normalize_plan_tasks(raw_tasks: Any) -> list[PlanTask]:
    """将计划工具提交的任务对象整理为状态机持久化使用的 PlanTask 列表。"""
    tasks: list[PlanTask] = []
    for index, raw_task in enumerate(raw_tasks, start=1):
        if not isinstance(raw_task, dict):
            continue
        status = str(raw_task.get("status", "pending"))
        if status not in VALID_TASK_STATUSES:
            status = "pending"
        task: PlanTask = {
            "id": str(raw_task.get("id") or f"task_{index}"),
            "title": str(raw_task.get("title") or f"任务 {index}"),
            "description": str(raw_task.get("description") or ""),
            "status": status,  # type: ignore[typeddict-item]
            "result": str(raw_task.get("result") or ""),
        }
        tasks.append(task)
    return tasks


def _activate_next_pending_task(tasks: list[PlanTask]) -> tuple[list[PlanTask], str | None]:
    """选择第一个待办任务并标记为执行中，返回更新后的任务列表和任务 ID。"""
    activated_task_id: str | None = None
    updated_tasks: list[PlanTask] = []
    for task in tasks:
        updated_task = dict(task)
        if activated_task_id is None and task["status"] == "pending":
            updated_task["status"] = "in_progress"
            activated_task_id = task["id"]
        updated_tasks.append(updated_task)  # type: ignore[arg-type]
    return updated_tasks, activated_task_id


def _find_current_task(state: AgentState) -> PlanTask | None:
    """根据 current_task_id 从计划任务列表中取出当前执行任务。"""
    current_task_id = state.get("current_task_id")
    for task in state.get("plan_tasks", []):
        if task["id"] == current_task_id:
            return task
    return None


def _update_task_status(
    tasks: list[PlanTask],
    task_id: str,
    status: TaskStatus,
    result: str,
) -> list[PlanTask]:
    """更新指定任务的终态和结果摘要，保持其他任务信息不变。"""
    updated_tasks: list[PlanTask] = []
    for task in tasks:
        updated_task = dict(task)
        if task["id"] == task_id:
            updated_task["status"] = status
            updated_task["result"] = result
        updated_tasks.append(updated_task)  # type: ignore[arg-type]
    return updated_tasks


def _build_active_task_context(state: AgentState) -> str | None:
    """生成注入 ReAct prompt 的当前任务上下文，约束 Executor 一次只处理一个任务。"""
    task = _find_current_task(state)
    if task is None:
        return None
    return (
        "当前处于计划模式，请只处理下面这一项 in_progress 任务。"
        "完成该任务后输出清晰结果，不要自行切换到其他任务。\n\n"
        f"任务 ID：{task['id']}\n"
        f"任务标题：{task['title']}\n"
        f"任务描述：{task['description']}\n"
        f"任务状态：{task['status']}"
    )


def _build_plan_final_summary(tasks: list[PlanTask]) -> str:
    """根据所有任务终态生成计划执行完成后的最终总结。"""
    lines = ["计划执行完成。", "", "任务状态："]
    for index, task in enumerate(tasks, start=1):
        result = task.get("result") or "无结果摘要"
        lines.append(f"{index}. [{task['status']}] {task['title']} - {result}")
    return "\n".join(lines)


def run(
    question: str,
    debug: bool = False,
    checkpoint_db_path: str | Path | None = None,
    use_plan_mode: bool = False,
) -> AIMessage:
    """运行一次性问答；启用 SQLite checkpointer 以支持工具审批和计划审批恢复。"""
    inputs = {"messages": [{"role": "user", "content": question}]}
    # 单次 run 也启用 checkpointer，因为 interrupt/resume 需要 thread_id
    # 找回暂停时的图状态。
    graph_config = {
        "configurable": {
            "thread_id": f"run-{uuid.uuid4()}",
            "use_plan_mode": use_plan_mode,
        }
    }

    with sqlite_checkpointer(checkpoint_db_path) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        return _invoke_graph(graph, inputs, config=graph_config, debug=debug)


def chat(
    thread_id: str = "default",
    debug: bool = False,
    checkpoint_db_path: str | Path | None = None,
    use_plan_mode: bool = False,
) -> None:
    """启动多轮对话；同一个 thread_id 会复用 LangGraph 历史状态。"""
    graph_config = {
        "configurable": {
            "thread_id": thread_id,
            "use_plan_mode": use_plan_mode,
        }
    }

    with sqlite_checkpointer(checkpoint_db_path) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        print("进入多轮对话模式。输入 exit、quit 或 q 结束。")
        print(f"thread_id: {thread_id}")
        print(f"checkpoint_db: {resolve_checkpoint_db_path(checkpoint_db_path)}")
        print(f"plan_mode: {use_plan_mode}")

        while True:
            question = input("\n你: ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                print("已结束多轮对话。")
                return
            if not question:
                continue

            inputs = {"messages": [{"role": "user", "content": question}]}
            final_message = _invoke_graph(
                graph,
                inputs,
                config=graph_config,
                debug=debug,
            )
            if not debug:
                print(f"\n助手: {final_message.content}")


def _invoke_graph(graph, inputs: dict, config: dict | None, debug: bool) -> AIMessage:
    """执行图直到得到最终 AIMessage，并处理 interrupt/resume 审批循环。"""
    if not debug:
        # invoke 会一次性跑到 END；如果遇到 interrupt，则人工确认后用 Command(resume=...)
        # 从中断点恢复。
        next_input: dict | Command = inputs
        while True:
            result = graph.invoke(next_input, config=config)
            interrupts = result.get("__interrupt__")
            if not interrupts:
                return result["messages"][-1]
            next_input = Command(resume=_prompt_for_interrupt_resume(interrupts))

    # debug 模式用 stream 查看每个节点的增量输出。
    # 这对学习 ReAct 很有用：可以看到 llm -> tools -> llm 的实际跳转。
    final_message: AIMessage | None = None
    next_input: dict | Command = inputs
    while True:
        interrupted = False
        for event in graph.stream(next_input, config=config, stream_mode="updates"):
            interrupts = event.get("__interrupt__")
            if interrupts:
                next_input = Command(resume=_prompt_for_interrupt_resume(interrupts))
                interrupted = True
                break

            for node_name, update in event.items():
                print(f"\n[{node_name}]")
                _print_debug_update(update)
                for message in update.get("messages", []):
                    if isinstance(message, AIMessage):
                        final_message = message
        if not interrupted:
            break

    if final_message is None:
        raise RuntimeError("图执行结束，但没有得到 AIMessage。")
    return final_message


def _print_debug_update(update: dict) -> None:
    """打印 debug stream 中一个节点的状态更新。"""
    for message in update.get("messages", []):
        print(_format_message(message))

    total_tokens = update.get("last_total_tokens")
    if total_tokens is not None:
        print(f"last_total_tokens: {total_tokens}")

    summary = update.get("session_summary")
    if summary:
        print(f"session_summary:\n{summary}")

    plan_status = update.get("plan_approval_status")
    if plan_status:
        print(f"plan_approval_status: {plan_status}")

    current_task_id = update.get("current_task_id")
    if current_task_id:
        print(f"current_task_id: {current_task_id}")

    plan_document = update.get("plan_document")
    if plan_document:
        print(f"plan_document:\n{plan_document}")

    plan_tasks = update.get("plan_tasks")
    if plan_tasks:
        print("plan_tasks:")
        print(json.dumps(plan_tasks, ensure_ascii=False, indent=2))

    compaction = update.get("context_compaction")
    if isinstance(compaction, dict):
        print("context_compaction:")
        print(json.dumps(compaction, ensure_ascii=False, indent=2))


def _prompt_for_interrupt_resume(interrupts) -> dict[str, Any]:
    """在 CLI 中展示 interrupt 审批请求，并返回 Command(resume=...) 所需数据。"""
    values = [getattr(interrupt_item, "value", interrupt_item) for interrupt_item in interrupts]
    for value in values:
        if isinstance(value, dict) and value.get("type") == "plan_approval":
            return _prompt_for_plan_approval(value)

    tool_call_ids = []
    for value in values:
        tool_call_ids.extend(
            _print_tool_approval_request(
                value,
                start_index=len(tool_call_ids) + 1,
            )
        )

    try:
        answer = input(
            "批准哪些工具调用？输入 a/yes 全部批准，r/no 全部拒绝，或编号如 1,3: "
        ).strip().lower()
    except (EOFError, OSError):
        answer = ""

    if answer in {"a", "all", "y", "yes"}:
        return {"approved": True}
    if answer in {"r", "reject", "n", "no", ""}:
        return {"approved": False}

    approved_call_ids = _parse_approved_indices(answer, tool_call_ids)
    return {"approved_call_ids": approved_call_ids}


def _prompt_for_plan_approval(value: dict[str, Any]) -> dict[str, Any]:
    """在 CLI 中展示计划书审批请求，并读取用户批准或反馈。"""
    print("\n[plan approval] 需要审核计划书")
    message = value.get("message")
    if message:
        print(message)
    plan_document = value.get("plan_document") or ""
    print("\n计划书:")
    print(plan_document)

    try:
        answer = input("是否批准计划？输入 a/yes 批准，其他内容将作为修改反馈: ").strip()
    except (EOFError, OSError):
        answer = ""

    if answer.lower() in {"a", "all", "y", "yes"}:
        return {"approved": True}
    return {"approved": False, "feedback": answer or "用户拒绝了当前计划。"}


def _print_tool_approval_request(value, start_index: int = 1) -> list[str]:
    """格式化打印 tool_guard 传出的结构化审批请求，并返回展示顺序中的 call id。"""
    print("\n[tool approval] 需要人工审核")
    if not isinstance(value, dict):
        print(value)
        return []

    message = value.get("message")
    if message:
        print(message)
    tool_calls = value.get("tool_calls")
    if not isinstance(tool_calls, list):
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return []

    tool_call_ids = []
    for offset, tool_call in enumerate(tool_calls):
        index = start_index + offset
        tool_call_id = tool_call.get("id")
        if isinstance(tool_call_id, str):
            tool_call_ids.append(tool_call_id)
        print(f"\n{index}. {tool_call.get('name')} id={tool_call_id}")
        reason = tool_call.get("reason")
        if reason:
            print(f"reason: {reason}")
        print("args:")
        print(json.dumps(tool_call.get("args") or {}, ensure_ascii=False, indent=2))
    return tool_call_ids


def _parse_approved_indices(answer: str, tool_call_ids: list[str]) -> list[str]:
    """把用户输入的编号列表转换为 tool_call id 列表。"""
    approved = []
    for raw_part in answer.replace("，", ",").split(","):
        part = raw_part.strip()
        if not part.isdigit():
            continue
        index = int(part)
        if 1 <= index <= len(tool_call_ids):
            approved.append(tool_call_ids[index - 1])
    return approved


def _format_message(message: BaseMessage) -> str:
    """把 LangChain message 格式化为适合 debug 输出的文本。"""
    parts = [f"{message.type}:"]

    reasoning = _extract_reasoning_text(message)
    if reasoning:
        parts.append(f"reasoning:\n{reasoning}")

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        parts.append(f"tool_calls={tool_calls}")

    total_tokens = extract_total_tokens(message)
    if total_tokens is not None:
        parts.append(f"total_tokens={total_tokens}")

    if message.content:
        parts.append(f"content:\n{message.content}")

    return "\n".join(parts)


def _extract_reasoning_text(message: BaseMessage) -> str:
    """兼容不同 OpenAI 类模型返回 reasoning 文本的位置。"""
    candidates = []
    for container_name in ("additional_kwargs", "response_metadata"):
        container = getattr(message, container_name, None)
        if isinstance(container, dict):
            candidates.extend(
                container.get(key)
                for key in (
                    "reasoning_content",
                    "reasoning",
                    "reasoning_text",
                    "thinking",
                    "thought",
                )
            )

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            text = _extract_text_from_reasoning_blocks(value)
            if text:
                return text
    return ""


def _extract_text_from_reasoning_blocks(blocks: list) -> str:
    """从字符串或字典块列表中提取 reasoning 文本。"""
    texts = []
    for block in blocks:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict):
            for key in ("text", "content", "summary"):
                value = block.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
                    break
    return "\n".join(texts).strip()
