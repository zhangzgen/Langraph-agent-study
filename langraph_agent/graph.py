from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langraph_agent.checkpoints import checkpoint_saver, describe_checkpoint_backend
from langraph_agent.config import config
from langraph_agent.context import (
    build_compacted_messages,
    build_summary_prompt,
    extract_total_tokens,
    should_compact_context,
)
from langraph_agent.llm import build_llm
from langraph_agent.models import AgentState
from langraph_agent.prompt import build_plan_prompt_messages, build_react_prompt_messages
from langraph_agent.skills.registry import discover_skills, format_skill_catalog
from langraph_agent.tool_guard import (
    approval_gate_node,
    build_tool_approval_request,
    classify_tool_calls_node,
    execute_tools_node,
    has_pending_approvals,
    has_tool_calls,
    normalize_approved_call_ids,
    should_auto_approve_tool_call,
)
from langraph_agent.tools import PLAN_TOOLS


def build_graph(
    with_memory: bool = False,
    checkpointer: BaseCheckpointSaver | None = None,
    compact_token_threshold: int = config.COMPACT_TOKEN_THRESHOLD,
    recent_messages_to_keep: int = config.RECENT_MESSAGES_TO_KEEP,
    plan_mode: bool = False,
):
    """构建 LangGraph ReAct 状态机，并按需启用 checkpointer。

    Description:
        创建主执行 ReAct 图；启用 plan_mode 时，在主执行节点前插入计划阶段
        ReAct 循环、计划书人工审核节点和按反馈重写计划的路由。
    Args:
        with_memory (bool): 未显式传入 checkpointer 时，是否启用内存 checkpoint。
        checkpointer (BaseCheckpointSaver | None): 外部传入的 LangGraph checkpoint 实例。
        compact_token_threshold (int): 触发上下文压缩的 token 阈值。
        recent_messages_to_keep (int): 上下文压缩后保留的最近消息数量。
        plan_mode (bool): 是否启用执行前计划模式。
    Returns:
        CompiledStateGraph: 可 invoke 或 stream 的 LangGraph 编译结果。
    """
    llm = build_llm()
    plan_llm = build_llm(tools=PLAN_TOOLS) if plan_mode else None
    summary_llm = build_llm(bind_tools=False)
    skill_catalog = format_skill_catalog(discover_skills())

    def call_llm(state: AgentState) -> dict[str, object]:
        """调用模型，让模型决定直接回答还是请求工具调用。"""
        prompt_messages = build_react_prompt_messages(
            skill_catalog=skill_catalog,
            session_summary=state.get("session_summary"),
            plan_document=state.get("plan_document") if plan_mode else None,
        )
        # 这里是 ReAct 中的“Reason/Act 决策”阶段：
        # 模型读取用户问题和历史消息，决定直接回答，还是返回 tool_calls。
        response = llm.invoke([*prompt_messages, *state["messages"]])
        update: dict[str, object] = {"messages": [response]}
        total_tokens = extract_total_tokens(response)
        if total_tokens is not None:
            update["last_total_tokens"] = total_tokens
        return update

    def call_plan_llm(state: AgentState) -> dict[str, object]:
        """调用计划阶段模型生成澄清问题、读取文件或输出计划书。

        Description:
            使用 plan 专用 prompt 和工具集运行前置 ReAct 决策，模型可以继续调用
            ask_human 或只读文件工具；无工具调用时，其输出会进入计划审核节点。
        Args:
            state (AgentState): 当前全局图状态，包含用户需求、澄清回答和历史消息。
        Returns:
            dict[str, object]: 包含计划阶段 AIMessage 和可选 token 统计的状态更新。
        """
        if plan_llm is None:
            raise RuntimeError("plan_mode 未启用，不能调用计划节点。")
        prompt_messages = build_plan_prompt_messages(
            skill_catalog=skill_catalog,
            session_summary=state.get("session_summary"),
        )
        response = plan_llm.invoke([*prompt_messages, *state["messages"]])
        update: dict[str, object] = {"messages": [response]}
        total_tokens = extract_total_tokens(response)
        if total_tokens is not None:
            update["last_total_tokens"] = total_tokens
        return update

    def execute_plan_tools(state: AgentState) -> dict[str, list[ToolMessage]]:
        """执行计划阶段的工具调用。

        Description:
            按模型返回的 tool_calls 顺序执行 ask_human 与只读文件工具。ask_human
            通过 LangGraph interrupt 暂停并收集用户回答，其他工具直接调用。
        Args:
            state (AgentState): 当前全局图状态，最后一条消息应为包含 tool_calls 的 AIMessage。
        Returns:
            dict[str, list[ToolMessage]]: 每个工具调用对应的 ToolMessage 状态更新。
        """
        last_message = state["messages"][-1]
        tool_calls = list(getattr(last_message, "tool_calls", None) or [])
        plan_tools_by_name = {tool.name: tool for tool in PLAN_TOOLS}
        messages = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name") or ""
            tool_call_id = tool_call.get("id") or tool_name
            args = tool_call.get("args") or {}
            tool = plan_tools_by_name.get(tool_name)
            if tool is None:
                content = f"未知计划工具，未执行: {tool_name}"
            elif tool_name == "ask_human":
                content = str(tool.invoke(args))
            elif not should_auto_approve_tool_call(tool_call):
                decision = interrupt(build_tool_approval_request([tool_call]))
                approved_ids = normalize_approved_call_ids(decision, [tool_call])
                if tool_call_id not in approved_ids:
                    content = f"用户未批准计划工具 {tool_name}，因此该工具调用没有执行。"
                else:
                    try:
                        content = str(tool.invoke(args))
                    except Exception as exc:
                        content = f"计划工具 {tool_name} 执行失败: {exc}"
            else:
                try:
                    content = str(tool.invoke(args))
                except Exception as exc:
                    content = f"计划工具 {tool_name} 执行失败: {exc}"
            messages.append(
                ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)
            )
        return {"messages": messages}

    def review_plan(state: AgentState) -> dict[str, object]:
        """请求用户审核计划书，并根据结果更新全局状态。

        Description:
            将计划阶段最后一条 AIMessage 作为计划书发给用户审核。用户直接回车或
            输入 yes 表示通过；否则把输入作为调整意见返回给计划模型重写计划书。
        Args:
            state (AgentState): 当前全局图状态，最后一条消息应为计划书 AIMessage。
        Returns:
            dict[str, object]: 计划书、审核结果以及供后续节点继续推理的消息更新。
        """
        plan_document = _extract_stream_text(state["messages"][-1]).strip()
        decision = interrupt(
            {
                "type": "plan_review",
                "message": "请审核计划书。直接回车或输入 yes 通过；否则输入调整意见。",
                "plan": plan_document,
            }
        )
        if isinstance(decision, dict) and decision.get("approved"):
            return {
                "plan_document": plan_document,
                "plan_approved": True,
                "messages": [
                    HumanMessage(content="用户已审核通过计划书。请严格按照计划书执行当前任务。")
                ],
            }
        feedback = ""
        if isinstance(decision, dict):
            feedback = str(decision.get("feedback") or "").strip()
        return {
            "plan_document": plan_document,
            "plan_approved": False,
            "messages": [
                HumanMessage(
                    content=(
                        "用户未通过计划书。请根据以下调整意见重新制定计划书："
                        f"{feedback or '用户要求重新调整计划书。'}"
                    )
                )
            ],
        }

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

    # StateGraph 定义“状态如何在节点之间流动”。
    # 这里把工具执行拆成三段，方便观察和扩展审批状态。
    builder = StateGraph(AgentState)

    # llm 节点：负责调用模型，让模型判断下一步。
    builder.add_node("llm", call_llm)
    if plan_mode:
        builder.add_node("plan_llm", call_plan_llm)
        builder.add_node("execute_plan_tools", execute_plan_tools)
        builder.add_node("review_plan", review_plan)
    builder.add_node("classify_tool_calls", classify_tool_calls_node)
    builder.add_node("approval_gate", approval_gate_node)
    builder.add_node("execute_tools", execute_tools_node)
    builder.add_node("summarize_and_compact", summarize_and_compact)

    # 图从 START 进入 llm；启用 plan 时先进入计划阶段。
    if plan_mode:
        builder.add_edge(START, "plan_llm")
        builder.add_conditional_edges(
            "plan_llm",
            lambda state: "execute_plan_tools"
            if has_tool_calls(state)
            else "review_plan",
            {
                "execute_plan_tools": "execute_plan_tools",
                "review_plan": "review_plan",
            },
        )
        builder.add_edge("execute_plan_tools", "plan_llm")
        builder.add_conditional_edges(
            "review_plan",
            lambda state: "llm" if state.get("plan_approved") else "plan_llm",
            {"llm": "llm", "plan_llm": "plan_llm"},
        )
    else:
        builder.add_edge(START, "llm")

    # 如果上一条 AIMessage 有 tool_calls，就进入工具状态机；否则结束。
    builder.add_conditional_edges(
        "llm",
        lambda state: _route_after_llm(
            state,
            compact_token_threshold=compact_token_threshold,
            recent_messages_to_keep=recent_messages_to_keep,
        ),
        {
            "classify_tool_calls": "classify_tool_calls",
            "summarize_and_compact": "summarize_and_compact",
            END: END,
        },
    )

    # 分类后，有待审核工具就暂停请求人工审批；否则直接执行自动批准的工具。
    builder.add_conditional_edges(
        "classify_tool_calls",
        _route_after_classify,
        {"approval_gate": "approval_gate", "execute_tools": "execute_tools"},
    )
    builder.add_edge("approval_gate", "execute_tools")

    # 工具执行完以后回到 llm。模型会看到 ToolMessage，再决定继续调用工具还是输出最终答案。
    builder.add_edge("execute_tools", "llm")
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


def _route_after_llm(
    state: AgentState,
    *,
    compact_token_threshold: int,
    recent_messages_to_keep: int,
) -> str:
    """根据模型输出决定进入工具状态机、压缩上下文或结束。"""
    if has_tool_calls(state):
        return "classify_tool_calls"
    if should_compact_context(
        state,
        token_threshold=compact_token_threshold,
        recent_messages_to_keep=recent_messages_to_keep,
    ):
        return "summarize_and_compact"
    return END


def _route_after_classify(state: AgentState) -> str:
    """根据分类结果决定先人工审批还是直接执行工具。"""
    return "approval_gate" if has_pending_approvals(state) else "execute_tools"


def run(
    question: str,
    debug: bool = False,
    checkpoint_db_path: str | Path | None = None,
    stream_output: bool = True,
    plan_mode: bool = False,
) -> AIMessage:
    """运行一次性问答；启用 checkpoint 以支持工具审批恢复。

    Description:
        使用临时 thread_id 执行一次用户请求。plan_mode 为 True 时，会先进入
        计划阶段澄清和计划书审核，再路由到主执行 ReAct。
    Args:
        question (str): 用户输入的任务需求。
        debug (bool): 是否打印 LangGraph 节点更新。
        checkpoint_db_path (str | Path | None): SQLite checkpoint 路径覆盖值。
        stream_output (bool): 是否流式打印模型正文。
        plan_mode (bool): 是否启用计划阶段。
    Returns:
        AIMessage: 图执行结束后的最终 AI 消息。
    """
    inputs = {"messages": [{"role": "user", "content": question}]}
    # 单次 run 也启用 checkpointer，因为 interrupt/resume 需要 thread_id
    # 找回暂停时的图状态。
    graph_config = {"configurable": {"thread_id": f"run-{uuid.uuid4()}"}}

    with checkpoint_saver(checkpoint_db_path) as checkpointer:
        graph = build_graph(checkpointer=checkpointer, plan_mode=plan_mode)
        return _invoke_graph(
            graph,
            inputs,
            config=graph_config,
            debug=debug,
            stream_output=stream_output,
        )


def chat(
    thread_id: str = "default",
    debug: bool = False,
    checkpoint_db_path: str | Path | None = None,
    stream_output: bool = True,
    plan_mode: bool = False,
) -> None:
    """启动多轮对话；同一个 thread_id 会复用 LangGraph 历史状态。

    Description:
        在命令行中持续读取用户输入，并用同一个 thread_id 复用 checkpoint
        历史。plan_mode 为 True 时，每轮用户需求都会先经过计划审核。
    Args:
        thread_id (str): 多轮会话使用的 checkpoint 会话 ID。
        debug (bool): 是否打印 LangGraph 节点更新。
        checkpoint_db_path (str | Path | None): SQLite checkpoint 路径覆盖值。
        stream_output (bool): 是否流式打印模型正文。
        plan_mode (bool): 是否启用计划阶段。
    Returns:
        None: 该函数直接在 CLI 中打印对话结果。
    """
    graph_config = {"configurable": {"thread_id": thread_id}}

    with checkpoint_saver(checkpoint_db_path) as checkpointer:
        graph = build_graph(checkpointer=checkpointer, plan_mode=plan_mode)
        print("进入多轮对话模式。输入 exit、quit 或 q 结束。")
        print(f"thread_id: {thread_id}")
        print(
            "checkpoint_backend: "
            f"{describe_checkpoint_backend(checkpoint_db_path)}"
        )

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
                stream_output=stream_output,
                stream_prefix="\n助手: ",
            )
            if not debug and not stream_output:
                print(f"\n助手: {final_message.content}")


def stream_answer(
    question: str,
    thread_id: str,
    on_text: Callable[[str], None],
    checkpoint_db_path: str | Path | None = None,
) -> AIMessage:
    """以回调方式流式运行一轮多轮对话。

    Description:
        为非终端渠道运行 LangGraph 对话，复用指定 thread_id 的 checkpoint
        历史，并在模型正文增长时向调用方回传累计文本。
    Args:
        question (str): 当前轮用户输入的问题。
        thread_id (str): 用于复用多轮对话历史的会话标识。
        on_text (Callable[[str], None]): 接收累计回答正文的回调函数。
        checkpoint_db_path (str | Path | None): SQLite checkpoint 路径覆盖值。
    Returns:
        AIMessage: 当前轮图执行结束后的最终 AI 消息。
    """
    inputs = {"messages": [{"role": "user", "content": question}]}
    graph_config = {"configurable": {"thread_id": thread_id}}
    final_message: AIMessage | None = None
    streamed_text = ""

    with checkpoint_saver(checkpoint_db_path) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        next_input: dict | Command = inputs
        while True:
            interrupted = False
            for mode, data in graph.stream(
                next_input,
                config=graph_config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    message, metadata = data
                    if _should_print_stream_message(message, metadata):
                        text = _extract_stream_text(message)
                        if text:
                            streamed_text += text
                            on_text(streamed_text)
                    continue

                if mode != "updates":
                    continue

                if data.get("__interrupt__"):
                    next_input = Command(resume={"approved": False})
                    interrupted = True
                    break

                for update in data.values():
                    for message in update.get("messages", []):
                        if isinstance(message, AIMessage):
                            final_message = message
            if not interrupted:
                break

    if final_message is None:
        raise RuntimeError("图执行结束，但没有得到 AIMessage。")

    final_text = _extract_stream_text(final_message)
    if final_text and final_text != streamed_text:
        on_text(final_text)
    return final_message


def _invoke_graph(
    graph,
    inputs: dict,
    config: dict | None,
    debug: bool,
    stream_output: bool = True,
    stream_prefix: str = "",
) -> AIMessage:
    """执行图直到得到最终 AIMessage，并处理 interrupt/resume 审批循环。"""
    if not debug:
        if stream_output:
            return _stream_graph_to_console(
                graph,
                inputs,
                config=config,
                prefix=stream_prefix,
            )

        next_input: dict | Command = inputs
        while True:
            result = graph.invoke(next_input, config=config)
            interrupts = result.get("__interrupt__")
            if not interrupts:
                return result["messages"][-1]
            next_input = Command(resume=_prompt_for_interrupt_resume(interrupts))

    # debug 模式同时监听 token 和节点更新：既能看到 SSE 风格输出，
    # 也保留 llm -> tools -> llm 的节点跳转视图。
    return _stream_debug_graph_to_console(graph, inputs, config=config)


def _stream_debug_graph_to_console(
    graph,
    inputs: dict,
    config: dict | None,
) -> AIMessage:
    """debug 模式下同时打印模型 delta 和 LangGraph 节点更新。"""
    final_message: AIMessage | None = None
    next_input: dict | Command = inputs
    streaming_text = False
    streamed_for_current_update = False

    while True:
        interrupted = False
        for mode, data in graph.stream(
            next_input,
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                message, metadata = data
                if _should_print_stream_message(message, metadata):
                    text = _extract_stream_text(message)
                    if text:
                        if not streaming_text:
                            print("\n[llm stream]")
                            streaming_text = True
                        print(text, end="", flush=True)
                        streamed_for_current_update = True
                continue

            if mode != "updates":
                continue

            if streaming_text:
                print()
                streaming_text = False

            interrupts = data.get("__interrupt__")
            if interrupts:
                streamed_for_current_update = False
                next_input = Command(resume=_prompt_for_interrupt_resume(interrupts))
                interrupted = True
                break

            for node_name, update in data.items():
                print(f"\n[{node_name}]")
                _print_debug_update(
                    update,
                    suppress_ai_content=streamed_for_current_update,
                )
                for message in update.get("messages", []):
                    if isinstance(message, AIMessage):
                        final_message = message
            streamed_for_current_update = False
        if not interrupted:
            break

    if final_message is None:
        raise RuntimeError("图执行结束，但没有得到 AIMessage。")
    return final_message


def _stream_graph_to_console(
    graph,
    inputs: dict,
    config: dict | None,
    prefix: str = "",
) -> AIMessage:
    """像 OpenAI SDK 的 SSE 示例一样，逐块打印模型 delta。"""
    final_message: AIMessage | None = None
    next_input: dict | Command = inputs
    printed_text = False

    while True:
        interrupted = False
        for mode, data in graph.stream(
            next_input,
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                message, metadata = data
                if _should_print_stream_message(message, metadata):
                    text = _extract_stream_text(message)
                    if text:
                        if not printed_text and prefix:
                            print(prefix, end="", flush=True)
                        print(text, end="", flush=True)
                        printed_text = True
                continue

            if mode != "updates":
                continue

            interrupts = data.get("__interrupt__")
            if interrupts:
                if printed_text:
                    print()
                    printed_text = False
                next_input = Command(resume=_prompt_for_interrupt_resume(interrupts))
                interrupted = True
                break

            for update in data.values():
                for message in update.get("messages", []):
                    if isinstance(message, AIMessage):
                        final_message = message
        if not interrupted:
            break

    if final_message is None:
        raise RuntimeError("图执行结束，但没有得到 AIMessage。")

    if printed_text:
        print()
    else:
        text = _extract_stream_text(final_message)
        if text:
            if prefix:
                print(prefix, end="", flush=True)
            print(text, flush=True)

    return final_message


def _should_print_stream_message(message: BaseMessage, metadata: dict) -> bool:
    """只把面向用户的 llm 节点 token 打到终端，跳过内部摘要等模型调用。"""
    return (
        isinstance(message, AIMessageChunk)
        and metadata.get("langgraph_node") in {"llm", "plan_llm"}
    )


def _extract_stream_text(message: BaseMessage) -> str:
    """提取 chunk 中新增的文本 delta，兼容字符串和 OpenAI 内容块。"""
    content = message.content
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            for key in ("text", "content"):
                value = block.get(key)
                if isinstance(value, str):
                    parts.append(value)
                    break
    return "".join(parts)


def _print_debug_update(
    update: dict,
    suppress_ai_content: bool = False,
) -> None:
    """打印 debug stream 中一个节点的状态更新。"""
    for message in update.get("messages", []):
        print(
            _format_message(
                message,
                suppress_content=suppress_ai_content
                and isinstance(message, AIMessage),
            )
        )

    total_tokens = update.get("last_total_tokens")
    if total_tokens is not None:
        print(f"last_total_tokens: {total_tokens}")

    summary = update.get("session_summary")
    if summary:
        print(f"session_summary:\n{summary}")

    compaction = update.get("context_compaction")
    if isinstance(compaction, dict):
        print("context_compaction:")
        print(json.dumps(compaction, ensure_ascii=False, indent=2))


def _prompt_for_interrupt_resume(interrupts) -> dict[str, object]:
    """在 CLI 中展示 interrupt 审批请求，并返回 Command(resume=...) 所需数据。"""
    tool_call_ids = []
    for interrupt_item in interrupts:
        value = getattr(interrupt_item, "value", interrupt_item)
        if isinstance(value, dict) and value.get("type") == "ask_human":
            return _prompt_for_ask_human(value)
        if isinstance(value, dict) and value.get("type") == "plan_review":
            return _prompt_for_plan_review(value)
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


def _prompt_for_ask_human(value: dict) -> dict[str, str]:
    """在 CLI 中展示 ask_human 提示并收集用户输入。

    Description:
        展示 ask_human 工具已格式化的选择题或说明题文本，使用 input 收集用户
        的选择编号或反馈内容，并作为 Command(resume=...) 数据返回。
    Args:
        value (dict): ask_human 工具通过 interrupt 发出的展示请求。
    Returns:
        dict[str, str]: 包含 answer 字段的中断恢复数据。
    """
    print("\n[ask-human] 计划阶段需要补充信息")
    print(str(value.get("display") or "请补充说明需求。"))
    try:
        answer = input("你的回答: ").strip()
    except (EOFError, OSError):
        answer = ""
    return {"answer": answer}


def _prompt_for_plan_review(value: dict) -> dict[str, object]:
    """在 CLI 中展示计划书并收集审核结论。

    Description:
        处理 plan_review interrupt 请求。用户直接回车或输入 yes 表示通过；
        其他非空输入会作为调整意见返回给计划模型。
    Args:
        value (dict): review_plan 节点发出的结构化请求，包含计划书正文。
    Returns:
        dict[str, object]: 包含 approved 或 feedback 的 Command(resume=...) 数据。
    """
    print("\n[plan review] 需要审核计划书")
    plan = str(value.get("plan") or "").strip()
    if plan:
        print(plan)
    message = value.get("message")
    if message:
        print(str(message))
    try:
        answer = input("审核意见: ").strip()
    except (EOFError, OSError):
        answer = ""
    if answer.lower() in {"", "y", "yes"}:
        return {"approved": True}
    return {"approved": False, "feedback": answer}


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


def _format_message(
    message: BaseMessage,
    suppress_content: bool = False,
) -> str:
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
        if suppress_content:
            parts.append("content: <streamed above>")
        else:
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
