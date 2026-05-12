from __future__ import annotations

import json
import uuid

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.types import Command

from langraph_agent.llm import build_llm
from langraph_agent.skills.registry import discover_skills, format_skill_catalog
from langraph_agent.tool_guard import guarded_tool_node


def build_graph(with_memory: bool = False):
    """构建 LangGraph ReAct 状态机，并按需启用 checkpointer。"""
    llm = build_llm()
    skill_catalog = format_skill_catalog(discover_skills())

    def call_llm(state: MessagesState) -> dict[str, list[BaseMessage]]:
        """调用模型，让模型决定直接回答还是请求工具调用。"""
        system_message = {
            "role": "system",
            "content": (
                "你是一个会使用工具的 ReAct agent。"
                "需要实时信息或计算时先调用工具，拿到工具结果后再给最终回答。"
                "\n\n你还具备动态 Skill 能力。启动时你只会看到每个 Skill 的 YAML 元数据。"
                "如果用户请求匹配某个 Skill 的 description，必须先调用 load_skill(skill_name) "
                "读取完整技能说明，再按照该 Skill 回答。"
                "如果不确定有哪些 Skill，可以调用 list_skills。"
                "Skill 是行为说明，不替代工具；需要计算或实时信息时仍然继续调用工具。"
                "\n\n你还可以使用 bash 执行本地命令和 Skill 自带脚本。"
                "执行 Skill 脚本时，先通过 load_skill 读取脚本说明，再用 bash 运行 scripts/ 下的脚本。"
                "不要尝试绕过确认；如果命令被拦截，需要向用户解释原因。"
                "\n\n你还可以使用文件工具查看或修改当前项目文件。"
                "read_file、list_directory、file_search 等只读工具通常会自动执行；"
                "write_file、copy_file、move_file、file_delete、bash 等高风险工具会先请求人工审核。"
                "当用户只要求查看、解释或搜索项目内容时，优先使用只读文件工具。"
                f"\n\n当前可用 Skill 元数据:\n{skill_catalog}"
            ),
        }
        # 这里是 ReAct 中的“Reason/Act 决策”阶段：
        # 模型读取用户问题和历史消息，决定直接回答，还是返回 tool_calls。
        response = llm.invoke([system_message, *state["messages"]])
        return {"messages": [response]}

    # StateGraph 定义“状态如何在节点之间流动”。
    # 这个示例只手写两个节点：llm 和 tools。
    builder = StateGraph(MessagesState)

    # llm 节点：负责调用模型，让模型判断下一步。
    builder.add_node("llm", call_llm)

    # tools 节点：先按白名单判断工具调用，风险工具通过 interrupt 请求人工审批。
    # 审批通过后才真正执行工具，并把结果作为 ToolMessage 追加回 messages。
    builder.add_node("tools", guarded_tool_node)

    # 图从 START 进入 llm。
    builder.add_edge(START, "llm")

    # 条件边是 ReAct 循环的核心：
    # - 如果上一条 AIMessage 有 tool_calls，则路由到 tools。
    # - 如果没有 tool_calls，则路由到 END，表示模型已经给出最终回答。
    builder.add_conditional_edges("llm", tools_condition)

    # 工具执行完以后回到 llm。
    # 模型会看到 ToolMessage，再决定继续调用工具还是输出最终答案。
    builder.add_edge("tools", "llm")

    if not with_memory:
        return builder.compile()

    # checkpointer 是 LangGraph 的“会话记忆”入口。
    # 同一个 thread_id 下，每次 invoke/stream 只需要传入新增消息；
    # LangGraph 会从 checkpointer 取出旧 State，再把新消息追加进去。
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def run(question: str, debug: bool = False) -> AIMessage:
    """运行一次性问答；启用 checkpointer 以支持工具审批恢复。"""
    graph = build_graph(with_memory=True)

    inputs = {"messages": [{"role": "user", "content": question}]}
    # 单次 run 也启用 checkpointer，因为 interrupt/resume 需要 thread_id
    # 找回暂停时的图状态。
    config = {"configurable": {"thread_id": f"run-{uuid.uuid4()}"}}
    return _invoke_graph(graph, inputs, config=config, debug=debug)


def chat(thread_id: str = "default", debug: bool = False) -> None:
    """启动多轮对话；同一个 thread_id 会复用 LangGraph 历史状态。"""
    graph = build_graph(with_memory=True)
    config = {"configurable": {"thread_id": thread_id}}

    print("进入多轮对话模式。输入 exit、quit 或 q 结束。")
    print(f"thread_id: {thread_id}")

    while True:
        question = input("\n你: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("已结束多轮对话。")
            return
        if not question:
            continue

        inputs = {"messages": [{"role": "user", "content": question}]}
        final_message = _invoke_graph(graph, inputs, config=config, debug=debug)
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
                for message in update.get("messages", []):
                    print(_format_message(message))
                    if isinstance(message, AIMessage):
                        final_message = message
        if not interrupted:
            break

    if final_message is None:
        raise RuntimeError("图执行结束，但没有得到 AIMessage。")
    return final_message


def _prompt_for_interrupt_resume(interrupts) -> dict[str, bool]:
    """在 CLI 中展示 interrupt 审批请求，并返回 Command(resume=...) 所需数据。"""
    for interrupt_item in interrupts:
        value = getattr(interrupt_item, "value", interrupt_item)
        _print_tool_approval_request(value)

    try:
        answer = input("是否批准执行以上工具调用？输入 y 或 yes 批准: ").strip().lower()
    except (EOFError, OSError):
        answer = ""
    return {"approved": answer in {"y", "yes"}}


def _print_tool_approval_request(value) -> None:
    """格式化打印 tool_guard 传出的结构化审批请求。"""
    print("\n[tool approval] 需要人工审核")
    if not isinstance(value, dict):
        print(value)
        return

    message = value.get("message")
    if message:
        print(message)
    tool_calls = value.get("tool_calls")
    if not isinstance(tool_calls, list):
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return

    for index, tool_call in enumerate(tool_calls, start=1):
        print(f"\n{index}. {tool_call.get('name')} id={tool_call.get('id')}")
        reason = tool_call.get("reason")
        if reason:
            print(f"reason: {reason}")
        print("args:")
        print(json.dumps(tool_call.get("args") or {}, ensure_ascii=False, indent=2))


def _format_message(message: BaseMessage) -> str:
    """把 LangChain message 格式化为适合 debug 输出的文本。"""
    parts = [f"{message.type}:"]

    reasoning = _extract_reasoning_text(message)
    if reasoning:
        parts.append(f"reasoning:\n{reasoning}")

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        parts.append(f"tool_calls={tool_calls}")

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
