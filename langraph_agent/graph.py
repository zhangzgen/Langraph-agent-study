from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langraph_agent.llm import build_llm
from langraph_agent.skills.registry import discover_skills, format_skill_catalog
from langraph_agent.tools import TOOLS


def build_graph(with_memory: bool = False):
    llm = build_llm()
    skill_catalog = format_skill_catalog(discover_skills())

    def call_llm(state: MessagesState) -> dict[str, list[BaseMessage]]:
        # MessagesState 是 LangGraph 内置 State，结构类似：
        # {"messages": [HumanMessage, AIMessage, ToolMessage, ...]}
        # 每个 node 只需要返回要追加到 state 的消息。
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

    # tools 节点：LangGraph 预置节点，负责执行 AIMessage.tool_calls。
    # 执行结果会变成 ToolMessage 追加回 messages。
    builder.add_node("tools", ToolNode(TOOLS))

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
    graph = build_graph()

    # 输入状态只需要包含用户消息。
    # 后续 AIMessage 和 ToolMessage 都由图中的节点自动追加。
    inputs = {"messages": [{"role": "user", "content": question}]}
    return _invoke_graph(graph, inputs, config=None, debug=debug)


def chat(thread_id: str = "default", debug: bool = False) -> None:
    # 多轮对话和单次调用的主要区别：
    # 1. graph 编译时启用 checkpointer。
    # 2. 每一轮调用都传同一个 thread_id。
    # 3. 输入只传本轮用户新消息，历史消息由 LangGraph 自动恢复。
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
    # 单次问答和多轮问答都可以复用这个执行函数。
    # debug=True 时打印每个 node 的增量输出，方便观察 ReAct 路由。
    if not debug:
        # invoke 会一次性跑完整张图，直到 END。
        result = graph.invoke(inputs, config=config)
        return result["messages"][-1]

    # debug 模式用 stream 查看每个节点的增量输出。
    # 这对学习 ReAct 很有用：可以看到 llm -> tools -> llm 的实际跳转。
    final_message: AIMessage | None = None
    for event in graph.stream(inputs, config=config, stream_mode="updates"):
        for node_name, update in event.items():
            print(f"\n[{node_name}]")
            for message in update.get("messages", []):
                print(_format_message(message))
                if isinstance(message, AIMessage):
                    final_message = message

    if final_message is None:
        raise RuntimeError("图执行结束，但没有得到 AIMessage。")
    return final_message


def _format_message(message: BaseMessage) -> str:
    # debug 打印时尽量把“模型推理字段 / 工具调用 / 最终内容”分开。
    # 不同 OpenAI 兼容模型暴露 reasoning 的字段名可能不同，所以这里做兼容式读取。
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
    # 一些模型会把“思考内容”放在 additional_kwargs 或 response_metadata。
    # 如果没有这些字段，就返回空字符串，debug 输出中也不会展示 reasoning 区块。
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
