from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from langraph_agent import graph as graph_module
from langraph_agent.tools import PLAN_TOOLS, ask_human


class AskState(TypedDict):
    result: str


def test_ask_human_schema_exposes_choose_list_or_question() -> None:
    """验证模型看到的是约定的 ask_human 入参。

    Description:
        工具 schema 应直接暴露 choose_list 和 question，而不再要求模型构造
        request/kind/choices 等中间结构。
    Args:
        无。
    Returns:
        None: 该测试只通过断言验证行为。
    """
    assert "choose_list" in ask_human.args
    assert "question" in ask_human.args
    assert "request" not in ask_human.args


def test_ask_human_choice_interrupt_formats_and_resumes_selection() -> None:
    """验证选择题会格式化输出并在恢复后解析编号。

    Description:
        通过 LangGraph 执行 ask_human，确认中断载荷包含格式化问题和选项，
        随后用 Command 恢复并将编号转换为实际选项文本。
    Args:
        无。
    Returns:
        None: 该测试只通过断言验证行为。
    """

    def ask_choice(_state: AskState) -> dict[str, str]:
        """执行测试用选择题工具调用。

        Description:
            调用 ask_human 触发图中断，供测试检查中断和恢复行为。
        Args:
            _state (AskState): 当前测试状态，本节点无需读取其中内容。
        Returns:
            dict[str, str]: 包含工具返回 JSON 的状态更新。
        """
        return {
            "result": ask_human.invoke(
                {"choose_list": {"选择优化方向": ["代码架构优化", "性能优化"]}}
            )
        }

    builder = StateGraph(AskState)
    builder.add_node("ask_choice", ask_choice)
    builder.add_edge(START, "ask_choice")
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "choice-test"}}

    interrupted = graph.invoke({"result": ""}, config=config)
    request = interrupted["__interrupt__"][0].value
    resumed = graph.invoke(Command(resume={"answer": "2"}), config=config)

    assert request["display"] == "选择优化方向\n1. 代码架构优化\n2. 性能优化"
    assert resumed["result"] == '{"type": "ask_human_answer", "answer": "性能优化"}'


def test_ask_human_question_interrupt_resumes_feedback() -> None:
    """验证说明题会接收用户反馈文本。

    Description:
        开放式提问应直接展示 question 文本，并将中断恢复后的反馈内容以
        结构化 JSON 返回给计划模型。
    Args:
        无。
    Returns:
        None: 该测试只通过断言验证行为。
    """

    def ask_question(_state: AskState) -> dict[str, str]:
        """执行测试用说明题工具调用。

        Description:
            调用 ask_human 触发开放式问题中断，供测试检查恢复结果。
        Args:
            _state (AskState): 当前测试状态，本节点无需读取其中内容。
        Returns:
            dict[str, str]: 包含工具返回 JSON 的状态更新。
        """
        return {"result": ask_human.invoke({"question": "请说明验收标准"})}

    builder = StateGraph(AskState)
    builder.add_node("ask_question", ask_question)
    builder.add_edge(START, "ask_question")
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "question-test"}}

    interrupted = graph.invoke({"result": ""}, config=config)
    request = interrupted["__interrupt__"][0].value
    resumed = graph.invoke(Command(resume={"answer": "测试全部通过"}), config=config)

    assert request["display"] == "请说明验收标准"
    assert resumed["result"] == '{"type": "ask_human_answer", "answer": "测试全部通过"}'


def test_prompt_for_ask_human_reads_terminal_input(monkeypatch, capsys) -> None:
    """验证 CLI 中断处理使用 input 接收选择或反馈。

    Description:
        CLI 需要展示工具生成的格式化文本，并把终端输入包装为中断恢复数据。
    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。
        capsys (pytest.CaptureFixture[str]): pytest 提供的标准输出捕获工具。
    Returns:
        None: 该测试只通过断言验证行为。
    """
    monkeypatch.setattr("builtins.input", lambda _prompt: "1")

    result = graph_module._prompt_for_ask_human(
        {"type": "ask_human", "display": "选择方向\n1. 性能优化"}
    )

    assert result == {"answer": "1"}
    assert "选择方向\n1. 性能优化" in capsys.readouterr().out


def test_plan_mode_routes_approved_plan_to_execution(monkeypatch) -> None:
    """验证 plan 模式审核通过后会进入主执行 ReAct。

    Description:
        使用假模型构造计划书输出和执行阶段输出，确认同一个图会先在
        review_plan 中断，审核通过后继续路由到 llm 节点。
    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。
    Returns:
        None: 该测试只通过断言验证行为。
    """

    class FakeLLM:
        def __init__(self, response: AIMessage) -> None:
            """初始化测试用假模型。

            Description:
                保存固定响应，供图执行时模拟不同阶段的模型输出。
            Args:
                response (AIMessage): 当前假模型每次调用时返回的消息。
            Returns:
                None: 初始化方法不返回值。
            """
            self.response = response

        def invoke(self, _messages):
            """返回预置模型响应。

            Description:
                模拟 LangChain 聊天模型的 invoke 接口，不解析输入消息。
            Args:
                _messages: 图节点传入的 prompt 和历史消息。
            Returns:
                AIMessage: 初始化时注入的固定响应。
            """
            return self.response

    plan_llm = FakeLLM(AIMessage(content="执行计划书\n1. 读取文件\n2. 修改实现"))
    agent_llm = FakeLLM(AIMessage(content="执行完成"))
    summary_llm = FakeLLM(AIMessage(content="摘要"))

    def fake_build_llm(*, bind_tools: bool = True, tools=None):
        """按工具参数返回对应阶段的假模型。

        Description:
            区分计划模型、执行模型和摘要模型，验证 plan 图的路由行为。
        Args:
            bind_tools (bool): 是否模拟绑定工具的模型。
            tools: 调用方传入的工具列表。
        Returns:
            FakeLLM: 对应阶段的假模型实例。
        """
        if not bind_tools:
            return summary_llm
        if tools == PLAN_TOOLS:
            return plan_llm
        return agent_llm

    monkeypatch.setattr(graph_module, "build_llm", fake_build_llm)
    monkeypatch.setattr(graph_module, "discover_skills", lambda: [])

    graph = graph_module.build_graph(checkpointer=MemorySaver(), plan_mode=True)
    config = {"configurable": {"thread_id": "plan-test"}}
    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="加 plan 模式")]},
        config=config,
    )

    assert interrupted["__interrupt__"][0].value["type"] == "plan_review"

    resumed = graph.invoke(Command(resume={"approved": True}), config=config)

    assert resumed["plan_approved"] is True
    assert resumed["plan_document"].startswith("执行计划书")
    assert resumed["messages"][-1].content == "执行完成"
