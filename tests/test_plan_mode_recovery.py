from __future__ import annotations

import json
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from langraph_agent.tools import planning as planning_module
from langraph_agent.tools import ask_human


class AskState(TypedDict):
    result: str


def test_ask_human_multiple_choices_accepts_serialized_dictionary() -> None:
    """验证多道选择题会逐题中断并聚合回答。

    Description:
        模拟模型把包含多道问题的 choose_list 作为 JSON 字符串输出，确认工具
        能恢复为字典，并按问题顺序逐题中断收集用户选择。
    Args:
        无。
    Returns:
        None: 该测试只通过断言验证行为。
    """

    def ask_multiple_choices(_state: AskState) -> dict[str, str]:
        """执行测试用多选择题工具调用。

        Description:
            通过字符串化的 choose_list 调用 ask_human，复现模型真实返回形态。
        Args:
            _state (AskState): 当前测试状态，本节点无需读取其中内容。
        Returns:
            dict[str, str]: 包含工具返回 JSON 的状态更新。
        """
        choose_list = {
            "选择版本": ["基础版", "增强版", "两个都优化"],
            "选择方向": ["架构", "玩法", "视觉"],
        }
        return {
            "result": ask_human.invoke(
                {"choose_list": json.dumps(choose_list, ensure_ascii=False)}
            )
        }

    builder = StateGraph(AskState)
    builder.add_node("ask_multiple_choices", ask_multiple_choices)
    builder.add_edge(START, "ask_multiple_choices")
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "multiple-choice-test"}}

    interrupted = graph.invoke({"result": ""}, config=config)
    first_request = interrupted["__interrupt__"][0].value
    second_interrupted = graph.invoke(Command(resume={"answer": "1"}), config=config)
    second_request = second_interrupted["__interrupt__"][0].value
    resumed = graph.invoke(Command(resume={"answer": "2"}), config=config)

    assert first_request["display"] == "选择版本\n1. 基础版\n2. 增强版\n3. 两个都优化"
    assert first_request["current"] == 1
    assert first_request["total"] == 2
    assert first_request["questions"]["选择版本"] == ["基础版", "增强版", "两个都优化"]
    assert first_request["questions"]["选择方向"] == ["架构", "玩法", "视觉"]
    assert second_request["display"] == "选择方向\n1. 架构\n2. 玩法\n3. 视觉"
    assert second_request["current"] == 2
    assert second_request["total"] == 2
    assert second_request["questions"]["选择版本"] == ["基础版", "增强版", "两个都优化"]
    assert second_request["questions"]["选择方向"] == ["架构", "玩法", "视觉"]
    assert resumed["result"] == (
        '{"type": "ask_human_answer", "answer": "选择版本: 基础版；选择方向: 玩法"}'
    )


def test_ask_human_validation_error_returns_tool_feedback() -> None:
    """验证非法选择题参数不会使计划流程抛出异常。

    Description:
        传入无法解析为字典的 choose_list 字符串，确认 StructuredTool 将
        Pydantic 校验异常转换为计划模型可以读取并据此重试的文本结果。
    Args:
        无。
    Returns:
        None: 该测试只通过断言验证行为。
    """
    result = ask_human.invoke({"choose_list": "not-json"})

    assert "ask_human 参数校验失败" in result
    assert "Input should be a valid dictionary" in result


def test_ask_human_runtime_error_returns_tool_feedback(monkeypatch) -> None:
    """验证提问执行异常会返回工具反馈但不抛出。

    Description:
        模拟交互函数发生非 LangGraph 中断异常，确认工具直接返回带异常类型的
        文本供计划模型读取，而不会终止当前执行链路。
    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。
    Returns:
        None: 该测试只通过断言验证行为。
    """

    def fail_interrupt(_request: dict[str, object]) -> None:
        """模拟计划工具执行时出现运行异常。

        Description:
            在工具准备暂停交互时主动抛出 RuntimeError，覆盖错误反馈路径。
        Args:
            _request (dict[str, object]): ask_human 准备发送的中断载荷。
        Returns:
            None: 本测试替身始终抛出异常而不会正常返回。
        """
        raise RuntimeError("terminal unavailable")

    monkeypatch.setattr(planning_module, "interrupt", fail_interrupt)

    result = ask_human.invoke({"question": "请说明范围"})

    assert result == "ask_human 执行失败，请修正参数或重试: RuntimeError: terminal unavailable"
