from __future__ import annotations

import pytest

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command


class FakeLLM:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.calls += 1
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


def _patch_graph_llms(monkeypatch, *, agent_llm: FakeLLM, planner_llm: FakeLLM, replanner_llm: FakeLLM) -> None:
    from langraph_agent import graph as graph_module

    false_llms = iter([FakeLLM([AIMessage(content="summary")]), planner_llm, replanner_llm])

    def fake_build_llm(*, bind_tools: bool = True):
        return agent_llm if bind_tools else next(false_llms)

    monkeypatch.setattr(graph_module, "build_llm", fake_build_llm)
    monkeypatch.setattr(graph_module, "discover_skills", lambda: [])
    monkeypatch.setattr(graph_module, "build_react_prompt_messages", lambda **_kwargs: [])


def _tool_response(tool_name: str, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": f"call_{tool_name}",
                "name": tool_name,
                "args": args,
            }
        ],
    )


def _planner_response(tasks: list[dict], plan_document: str = "测试计划书") -> AIMessage:
    return _tool_response(
        "create_plan",
        {"tasks": tasks, "plan_document": plan_document},
    )


def _replanner_response(tool_name: str, task_id: str, text: str) -> AIMessage:
    arg_name = "result" if tool_name == "complete_task" else "reason"
    return _tool_response(tool_name, {"task_id": task_id, arg_name: text})


def test_plan_mode_disabled_uses_original_react_path(monkeypatch) -> None:
    from langraph_agent import graph as graph_module

    agent_llm = FakeLLM([AIMessage(content="react answer")])
    _patch_graph_llms(
        monkeypatch,
        agent_llm=agent_llm,
        planner_llm=FakeLLM([_planner_response([])]),
        replanner_llm=FakeLLM([_replanner_response("complete_task", "task_1", "ok")]),
    )

    graph = graph_module.build_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content="hello")]},
        config={"configurable": {"use_plan_mode": False}},
    )

    assert result["messages"][-1].content == "react answer"
    assert "plan_tasks" not in result
    assert agent_llm.calls == 1


def test_planner_generates_plan_and_interrupts_for_review(monkeypatch) -> None:
    from langraph_agent import graph as graph_module

    _patch_graph_llms(
        monkeypatch,
        agent_llm=FakeLLM([AIMessage(content="unused")]),
        planner_llm=FakeLLM([
            _planner_response(
                [
                    {
                        "id": "task_1",
                        "title": "分析需求",
                        "description": "梳理用户目标",
                        "status": "pending",
                        "result": "",
                    }
                ],
                "1. 先分析需求",
            )
        ]),
        replanner_llm=FakeLLM([_replanner_response("complete_task", "task_1", "ok")]),
    )

    graph = graph_module.build_graph(checkpointer=MemorySaver())
    result = graph.invoke(
        {"messages": [HumanMessage(content="做一个计划")]},
        config={"configurable": {"thread_id": "plan-review", "use_plan_mode": True}},
    )

    assert result["plan_approval_status"] == "pending_review"
    assert result["plan_document"] == "1. 先分析需求"
    assert result["plan_tasks"][0]["status"] == "pending"
    assert result["__interrupt__"]
    assert result["__interrupt__"][0].value["type"] == "plan_approval"


def test_plan_approval_activates_first_task_before_tool_review(monkeypatch) -> None:
    from langraph_agent import graph as graph_module

    _patch_graph_llms(
        monkeypatch,
        agent_llm=FakeLLM([
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_bash",
                        "name": "bash",
                        "args": {"command": "echo ok"},
                    }
                ],
            )
        ]),
        planner_llm=FakeLLM([
            _planner_response(
                [
                    {
                        "id": "task_1",
                        "title": "执行命令",
                        "description": "运行命令并观察结果",
                        "status": "pending",
                        "result": "",
                    }
                ]
            )
        ]),
        replanner_llm=FakeLLM([_replanner_response("complete_task", "task_1", "ok")]),
    )

    graph = graph_module.build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "plan-approved", "use_plan_mode": True}}
    graph.invoke({"messages": [HumanMessage(content="执行一个任务")]}, config=config)
    result = graph.invoke(Command(resume={"approved": True}), config=config)

    assert result["plan_approval_status"] == "approved"
    assert result["current_task_id"] == "task_1"
    assert result["plan_tasks"][0]["status"] == "in_progress"
    assert result["__interrupt__"][0].value["type"] == "tool_approval"


def test_replanner_marks_completed_and_finishes_plan(monkeypatch) -> None:
    from langraph_agent import graph as graph_module

    _patch_graph_llms(
        monkeypatch,
        agent_llm=FakeLLM([AIMessage(content="任务已经完成")]),
        planner_llm=FakeLLM([
            _planner_response(
                [
                    {
                        "id": "task_1",
                        "title": "完成任务",
                        "description": "输出结果",
                        "status": "pending",
                        "result": "",
                    }
                ]
            )
        ]),
        replanner_llm=FakeLLM([
            _replanner_response("complete_task", "task_1", "已完成输出")
        ]),
    )

    graph = graph_module.build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "plan-complete", "use_plan_mode": True}}
    graph.invoke({"messages": [HumanMessage(content="执行计划")]}, config=config)
    result = graph.invoke(Command(resume={"approved": True}), config=config)

    assert result["current_task_id"] is None
    assert result["plan_tasks"][0]["status"] == "completed"
    assert result["plan_tasks"][0]["result"] == "已完成输出"
    assert "计划执行完成" in result["messages"][-1].content


@pytest.mark.parametrize("status,tool_name", [("failed", "fail_task"), ("skipped", "skip_task")])
def test_replanner_supports_failed_and_skipped_terminal_statuses(
    monkeypatch,
    status: str,
    tool_name: str,
) -> None:
    from langraph_agent import graph as graph_module

    _patch_graph_llms(
        monkeypatch,
        agent_llm=FakeLLM([AIMessage(content="任务进入终态")]),
        planner_llm=FakeLLM([
            _planner_response(
                [
                    {
                        "id": "task_1",
                        "title": "终态任务",
                        "description": "验证终态",
                        "status": "pending",
                        "result": "",
                    }
                ]
            )
        ]),
        replanner_llm=FakeLLM([
            _replanner_response(tool_name, "task_1", f"任务状态为 {status}")
        ]),
    )

    graph = graph_module.build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": f"plan-{status}", "use_plan_mode": True}}
    graph.invoke({"messages": [HumanMessage(content="执行计划")]}, config=config)
    result = graph.invoke(Command(resume={"approved": True}), config=config)

    assert result["current_task_id"] is None
    assert result["plan_tasks"][0]["status"] == status
    assert result["plan_tasks"][0]["result"] == f"任务状态为 {status}"


def test_planner_can_read_existing_tasks_before_creating_plan(monkeypatch) -> None:
    from langraph_agent import graph as graph_module

    planner_llm = FakeLLM([
        _tool_response("get_plan_tasks", {}),
        _planner_response(
            [
                {
                    "id": "task_1",
                    "title": "读取后规划",
                    "description": "先读取已有任务再创建计划",
                    "status": "pending",
                    "result": "",
                }
            ],
            "读取已有任务后生成计划",
        ),
    ])
    _patch_graph_llms(
        monkeypatch,
        agent_llm=FakeLLM([AIMessage(content="unused")]),
        planner_llm=planner_llm,
        replanner_llm=FakeLLM([_replanner_response("complete_task", "task_1", "ok")]),
    )

    graph = graph_module.build_graph(checkpointer=MemorySaver())
    result = graph.invoke(
        {"messages": [HumanMessage(content="做一个计划")]},
        config={"configurable": {"thread_id": "plan-read-tools", "use_plan_mode": True}},
    )

    assert planner_llm.calls == 2
    assert result["plan_document"] == "读取已有任务后生成计划"
    assert result["plan_tasks"][0]["title"] == "读取后规划"
