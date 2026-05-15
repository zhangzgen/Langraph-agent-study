from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from langraph_agent import tool_guard
from langraph_agent.models import AgentState
from langraph_agent.tools import FILESYSTEM_TOOLS
from langraph_agent.tools import shell


def test_filesystem_tools_are_registered() -> None:
    names = {tool.name for tool in FILESYSTEM_TOOLS}

    assert {
        "edit_file",
        "glob",
        "grep",
        "ls",
        "read_file",
        "write_file",
    } <= names


def test_read_file_is_auto_approved_for_normal_project_path() -> None:
    tool_call = {
        "id": "call_read",
        "name": "read_file",
        "args": {"file_path": "README.md"},
    }

    assert tool_guard.should_auto_approve_tool_call(tool_call)


def test_read_file_requires_review_for_sensitive_path() -> None:
    tool_call = {
        "id": "call_env",
        "name": "read_file",
        "args": {"file_path": ".env"},
    }

    assert not tool_guard.should_auto_approve_tool_call(tool_call)


def test_write_file_requires_review() -> None:
    tool_call = {
        "id": "call_write",
        "name": "write_file",
        "args": {"file_path": "notes.txt", "content": "hello"},
    }

    assert not tool_guard.should_auto_approve_tool_call(tool_call)


def test_edit_file_requires_review() -> None:
    tool_call = {
        "id": "call_edit",
        "name": "edit_file",
        "args": {
            "file_path": "notes.txt",
            "old_string": "hello",
            "new_string": "hi",
        },
    }

    assert not tool_guard.should_auto_approve_tool_call(tool_call)


def test_normalize_approved_call_ids_supports_approve_all() -> None:
    review_required = [
        {"id": "call_1", "name": "write_file", "args": {}},
        {"id": "call_2", "name": "bash", "args": {}},
    ]

    assert tool_guard.normalize_approved_call_ids(
        {"approved": True},
        review_required,
    ) == {"call_1", "call_2"}


def test_normalize_approved_call_ids_supports_specific_ids() -> None:
    review_required = [
        {"id": "call_1", "name": "write_file", "args": {}},
        {"id": "call_2", "name": "bash", "args": {}},
    ]

    assert tool_guard.normalize_approved_call_ids(
        {"approved_call_ids": ["call_2", "unknown"]},
        review_required,
    ) == {"call_2"}


def test_graph_approved_bash_execution_skips_interactive_confirmation(
    monkeypatch,
) -> None:
    def fail_if_called(tool_name: str, command: str) -> bool:
        raise AssertionError("graph-approved bash should not prompt again")

    monkeypatch.setattr(shell, "_confirm_execution", fail_if_called)

    with shell.graph_approved_execution():
        result = shell.bash.invoke({"command": "printf hello", "timeout_seconds": 1})

    assert "exit_code: 0" in result
    assert "stdout:\nhello" in result


def test_tool_approval_state_machine_interrupts_and_resumes_with_rejection() -> None:
    def request_write_file(_state: AgentState) -> dict[str, list[AIMessage]]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_write",
                            "name": "write_file",
                            "args": {"file_path": "tmp.txt", "content": "hello"},
                        }
                    ],
                )
            ]
        }

    builder = StateGraph(AgentState)
    builder.add_node("llm", request_write_file)
    builder.add_node("classify_tool_calls", tool_guard.classify_tool_calls_node)
    builder.add_node("approval_gate", tool_guard.approval_gate_node)
    builder.add_node("execute_tools", tool_guard.execute_tools_node)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", "classify_tool_calls")
    builder.add_edge("classify_tool_calls", "approval_gate")
    builder.add_edge("approval_gate", "execute_tools")
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "approval-test"}}

    interrupted = graph.invoke({"messages": []}, config=config)

    assert "__interrupt__" in interrupted
    approval_request = interrupted["__interrupt__"][0].value
    assert approval_request["type"] == "tool_approval"
    assert approval_request["tool_calls"][0]["name"] == "write_file"

    resumed = graph.invoke(Command(resume={"approved": False}), config=config)

    assert "用户未批准执行工具 write_file" in resumed["messages"][-1].content
    assert resumed["tool_audit_log"][-1]["status"] == "rejected"


def test_tool_approval_state_machine_supports_partial_approval() -> None:
    def request_two_tools(_state: AgentState) -> dict[str, list[AIMessage]]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_write",
                            "name": "write_file",
                            "args": {"file_path": "tmp.txt", "content": "hello"},
                        },
                        {
                            "id": "call_bash",
                            "name": "bash",
                            "args": {"command": "printf approved", "timeout_seconds": 1},
                        },
                    ],
                )
            ]
        }

    builder = StateGraph(AgentState)
    builder.add_node("llm", request_two_tools)
    builder.add_node("classify_tool_calls", tool_guard.classify_tool_calls_node)
    builder.add_node("approval_gate", tool_guard.approval_gate_node)
    builder.add_node("execute_tools", tool_guard.execute_tools_node)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", "classify_tool_calls")
    builder.add_edge("classify_tool_calls", "approval_gate")
    builder.add_edge("approval_gate", "execute_tools")
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "partial-approval-test"}}

    interrupted = graph.invoke({"messages": []}, config=config)
    approval_request = interrupted["__interrupt__"][0].value

    assert [item["id"] for item in approval_request["tool_calls"]] == [
        "call_write",
        "call_bash",
    ]

    resumed = graph.invoke(
        Command(resume={"approved_call_ids": ["call_bash"]}),
        config=config,
    )

    tool_messages = resumed["messages"][-2:]
    assert "stdout:\napproved" in tool_messages[0].content
    assert "用户未批准执行工具 write_file" in tool_messages[1].content
    assert [entry["status"] for entry in resumed["tool_audit_log"][-3:]] == [
        "rejected",
        "approved",
        "executed",
    ]
