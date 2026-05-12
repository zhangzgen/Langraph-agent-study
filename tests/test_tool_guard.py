from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.types import Command

from langraph_agent import tool_guard
from langraph_agent.tools import FILESYSTEM_TOOLS
from langraph_agent.tools import shell


def test_filesystem_tools_are_registered() -> None:
    names = {tool.name for tool in FILESYSTEM_TOOLS}

    assert {
        "copy_file",
        "file_delete",
        "file_search",
        "list_directory",
        "move_file",
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
        "args": {"file_path": "notes.txt", "text": "hello"},
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


def test_guarded_tool_node_interrupts_and_resumes_with_rejection() -> None:
    def request_write_file(_state: MessagesState) -> dict[str, list[AIMessage]]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_write",
                            "name": "write_file",
                            "args": {"file_path": "tmp.txt", "text": "hello"},
                        }
                    ],
                )
            ]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("llm", request_write_file)
    builder.add_node("tools", tool_guard.guarded_tool_node)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", "tools")
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "approval-test"}}

    interrupted = graph.invoke({"messages": []}, config=config)

    assert "__interrupt__" in interrupted
    approval_request = interrupted["__interrupt__"][0].value
    assert approval_request["type"] == "tool_approval"
    assert approval_request["tool_calls"][0]["name"] == "write_file"

    resumed = graph.invoke(Command(resume={"approved": False}), config=config)

    assert "用户未批准执行工具 write_file" in resumed["messages"][-1].content
