from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
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


def test_mcp_tool_requires_review_by_default() -> None:
    """验证未显式授权的 MCP 工具默认需要人工审核。

    Description:
        模拟 MCP fetch 工具调用，确认它不在默认自动审批白名单中时不会自动执行。
    Args:
        None: 该测试不接收外部参数。
    Returns:
        None: 该测试通过断言验证审批策略。
    """
    tool_call = {
        "id": "call_fetch",
        "name": "fetch",
        "args": {"url": "https://example.com"},
    }

    assert not tool_guard.should_auto_approve_tool_call(tool_call)


def test_explicit_safe_mcp_tool_can_be_auto_approved() -> None:
    """验证配置显式授权的 MCP 工具可以自动审批。

    Description:
        模拟 time MCP 的只读工具调用，确认传入额外白名单后可沿用自动执行链路。
    Args:
        None: 该测试不接收外部参数。
    Returns:
        None: 该测试通过断言验证审批策略。
    """
    tool_call = {
        "id": "call_time",
        "name": "get_current_time",
        "args": {"timezone": "Asia/Shanghai"},
    }

    assert tool_guard.should_auto_approve_tool_call(
        tool_call,
        auto_approved_tools={"get_current_time"},
    )


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


def test_execute_tools_node_supports_async_only_tools() -> None:
    """验证执行器兼容 MCP 风格异步工具。

    Description:
        构造只支持 ainvoke 的 StructuredTool，确认运行时工具执行节点会自动切换到异步调用路径。
    Args:
        None: 该测试不接收外部参数。
    Returns:
        None: 该测试通过断言验证工具执行结果。
    """

    async def async_echo(text: str) -> str:
        """返回输入文本。

        Description:
            模拟 MCP adapter 生成的异步 LangChain 工具。
        Args:
            text (str): 需要原样返回的文本。
        Returns:
            str: 原始输入文本。
        """
        return f"async:{text}"

    async_tool = StructuredTool.from_function(
        coroutine=async_echo,
        name="async_echo",
        description="Echo text asynchronously.",
    )
    execute_node = tool_guard.build_execute_tools_node([async_tool])

    result = execute_node(
        {
            "messages": [],
            "approved_tool_calls": [
                {
                    "id": "call_async",
                    "name": "async_echo",
                    "args": {"text": "ok"},
                }
            ],
            "rejected_tool_calls": [],
            "pending_approvals": [],
            "tool_audit_log": [],
        }
    )

    assert result["messages"][0].content == "async:ok"
    assert result["tool_audit_log"][-1]["status"] == "executed"


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
