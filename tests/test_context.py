from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from langraph_agent.context import (
    TOOL_RESULT_PLACEHOLDER,
    build_compacted_messages,
    build_tool_result_placeholder_messages,
    extract_total_tokens,
    select_recent_messages,
    should_compact_context,
    should_expire_tool_results,
)


def test_extract_total_tokens_prefers_usage_metadata() -> None:
    message = AIMessage(
        content="ok",
        response_metadata={"token_usage": {"total_tokens": 10}},
        usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
    )

    assert extract_total_tokens(message) == 25


def test_should_compact_context_requires_final_ai_message_over_threshold() -> None:
    state = {
        "messages": [
            HumanMessage(content=f"message {index}") for index in range(4)
        ]
        + [AIMessage(content="final")],
        "last_total_tokens": 100,
    }

    assert should_compact_context(
        state,
        token_threshold=50,
        recent_messages_to_keep=2,
    )


def test_should_compact_context_skips_pending_tool_calls() -> None:
    state = {
        "messages": [
            HumanMessage(content="run command"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "bash",
                        "args": {"command": "pwd"},
                    }
                ],
            ),
        ],
        "last_total_tokens": 100,
    }

    assert not should_compact_context(
        state,
        token_threshold=50,
        recent_messages_to_keep=1,
    )


def test_should_expire_tool_results_requires_expired_model_time_and_assistant() -> None:
    state = {
        "messages": [
            HumanMessage(content="run"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "bash", "args": {}}],
            ),
            ToolMessage(content="secret output", tool_call_id="call_1", name="bash"),
            HumanMessage(content="continue"),
        ],
        "last_model_or_turn_at": 100.0,
    }

    assert should_expire_tool_results(state, ttl_seconds=300, now=401.0)
    assert not should_expire_tool_results(state, ttl_seconds=300, now=399.0)

    state_without_assistant = {
        "messages": [
            HumanMessage(content="run"),
            ToolMessage(content="secret output", tool_call_id="call_1", name="bash"),
        ],
        "last_model_or_turn_at": 100.0,
    }
    assert not should_expire_tool_results(
        state_without_assistant,
        ttl_seconds=300,
        now=401.0,
    )


def test_build_tool_result_placeholder_messages_replaces_all_tool_results() -> None:
    messages = [
        HumanMessage(content="run"),
        AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "bash", "args": {}}],
        ),
        ToolMessage(content="first", tool_call_id="call_1", name="bash"),
        AIMessage(
            content="",
            tool_calls=[{"id": "call_2", "name": "calculator", "args": {}}],
        ),
        ToolMessage(content="second", tool_call_id="call_2", name="calculator"),
    ]

    compacted = add_messages(messages, build_tool_result_placeholder_messages(messages))

    tool_messages = [message for message in compacted if isinstance(message, ToolMessage)]
    assert [message.content for message in tool_messages] == [
        TOOL_RESULT_PLACEHOLDER,
        TOOL_RESULT_PLACEHOLDER,
    ]
    assert [message.tool_call_id for message in tool_messages] == ["call_1", "call_2"]


def test_select_recent_messages_does_not_start_with_orphan_tool_message() -> None:
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "calculator",
                "args": {"expression": "2+2"},
            }
        ],
    )
    messages = [
        HumanMessage(content="old"),
        HumanMessage(content="calculate"),
        ai_message,
        ToolMessage(content="4", tool_call_id="call_1", name="calculator"),
        AIMessage(content="answer"),
    ]

    recent = select_recent_messages(messages, max_messages=2)

    assert recent[0] is ai_message
    assert len(recent) == 3


def test_build_compacted_messages_replaces_state_messages() -> None:
    messages = [
        HumanMessage(content="old"),
        HumanMessage(content="new"),
        AIMessage(content="answer"),
    ]

    update = build_compacted_messages(messages, recent_messages_to_keep=2)
    compacted = add_messages(messages, update)

    assert [message.content for message in compacted] == ["new", "answer"]


def test_graph_compacts_messages_after_final_high_token_response(monkeypatch) -> None:
    from langraph_agent import graph as graph_module

    class FakeLLM:
        def __init__(self, response: AIMessage) -> None:
            self.response = response

        def invoke(self, _messages):
            return self.response

    agent_llm = FakeLLM(
        AIMessage(
            content="final answer",
            usage_metadata={
                "input_tokens": 80,
                "output_tokens": 20,
                "total_tokens": 100,
            },
        )
    )
    summary_llm = FakeLLM(AIMessage(content="压缩后的摘要"))

    def fake_build_llm(*, bind_tools: bool = True, tools=None):
        return agent_llm if bind_tools else summary_llm

    monkeypatch.setattr(graph_module, "build_llm", fake_build_llm)
    monkeypatch.setattr(graph_module, "discover_skills", lambda: [])
    monkeypatch.setattr(graph_module, "load_mcp_tools_sync", lambda: [])
    monkeypatch.setattr(graph_module, "get_mcp_auto_approved_tools", lambda: set())

    graph = graph_module.build_graph(
        compact_token_threshold=50,
        recent_messages_to_keep=2,
    )
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="old 1"),
                HumanMessage(content="old 2"),
                HumanMessage(content="latest"),
            ]
        }
    )

    assert result["session_summary"] == "压缩后的摘要"
    assert result["context_compaction"] == {
        "previous_message_count": 4,
        "kept_message_count": 2,
        "removed_message_count": 2,
        "last_total_tokens": 100,
        "token_threshold": 50,
    }
    assert [message.content for message in result["messages"]] == [
        "latest",
        "final answer",
    ]


def test_graph_expires_tool_results_before_main_llm(monkeypatch) -> None:
    from langraph_agent import graph as graph_module

    class RecordingLLM:
        def __init__(self, response: AIMessage) -> None:
            self.response = response
            self.seen_messages = []

        def invoke(self, messages):
            self.seen_messages.append(messages)
            return self.response

    agent_llm = RecordingLLM(AIMessage(content="final answer"))
    summary_llm = RecordingLLM(AIMessage(content="摘要"))

    def fake_build_llm(*, bind_tools: bool = True, tools=None):
        return agent_llm if bind_tools else summary_llm

    monkeypatch.setattr(graph_module, "build_llm", fake_build_llm)
    monkeypatch.setattr(graph_module, "discover_skills", lambda: [])
    monkeypatch.setattr(graph_module, "load_mcp_tools_sync", lambda: [])
    monkeypatch.setattr(graph_module, "get_mcp_auto_approved_tools", lambda: set())

    graph = graph_module.build_graph(kv_cache_ttl_seconds=1)
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="old"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "call_1", "name": "bash", "args": {}}],
                ),
                ToolMessage(
                    content="secret output",
                    tool_call_id="call_1",
                    name="bash",
                ),
                HumanMessage(content="continue"),
            ],
            "last_model_or_turn_at": 0.0,
        }
    )

    tool_messages = [
        message
        for message in agent_llm.seen_messages[0]
        if isinstance(message, ToolMessage)
    ]
    assert [message.content for message in tool_messages] == [TOOL_RESULT_PLACEHOLDER]
    assert result["tool_result_compaction"]["replaced_count"] == 1


def test_graph_skips_tool_result_expiry_when_disabled(monkeypatch) -> None:
    from langraph_agent import graph as graph_module

    class RecordingLLM:
        def __init__(self, response: AIMessage) -> None:
            self.response = response
            self.seen_messages = []

        def invoke(self, messages):
            self.seen_messages.append(messages)
            return self.response

    agent_llm = RecordingLLM(AIMessage(content="final answer"))
    summary_llm = RecordingLLM(AIMessage(content="摘要"))

    def fake_build_llm(*, bind_tools: bool = True, tools=None):
        return agent_llm if bind_tools else summary_llm

    monkeypatch.setattr(graph_module, "build_llm", fake_build_llm)
    monkeypatch.setattr(graph_module, "discover_skills", lambda: [])
    monkeypatch.setattr(graph_module, "load_mcp_tools_sync", lambda: [])
    monkeypatch.setattr(graph_module, "get_mcp_auto_approved_tools", lambda: set())

    graph = graph_module.build_graph(
        tool_result_compaction_enabled=False,
        kv_cache_ttl_seconds=1,
    )
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="old"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "call_1", "name": "bash", "args": {}}],
                ),
                ToolMessage(
                    content="secret output",
                    tool_call_id="call_1",
                    name="bash",
                ),
                HumanMessage(content="continue"),
            ],
            "last_model_or_turn_at": 0.0,
        }
    )

    tool_messages = [
        message
        for message in agent_llm.seen_messages[0]
        if isinstance(message, ToolMessage)
    ]
    assert [message.content for message in tool_messages] == ["secret output"]
    assert "tool_result_compaction" not in result


def test_plan_mode_expires_tool_results_before_plan_llm(monkeypatch) -> None:
    from langgraph.checkpoint.memory import MemorySaver

    from langraph_agent import graph as graph_module
    from langraph_agent.tools import PLAN_TOOLS

    class RecordingLLM:
        def __init__(self, response: AIMessage) -> None:
            self.response = response
            self.seen_messages = []

        def invoke(self, messages):
            self.seen_messages.append(messages)
            return self.response

    plan_llm = RecordingLLM(AIMessage(content="执行计划书"))
    agent_llm = RecordingLLM(AIMessage(content="执行完成"))
    summary_llm = RecordingLLM(AIMessage(content="摘要"))

    def fake_build_llm(*, bind_tools: bool = True, tools=None):
        if not bind_tools:
            return summary_llm
        if tools == PLAN_TOOLS:
            return plan_llm
        return agent_llm

    monkeypatch.setattr(graph_module, "build_llm", fake_build_llm)
    monkeypatch.setattr(graph_module, "discover_skills", lambda: [])
    monkeypatch.setattr(graph_module, "load_mcp_tools_sync", lambda: [])
    monkeypatch.setattr(graph_module, "get_mcp_auto_approved_tools", lambda: set())

    graph = graph_module.build_graph(
        checkpointer=MemorySaver(),
        kv_cache_ttl_seconds=1,
        plan_mode=True,
    )
    interrupted = graph.invoke(
        {
            "messages": [
                HumanMessage(content="old"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "call_1", "name": "read_file", "args": {}}],
                ),
                ToolMessage(
                    content="file content",
                    tool_call_id="call_1",
                    name="read_file",
                ),
                HumanMessage(content="new task"),
            ],
            "last_model_or_turn_at": 0.0,
            "plan_approved": True,
        },
        config={"configurable": {"thread_id": "plan-expire-test"}},
    )

    assert interrupted["__interrupt__"][0].value["type"] == "plan_review"
    tool_messages = [
        message
        for message in plan_llm.seen_messages[0]
        if isinstance(message, ToolMessage)
    ]
    assert [message.content for message in tool_messages] == [TOOL_RESULT_PLACEHOLDER]


def test_debug_update_prints_compaction_info(capsys) -> None:
    from langraph_agent.graph import _print_debug_update

    _print_debug_update(
        {
            "session_summary": "压缩后的摘要",
            "context_compaction": {
                "previous_message_count": 10,
                "kept_message_count": 4,
                "removed_message_count": 6,
                "last_total_tokens": 9000,
                "token_threshold": 8000,
            },
        }
    )

    output = capsys.readouterr().out
    assert "session_summary:" in output
    assert "压缩后的摘要" in output
    assert "context_compaction:" in output
    assert '"removed_message_count": 6' in output
