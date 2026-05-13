from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from langraph_agent.context import (
    build_compacted_messages,
    extract_total_tokens,
    select_recent_messages,
    should_compact_context,
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

    def fake_build_llm(*, bind_tools: bool = True):
        return agent_llm if bind_tools else summary_llm

    monkeypatch.setattr(graph_module, "build_llm", fake_build_llm)
    monkeypatch.setattr(graph_module, "discover_skills", lambda: [])

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
