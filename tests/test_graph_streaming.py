from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk

from langraph_agent.graph import _stream_debug_graph_to_console, _stream_graph_to_console


class FakeStreamingGraph:
    def stream(self, next_input, config, stream_mode):
        assert next_input == {"messages": [{"role": "user", "content": "hi"}]}
        assert config == {"configurable": {"thread_id": "test"}}
        assert stream_mode == ["messages", "updates"]
        yield (
            "messages",
            (AIMessageChunk(content="hel"), {"langgraph_node": "llm"}),
        )
        yield (
            "messages",
            (
                AIMessageChunk(content="hidden"),
                {"langgraph_node": "summarize_and_compact"},
            ),
        )
        yield (
            "messages",
            (
                AIMessageChunk(content=[{"type": "text", "text": "lo"}]),
                {"langgraph_node": "llm"},
            ),
        )
        yield ("updates", {"llm": {"messages": [AIMessage(content="hello")]}})


def test_stream_graph_to_console_prints_llm_chunks_only(capsys):
    final_message = _stream_graph_to_console(
        FakeStreamingGraph(),
        {"messages": [{"role": "user", "content": "hi"}]},
        config={"configurable": {"thread_id": "test"}},
    )

    assert final_message.content == "hello"
    assert capsys.readouterr().out == "hello\n"


def test_stream_graph_to_console_prints_prefix_with_first_chunk(capsys):
    _stream_graph_to_console(
        FakeStreamingGraph(),
        {"messages": [{"role": "user", "content": "hi"}]},
        config={"configurable": {"thread_id": "test"}},
        prefix="assistant: ",
    )

    assert capsys.readouterr().out == "assistant: hello\n"


class FakeNonChunkGraph:
    def stream(self, next_input, config, stream_mode):
        yield ("updates", {"llm": {"messages": [AIMessage(content="fallback")]}})


def test_stream_graph_to_console_falls_back_to_final_message(capsys):
    final_message = _stream_graph_to_console(
        FakeNonChunkGraph(),
        {"messages": [{"role": "user", "content": "hi"}]},
        config={"configurable": {"thread_id": "test"}},
        prefix="assistant: ",
    )

    assert final_message.content == "fallback"
    assert capsys.readouterr().out == "assistant: fallback\n"

def test_stream_debug_graph_to_console_streams_and_keeps_node_updates(capsys):
    final_message = _stream_debug_graph_to_console(
        FakeStreamingGraph(),
        {"messages": [{"role": "user", "content": "hi"}]},
        config={"configurable": {"thread_id": "test"}},
    )

    output = capsys.readouterr().out
    assert final_message.content == "hello"
    assert "[llm stream]" in output
    assert "hello" in output
    assert "[llm]" in output
    assert "content: <streamed above>" in output
    assert "hidden" not in output


def test_stream_debug_graph_to_console_keeps_content_without_chunks(capsys):
    final_message = _stream_debug_graph_to_console(
        FakeNonChunkGraph(),
        {"messages": [{"role": "user", "content": "hi"}]},
        config={"configurable": {"thread_id": "test"}},
    )

    output = capsys.readouterr().out
    assert final_message.content == "fallback"
    assert "[llm stream]" not in output
    assert "[llm]" in output
    assert "content:\nfallback" in output
