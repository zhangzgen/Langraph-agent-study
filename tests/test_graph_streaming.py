from __future__ import annotations

from contextlib import contextmanager

from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.types import Command

from langraph_agent import graph as graph_module
from langraph_agent.graph import (
    _stream_debug_graph_to_console,
    _stream_graph_to_console,
    stream_answer_until_interrupt,
)


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


class FakeInterruptGraph:
    """模拟需要恢复一次审批中断后继续回答的图。"""

    def __init__(self) -> None:
        """初始化调用记录。

        Description:
            创建用于验证第二次 stream 输入为拒绝审批 Command 的记录列表。
        Args:
            无。
        Returns:
            None: 该方法仅初始化测试状态。
        """
        self.inputs: list[object] = []

    def stream(self, next_input, config, stream_mode):
        """产生一次中断和一次最终回答事件流。

        Description:
            首次执行发出审批中断；恢复执行时断言渠道已自动拒绝，再返回回答。
        Args:
            next_input (object): 当前图输入或中断恢复命令。
            config (dict): 当前测试使用的 thread 配置。
            stream_mode (list[str]): 请求监听的事件类别。
        Returns:
            Iterator[tuple[str, object]]: 固定的 LangGraph 流式事件序列。
        """
        self.inputs.append(next_input)
        assert config == {"configurable": {"thread_id": "test"}}
        assert stream_mode == ["messages", "updates"]
        if len(self.inputs) == 1:
            yield ("updates", {"__interrupt__": [object()]})
            return
        assert isinstance(next_input, Command)
        yield (
            "messages",
            (AIMessageChunk(content="rejected"), {"langgraph_node": "llm"}),
        )
        yield ("updates", {"llm": {"messages": [AIMessage(content="rejected")]}})


class FakeChannelPauseGraph:
    """模拟飞书渠道在工具分类后暂停审批的图。"""

    def stream(self, next_input, config, stream_mode):
        """产生工具概要更新及结构化审批中断。

        Description:
            让渠道入口可以验证工具名回调以及不会自动发送恢复命令的行为。
        Args:
            next_input (object): 当前用户输入。
            config (dict): 当前测试使用的 thread 配置。
            stream_mode (list[str]): 请求监听的事件类别。
        Returns:
            Iterator[tuple[str, object]]: 固定分类更新与中断事件流。
        """
        assert next_input == {"messages": [{"role": "user", "content": "hi"}]}
        assert config == {"configurable": {"thread_id": "test"}}
        assert stream_mode == ["messages", "updates"]
        yield (
            "updates",
            {
                "classify_tool_calls": {
                    "approved_tool_calls": [{"id": "auto", "name": "read_file"}],
                    "pending_approvals": [
                        {"tool_call_id": "review", "tool_name": "write_file"}
                    ],
                }
            },
        )

        class InterruptItem:
            value = {
                "type": "tool_approval",
                "tool_calls": [{"id": "review", "name": "write_file"}],
            }

        yield ("updates", {"__interrupt__": [InterruptItem()]})


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


def test_stream_answer_reports_accumulated_text_without_console(
    monkeypatch,
) -> None:
    """验证渠道流式入口会回调累计正文。

    Description:
        飞书等非终端渠道需要得到累计文本以覆盖更新卡片，而不能依赖控制台打印。
    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。
    Returns:
        None: 测试通过断言校验回调正文及最终消息。
    """
    calls: list[str] = []

    @contextmanager
    def fake_checkpoint_saver(_db_path=None):
        """提供不访问真实数据库的 checkpoint 替身。

        Description:
            为 stream_answer 测试提供可传给 build_graph 的固定上下文对象。
        Args:
            _db_path (str | None): 未在测试中使用的路径参数。
        Returns:
            Iterator[object]: 固定 checkpointer 替身的上下文迭代器。
        """
        yield object()

    def fake_build_graph(*, checkpointer):
        """返回提供流式消息的测试图。

        Description:
            断言渠道执行向图构建器传入 checkpoint，并返回固定事件流。
        Args:
            checkpointer (object): checkpoint_saver 产生的替身对象。
        Returns:
            FakeStreamingGraph: 具有固定流式输出的测试图实例。
        """
        assert checkpointer is not None
        return FakeStreamingGraph()

    monkeypatch.setattr(graph_module, "checkpoint_saver", fake_checkpoint_saver)
    monkeypatch.setattr(graph_module, "build_graph", fake_build_graph)

    message = graph_module.stream_answer("hi", "test", calls.append)

    assert message.content == "hello"
    assert calls == ["hel", "hello"]


def test_stream_answer_rejects_interrupt_and_continues(
    monkeypatch,
) -> None:
    """验证飞书渠道遇到工具审批时不会遗留暂停会话。

    Description:
        模拟图产生审批 interrupt，确认渠道以拒绝决定恢复执行并最终回传回答。
    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。
    Returns:
        None: 测试通过断言校验恢复路径和输出正文。
    """
    fake_graph = FakeInterruptGraph()
    calls: list[str] = []

    @contextmanager
    def fake_checkpoint_saver(_db_path=None):
        """提供审批恢复测试使用的 checkpoint 替身。

        Description:
            使渠道入口无需建立数据库即可编译模拟图。
        Args:
            _db_path (str | None): 未在测试中使用的路径参数。
        Returns:
            Iterator[object]: 固定 checkpointer 替身的上下文迭代器。
        """
        yield object()

    def fake_build_graph(*, checkpointer):
        """返回审批中断模拟图。

        Description:
            复用单个模拟实例以记录中断前后的两次 stream 输入。
        Args:
            checkpointer (object): checkpoint_saver 产生的替身对象。
        Returns:
            FakeInterruptGraph: 审批中断模拟图。
        """
        assert checkpointer is not None
        return fake_graph

    monkeypatch.setattr(graph_module, "checkpoint_saver", fake_checkpoint_saver)
    monkeypatch.setattr(graph_module, "build_graph", fake_build_graph)

    message = graph_module.stream_answer("hi", "test", calls.append)

    assert message.content == "rejected"
    assert calls == ["rejected"]
    assert len(fake_graph.inputs) == 2


def test_stream_answer_until_interrupt_returns_pending_tool_request(
    monkeypatch,
) -> None:
    """验证飞书渠道会保留审批中断并上报工具名称。

    Description:
        模拟分类节点与审批暂停，确认新的交互渠道入口不会自动拒绝恢复，
        且回调数据不包含工具参数或执行结果。
    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。
    Returns:
        None: 测试通过断言校验中断结果与工具概要。
    """
    reported_tools: list[list[dict]] = []

    @contextmanager
    def fake_checkpoint_saver(_db_path=None):
        """提供交互暂停测试的 checkpoint 替身。

        Description:
            避免测试依赖实际 SQLite 数据库文件。
        Args:
            _db_path (str | None): 测试未使用的路径参数。
        Returns:
            Iterator[object]: 固定的 checkpointer 哨兵对象。
        """
        yield object()

    def fake_build_graph(*, checkpointer):
        """返回飞书审批暂停模拟图。

        Description:
            校验入口构建图时携带 checkpointer，并返回固定事件流。
        Args:
            checkpointer (object): checkpoint 上下文产生的哨兵对象。
        Returns:
            FakeChannelPauseGraph: 固定的渠道暂停图。
        """
        assert checkpointer is not None
        return FakeChannelPauseGraph()

    monkeypatch.setattr(graph_module, "checkpoint_saver", fake_checkpoint_saver)
    monkeypatch.setattr(graph_module, "build_graph", fake_build_graph)

    outcome = stream_answer_until_interrupt(
        "hi",
        "test",
        lambda _text: None,
        reported_tools.append,
    )

    assert outcome.final_message is None
    assert outcome.interrupt_request["type"] == "tool_approval"
    assert reported_tools == [
        [
            {
                "id": "auto",
                "name": "read_file",
                "approval_required": False,
                "display_content": "",
            },
            {
                "id": "review",
                "name": "write_file",
                "approval_required": True,
                "display_content": "",
            },
        ]
    ]
