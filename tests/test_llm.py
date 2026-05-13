from __future__ import annotations

from langraph_agent import llm as llm_module


def test_build_llm_disables_thinking_by_default(monkeypatch) -> None:
    calls = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs) -> None:
            calls.update(kwargs)

        def bind_tools(self, tools):
            calls["tools"] = tools
            return self

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)

    llm_module.build_llm()

    assert calls["extra_body"] == {"thinking": {"type": "disabled"}}
    assert calls["tools"] == llm_module.TOOLS


def test_build_llm_uses_central_extra_body_config(monkeypatch) -> None:
    calls = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs) -> None:
            calls.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(
        llm_module,
        "OPENAI_EXTRA_BODY",
        {"thinking": {"type": "enabled"}},
    )

    llm_module.build_llm(bind_tools=False)

    assert calls["extra_body"] == {"thinking": {"type": "enabled"}}
