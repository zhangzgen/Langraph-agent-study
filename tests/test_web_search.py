from __future__ import annotations

import importlib

import pytest

from langraph_agent.config import config
from langraph_agent.tools import BASE_TOOLS


def test_web_search_is_registered() -> None:
    assert any(tool.name == "web_search" for tool in BASE_TOOLS)
    assert any(tool.name == "web_extract" for tool in BASE_TOOLS)


def test_web_search_reports_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    web_search_module = importlib.import_module("langraph_agent.tools.web_search")
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")

    result = web_search_module.web_search.invoke({"query": "LangGraph latest release"})

    assert "缺少 TAVILY_API_KEY" in result


def test_web_extract_reports_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    web_search_module = importlib.import_module("langraph_agent.tools.web_search")
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")

    result = web_search_module.web_extract.invoke({"url": "https://example.com"})

    assert "缺少 TAVILY_API_KEY" in result


def test_web_search_formats_tavily_results(monkeypatch: pytest.MonkeyPatch) -> None:
    web_search_module = importlib.import_module("langraph_agent.tools.web_search")
    monkeypatch.setattr(config, "TAVILY_API_KEY", "test-key")

    class FakeTavilySearch:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            assert kwargs["tavily_api_key"] == "test-key"

        def invoke(self, payload: dict[str, str]) -> dict:
            assert payload == {"query": "LangGraph latest release"}
            return {
                "answer": "LangGraph released a new version.",
                "results": [
                    {
                        "title": "LangGraph release notes",
                        "url": "https://example.com/langgraph",
                        "content": "Release notes summary",
                        "score": 0.91,
                    }
                ],
            }

    monkeypatch.setattr(web_search_module, "TavilySearch", FakeTavilySearch)

    result = web_search_module.web_search.invoke(
        {"query": "LangGraph latest release", "max_results": 3}
    )

    assert "answer:\nLangGraph released a new version." in result
    assert "LangGraph release notes" in result
    assert "https://example.com/langgraph" in result
    assert "raw_content:" not in result


def test_web_extract_formats_extracted_content(monkeypatch: pytest.MonkeyPatch) -> None:
    web_search_module = importlib.import_module("langraph_agent.tools.web_search")
    monkeypatch.setattr(config, "TAVILY_API_KEY", "test-key")

    class FakeTavilyExtract:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            assert kwargs["tavily_api_key"] == "test-key"

        def invoke(self, payload: dict) -> dict:
            assert payload == {"urls": ["https://example.com/langgraph"]}
            return {
                "results": [
                    {
                        "title": "LangGraph docs",
                        "url": "https://example.com/langgraph",
                        "raw_content": "Full extracted page content",
                    }
                ]
            }

    monkeypatch.setattr(web_search_module, "TavilyExtract", FakeTavilyExtract)

    result = web_search_module.web_extract.invoke(
        {"url": "https://example.com/langgraph"}
    )

    assert "LangGraph docs" in result
    assert "https://example.com/langgraph" in result
    assert "content:\nFull extracted page content" in result
