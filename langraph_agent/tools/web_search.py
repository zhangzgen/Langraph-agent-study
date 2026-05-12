from __future__ import annotations

import json
import os
from typing import Any, Literal

from langchain_core.tools import tool
from langchain_tavily import TavilyExtract, TavilySearch


@tool
def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
) -> str:
    """使用 Tavily 搜索互联网实时信息，返回搜索摘要和来源链接，不抓取网页全文。"""
    if not _has_tavily_api_key():
        return _missing_tavily_api_key_message("Web search")

    bounded_max_results = min(max(max_results, 1), 10)
    search_tool = TavilySearch(
        max_results=bounded_max_results,
        search_depth="basic",
        topic=topic,
        include_answer=True,
        # Tavily Search 返回的 content 是搜索摘要/片段，不是网页全文。
        # 这里保持 False，避免搜索工具输出过长；用户给定 URL 时使用 web_extract。
        include_raw_content=False,
    )
    result = search_tool.invoke({"query": query})
    return _format_tavily_result(result)


@tool
def web_extract(
    url: str,
    extract_depth: Literal["basic", "advanced"] = "basic",
    query: str | None = None,
    content_limit: int = 12000,
) -> str:
    """提取指定网页 URL 的正文内容；适合用户直接给链接并要求阅读、总结或分析。"""
    if not _has_tavily_api_key():
        return _missing_tavily_api_key_message("Web extract")

    extract_tool = TavilyExtract(
        extract_depth=extract_depth,
        include_images=False,
        format="markdown",
    )
    payload: dict[str, Any] = {"urls": [url]}
    if query:
        payload["query"] = query
    result = extract_tool.invoke(payload)
    return _format_tavily_extract_result(result, content_limit=content_limit)


def _has_tavily_api_key() -> bool:
    return bool(os.getenv("TAVILY_API_KEY"))


def _missing_tavily_api_key_message(tool_name: str) -> str:
    return (
        f"{tool_name} 未配置: 缺少 TAVILY_API_KEY。"
        "请在 .env 中设置 TAVILY_API_KEY 后重试。"
    )


def _format_tavily_result(result: Any) -> str:
    if not isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False, indent=2)

    parts = []
    answer = result.get("answer")
    if answer:
        parts.append(f"answer:\n{answer}")

    results = result.get("results")
    if isinstance(results, list) and results:
        parts.append("sources:")
        for index, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                parts.append(f"{index}. {item}")
                continue
            title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            content = item.get("content") or item.get("snippet") or ""
            score = item.get("score")
            score_text = f" score={score}" if score is not None else ""
            parts.append(f"{index}. {title}{score_text}\nurl: {url}\ncontent: {content}")

    if parts:
        return "\n\n".join(parts)
    return json.dumps(result, ensure_ascii=False, indent=2)


def _format_tavily_extract_result(result: Any, content_limit: int = 12000) -> str:
    if not isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False, indent=2)

    parts = []
    results = result.get("results")
    if isinstance(results, list) and results:
        for index, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                parts.append(f"{index}. {item}")
                continue
            url = item.get("url") or ""
            title = item.get("title") or "Untitled"
            content = item.get("raw_content") or item.get("content") or ""
            source_parts = [f"{index}. {title}", f"url: {url}"]
            if content:
                source_parts.append(
                    f"content:\n{_truncate_text(content, content_limit)}"
                )
            parts.append("\n".join(source_parts))

    failed_results = result.get("failed_results")
    if isinstance(failed_results, list) and failed_results:
        parts.append("failed_results:")
        for item in failed_results:
            parts.append(json.dumps(item, ensure_ascii=False))

    if parts:
        return "\n\n".join(parts)
    return json.dumps(result, ensure_ascii=False, indent=2)


def _truncate_text(value: str, limit: int) -> str:
    bounded_limit = min(max(limit, 500), 30000)
    if len(value) <= bounded_limit:
        return value
    return f"{value[:bounded_limit]}\n...[truncated {len(value) - bounded_limit} chars]"
