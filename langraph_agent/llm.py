from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from langraph_agent.config import XIAOMI_DEFAULT_BASE_URL, XIAOMI_DEFAULT_MODEL
from langraph_agent.tools import TOOLS


def build_llm() -> ChatOpenAI:
    # 真实 AK 从 .env 或 shell 环境变量读取，避免写入代码仓库。
    api_key = os.getenv("XIAOMI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少 XIAOMI_API_KEY。请复制 .env.example 为 .env，并填入你的真实 AK。"
        )

    # 小米的接口兼容 OpenAI Chat Completions，因此可以直接使用 ChatOpenAI。
    # bind_tools 会把工具 schema 附到请求里，模型如果需要工具，会返回 tool_calls。
    return ChatOpenAI(
        model=os.getenv("XIAOMI_MODEL", XIAOMI_DEFAULT_MODEL),
        api_key=api_key,
        base_url=os.getenv("XIAOMI_BASE_URL", XIAOMI_DEFAULT_BASE_URL),
        temperature=0,
    ).bind_tools(TOOLS)
