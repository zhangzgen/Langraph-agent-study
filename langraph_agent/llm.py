from __future__ import annotations

from langchain_openai import ChatOpenAI

from langraph_agent.config import config
from langraph_agent.tools import TOOLS


def build_llm(*, bind_tools: bool = True) -> ChatOpenAI:
    # 真实 AK 从 .env 或 shell 环境变量读取，避免写入代码仓库。
    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "缺少 OPENAI_API_KEY。请复制 .env.example 为 .env，并填入你的真实 AK。"
        )

    # 接口兼容 OpenAI Chat Completions，因此可以直接使用 ChatOpenAI。
    # bind_tools 会把工具 schema 附到请求里，模型如果需要工具，会返回 tool_calls。
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        temperature=config.OPENAI_TEMPERATURE,
        extra_body=config.OPENAI_EXTRA_BODY,
        streaming=True,
    )
    if bind_tools:
        return llm.bind_tools(TOOLS)
    return llm
