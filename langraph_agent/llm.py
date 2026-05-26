from __future__ import annotations

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from langraph_agent.config import config
from langraph_agent.tools import TOOLS


def build_llm(
    *,
    bind_tools: bool = True,
    tools: list[BaseTool] | None = None,
) -> ChatOpenAI:
    """构建项目使用的 OpenAI 兼容聊天模型。

    Description:
        按统一配置初始化 ChatOpenAI，并在需要时绑定指定工具集。未显式传入工具时，
        默认绑定主执行 Agent 使用的完整工具列表。
    Args:
        bind_tools (bool): 是否为模型绑定 LangChain 工具 schema。
        tools (list[BaseTool] | None): 可选工具列表；为空时使用项目默认工具集。
    Returns:
        ChatOpenAI: 已初始化的聊天模型实例；当 bind_tools 为 True 时返回绑定工具后的模型。
    """
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
        return llm.bind_tools(TOOLS if tools is None else tools)
    return llm
