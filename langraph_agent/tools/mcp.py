from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from langraph_agent.config import config


MCP_METADATA_KEYS = {"auto_approve_tools", "description", "enabled"}


@dataclass(frozen=True)
class MCPToolConfig:
    """MCP 工具运行时配置。

    Description:
        保存传给 MultiServerMCPClient 的连接配置，以及允许自动执行的 MCP 工具名称集合。
    Args:
        connections (dict[str, dict[str, Any]]): MCP server 名称到连接参数的映射。
        auto_approved_tools (set[str]): 配置中显式标记为安全只读的 MCP 工具名称。
    Returns:
        MCPToolConfig: 不可变 MCP 工具配置对象。
    """

    connections: dict[str, dict[str, Any]]
    auto_approved_tools: set[str]


def load_mcp_config(config_path: str | Path | None = None) -> MCPToolConfig:
    """读取 MCP server 配置文件。

    Description:
        从 mcp_servers.json 读取 stdio、http 或 sse MCP server 配置，并剥离仅供本项目审批策略使用的元数据字段。
    Args:
        config_path (str | Path | None): MCP 配置文件路径；为空时使用全局配置。
    Returns:
        MCPToolConfig: 包含 MCP 连接配置和自动审批工具名称的配置对象。
    """
    path = Path(config_path or config.MCP_SERVERS_CONFIG_PATH)
    if not path.exists():
        return MCPToolConfig(connections={}, auto_approved_tools=set())

    with path.open(encoding="utf-8") as file:
        raw_config = json.load(file)

    servers = raw_config.get("servers", {})
    if not isinstance(servers, dict):
        raise ValueError("mcp_servers.json 的 servers 字段必须是对象。")

    connections: dict[str, dict[str, Any]] = {}
    auto_approved_tools: set[str] = set()
    for server_name, server_config in servers.items():
        if not isinstance(server_config, dict):
            raise ValueError(f"MCP server {server_name} 的配置必须是对象。")
        if server_config.get("enabled", True) is False:
            continue

        safe_tools = server_config.get("auto_approve_tools", [])
        if safe_tools:
            if not isinstance(safe_tools, list) or not all(
                isinstance(item, str) for item in safe_tools
            ):
                raise ValueError(
                    f"MCP server {server_name} 的 auto_approve_tools 必须是字符串数组。"
                )
            auto_approved_tools.update(safe_tools)

        connections[server_name] = {
            key: value
            for key, value in server_config.items()
            if key not in MCP_METADATA_KEYS
        }

    return MCPToolConfig(
        connections=connections,
        auto_approved_tools=auto_approved_tools,
    )


async def load_mcp_tools(config_path: str | Path | None = None) -> list[BaseTool]:
    """异步加载 MCP 工具。

    Description:
        使用 langchain-mcp-adapters 的 MultiServerMCPClient 连接配置中的 MCP servers，并将外部 MCP tools 转换为 LangChain BaseTool。
    Args:
        config_path (str | Path | None): MCP 配置文件路径；为空时使用全局配置。
    Returns:
        list[BaseTool]: 从所有启用 MCP server 加载到的 LangChain 工具列表。
    """
    mcp_config = load_mcp_config(config_path)
    if not mcp_config.connections:
        return []

    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(mcp_config.connections)
    return list(await client.get_tools())


def load_mcp_tools_sync(config_path: str | Path | None = None) -> list[BaseTool]:
    """同步加载 MCP 工具。

    Description:
        为当前同步 LangGraph 构图路径包装异步 MCP 加载逻辑；当调用方已经处于事件循环中时，改用临时线程执行，避免 asyncio.run 嵌套报错。
    Args:
        config_path (str | Path | None): MCP 配置文件路径；为空时使用全局配置。
    Returns:
        list[BaseTool]: 可直接绑定到模型并交给执行器调用的 MCP 工具列表。
    """
    coroutine = load_mcp_tools(config_path)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    return _run_coroutine_in_thread(coroutine)


def get_mcp_auto_approved_tools(config_path: str | Path | None = None) -> set[str]:
    """读取 MCP 自动审批工具名。

    Description:
        从 MCP 配置文件中提取 auto_approve_tools，仅用于扩展主工具审批白名单。
    Args:
        config_path (str | Path | None): MCP 配置文件路径；为空时使用全局配置。
    Returns:
        set[str]: 配置中显式声明为安全只读的 MCP 工具名称集合。
    """
    return load_mcp_config(config_path).auto_approved_tools


def _run_coroutine_in_thread(coroutine: Any) -> list[BaseTool]:
    """在线程中执行异步 MCP 加载协程。

    Description:
        当当前线程已有运行中的事件循环时，创建临时线程并在其中运行 asyncio.run，将结果或异常同步带回调用方。
    Args:
        coroutine (Any): 待执行的异步协程对象。
    Returns:
        list[BaseTool]: 协程返回的 LangChain 工具列表。
    """
    result: list[BaseTool] = []
    error: BaseException | None = None

    def runner() -> None:
        """执行线程内事件循环。

        Description:
            在线程内运行传入协程，并把结果或异常写回外层作用域。
        Args:
            None: 该闭包不接收外部参数。
        Returns:
            None: 通过外层变量传递执行结果。
        """
        nonlocal result, error
        try:
            result = asyncio.run(coroutine)
        except BaseException as exc:
            error = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error is not None:
        raise error
    return result
