from __future__ import annotations

import json

from langraph_agent.tools.mcp import load_mcp_config


def test_load_mcp_config_extracts_connections_and_auto_approved_tools(tmp_path) -> None:
    """验证 MCP 配置解析结果。

    Description:
        确认配置解析会保留 MultiServerMCPClient 需要的连接字段，并剥离本项目审批策略使用的元数据字段。
    Args:
        tmp_path: pytest 提供的临时目录路径。
    Returns:
        None: 该测试通过断言验证行为。
    """
    config_path = tmp_path / "mcp_servers.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "time": {
                        "transport": "stdio",
                        "command": "uvx",
                        "args": ["mcp-server-time"],
                        "auto_approve_tools": ["get_current_time"],
                        "description": "local time tools",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    mcp_config = load_mcp_config(config_path)

    assert mcp_config.connections == {
        "time": {
            "transport": "stdio",
            "command": "uvx",
            "args": ["mcp-server-time"],
        }
    }
    assert mcp_config.auto_approved_tools == {"get_current_time"}


def test_load_mcp_config_skips_disabled_servers(tmp_path) -> None:
    """验证禁用 MCP server 不会进入运行时配置。

    Description:
        当 server 配置 enabled 为 false 时，连接配置和自动审批工具名都不会被加载。
    Args:
        tmp_path: pytest 提供的临时目录路径。
    Returns:
        None: 该测试通过断言验证行为。
    """
    config_path = tmp_path / "mcp_servers.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "fetch": {
                        "enabled": False,
                        "transport": "stdio",
                        "command": "uvx",
                        "args": ["mcp-server-fetch"],
                        "auto_approve_tools": ["fetch"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    mcp_config = load_mcp_config(config_path)

    assert mcp_config.connections == {}
    assert mcp_config.auto_approved_tools == set()
