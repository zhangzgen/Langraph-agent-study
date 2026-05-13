from __future__ import annotations

from langchain_community.agent_toolkits import FileManagementToolkit

from langraph_agent.config import config


# FileManagementToolkit 是 LangChain 提供的现成文件工具集合。
# root_dir 把所有相对路径限制在当前项目内，避免模型访问项目外文件。
# 写入/删除/移动等风险操作不会在这里拦截，而是统一交给 tool_guard 的
# LangGraph interrupt 审批节点处理。
FILESYSTEM_TOOLS = FileManagementToolkit(
    root_dir=str(config.PROJECT_ROOT),
    selected_tools=[
        "copy_file",
        "file_delete",
        "file_search",
        "list_directory",
        "move_file",
        "read_file",
        "write_file",
    ],
).get_tools()
