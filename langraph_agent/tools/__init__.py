from __future__ import annotations

from langraph_agent.tools.basic import calculator, current_time
from langraph_agent.tools.filesystem import (
    FILESYSTEM_TOOLS,
    edit_file,
    glob,
    grep,
    ls,
    read_file,
    write_file,
)
from langraph_agent.tools.planning import ask_human
from langraph_agent.tools.shell import bash
from langraph_agent.tools.skill_tools import list_skills, load_skill
from langraph_agent.tools.web_search import web_extract, web_search


# 基础工具列表会同时传给两处：
# 1. llm.bind_tools(BASE_TOOLS): 告诉模型“你可以调用这些普通工具”。
# 2. guarded_tool_node: 按白名单和人工审核结果执行模型请求的工具调用。
BASE_TOOLS = [calculator, current_time, web_search, web_extract]
SKILL_TOOLS = [list_skills, load_skill]
EXECUTION_TOOLS = [bash]
PLAN_TOOLS = [ask_human, ls, read_file, glob, grep]
TOOLS = [*BASE_TOOLS, *SKILL_TOOLS, *EXECUTION_TOOLS, *FILESYSTEM_TOOLS]

__all__ = [
    "BASE_TOOLS",
    "EXECUTION_TOOLS",
    "FILESYSTEM_TOOLS",
    "PLAN_TOOLS",
    "SKILL_TOOLS",
    "TOOLS",
    "ask_human",
    "bash",
    "calculator",
    "current_time",
    "edit_file",
    "glob",
    "grep",
    "ls",
    "list_skills",
    "load_skill",
    "read_file",
    "web_extract",
    "web_search",
    "write_file",
]
