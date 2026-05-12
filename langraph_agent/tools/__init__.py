from __future__ import annotations

from langraph_agent.tools.basic import calculator, current_time
from langraph_agent.tools.shell import bash
from langraph_agent.tools.skill_tools import list_skills, load_skill
from langraph_agent.tools.web_search import web_extract, web_search


# 基础工具列表会同时传给两处：
# 1. llm.bind_tools(BASE_TOOLS): 告诉模型“你可以调用这些普通工具”。
# 2. ToolNode(TOOLS): 真正执行模型请求的所有工具调用，包括 Skill 工具。
BASE_TOOLS = [calculator, current_time, web_search, web_extract]
SKILL_TOOLS = [list_skills, load_skill]
EXECUTION_TOOLS = [bash]
TOOLS = [*BASE_TOOLS, *SKILL_TOOLS, *EXECUTION_TOOLS]

__all__ = [
    "BASE_TOOLS",
    "EXECUTION_TOOLS",
    "SKILL_TOOLS",
    "TOOLS",
    "bash",
    "calculator",
    "current_time",
    "list_skills",
    "load_skill",
    "web_extract",
    "web_search",
]
