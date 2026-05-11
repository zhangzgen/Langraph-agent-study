from __future__ import annotations

from langchain_core.tools import tool

from langraph_agent.skills.registry import (
    discover_skills,
    find_skill,
    format_skill_catalog,
    split_skill_file,
)


@tool
def list_skills() -> str:
    """列出当前项目 skills 目录中可用 Skill 的名称和描述。"""
    skills = discover_skills()
    if not skills:
        return "当前没有发现可用 Skill。"
    return format_skill_catalog(skills)


@tool
def load_skill(skill_name: str) -> str:
    """按 Skill 名称加载完整 SKILL.md 说明正文。"""
    # 这是 Skill 机制的“按需加载”入口。
    # 模型启动时只看到 YAML 元数据；判断需要某个 Skill 后，再调用这个工具加载正文。
    skill = find_skill(skill_name)
    if skill is None:
        available = ", ".join(item.name for item in discover_skills()) or "无"
        return f"没有找到 Skill: {skill_name}。可用 Skill: {available}"

    content = skill.path.read_text(encoding="utf-8")
    metadata, body = split_skill_file(content)
    description = metadata.get("description", skill.description)
    return (
        f"Skill 名称: {skill.name}\n"
        f"Skill 描述: {description}\n\n"
        f"完整说明:\n{body.strip()}"
    )
