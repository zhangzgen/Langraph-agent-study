from __future__ import annotations

import os
from pathlib import Path

import yaml

from langraph_agent.config import DEFAULT_SKILLS_DIR
from langraph_agent.models import SkillMetadata


def get_skills_dir() -> Path:
    # 可通过 AGENT_SKILLS_DIR 覆盖默认目录，便于后续实验不同 Skill 集合。
    return Path(os.getenv("AGENT_SKILLS_DIR", DEFAULT_SKILLS_DIR)).expanduser()


def discover_skills() -> list[SkillMetadata]:
    # 只读取 SKILL.md 的 YAML frontmatter，不加载正文。
    # 这对应 Skill 的第一层 progressive disclosure：name + description。
    skills_dir = get_skills_dir()
    if not skills_dir.exists():
        return []

    skills: list[SkillMetadata] = []
    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        try:
            metadata, _body = split_skill_file(skill_file.read_text(encoding="utf-8"))
        except ValueError:
            continue

        name = metadata.get("name")
        description = metadata.get("description")
        if not isinstance(name, str) or not isinstance(description, str):
            continue
        skills.append(
            SkillMetadata(
                name=name.strip(),
                description=description.strip(),
                path=skill_file,
            )
        )
    return skills


def find_skill(skill_name: str) -> SkillMetadata | None:
    normalized = skill_name.strip().lower()
    for skill in discover_skills():
        if skill.name.lower() == normalized:
            return skill
    return None


def split_skill_file(content: str) -> tuple[dict, str]:
    if not content.startswith("---\n"):
        raise ValueError("SKILL.md 必须以 YAML frontmatter 开始。")

    marker = "\n---\n"
    end = content.find(marker, 4)
    if end == -1:
        raise ValueError("SKILL.md 缺少 YAML frontmatter 结束标记。")

    frontmatter = content[4:end]
    body = content[end + len(marker) :]
    metadata = yaml.safe_load(frontmatter) or {}
    if not isinstance(metadata, dict):
        raise ValueError("SKILL.md frontmatter 必须是 YAML mapping。")
    return metadata, body


def format_skill_catalog(skills: list[SkillMetadata]) -> str:
    if not skills:
        return "无"
    return "\n".join(
        f"- name: {skill.name}\n  description: {skill.description}"
        for skill in skills
    )
