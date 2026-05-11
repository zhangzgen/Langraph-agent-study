from __future__ import annotations

from pathlib import Path

import pytest

from langraph_agent.skills.registry import (
    discover_skills,
    find_skill,
    format_skill_catalog,
    get_skills_dir,
    split_skill_file,
)


def write_skill(root: Path, folder: str, frontmatter: str, body: str = "# Body") -> None:
    skill_dir = root / folder
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\n{frontmatter}\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_get_skills_dir_uses_environment_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_SKILLS_DIR", str(tmp_path))

    assert get_skills_dir() == tmp_path


def test_discover_skills_reads_valid_frontmatter_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_SKILLS_DIR", str(tmp_path))
    write_skill(tmp_path, "alpha", "name: alpha\ndescription: Alpha skill")
    write_skill(tmp_path, "missing-description", "name: broken")
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    (broken_dir / "SKILL.md").write_text("not frontmatter", encoding="utf-8")

    skills = discover_skills()

    assert len(skills) == 1
    assert skills[0].name == "alpha"
    assert skills[0].description == "Alpha skill"
    assert skills[0].path == tmp_path / "alpha" / "SKILL.md"


def test_find_skill_is_case_insensitive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_SKILLS_DIR", str(tmp_path))
    write_skill(tmp_path, "alpha", "name: Alpha\ndescription: Alpha skill")

    assert find_skill(" alpha ").name == "Alpha"


def test_split_skill_file_returns_metadata_and_body() -> None:
    metadata, body = split_skill_file(
        "---\nname: demo\ndescription: Demo skill\n---\n\n# Demo\nDetails\n"
    )

    assert metadata == {"name": "demo", "description": "Demo skill"}
    assert body == "\n# Demo\nDetails\n"


@pytest.mark.parametrize(
    "content",
    [
        "name: demo\n---\nbody",
        "---\n- not\n- mapping\n---\nbody",
        "---\nname: demo\nbody",
    ],
)
def test_split_skill_file_rejects_invalid_frontmatter(content: str) -> None:
    with pytest.raises(ValueError):
        split_skill_file(content)


def test_format_skill_catalog_handles_empty_list() -> None:
    assert format_skill_catalog([]) == "无"
