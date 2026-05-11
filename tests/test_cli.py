from __future__ import annotations

import sys
from pathlib import Path

import pytest

from langraph_agent import cli


def test_cli_list_skills_prints_catalog_without_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    skill_dir = tmp_path / "demo-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo skill\n---\n\n# Demo\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_SKILLS_DIR", str(tmp_path))
    monkeypatch.delenv("XIAOMI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["react-agent", "--list-skills"])

    cli.main()

    captured = capsys.readouterr()
    assert "- name: demo-skill" in captured.out
    assert "description: Demo skill" in captured.out
