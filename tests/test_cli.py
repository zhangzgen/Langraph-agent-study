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


def test_cli_chat_passes_checkpoint_db_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from langraph_agent import graph

    calls = {}
    db_path = tmp_path / "checkpoints.sqlite"

    def fake_chat(
        thread_id: str,
        debug: bool,
        checkpoint_db_path: str | None,
    ) -> None:
        calls["thread_id"] = thread_id
        calls["debug"] = debug
        calls["checkpoint_db_path"] = checkpoint_db_path

    monkeypatch.setattr(graph, "chat", fake_chat)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "react-agent",
            "--chat",
            "--debug",
            "--thread-id",
            "demo-thread",
            "--checkpoint-db",
            str(db_path),
        ],
    )

    cli.main()

    assert calls == {
        "thread_id": "demo-thread",
        "debug": True,
        "checkpoint_db_path": str(db_path),
    }


def test_cli_run_passes_checkpoint_db_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from langchain_core.messages import AIMessage

    from langraph_agent import graph

    calls = {}
    db_path = tmp_path / "checkpoints.sqlite"

    def fake_run(
        question: str,
        debug: bool,
        checkpoint_db_path: str | None,
    ) -> AIMessage:
        calls["question"] = question
        calls["debug"] = debug
        calls["checkpoint_db_path"] = checkpoint_db_path
        return AIMessage(content="ok")

    monkeypatch.setattr(graph, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["react-agent", "hello", "--checkpoint-db", str(db_path)],
    )

    cli.main()

    assert calls == {
        "question": "hello",
        "debug": False,
        "checkpoint_db_path": str(db_path),
    }
    assert "ok" in capsys.readouterr().out
