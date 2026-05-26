from __future__ import annotations

import sys
from pathlib import Path

import pytest

from langraph_agent import cli
from langraph_agent.config import config


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
    monkeypatch.setattr(config, "SKILLS_DIR", tmp_path)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
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
        stream_output: bool = True,
        plan_mode: bool = False,
    ) -> None:
        calls["thread_id"] = thread_id
        calls["debug"] = debug
        calls["checkpoint_db_path"] = checkpoint_db_path
        calls["stream_output"] = stream_output
        calls["plan_mode"] = plan_mode

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
        "stream_output": True,
        "plan_mode": False,
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
        stream_output: bool = True,
        plan_mode: bool = False,
    ) -> AIMessage:
        """记录 CLI 传入 graph.run 的参数。

        Description:
            测试替身函数不执行真实图，只保存参数并返回固定消息。
        Args:
            question (str): CLI 解析出的用户问题。
            debug (bool): CLI 解析出的 debug 开关。
            checkpoint_db_path (str | None): CLI 解析出的 checkpoint 路径。
            stream_output (bool): CLI 传入的流式输出开关。
            plan_mode (bool): CLI 解析出的 plan 模式开关。
        Returns:
            AIMessage: 固定的测试响应消息。
        """
        calls["question"] = question
        calls["debug"] = debug
        calls["checkpoint_db_path"] = checkpoint_db_path
        calls["stream_output"] = stream_output
        calls["plan_mode"] = plan_mode
        if stream_output:
            print("ok", end="")
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
        "stream_output": True,
        "plan_mode": False,
    }
    assert "ok" in capsys.readouterr().out


def test_cli_run_passes_plan_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 --plan 会启用计划模式。

    Description:
        CLI 需要把 --plan 参数传递给 graph.run，使一次性任务先进入计划审核流程。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。
        capsys (pytest.CaptureFixture[str]): pytest 提供的标准输出捕获工具。

    Returns:
        None: 该测试只通过断言验证行为。
    """
    from langchain_core.messages import AIMessage

    from langraph_agent import graph

    calls = {}

    def fake_run(
        question: str,
        debug: bool,
        checkpoint_db_path: str | None,
        stream_output: bool = True,
        plan_mode: bool = False,
    ) -> AIMessage:
        """记录 --plan 场景下 CLI 传入 graph.run 的参数。

        Description:
            测试替身函数不执行真实图，只验证 plan_mode 会被设置为 True。
        Args:
            question (str): CLI 解析出的用户问题。
            debug (bool): CLI 解析出的 debug 开关。
            checkpoint_db_path (str | None): CLI 解析出的 checkpoint 路径。
            stream_output (bool): CLI 传入的流式输出开关。
            plan_mode (bool): CLI 解析出的 plan 模式开关。
        Returns:
            AIMessage: 固定的测试响应消息。
        """
        calls["question"] = question
        calls["debug"] = debug
        calls["checkpoint_db_path"] = checkpoint_db_path
        calls["stream_output"] = stream_output
        calls["plan_mode"] = plan_mode
        print("ok", end="")
        return AIMessage(content="ok")

    monkeypatch.setattr(graph, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["react-agent", "--plan", "hello"])

    cli.main()

    assert calls == {
        "question": "hello",
        "debug": False,
        "checkpoint_db_path": None,
        "stream_output": True,
        "plan_mode": True,
    }
    assert "ok" in capsys.readouterr().out
