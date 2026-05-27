from __future__ import annotations

from unittest.mock import Mock

import pytest
from langchain_core.prompts import ChatPromptTemplate

from langraph_agent import prompt as prompt_module
from langraph_agent.config import config


@pytest.fixture(autouse=True)
def clear_prompt_cache():
    """在每个提示词测试前后清空模板加载缓存。

    Description:
        隔离使用临时目录和模拟远程客户端的测试，避免缓存模板影响后续用例。
    Args:
        无。
    Returns:
        Generator[None, None, None]: 在测试执行前后触发缓存清理的夹具迭代器。
    """
    prompt_module.load_prompt.cache_clear()
    yield
    prompt_module.load_prompt.cache_clear()


def test_load_prompt_prefers_langsmith_template(monkeypatch) -> None:
    """验证配置远程模板时优先使用 LangSmith 返回内容。

    Description:
        以模拟的 LangSmith 客户端返回远程模板，确认加载器不读取本地回退内容。
    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。
    Returns:
        None: 该测试通过断言校验加载优先级。
    """
    remote_prompt = ChatPromptTemplate.from_messages([("system", "远程 {value}")])
    client = Mock()
    client.pull_prompt.return_value = remote_prompt
    monkeypatch.setattr(prompt_module, "Client", Mock(return_value=client))

    prompt = prompt_module.load_prompt("remote-prompt", "missing.txt")
    messages = prompt.invoke({"value": "模板"}).to_messages()

    assert messages[0].content == "远程 模板"
    client.pull_prompt.assert_called_once_with("remote-prompt")


def test_load_prompt_falls_back_to_local_file(monkeypatch, tmp_path) -> None:
    """验证远程模板不可用时读取 prompts 下的本地模板。

    Description:
        模拟 LangSmith 拉取失败，并使用临时项目目录中的模板验证本地渲染结果。
    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。
        tmp_path (pathlib.Path): pytest 创建的临时项目根目录。
    Returns:
        None: 该测试通过断言校验本地回退行为。
    """
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "fallback.txt").write_text("本地 {value}", encoding="utf-8")
    client = Mock()
    client.pull_prompt.side_effect = RuntimeError("remote unavailable")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(prompt_module, "Client", Mock(return_value=client))

    prompt = prompt_module.load_prompt("missing-prompt", "fallback.txt")
    messages = prompt.invoke({"value": "模板"}).to_messages()

    assert messages[0].content == "本地 模板"
    client.pull_prompt.assert_called_once_with("missing-prompt")


def test_plan_prompt_reads_local_template_without_remote_id(monkeypatch, tmp_path) -> None:
    """验证计划提示词在未配置远程标识时由本地文件渲染。

    Description:
        提供包含动态参数的临时计划模板，确认构建入口继续注入 Skill 与会话摘要。
    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。
        tmp_path (pathlib.Path): pytest 创建的临时项目根目录。
    Returns:
        None: 该测试通过断言校验计划提示词渲染内容。
    """
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "plan.txt").write_text(
        "计划 {skill_catalog}{session_summary_block}", encoding="utf-8"
    )
    client_factory = Mock()
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "PLAN_PROMPT_ID", "")
    monkeypatch.setattr(prompt_module, "Client", client_factory)

    messages = prompt_module.build_plan_prompt_messages(
        skill_catalog="技能目录",
        session_summary="已有摘要",
    )

    assert messages[0].content == "计划 技能目录\n\n当前会话摘要:\n已有摘要"
    client_factory.assert_not_called()
