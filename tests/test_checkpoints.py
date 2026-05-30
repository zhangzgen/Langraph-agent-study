from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest
from psycopg import OperationalError

from langraph_agent import checkpoints as checkpoint_module
from langraph_agent.checkpoints import (
    checkpoint_saver,
    describe_checkpoint_backend,
    resolve_checkpoint_database_url,
    resolve_checkpoint_db_path,
    sqlite_checkpointer,
)
from langraph_agent.config import config


def test_resolve_checkpoint_db_path_uses_config_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "custom.sqlite"
    monkeypatch.setattr(config, "CHECKPOINT_DB_PATH", str(db_path))

    assert resolve_checkpoint_db_path() == db_path


def test_resolve_checkpoint_db_path_supports_sqlite_memory() -> None:
    assert resolve_checkpoint_db_path(":memory:") == ":memory:"


def test_sqlite_checkpointer_creates_checkpoint_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "checkpoints.sqlite"

    with sqlite_checkpointer(db_path):
        pass

    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {"checkpoints", "writes"} <= table_names


def test_resolve_checkpoint_database_url_uses_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 PostgreSQL checkpoint URL 从 config 读取。

    Description:
        PostgreSQL 连接串不再从 CLI 传入，checkpoint 模块应直接读取配置对象。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。

    Returns:
        None: 该测试只通过断言验证行为。
    """
    monkeypatch.setattr(
        config,
        "CHECKPOINT_DATABASE_URL",
        " postgresql://localhost/langraph_agent ",
    )

    assert resolve_checkpoint_database_url() == "postgresql://localhost/langraph_agent"


def test_resolve_checkpoint_database_url_returns_none_when_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证空 PostgreSQL checkpoint URL 会被视为未配置。

    Description:
        空字符串或仅包含空白字符的配置不应启用 PostgreSQL checkpoint。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。

    Returns:
        None: 该测试只通过断言验证行为。
    """
    monkeypatch.setattr(config, "CHECKPOINT_DATABASE_URL", "   ")

    assert resolve_checkpoint_database_url() is None


def test_checkpoint_saver_uses_postgres_when_config_url_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证通用 checkpoint saver 会优先选择 PostgreSQL。

    Description:
        当存在 PostgreSQL URL 时，checkpoint_saver 应调用 postgres_checkpointer，
        而不是继续创建 SQLite checkpointer。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。

    Returns:
        None: 该测试只通过断言验证行为。
    """
    calls = {}
    sentinel = object()

    @contextmanager
    def fake_postgres_checkpointer():
        """提供测试用 PostgreSQL checkpointer 替身。

        Description:
            记录 PostgreSQL checkpointer 被调用，并返回固定哨兵对象用于断言。

        Args:
            无。

        Returns:
            Iterator[object]: 包含哨兵对象的上下文管理器迭代器。
        """
        calls["called"] = True
        yield sentinel

    monkeypatch.setattr(
        checkpoint_module,
        "postgres_checkpointer",
        fake_postgres_checkpointer,
    )

    monkeypatch.setattr(
        config,
        "CHECKPOINT_DATABASE_URL",
        "postgresql://localhost/langraph_agent",
    )

    with checkpoint_saver() as saver:
        assert saver is sentinel

    assert calls == {"called": True}


def test_checkpoint_saver_falls_back_to_sqlite_when_postgres_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 PostgreSQL 启动连接失败时回退 SQLite。

    Description:
        当配置了 PostgreSQL URL 但启动阶段无法建立连接时，checkpoint_saver
        应使用传入的 SQLite 路径继续创建本地 checkpoint 后端。

    Args:
        tmp_path (Path): pytest 提供的临时目录路径。
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。

    Returns:
        None: 该测试通过断言数据库文件和表结构验证回退行为。
    """
    db_path = tmp_path / "fallback.sqlite"

    @contextmanager
    def fake_postgres_checkpointer():
        """模拟不可用的 PostgreSQL checkpointer。

        Description:
            在进入上下文时抛出连接错误，用于验证 checkpoint_saver 的 SQLite
            回退逻辑是否生效。

        Args:
            无。

        Returns:
            Iterator[object]: 该上下文管理器用于测试异常路径，不会实际返回对象。
        """
        raise OperationalError("database unavailable")
        yield

    monkeypatch.setattr(
        checkpoint_module,
        "postgres_checkpointer",
        fake_postgres_checkpointer,
    )
    monkeypatch.setattr(
        config,
        "CHECKPOINT_DATABASE_URL",
        "postgresql://localhost/langraph_agent",
    )

    with checkpoint_saver(db_path):
        pass

    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {"checkpoints", "writes"} <= table_names


def test_describe_checkpoint_backend_masks_configured_postgres_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 CLI 展示 PostgreSQL URL 时会隐藏密码。

    Description:
        后端描述需要便于排查当前连接目标，但不能把数据库密码直接打印到终端。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。

    Returns:
        None: 该测试只通过断言验证行为。
    """
    monkeypatch.setattr(
        config,
        "CHECKPOINT_DATABASE_URL",
        "postgresql://langraph_agent:secret@localhost:5432/langraph_agent",
    )

    description = describe_checkpoint_backend()

    assert description == (
        "postgres: postgresql://langraph_agent:***@localhost:5432/langraph_agent"
    )
    assert "secret" not in description
