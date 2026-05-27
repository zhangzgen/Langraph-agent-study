from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from psycopg_pool import PoolTimeout

from langraph_agent import feishu_approvals as approval_module
from langraph_agent.config import config
from langraph_agent.feishu_approvals import (
    FeishuApprovalStore,
    resolve_approval_database_url,
)


def test_resolve_approval_database_url_uses_dedicated_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证审批存储优先采用自身 PostgreSQL 配置。

    Description:
        设置飞书审批专用连接串后，解析函数应返回规范化后的地址，避免意外
        使用其他数据库后端。
    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。
    Returns:
        None: 测试仅通过断言校验解析结果。
    """
    monkeypatch.setattr(
        config,
        "FEISHU_APPROVAL_DATABASE_URL",
        " postgresql://localhost/feishu_approval ",
    )

    assert (
        resolve_approval_database_url()
        == "postgresql://localhost/feishu_approval"
    )


def test_explicit_sqlite_path_does_not_open_postgres_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证显式 SQLite 路径继续保留原有本地后端行为。

    Description:
        即使配置了 PostgreSQL，测试或调用方传入 SQLite 文件路径时也不得创建
        连接池，从而保持原有可控的本地存储覆盖能力。
    Args:
        tmp_path (Path): pytest 提供的临时目录。
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。
    Returns:
        None: 测试仅通过断言校验 SQLite 写入结果。
    """
    monkeypatch.setattr(
        config,
        "FEISHU_APPROVAL_DATABASE_URL",
        "postgresql://localhost/feishu_approval",
    )

    def fail_if_pool_created(*args: Any, **kwargs: Any) -> Any:
        """阻止显式 SQLite 测试意外创建连接池。

        Description:
            在 PostgreSQL 池被错误实例化时立即使测试失败。
        Args:
            *args (Any): 被拦截的连接池位置参数。
            **kwargs (Any): 被拦截的连接池关键字参数。
        Returns:
            Any: 正确执行路径不会返回结果。
        """
        raise AssertionError("显式 SQLite 路径不应创建 PostgreSQL 连接池")

    monkeypatch.setattr(approval_module, "ConnectionPool", fail_if_pool_created)
    store = FeishuApprovalStore(tmp_path / "approvals.sqlite")
    store.create_session("card-1", "chat-1", "feishu:chat-1")

    assert store.get_session("card-1")["status"] == "generating"


def test_postgres_startup_timeout_falls_back_to_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 PostgreSQL 不可连接时启动阶段回退 SQLite。

    Description:
        模拟连接池预热超时，确认审批存储仍能通过配置的 SQLite 文件接收会话
        写入，同时释放未成功启用的池对象。
    Args:
        tmp_path (Path): pytest 提供的临时目录。
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。
    Returns:
        None: 测试仅通过断言校验回退行为。
    """
    lifecycle: list[str] = []

    class UnavailablePool:
        """模拟无法建立数据库连接的 PostgreSQL 池。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """记录模拟池的构造动作。

            Description:
                保留创建轨迹以供测试断言，不实际建立网络连接。
            Args:
                *args (Any): 连接池位置参数。
                **kwargs (Any): 连接池关键字参数。
            Returns:
                None: 仅初始化测试状态。
            """
            lifecycle.append("created")

        def open(self, wait: bool, timeout: float) -> None:
            """模拟池预热连接超时。

            Description:
                抛出与真实 PostgreSQL 不可用时一致的池等待超时异常。
            Args:
                wait (bool): 是否等待预热完成。
                timeout (float): 最大等待秒数。
            Returns:
                None: 该方法始终抛出异常。
            """
            raise PoolTimeout("database unavailable")

        def close(self) -> None:
            """记录失败池被关闭。

            Description:
                模拟释放启动失败后的池资源。
            Args:
                无。
            Returns:
                None: 仅更新调用记录。
            """
            lifecycle.append("closed")

    monkeypatch.setattr(approval_module, "ConnectionPool", UnavailablePool)
    monkeypatch.setattr(
        config,
        "FEISHU_APPROVAL_DATABASE_URL",
        "postgresql://localhost/feishu_approval",
    )
    monkeypatch.setattr(config, "FEISHU_APPROVAL_DB_PATH", str(tmp_path / "fallback.sqlite"))
    monkeypatch.setattr(config, "FEISHU_APPROVAL_POOL_TIMEOUT_SECONDS", 0.1)

    store = FeishuApprovalStore()
    store.create_session("card-fallback", "chat-1", "feishu:chat-1")

    assert lifecycle == ["created", "closed"]
    assert store.get_session("card-fallback")["status"] == "generating"
    assert (tmp_path / "fallback.sqlite").exists()


def test_postgres_store_prewarms_and_closes_connection_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 PostgreSQL 路径复用预热连接池并在关闭时释放资源。

    Description:
        使用无网络的记录型连接池执行初始化，确认后端创建表结构时从池借用
        连接，并按配置传递池大小，退出时显式关闭池。
    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的运行时替换工具。
    Returns:
        None: 测试仅通过断言校验连接池生命周期。
    """
    lifecycle: list[str] = []
    pool_options: dict[str, Any] = {}
    statements: list[str] = []

    class RecordingConnection:
        """记录初始化 SQL 的测试数据库连接。"""

        def execute(
            self,
            statement: str,
            parameters: tuple[Any, ...] = (),
        ) -> RecordingConnection:
            """保存执行过的 PostgreSQL 语句。

            Description:
                接收初始化 DDL 并把归一化文本追加到断言列表。
            Args:
                statement (str): 待记录的 SQL 语句。
                parameters (tuple[Any, ...]): SQL 绑定参数。
            Returns:
                RecordingConnection: 可链式作为模拟游标返回的当前对象。
            """
            statements.append(" ".join(statement.split()))
            return self

    class RecordingPool:
        """记录 PostgreSQL 池初始化与关闭动作的测试替身。"""

        def __init__(self, conninfo: str, **kwargs: Any) -> None:
            """保存池配置而不访问真实 PostgreSQL。

            Description:
                捕获连接串和池大小设置，供测试校验性能相关配置是否生效。
            Args:
                conninfo (str): PostgreSQL 连接串。
                **kwargs (Any): 连接池配置参数。
            Returns:
                None: 仅记录构造输入。
            """
            pool_options.update({"conninfo": conninfo, **kwargs})

        def open(self, wait: bool, timeout: float) -> None:
            """记录池预热动作。

            Description:
                模拟可用 PostgreSQL 池成功完成预热。
            Args:
                wait (bool): 是否等待预热完成。
                timeout (float): 预热最大等待秒数。
            Returns:
                None: 仅记录生命周期动作。
            """
            lifecycle.append("opened")

        @contextmanager
        def connection(self, timeout: float) -> Any:
            """提供初始化 DDL 所用的模拟事务连接。

            Description:
                返回记录型连接，以验证表结构初始化通过连接池完成。
            Args:
                timeout (float): 从池借用连接的等待秒数。
            Returns:
                Any: 可执行 SQL 的模拟连接上下文。
            """
            lifecycle.append("borrowed")
            yield RecordingConnection()

        def close(self) -> None:
            """记录连接池资源释放动作。

            Description:
                模拟服务关闭阶段释放 PostgreSQL 连接。
            Args:
                无。
            Returns:
                None: 仅记录生命周期动作。
            """
            lifecycle.append("closed")

    monkeypatch.setattr(approval_module, "ConnectionPool", RecordingPool)
    monkeypatch.setattr(config, "FEISHU_APPROVAL_POOL_MIN_SIZE", 1)
    monkeypatch.setattr(config, "FEISHU_APPROVAL_POOL_MAX_SIZE", 5)
    monkeypatch.setattr(config, "FEISHU_APPROVAL_POOL_TIMEOUT_SECONDS", 2.0)

    store = FeishuApprovalStore(database_url="postgresql://localhost/feishu_approval")
    store.close()

    assert lifecycle == ["opened", "borrowed", "closed"]
    assert pool_options["min_size"] == 1
    assert pool_options["max_size"] == 5
    assert any("CREATE TABLE IF NOT EXISTS feishu_action_events" in item for item in statements)
