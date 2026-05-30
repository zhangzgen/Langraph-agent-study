from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from psycopg import OperationalError

from langraph_agent.config import config


LOGGER = logging.getLogger(__name__)


def resolve_checkpoint_db_path(db_path: str | Path | None = None) -> str | Path:
    """解析 CLI 使用的 SQLite checkpoint 数据库路径。

    Description:
        将传入的 SQLite checkpoint 路径或默认配置解析为可直接连接的路径。

    Args:
        db_path (str | Path | None): 用户显式传入的 SQLite 数据库路径。

    Returns:
        str | Path: 解析后的 SQLite 数据库路径；`:memory:` 会保持字符串形式。
    """
    raw_path = db_path or config.CHECKPOINT_DB_PATH
    if str(raw_path) == config.SQLITE_IN_MEMORY:
        return config.SQLITE_IN_MEMORY

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return config.PROJECT_ROOT / path


def resolve_checkpoint_database_url() -> str | None:
    """解析 PostgreSQL checkpoint 数据库连接串。

    Description:
        从 config.CHECKPOINT_DATABASE_URL 读取 PostgreSQL 连接串，空字符串会被
        视为未启用 PostgreSQL checkpoint。

    Args:
        无。

    Returns:
        str | None: 可用于连接 PostgreSQL 的连接串；未配置时返回 None。
    """
    raw_url = config.CHECKPOINT_DATABASE_URL
    if raw_url is None:
        return None

    stripped_url = raw_url.strip()
    return stripped_url or None


@contextmanager
def sqlite_checkpointer(
    db_path: str | Path | None = None,
) -> Iterator[BaseCheckpointSaver]:
    """创建 SQLite checkpointer，并在图运行期间保持连接打开。

    Description:
        初始化 SQLite checkpoint 表结构，并把连接生命周期绑定到上下文管理器。

    Args:
        db_path (str | Path | None): SQLite checkpoint 数据库路径。

    Returns:
        Iterator[BaseCheckpointSaver]: 可传给 LangGraph compile 的 checkpointer。
    """
    resolved_path = resolve_checkpoint_db_path(db_path)
    if isinstance(resolved_path, Path):
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(resolved_path, check_same_thread=False)
    try:
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        yield checkpointer
    finally:
        conn.close()


@contextmanager
def postgres_checkpointer() -> Iterator[BaseCheckpointSaver]:
    """创建 PostgreSQL checkpointer，并在图运行期间保持连接打开。

    Description:
        使用 LangGraph 的 PostgresSaver 初始化 checkpoint 表结构，适合 Docker
        或远程 PostgreSQL 场景下跨进程保存对话状态。

    Args:
        无。

    Returns:
        Iterator[BaseCheckpointSaver]: 可传给 LangGraph compile 的 checkpointer。
    """
    resolved_url = resolve_checkpoint_database_url()
    if resolved_url is None:
        raise ValueError("PostgreSQL checkpoint requires a database URL.")

    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(resolved_url) as checkpointer:
        checkpointer.setup()
        yield checkpointer


@contextmanager
def checkpoint_saver(
    db_path: str | Path | None = None,
) -> Iterator[BaseCheckpointSaver]:
    """根据配置创建当前会话使用的 checkpoint saver。

    Description:
        当配置了 PostgreSQL 连接串时使用 PostgresSaver；否则回退到项目原有的
        SQLite saver，保证未配置 PostgreSQL 的本地使用方式不变。

    Args:
        db_path (str | Path | None): SQLite checkpoint 数据库路径。

    Returns:
        Iterator[BaseCheckpointSaver]: 可传给 LangGraph compile 的 checkpointer。
    """
    if resolve_checkpoint_database_url() is not None:
        try:
            with postgres_checkpointer() as checkpointer:
                yield checkpointer
            return
        except OperationalError:
            LOGGER.warning("LangGraph checkpoint PostgreSQL 启动连接失败，将回退到 SQLite。")

    with sqlite_checkpointer(db_path) as checkpointer:
        yield checkpointer


def describe_checkpoint_backend(
    db_path: str | Path | None = None,
) -> str:
    """生成当前 checkpoint 后端的人类可读描述。

    Description:
        用于 CLI 输出当前使用的 checkpoint 后端，并在展示 PostgreSQL URL 时隐藏密码。

    Args:
        db_path (str | Path | None): SQLite checkpoint 数据库路径。

    Returns:
        str: 当前 checkpoint 后端的简短说明。
    """
    resolved_url = resolve_checkpoint_database_url()
    if resolved_url is None:
        return f"sqlite: {resolve_checkpoint_db_path(db_path)}"

    parsed_url = urlsplit(resolved_url)
    if parsed_url.password is None:
        return f"postgres: {resolved_url}"

    username = parsed_url.username or ""
    hostname = parsed_url.hostname or ""
    port = f":{parsed_url.port}" if parsed_url.port is not None else ""
    netloc = f"{username}:***@{hostname}{port}"
    return f"postgres: {urlunsplit(parsed_url._replace(netloc=netloc))}"
