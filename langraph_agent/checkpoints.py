from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from langraph_agent.config import PROJECT_ROOT


CHECKPOINT_DB_ENV = "LANGRAPH_CHECKPOINT_DB_PATH"
DEFAULT_CHECKPOINT_DB_PATH = PROJECT_ROOT / "data" / "checkpoints.sqlite"
SQLITE_IN_MEMORY = ":memory:"


def resolve_checkpoint_db_path(db_path: str | Path | None = None) -> str | Path:
    """解析 CLI 使用的 SQLite checkpoint 数据库路径。"""
    raw_path = db_path or os.getenv(CHECKPOINT_DB_ENV) or DEFAULT_CHECKPOINT_DB_PATH
    if str(raw_path) == SQLITE_IN_MEMORY:
        return SQLITE_IN_MEMORY

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@contextmanager
def sqlite_checkpointer(
    db_path: str | Path | None = None,
) -> Iterator[BaseCheckpointSaver]:
    """创建 SQLite checkpointer，并在图运行期间保持连接打开。"""
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
