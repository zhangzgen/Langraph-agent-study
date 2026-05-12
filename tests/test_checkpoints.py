from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from langraph_agent.checkpoints import resolve_checkpoint_db_path, sqlite_checkpointer


def test_resolve_checkpoint_db_path_uses_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "custom.sqlite"
    monkeypatch.setenv("LANGRAPH_CHECKPOINT_DB_PATH", str(db_path))

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
