from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Literal, TypedDict

from psycopg import Connection, OperationalError
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from langraph_agent.config import config


LOGGER = logging.getLogger(__name__)

SessionStatus = Literal[
    "generating",
    "pending_approval",
    "executing",
    "completed",
    "failed",
]
DecisionStatus = Literal["pending", "auto_approved", "approved", "rejected"]


class CardSession(TypedDict):
    card_id: str
    chat_id: str
    thread_id: str
    status: SessionStatus
    answer_text: str
    sequence: int


class DisplayToolCall(TypedDict):
    tool_call_id: str
    tool_name: str
    approval_required: bool
    status: DecisionStatus
    display_content: str
    display_order: int


class CardContentBlock(TypedDict):
    block_id: int
    block_type: Literal["text", "tool"]
    content: str
    tool_call_id: str | None


def resolve_approval_db_path(db_path: str | Path | None = None) -> Path:
    """解析飞书审批数据库路径。

    Description:
        将显式路径或配置中的相对路径转换为可用于 SQLite 连接的绝对路径。
    Args:
        db_path (str | Path | None): 可选的审批数据库路径覆盖值。
    Returns:
        Path: 已解析的 SQLite 数据库文件绝对路径。
    """
    raw_path = db_path or config.FEISHU_APPROVAL_DB_PATH
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def resolve_approval_database_url(database_url: str | None = None) -> str | None:
    """解析飞书审批使用的 PostgreSQL 连接串。

    Description:
        优先使用显式覆盖值，否则读取审批专用连接串配置。专用配置在未通过
        环境变量声明时由全局配置沿用 checkpoint PostgreSQL 地址。
    Args:
        database_url (str | None): 可选的审批 PostgreSQL 连接串覆盖值。
    Returns:
        str | None: 去除空白后的 PostgreSQL 连接串；配置为空时返回 None。
    """
    raw_url = (
        database_url
        if database_url is not None
        else config.FEISHU_APPROVAL_DATABASE_URL
    )
    stripped_url = raw_url.strip() if raw_url else ""
    return stripped_url or None


class FeishuApprovalStore:
    """持久化保存飞书卡片审批会话及按钮幂等状态。"""

    def __init__(
        self,
        db_path: str | Path | None = None,
        database_url: str | None = None,
    ) -> None:
        """初始化审批存储并创建数据表。

        Description:
            默认尝试使用池化 PostgreSQL 保存卡片会话、工具调用与已处理动作；
            未配置数据库或启动时连接不可用时回退到原有 SQLite 文件。显式提供
            SQLite 路径时直接选择 SQLite，便于测试和本地独立运行。
        Args:
            db_path (str | Path | None): 可选的审批数据库文件路径。
            database_url (str | None): 可选的 PostgreSQL 连接串覆盖值。
        Returns:
            None: 初始化仅建立持久化结构。
        """
        self._path = resolve_approval_db_path(db_path)
        self._lock = threading.RLock()
        self._pool: ConnectionPool | None = None
        resolved_url = resolve_approval_database_url(database_url)
        if resolved_url is not None and (db_path is None or database_url is not None):
            pool = ConnectionPool(
                resolved_url,
                kwargs={"row_factory": dict_row},
                min_size=min(
                    max(config.FEISHU_APPROVAL_POOL_MIN_SIZE, 0),
                    max(config.FEISHU_APPROVAL_POOL_MAX_SIZE, 1),
                ),
                max_size=max(config.FEISHU_APPROVAL_POOL_MAX_SIZE, 1),
                timeout=max(config.FEISHU_APPROVAL_POOL_TIMEOUT_SECONDS, 0.1),
                reconnect_timeout=max(config.FEISHU_APPROVAL_POOL_TIMEOUT_SECONDS, 0.1),
                open=False,
                name="feishu-approvals",
            )
            try:
                pool.open(
                    wait=True,
                    timeout=max(config.FEISHU_APPROVAL_POOL_TIMEOUT_SECONDS, 0.1),
                )
            except (OperationalError, PoolTimeout):
                pool.close()
                LOGGER.warning("飞书审批 PostgreSQL 启动连接失败，将回退到 SQLite。")
            else:
                self._pool = pool
                try:
                    self._setup_postgres()
                except Exception:
                    self.close()
                    raise
                return
        self._setup_sqlite()

    def _setup_sqlite(self) -> None:
        """初始化 SQLite 回退存储结构。

        Description:
            创建兼容原实现的数据表，并启用 WAL 与合理等待时间以降低本地并发
            流式更新和按钮动作之间的写锁冲突。
        Args:
            无。
        Returns:
            None: 初始化完成后 SQLite 后端可接受审批读写。
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS feishu_card_sessions (
                    card_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    answer_text TEXT NOT NULL DEFAULT '',
                    sequence INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS feishu_tool_approvals (
                    card_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    approval_required INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    display_content TEXT NOT NULL DEFAULT '',
                    display_order INTEGER NOT NULL,
                    PRIMARY KEY (card_id, tool_call_id),
                    FOREIGN KEY (card_id) REFERENCES feishu_card_sessions(card_id)
                );
                CREATE TABLE IF NOT EXISTS feishu_card_blocks (
                    block_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    card_id TEXT NOT NULL,
                    block_type TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    tool_call_id TEXT,
                    UNIQUE (card_id, tool_call_id),
                    FOREIGN KEY (card_id) REFERENCES feishu_card_sessions(card_id)
                );
                CREATE TABLE IF NOT EXISTS feishu_action_events (
                    action_key TEXT PRIMARY KEY,
                    card_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_feishu_active_sessions_chat
                    ON feishu_card_sessions(chat_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_feishu_tool_card_status_order
                    ON feishu_tool_approvals(card_id, status, display_order);
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(feishu_tool_approvals)")
            }
            if "display_content" not in columns:
                conn.execute(
                    """
                    ALTER TABLE feishu_tool_approvals
                    ADD COLUMN display_content TEXT NOT NULL DEFAULT ''
                    """
                )
            conn.execute(
                """
                INSERT OR IGNORE INTO feishu_card_blocks(card_id, block_type, tool_call_id)
                SELECT card_id, 'tool', tool_call_id
                FROM feishu_tool_approvals
                ORDER BY card_id, display_order
                """
            )

    def _setup_postgres(self) -> None:
        """初始化 PostgreSQL 审批存储结构。

        Description:
            在连接池已预热后建立审批、内容块和幂等事件表及查询索引，并兼容
            已部署过的缺少展示内容字段的旧表结构。
        Args:
            无。
        Returns:
            None: 初始化完成后 PostgreSQL 后端可接受审批读写。
        """
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feishu_card_sessions (
                    card_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    answer_text TEXT NOT NULL DEFAULT '',
                    sequence INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feishu_tool_approvals (
                    card_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    approval_required INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    display_content TEXT NOT NULL DEFAULT '',
                    display_order INTEGER NOT NULL,
                    PRIMARY KEY (card_id, tool_call_id),
                    FOREIGN KEY (card_id) REFERENCES feishu_card_sessions(card_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feishu_card_blocks (
                    block_id BIGSERIAL PRIMARY KEY,
                    card_id TEXT NOT NULL,
                    block_type TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    tool_call_id TEXT,
                    UNIQUE (card_id, tool_call_id),
                    FOREIGN KEY (card_id) REFERENCES feishu_card_sessions(card_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feishu_action_events (
                    action_key TEXT PRIMARY KEY,
                    card_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                ALTER TABLE feishu_tool_approvals
                ADD COLUMN IF NOT EXISTS display_content TEXT NOT NULL DEFAULT ''
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feishu_active_sessions_chat
                ON feishu_card_sessions(chat_id, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feishu_tool_card_status_order
                ON feishu_tool_approvals(card_id, status, display_order)
                """
            )
            conn.execute(
                """
                INSERT INTO feishu_card_blocks(card_id, block_type, tool_call_id)
                SELECT card_id, 'tool', tool_call_id
                FROM feishu_tool_approvals
                ORDER BY card_id, display_order
                ON CONFLICT (card_id, tool_call_id) DO NOTHING
                """
            )

    def close(self) -> None:
        """关闭审批存储持有的 PostgreSQL 连接池。

        Description:
            服务退出时释放已建立的 PostgreSQL 连接；SQLite 短连接后端无需额外
            资源释放，因此该操作对 SQLite 为无副作用调用。
        Args:
            无。
        Returns:
            None: 已存在的数据库连接资源在返回前完成释放。
        """
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def create_session(self, card_id: str, chat_id: str, thread_id: str) -> CardSession:
        """创建一张回答卡片对应的会话记录。

        Description:
            在 Agent 开始生成前保存卡片到聊天与 checkpoint thread 的映射。
        Args:
            card_id (str): CardKit 卡片实体标识。
            chat_id (str): 飞书单聊会话标识。
            thread_id (str): LangGraph checkpoint 会话标识。
        Returns:
            CardSession: 新创建的生成中会话状态。
        """
        with self._operation_lock(), self._connect() as conn:
            self._execute(
                conn,
                """
                INSERT INTO feishu_card_sessions(card_id, chat_id, thread_id, status)
                VALUES (?, ?, ?, 'generating')
                """,
                (card_id, chat_id, thread_id),
            )
        return self.get_session(card_id)

    def get_session(self, card_id: str) -> CardSession:
        """读取指定卡片的持久化会话。

        Description:
            查询用于渲染卡片或恢复 LangGraph 的基础会话字段。
        Args:
            card_id (str): CardKit 卡片实体标识。
        Returns:
            CardSession: 与卡片关联的当前状态。
        """
        with self._operation_lock(), self._connect() as conn:
            row = self._execute(
                conn,
                """
                SELECT card_id, chat_id, thread_id, status, answer_text, sequence
                FROM feishu_card_sessions WHERE card_id = ?
                """,
                (card_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"找不到飞书审批卡片: {card_id}")
        return CardSession(
            card_id=row["card_id"],
            chat_id=row["chat_id"],
            thread_id=row["thread_id"],
            status=row["status"],
            answer_text=row["answer_text"],
            sequence=row["sequence"],
        )

    def find_active_session(self, chat_id: str) -> CardSession | None:
        """查询聊天中尚未结束的卡片任务。

        Description:
            防止同一 checkpoint thread 在审批暂停期间接收新的并行用户轮次。
        Args:
            chat_id (str): 飞书单聊会话标识。
        Returns:
            CardSession | None: 最近的活动会话；不存在时返回 None。
        """
        with self._operation_lock(), self._connect() as conn:
            row = self._execute(
                conn,
                """
                SELECT card_id FROM feishu_card_sessions
                WHERE chat_id = ? AND status IN ('generating', 'pending_approval', 'executing')
                ORDER BY created_at DESC LIMIT 1
                """,
                (chat_id,),
            ).fetchone()
        return self.get_session(row["card_id"]) if row is not None else None

    def list_recoverable_sessions(self) -> list[CardSession]:
        """列出存在已落库审批动作但仍需继续处理的会话。

        Description:
            定位部分审批已决定但卡片可能尚未刷新下一按钮的待审批卡片，以及
            已没有待审项却仍未完成恢复的执行中卡片，供服务重启后继续处理。
        Args:
            无。
        Returns:
            list[CardSession]: 可继续恢复执行的持久化卡片会话列表。
        """
        with self._operation_lock(), self._connect() as conn:
            rows = self._execute(
                conn,
                """
                SELECT s.card_id
                FROM feishu_card_sessions s
                WHERE (
                    s.status = 'pending_approval'
                    AND EXISTS (
                        SELECT 1 FROM feishu_tool_approvals t
                        WHERE t.card_id = s.card_id
                          AND t.approval_required = 1
                          AND t.status IN ('approved', 'rejected')
                    )
                ) OR (
                    s.status = 'executing'
                    AND EXISTS (
                        SELECT 1 FROM feishu_tool_approvals t
                        WHERE t.card_id = s.card_id AND t.approval_required = 1
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM feishu_tool_approvals t
                        WHERE t.card_id = s.card_id AND t.status = 'pending'
                    )
                )
                ORDER BY s.created_at
                """
            ).fetchall()
        return [self.get_session(row["card_id"]) for row in rows]

    def list_abandoned_sessions(self) -> list[CardSession]:
        """列出服务重启后无法继续执行的活动会话。

        Description:
            找出没有审批恢复入口的生成中或执行中卡片，以及缺少可展示审批项的
            异常待审批卡片。这些会话无法从 checkpoint 安全补交原输入，应在
            服务启动时结束，避免持续拦截同一聊天的新问题。
        Args:
            无。
        Returns:
            list[CardSession]: 应被收敛为失败状态的持久化卡片会话列表。
        """
        with self._operation_lock(), self._connect() as conn:
            rows = self._execute(
                conn,
                """
                SELECT s.card_id
                FROM feishu_card_sessions s
                WHERE s.status = 'generating'
                   OR (
                       s.status = 'executing'
                       AND NOT (
                           EXISTS (
                               SELECT 1 FROM feishu_tool_approvals t
                               WHERE t.card_id = s.card_id
                                 AND t.approval_required = 1
                           )
                           AND NOT EXISTS (
                               SELECT 1 FROM feishu_tool_approvals t
                               WHERE t.card_id = s.card_id AND t.status = 'pending'
                           )
                       )
                   )
                   OR (
                       s.status = 'pending_approval'
                       AND NOT EXISTS (
                           SELECT 1 FROM feishu_tool_approvals t
                           WHERE t.card_id = s.card_id
                             AND t.approval_required = 1
                       )
                   )
                ORDER BY s.created_at
                """
            ).fetchall()
        return [self.get_session(row["card_id"]) for row in rows]

    def update_answer(self, card_id: str, text: str) -> None:
        """保存卡片当前已展示的模型正文。

        Description:
            使审批暂停或进程恢复后仍能以同一张卡片呈现此前生成的文本。
        Args:
            card_id (str): CardKit 卡片实体标识。
            text (str): 当前完整的可见回答正文。
        Returns:
            None: 状态通过数据库持久化。
        """
        with self._operation_lock(), self._connect() as conn:
            self._execute(
                conn,
                """
                UPDATE feishu_card_sessions
                SET answer_text = ?, updated_at = CURRENT_TIMESTAMP
                WHERE card_id = ?
                """,
                (text, card_id),
            )

    def append_text_block(self, card_id: str, content: str) -> int:
        """追加模型输出文本块。

        Description:
            在卡片内容时间线末尾新增一段模型正文，用于在工具调用前后分别保留
            输出顺序，并让后续 token 继续更新当前段落。
        Args:
            card_id (str): CardKit 卡片实体标识。
            content (str): 当前文本块初始展示内容。
        Returns:
            int: 新建内容块的数据库标识。
        """
        with self._operation_lock(), self._connect() as conn:
            row = self._execute(
                conn,
                """
                INSERT INTO feishu_card_blocks(card_id, block_type, content)
                VALUES (?, 'text', ?)
                RETURNING block_id
                """,
                (card_id, content),
            ).fetchone()
        return int(row["block_id"])

    def update_text_block(self, block_id: int, content: str) -> None:
        """更新正在流式生成的文本块。

        Description:
            仅覆盖当前生成段落的累计文本，已完成的历史正文和工具块保持原顺序不变。
        Args:
            block_id (int): 当前文本内容块标识。
            content (str): 当前段落截至此刻的完整正文。
        Returns:
            None: 更新内容持久化到卡片时间线。
        """
        with self._operation_lock(), self._connect() as conn:
            self._execute(
                conn,
                "UPDATE feishu_card_blocks SET content = ? WHERE block_id = ?",
                (content, block_id),
            )

    def list_content_blocks(self, card_id: str) -> list[CardContentBlock]:
        """读取卡片按发生顺序保存的内容块。

        Description:
            返回模型正文和工具调用占位块组成的时间线，供渲染时按从上到下布局。
        Args:
            card_id (str): CardKit 卡片实体标识。
        Returns:
            list[CardContentBlock]: 按创建先后排列的卡片内容块列表。
        """
        with self._operation_lock(), self._connect() as conn:
            rows = self._execute(
                conn,
                """
                SELECT block_id, block_type, content, tool_call_id
                FROM feishu_card_blocks WHERE card_id = ? ORDER BY block_id
                """,
                (card_id,),
            ).fetchall()
        return [
            CardContentBlock(
                block_id=row["block_id"],
                block_type=row["block_type"],
                content=row["content"],
                tool_call_id=row["tool_call_id"],
            )
            for row in rows
        ]

    def set_status(self, card_id: str, status: SessionStatus) -> None:
        """更新卡片所处的交互阶段。

        Description:
            持久化生成中、待审批、执行中、完成或失败状态，供卡片渲染和并发拦截使用。
        Args:
            card_id (str): CardKit 卡片实体标识。
            status (SessionStatus): 目标状态值。
        Returns:
            None: 状态通过数据库持久化。
        """
        with self._operation_lock(), self._connect() as conn:
            self._execute(
                conn,
                """
                UPDATE feishu_card_sessions
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE card_id = ?
                """,
                (status, card_id),
            )

    def next_sequence(self, card_id: str) -> int:
        """原子递增卡片更新序号。

        Description:
            为 CardKit 的流式内容或整卡更新提供严格递增的持久化 sequence。
        Args:
            card_id (str): CardKit 卡片实体标识。
        Returns:
            int: 本次远端卡片更新应使用的新序号。
        """
        with self._operation_lock(), self._connect() as conn:
            row = self._execute(
                conn,
                """
                UPDATE feishu_card_sessions
                SET sequence = sequence + 1, updated_at = CURRENT_TIMESTAMP
                WHERE card_id = ?
                RETURNING sequence
                """,
                (card_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"找不到飞书审批卡片: {card_id}")
        return int(row["sequence"])

    def record_tool_calls(self, card_id: str, tool_calls: list[dict[str, Any]]) -> None:
        """保存卡片中需要展示的工具调用。

        Description:
            仅保存工具名称、调用标识和审批属性，不存储或展示工具参数与执行结果。
        Args:
            card_id (str): CardKit 卡片实体标识。
            tool_calls (list[dict[str, Any]]): 渠道运行器识别出的工具调用概要。
        Returns:
            None: 工具展示状态通过数据库持久化。
        """
        with self._operation_lock(), self._connect() as conn:
            current_count = self._execute(
                conn,
                """
                SELECT COUNT(*) AS item_count
                FROM feishu_tool_approvals WHERE card_id = ?
                """,
                (card_id,),
            ).fetchone()["item_count"]
            for offset, tool_call in enumerate(tool_calls, start=1):
                tool_call_id = str(tool_call.get("id") or tool_call.get("name") or "")
                tool_name = str(tool_call.get("name") or "")
                approval_required = bool(tool_call.get("approval_required"))
                initial_status = "pending" if approval_required else "auto_approved"
                self._execute(
                    conn,
                    """
                    INSERT INTO feishu_tool_approvals(
                        card_id, tool_call_id, tool_name, approval_required,
                        status, display_content, display_order
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (card_id, tool_call_id) DO NOTHING
                    """,
                    (
                        card_id,
                        tool_call_id,
                        tool_name,
                        int(approval_required),
                        initial_status,
                        str(tool_call.get("display_content") or ""),
                        current_count + offset,
                    ),
                )
                self._execute(
                    conn,
                    """
                    INSERT INTO feishu_card_blocks(
                        card_id, block_type, tool_call_id
                    ) VALUES (?, 'tool', ?)
                    ON CONFLICT (card_id, tool_call_id) DO NOTHING
                    """,
                    (card_id, tool_call_id),
                )

    def list_tool_calls(self, card_id: str) -> list[DisplayToolCall]:
        """读取卡片内按出现顺序展示的工具列表。

        Description:
            返回用于卡片状态列表渲染和下一项审批选择的精简工具记录。
        Args:
            card_id (str): CardKit 卡片实体标识。
        Returns:
            list[DisplayToolCall]: 按出现顺序排列的工具展示条目。
        """
        with self._operation_lock(), self._connect() as conn:
            rows = self._execute(
                conn,
                """
                SELECT tool_call_id, tool_name, approval_required, status,
                       display_content, display_order
                FROM feishu_tool_approvals WHERE card_id = ?
                ORDER BY display_order
                """,
                (card_id,),
            ).fetchall()
        return [
            DisplayToolCall(
                tool_call_id=row["tool_call_id"],
                tool_name=row["tool_name"],
                approval_required=bool(row["approval_required"]),
                status=row["status"],
                display_content=row["display_content"],
                display_order=row["display_order"],
            )
            for row in rows
        ]

    def decide_tool(
        self,
        card_id: str,
        tool_call_id: str,
        approved: bool,
        action_key: str,
    ) -> bool:
        """提交当前待审核工具的一项审批决定。

        Description:
            在单个事务中验证工具仍为首项待审记录、记录动作幂等键并更新通过或
            拒绝状态；重复点击或越序按钮不会改变已有结论。
        Args:
            card_id (str): CardKit 卡片实体标识。
            tool_call_id (str): 用户当前审核的工具调用标识。
            approved (bool): True 表示批准，False 表示拒绝。
            action_key (str): 按钮携带的稳定幂等键。
        Returns:
            bool: 本次动作首次成功应用时返回 True，否则返回 False。
        """
        with self._operation_lock(), self._connect() as conn:
            if self._execute(
                conn,
                "SELECT 1 FROM feishu_action_events WHERE action_key = ?",
                (action_key,),
            ).fetchone():
                return False
            pending_sql = """
                SELECT tool_call_id FROM feishu_tool_approvals
                WHERE card_id = ? AND status = 'pending'
                ORDER BY display_order LIMIT 1
            """
            if self._pool is not None:
                pending_sql += " FOR UPDATE"
            pending = self._execute(
                conn,
                pending_sql,
                (card_id,),
            ).fetchone()
            if pending is None or pending["tool_call_id"] != tool_call_id:
                return False
            decision = "approved" if approved else "rejected"
            self._execute(
                conn,
                """
                UPDATE feishu_tool_approvals SET status = ?
                WHERE card_id = ? AND tool_call_id = ?
                """,
                (decision, card_id, tool_call_id),
            )
            self._execute(
                conn,
                """
                INSERT INTO feishu_action_events(action_key, card_id, tool_call_id, decision)
                VALUES (?, ?, ?, ?)
                """,
                (action_key, card_id, tool_call_id, decision),
            )
        return True

    def approved_call_ids(self, card_id: str) -> list[str]:
        """返回本轮由用户批准的工具调用标识。

        Description:
            构造 LangGraph `Command(resume=...)` 所需的批准调用列表。
        Args:
            card_id (str): CardKit 卡片实体标识。
        Returns:
            list[str]: 按展示顺序排列的已批准工具调用 ID。
        """
        return [
            item["tool_call_id"]
            for item in self.list_tool_calls(card_id)
            if item["approval_required"] and item["status"] == "approved"
        ]

    def _operation_lock(self) -> Any:
        """创建当前后端所需的操作级锁上下文。

        Description:
            SQLite 依赖进程内锁保持多语句审批事务的顺序；PostgreSQL 通过数据库
            事务和行锁处理并发，避免串行化不同卡片的交互请求。
        Args:
            无。
        Returns:
            Any: 可作为 `with` 上下文使用的同步互斥或空操作上下文。
        """
        return self._lock if self._pool is None else nullcontext()

    def _execute(
        self,
        conn: sqlite3.Connection | Connection,
        statement: str,
        parameters: tuple[Any, ...] = (),
    ) -> Any:
        """执行兼容当前数据库后端的参数化 SQL。

        Description:
            业务 SQL 使用 SQLite 风格占位符编写；在 PostgreSQL 后端执行时转换为
            psycopg 参数格式，以保持审批逻辑只有一份实现。
        Args:
            conn (sqlite3.Connection | Connection): 当前事务使用的数据库连接。
            statement (str): 待执行的参数化 SQL 语句。
            parameters (tuple[Any, ...]): 绑定到 SQL 占位符的参数序列。
        Returns:
            Any: 数据库驱动返回的游标对象，可继续读取查询结果。
        """
        if self._pool is not None:
            statement = statement.replace("?", "%s")
        return conn.execute(statement, parameters)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection | Connection]:
        """取得一次短事务使用的数据库连接。

        Description:
            PostgreSQL 从已预热连接池借用连接并由上下文自动提交或回滚；SQLite
            则创建启用字典行访问、外键校验和忙等待的短连接。
        Args:
            无。
        Returns:
            Iterator[sqlite3.Connection | Connection]: 当前操作可使用的事务连接。
        """
        if self._pool is not None:
            with self._pool.connection(
                timeout=max(config.FEISHU_APPROVAL_POOL_TIMEOUT_SECONDS, 0.1)
            ) as conn:
                yield conn
            return

        conn = sqlite3.connect(
            self._path,
            timeout=max(config.FEISHU_APPROVAL_POOL_TIMEOUT_SECONDS, 0.1),
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            f"PRAGMA busy_timeout = {int(max(config.FEISHU_APPROVAL_POOL_TIMEOUT_SECONDS, 0.1) * 1000)}"
        )
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
