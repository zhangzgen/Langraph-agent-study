from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


class Config:
    PROJECT_ROOT = Path(
        os.getenv(
            "LANGRAPH_PROJECT_ROOT",
            str(Path(__file__).resolve().parent.parent),
        )
    ).expanduser()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv(
        "OPENAI_BASE_URL",
        "https://token-plan-cn.xiaomimimo.com/v1",
    )
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "mimo-v2.5-pro")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    OPENAI_THINKING_TYPE = os.getenv("OPENAI_THINKING_TYPE", "disabled")
    OPENAI_EXTRA_BODY = {"thinking": {"type": OPENAI_THINKING_TYPE}}

    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    TAVILY_EXTRACT_CONTENT_LIMIT = int(
        os.getenv("TAVILY_EXTRACT_CONTENT_LIMIT", "12000")
    )

    SKILLS_DIR = Path(
        os.getenv("AGENT_SKILLS_DIR", str(PROJECT_ROOT / "skills"))
    ).expanduser()

    CHECKPOINT_DB_PATH = os.getenv(
        "LANGRAPH_CHECKPOINT_DB_PATH",
        "data/checkpoints.sqlite",
    )
    CHECKPOINT_DATABASE_URL = os.getenv("LANGRAPH_CHECKPOINT_DATABASE_URL", "")
    SQLITE_IN_MEMORY = os.getenv("LANGRAPH_SQLITE_IN_MEMORY", ":memory:")

    COMPACT_TOKEN_THRESHOLD = int(
        os.getenv("LANGRAPH_COMPACT_TOKEN_THRESHOLD", "8000")
    )
    RECENT_MESSAGES_TO_KEEP = int(os.getenv("LANGRAPH_RECENT_MESSAGES_TO_KEEP", "8"))

    COMMAND_TIMEOUT_SECONDS = int(os.getenv("LANGRAPH_COMMAND_TIMEOUT_SECONDS", "30"))
    OUTPUT_LIMIT = int(os.getenv("LANGRAPH_OUTPUT_LIMIT", "8000"))

    FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
    FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
    FEISHU_BASE_URL = os.getenv("FEISHU_BASE_URL", "https://open.feishu.cn").rstrip("/")
    FEISHU_CARD_UPDATE_INTERVAL_MS = int(
        os.getenv("FEISHU_CARD_UPDATE_INTERVAL_MS", "250")
    )
    FEISHU_WORKER_COUNT = int(os.getenv("FEISHU_WORKER_COUNT", "4"))
    FEISHU_APPROVAL_DB_PATH = os.getenv(
        "FEISHU_APPROVAL_DB_PATH",
        "data/feishu_approvals.sqlite",
    )

    REACT_PROMPT_ID = os.getenv(
        "LANGRAPH_REACT_PROMPT_ID",
        "langraph-agent-react-system",
    )
    PLAN_PROMPT_ID = os.getenv("LANGRAPH_PLAN_PROMPT_ID", "")
    SUMMARY_PROMPT_ID = os.getenv(
        "LANGRAPH_SUMMARY_PROMPT_ID",
        "langraph-agent-summary",
    )


config = Config()
