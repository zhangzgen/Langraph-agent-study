from __future__ import annotations

from pathlib import Path


OPENAI_DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
OPENAI_DEFAULT_MODEL = "mimo-v2.5-pro"
OPENAI_THINKING_TYPE = "disabled"
OPENAI_EXTRA_BODY = {"thinking": {"type": OPENAI_THINKING_TYPE}}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SKILLS_DIR = PROJECT_ROOT / "skills"
COMMAND_TIMEOUT_SECONDS = 30
OUTPUT_LIMIT = 8000
