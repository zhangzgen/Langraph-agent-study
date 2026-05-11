from __future__ import annotations

from pathlib import Path


XIAOMI_DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
XIAOMI_DEFAULT_MODEL = "mimo-v2.5-pro"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SKILLS_DIR = PROJECT_ROOT / "skills"
COMMAND_TIMEOUT_SECONDS = 30
OUTPUT_LIMIT = 8000
