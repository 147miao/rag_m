"""E2E 测试辅助：真实服务凭据优先取环境变量，其次仓库根 .env（只读，不打印、不写回）。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_secret(name: str) -> str:
    """返回凭据值：环境变量优先，其次 .env；都没有时返回空串。"""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return ""
    return (dotenv_values(env_file).get(name) or "").strip()
