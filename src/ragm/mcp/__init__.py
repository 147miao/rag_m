"""MCP 工具接入（v1.1 W2）。

模块布局：
- `server`：自写 stdio MCP Server（list_dir/read_file/write_note），所有路径限制在
  MCP_SANDBOX_ROOT 内；
- `client`：每次调用拉起一个 stdio 子进程的 MCPClientManager（同步 wrapper 供非
  async 调用方使用）；
- `confirmation`：敏感工具确认 token（HMAC + TTL + 原子单次消费）。

默认关闭（MCP_ENABLED=false）：lifespan 只装配确认存储，`app.state.mcp` 为 None，
MCP 路由返回 503。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .confirmation import SENSITIVE_TOOLS

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_DIRNAME = "mcp_sandbox"


def resolve_sandbox_root(settings: Any) -> Path:
    """解析 MCP 沙箱根：优先 MCP_SANDBOX_ROOT，空则落到 data_root/data/mcp_sandbox。"""
    raw = str(getattr(settings, "mcp_sandbox_root", "") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    data_root = getattr(settings, "data_root", None)
    if data_root is None:
        raise RuntimeError("MCP sandbox root is not configured (MCP_SANDBOX_ROOT)")
    return (Path(data_root) / "data" / DEFAULT_SANDBOX_DIRNAME).resolve()


def init_mcp(app: Any, settings: Any) -> None:
    """lifespan 装配入口：初始化确认存储；启用时再创建 stdio client manager。"""
    from .confirmation import ConfirmationStore

    app.state.mcp_confirmations = ConfirmationStore(ttl_s=settings.mcp_confirm_ttl_s)
    app.state.mcp = None
    if not getattr(settings, "mcp_enabled", False):
        logger.info("MCP disabled (MCP_ENABLED=false)")
        return
    try:
        root = resolve_sandbox_root(settings)
        root.mkdir(parents=True, exist_ok=True)
        from .client import MCPClientManager

        app.state.mcp = MCPClientManager(root)
    except Exception:
        logger.warning("MCP init failed; MCP API will return 503", exc_info=True)
    else:
        logger.info("MCP ready: sandbox=%s", app.state.mcp.sandbox_root)


__all__ = ["DEFAULT_SANDBOX_DIRNAME", "SENSITIVE_TOOLS", "init_mcp", "resolve_sandbox_root"]
