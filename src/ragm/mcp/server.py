"""自写 MCP Server：list_dir / read_file / write_note。

安全边界：所有工具只接受沙箱根（环境变量 MCP_SANDBOX_ROOT）内的路径。相对路径按
沙箱根解析，绝对路径也必须落在根内；`Path.resolve()` 会展开 `..` 与符号链接，
因此越界与 symlink 逃逸都抛 `ToolError`。

运行方式（由 client 拉起）：`python -m ragm.mcp.server`，stdio transport。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .. import __version__

SANDBOX_ENV = "MCP_SANDBOX_ROOT"
SERVER_NAME = "ragm-mcp"
DEFAULT_MAX_CHARS = 4000
MAX_READ_CHARS = 1_000_000


def _sandbox_root() -> Path:
    raw = os.environ.get(SANDBOX_ENV, "").strip()
    if not raw:
        raise ToolError(f"{SANDBOX_ENV} is not set; refusing filesystem access")
    try:
        root = Path(raw).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ToolError(f"invalid {SANDBOX_ENV}: {exc}") from exc
    if not root.is_dir():
        raise ToolError(f"{SANDBOX_ENV} is not a directory: {root}")
    return root


def resolve_path(raw: str) -> tuple[Path, Path]:
    """返回 (sandbox_root, resolved_path)；越界或非法路径抛 ToolError。"""
    root = _sandbox_root()
    text = str(raw or "").strip()
    candidate = root if not text else Path(text).expanduser()
    if text and not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ToolError(f"invalid path {raw!r}: {exc}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ToolError(f"path escapes sandbox root: {raw!r}") from exc
    return root, resolved


def _relative(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return "."
    return rel.as_posix() or "."


def list_dir(path: str = ".") -> dict[str, Any]:
    """列出沙箱内目录（默认根目录）。"""
    root, target = resolve_path(path)
    if not target.exists():
        raise ToolError(f"directory not found: {_relative(root, target)}")
    if not target.is_dir():
        raise ToolError(f"not a directory: {_relative(root, target)}")
    entries: list[dict[str, Any]] = []
    try:
        for item in sorted(target.iterdir(), key=lambda p: p.name):
            try:
                is_dir = item.is_dir()
                size = None if is_dir else item.stat().st_size
            except OSError:
                is_dir = False
                size = None
            entries.append({"name": item.name, "type": "dir" if is_dir else "file", "size": size})
    except OSError as exc:
        raise ToolError(f"cannot list directory {_relative(root, target)}: {exc}") from exc
    return {"path": _relative(root, target), "entries": entries}


def read_file(path: str, max_chars: int = DEFAULT_MAX_CHARS) -> dict[str, Any]:
    """读取沙箱内 UTF-8 文本，最多返回 max_chars 个字符。"""
    root, target = resolve_path(path)
    if not target.exists():
        raise ToolError(f"file not found: {_relative(root, target)}")
    if not target.is_file():
        raise ToolError(f"not a file: {_relative(root, target)}")
    try:
        limit = int(max_chars)
    except (TypeError, ValueError) as exc:
        raise ToolError("max_chars must be an integer") from exc
    if limit < 1:
        raise ToolError("max_chars must be >= 1")
    limit = min(limit, MAX_READ_CHARS)
    try:
        with target.open("r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(limit + 1)
        size = target.stat().st_size
    except OSError as exc:
        raise ToolError(f"cannot read file {_relative(root, target)}: {exc}") from exc
    return {
        "path": _relative(root, target),
        "text": text[:limit],
        "truncated": len(text) > limit,
        "size": size,
    }


def write_note(path: str, text: str) -> dict[str, Any]:
    """在沙箱内写 UTF-8 文本；父目录不存在时自动创建。"""
    root, target = resolve_path(path)
    if target == root:
        raise ToolError("cannot overwrite sandbox root")
    if target.exists() and target.is_dir():
        raise ToolError(f"target is a directory: {_relative(root, target)}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = str(text)
        target.write_text(data, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"cannot write file {_relative(root, target)}: {exc}") from exc
    return {"path": _relative(root, target), "written": True, "bytes": len(data.encode("utf-8"))}


def create_server() -> MCPServer:
    """构造 MCP Server（in-memory 测试与 stdio 子进程共用）。"""
    server: MCPServer = MCPServer(
        name=SERVER_NAME,
        version=__version__,
        description="RAG-M sandboxed MCP tools",
        log_level="WARNING",
    )
    server.add_tool(
        list_dir,
        description="List entries of a directory inside the configured sandbox root.",
    )
    server.add_tool(
        read_file,
        description="Read a UTF-8 text file inside the sandbox root (optionally truncated).",
    )
    server.add_tool(
        write_note,
        description="Write a UTF-8 note inside the sandbox root; requires user confirmation at the API layer.",
    )
    return server


def main() -> None:
    create_server().run("stdio")


if __name__ == "__main__":
    main()
