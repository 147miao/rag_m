"""敏感 MCP 工具确认 token。

token = HMAC-SHA256(secret, "tool|args_json|exp")；secret 优先取环境变量
MCP_CONFIRM_SECRET，缺失时使用进程内随机值（重启后旧 token 全部失效）。

待确认表 `pending` 由 threading.Lock 保护，consume 时在同一把锁内原子 pop；
已消费/已过期 token 进入墓碑表，分别映射 409 / 410；完全未知的 token 映射 404。
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SENSITIVE_TOOLS: frozenset[str] = frozenset({"write_note"})
SECRET_ENV = "MCP_CONFIRM_SECRET"
DEFAULT_TTL_S = 60.0
TOMBSTONE_TTL_S = 3600.0
_PROCESS_SECRET = secrets.token_bytes(32)


class ConfirmationError(Exception):
    """确认 token 校验失败基类。"""

    status_code = 400


class Expired(ConfirmationError):
    status_code = 410

    def __init__(self, message: str = "confirmation token expired") -> None:
        super().__init__(message)


class Used(ConfirmationError):
    status_code = 409

    def __init__(self, message: str = "confirmation token already used") -> None:
        super().__init__(message)


class Unknown(ConfirmationError):
    status_code = 404

    def __init__(self, message: str = "unknown confirmation token") -> None:
        super().__init__(message)


# 兼容更显式的 import 名
ExpiredToken = Expired
UsedToken = Used
UnknownToken = Unknown


@dataclass(frozen=True)
class PendingCall:
    tool: str
    args: dict[str, Any]
    exp: float


def _canonical_args(args: dict[str, Any]) -> str:
    return json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _resolve_secret(secret: str | bytes | None) -> bytes:
    if secret is None:
        raw = os.environ.get(SECRET_ENV, "").strip()
        return raw.encode("utf-8") if raw else _PROCESS_SECRET
    if isinstance(secret, str):
        return secret.encode("utf-8")
    return bytes(secret)


class ConfirmationStore:
    def __init__(
        self,
        ttl_s: float | None = None,
        *,
        secret: str | bytes | None = None,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self.ttl_s = float(DEFAULT_TTL_S if ttl_s is None else ttl_s)
        self._secret = _resolve_secret(secret)
        self._now = now_fn or time.time
        self.pending: dict[str, PendingCall] = {}
        self._tombstones: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def _sign(self, tool: str, args: dict[str, Any], exp: float) -> str:
        message = f"{tool}|{_canonical_args(args)}|{exp:.6f}".encode()
        return hmac.new(self._secret, message, hashlib.sha256).hexdigest()

    def issue(self, tool: str, args: dict[str, Any], ttl: float | None = None) -> str:
        """登记一次待确认调用并返回 HMAC token。"""
        if not tool:
            raise ValueError("tool is required")
        if not isinstance(args, dict):
            raise TypeError("args must be a dict")
        exp = self._now() + float(self.ttl_s if ttl is None else ttl)
        call = PendingCall(tool=tool, args=copy.deepcopy(args), exp=exp)
        token = self._sign(call.tool, call.args, call.exp)
        with self._lock:
            now = self._now()
            self._prune(now)
            self.pending[token] = call
        return token

    def consume_call(self, token: str) -> PendingCall:
        """原子消费 token，返回完整调用（tool + args）；失败抛具体异常。"""
        value = str(token or "")
        with self._lock:
            now = self._now()
            tombstone = self._tombstones.get(value)
            if tombstone is not None:
                status, _ = tombstone
                raise Used() if status == "used" else Expired()
            call = self.pending.get(value)
            if call is None:
                raise Unknown()
            if not hmac.compare_digest(value, self._sign(call.tool, call.args, call.exp)):
                raise Unknown()
            del self.pending[value]
            if now >= call.exp:
                self._tombstones[value] = ("expired", now)
                raise Expired()
            self._tombstones[value] = ("used", now)
            return PendingCall(tool=call.tool, args=copy.deepcopy(call.args), exp=call.exp)

    def consume(self, token: str) -> dict[str, Any]:
        """原子消费 token，只返回 args（兼容任务接口约定）。"""
        return copy.deepcopy(self.consume_call(token).args)

    def pending_count(self) -> int:
        with self._lock:
            return len(self.pending)

    def clear(self) -> None:
        with self._lock:
            self.pending.clear()
            self._tombstones.clear()

    def _prune(self, now: float) -> None:
        for value, (_, at) in list(self._tombstones.items()):
            if now - at > TOMBSTONE_TTL_S:
                del self._tombstones[value]
        for value, call in list(self.pending.items()):
            if now > call.exp + TOMBSTONE_TTL_S:
                del self.pending[value]
                self._tombstones[value] = ("expired", now)


__all__ = [
    "DEFAULT_TTL_S",
    "SENSITIVE_TOOLS",
    "ConfirmationError",
    "ConfirmationStore",
    "Expired",
    "ExpiredToken",
    "PendingCall",
    "Unknown",
    "UnknownToken",
    "Used",
    "UsedToken",
]
