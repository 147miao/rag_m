"""LangSmith 追踪：配置环境变量 + 懒加载 traceable 装饰器（未启用时透明降级）。

注意：装饰器在 import 时注册，但配置在应用启动时才读取 .env，
因此这里用"调用时判断 + 缓存"的方式，保证启动后开启追踪能生效。
"""
from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from typing import Any, TypeVar

from ..config import Settings

logger = logging.getLogger("ragm.tracing")

F = TypeVar("F", bound=Callable[..., Any])

_enabled = False
_traced_cache: dict[tuple[int, str, str], Callable[..., Any]] = {}


def configure_tracing(settings: Settings) -> bool:
    """在创建 LangGraph 之前调用；返回是否启用。"""
    global _enabled
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        _enabled = False
        logger.info("langsmith tracing disabled (no key or flag)")
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    _enabled = True
    logger.info("langsmith tracing enabled: project=%s endpoint=%s", settings.langsmith_project, settings.langsmith_endpoint)
    return True


def tracing_enabled() -> bool:
    return _enabled


def _traced(fn: Callable[..., Any], run_type: str, name: str | None) -> Callable[..., Any]:
    key = (id(fn), run_type, name or "")
    if key not in _traced_cache:
        from langsmith import traceable as ls_traceable

        _traced_cache[key] = ls_traceable(run_type=run_type, name=name or fn.__name__)(fn)
    return _traced_cache[key]


def traceable(*, run_type: str = "chain", name: str | None = None) -> Callable[[F], F]:
    """启用时用 langsmith.traceable 包裹，否则原样调用（判断在调用时进行）。"""
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _enabled:
                return fn(*args, **kwargs)
            try:
                return _traced(fn, run_type, name)(*args, **kwargs)
            except Exception:
                logger.warning("langsmith trace failed, fallback to direct call", exc_info=True)
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
