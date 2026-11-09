"""W3 LLM 降级：多 provider 顺序回退、首 token 超时、熔断与 health 状态。"""
from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from ragm.api import routes
from ragm.api.main import create_app
from ragm.api.schemas import HealthCheck
from ragm.config import Settings, get_settings
from ragm.rag.llm import LLMError
from ragm.rag.llm_router import CircuitBreaker, FallbackLLM, build_llm


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeClient:
    """可注入的 LLMClient 替身：记录调用次数并按配置延迟/抛错/流式输出。"""

    def __init__(
        self,
        name: str,
        *,
        chat_result: dict | None = None,
        chat_error: Exception | None = None,
        chat_delay: float = 0.0,
        tools_result: dict | None = None,
        tools_error: Exception | None = None,
        tools_delay: float = 0.0,
        stream_tokens: tuple[str, ...] = (),
        stream_delay: float = 0.0,
        stream_error_before: Exception | None = None,
        stream_error_after: Exception | None = None,
    ) -> None:
        self.name = name
        self.chat_result = chat_result
        self.chat_error = chat_error
        self.chat_delay = chat_delay
        self.tools_result = tools_result
        self.tools_error = tools_error
        self.tools_delay = tools_delay
        self.stream_tokens = stream_tokens
        self.stream_delay = stream_delay
        self.stream_error_before = stream_error_before
        self.stream_error_after = stream_error_after
        self.chat_calls = 0
        self.tools_calls = 0
        self.stream_calls = 0

    def chat(self, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False) -> dict:
        self.chat_calls += 1
        if self.chat_delay:
            time.sleep(self.chat_delay)
        if self.chat_error is not None:
            raise self.chat_error
        return dict(self.chat_result or {"text": self.name, "tokens_in": 1, "tokens_out": 1})

    def chat_tools(self, messages: list[dict], tools: list[dict], max_tokens: int = 1024) -> dict:
        self.tools_calls += 1
        if self.tools_delay:
            time.sleep(self.tools_delay)
        if self.tools_error is not None:
            raise self.tools_error
        return dict(self.tools_result or {"text": self.name, "tool_calls": [], "tokens_in": 1, "tokens_out": 1})

    def stream_chat(self, messages: list[dict], max_tokens: int = 1024):
        self.stream_calls += 1
        if self.stream_error_before is not None:
            raise self.stream_error_before

        def gen():
            if self.stream_delay:
                time.sleep(self.stream_delay)
            yield from self.stream_tokens
            if self.stream_error_after is not None:
                raise self.stream_error_after

        return gen()


# ---------- chat / chat_tools ----------
def test_chat_falls_back_on_error() -> None:
    primary = FakeClient("primary", chat_error=RuntimeError("primary down"))
    backup = FakeClient("backup", chat_result={"text": "backup answer", "tokens_in": 2, "tokens_out": 3})
    llm = FallbackLLM([primary, backup])

    out = llm.chat([{"role": "user", "content": "hi"}])

    assert out["text"] == "backup answer"
    assert primary.chat_calls == 1
    assert backup.chat_calls == 1
    assert llm.last_provider == "backup"
    assert llm.provider_status() == {
        "primary": {"state": "closed", "failures": 1},
        "backup": {"state": "closed", "failures": 0},
    }


def test_chat_timeout_falls_back() -> None:
    primary = FakeClient("primary", chat_delay=0.2)
    backup = FakeClient("backup", chat_result={"text": "backup", "tokens_in": 0, "tokens_out": 0})
    llm = FallbackLLM([primary, backup], chat_timeout=0.01)

    out = llm.chat([{"role": "user", "content": "hi"}])

    assert out["text"] == "backup"
    assert llm.last_provider == "backup"
    assert llm.provider_status()["primary"]["failures"] == 1


def test_chat_tools_timeout_falls_back() -> None:
    primary = FakeClient("primary", tools_delay=0.2)
    backup = FakeClient("backup", tools_result={"text": "", "tool_calls": [{"id": "c1"}], "tokens_in": 1, "tokens_out": 2})
    llm = FallbackLLM([primary, backup], chat_timeout=0.01)

    out = llm.chat_tools([{"role": "user", "content": "hi"}], [{"type": "function"}], max_tokens=600)

    assert out["tool_calls"] == [{"id": "c1"}]
    assert primary.tools_calls == 1
    assert backup.tools_calls == 1
    assert llm.last_provider == "backup"


def test_all_providers_fail_raises_llm_error() -> None:
    llm = FallbackLLM(
        [
            FakeClient("a", chat_error=RuntimeError("a down")),
            FakeClient("b", chat_error=RuntimeError("b down")),
        ]
    )

    with pytest.raises(LLMError, match="all LLM providers failed during chat"):
        llm.chat([{"role": "user", "content": "hi"}])


# ---------- streaming ----------
def test_stream_first_token_timeout_falls_back() -> None:
    primary = FakeClient("primary", stream_tokens=("p",), stream_delay=0.2)
    backup = FakeClient("backup", stream_tokens=("a", "b"))
    llm = FallbackLLM([primary, backup], first_token_timeout=0.01)

    assert list(llm.stream_chat([{"role": "user", "content": "hi"}])) == ["a", "b"]
    assert llm.last_provider == "backup"
    assert llm.provider_status()["primary"]["failures"] == 1


def test_stream_first_token_error_falls_back() -> None:
    primary = FakeClient("primary", stream_error_before=RuntimeError("connect failed"))
    backup = FakeClient("backup", stream_tokens=("a",))
    llm = FallbackLLM([primary, backup])

    assert list(llm.stream_chat([{"role": "user", "content": "hi"}])) == ["a"]
    assert primary.stream_calls == 1
    assert backup.stream_calls == 1
    assert llm.last_provider == "backup"


def test_stream_after_first_token_does_not_switch() -> None:
    primary = FakeClient("primary", stream_tokens=("hello",), stream_error_after=RuntimeError("late failure"))
    backup = FakeClient("backup", stream_tokens=("should-not",))
    llm = FallbackLLM([primary, backup])

    iterator = llm.stream_chat([{"role": "user", "content": "hi"}])
    assert next(iterator) == "hello"
    with pytest.raises(RuntimeError, match="late failure"):
        next(iterator)

    assert backup.stream_calls == 0
    assert llm.last_provider == "primary"
    assert llm.provider_status()["primary"]["failures"] == 1


# ---------- circuit breaker ----------
def test_circuit_breaker_threshold_cooldown_and_half_open() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(fail_threshold=3, cooldown=60, clock=clock)

    assert breaker.allow_request() is True
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.failures == 3
    assert breaker.allow_request() is False

    clock.advance(60)
    assert breaker.state == "half_open"
    assert breaker.allow_request() is True
    # 半开只放行一个探针
    assert breaker.allow_request() is False
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.allow_request() is False

    clock.advance(60)
    assert breaker.allow_request() is True
    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.failures == 0
    assert breaker.allow_request() is True


def test_circuit_breaker_uses_module_monotonic(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr("ragm.rag.llm_router.time.monotonic", lambda: now[0])
    breaker = CircuitBreaker(fail_threshold=2, cooldown=10)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.allow_request() is False
    now[0] += 10
    assert breaker.allow_request() is True
    assert breaker.state == "half_open"


def test_open_provider_skipped_then_recovers() -> None:
    clock = FakeClock()
    primary = FakeClient("primary", chat_error=RuntimeError("down"))
    backup = FakeClient("backup", chat_result={"text": "backup", "tokens_in": 0, "tokens_out": 0})
    llm = FallbackLLM([primary, backup], fail_threshold=3, cooldown=60, clock=clock)

    for _ in range(3):
        assert llm.chat([])["text"] == "backup"
    assert primary.chat_calls == 3
    assert llm.provider_status()["primary"]["state"] == "open"

    # 熔断打开后不再请求 primary
    assert llm.chat([])["text"] == "backup"
    assert primary.chat_calls == 3
    assert backup.chat_calls == 4

    # 冷却结束半开探测；primary 恢复后重新闭合
    clock.advance(61)
    primary.chat_error = None
    assert llm.chat([])["text"] == "primary"
    assert llm.last_provider == "primary"
    assert llm.provider_status()["primary"] == {"state": "closed", "failures": 0}


# ---------- config / key handling ----------
def test_missing_api_key_skips_provider(monkeypatch, caplog) -> None:
    monkeypatch.delenv("RAGM_TEST_MISSING_KEY", raising=False)
    caplog.set_level(logging.WARNING, logger="ragm.rag.llm_router")
    llm = FallbackLLM(
        [
            {
                "name": "missing",
                "base_url": "http://127.0.0.1:1/v1",
                "model": "m",
                "api_key_env": "RAGM_TEST_MISSING_KEY",
            },
            {"name": "ok", "base_url": "http://127.0.0.1:2/v1", "model": "m", "api_key": "local"},
        ]
    )

    assert llm.provider_names == ["ok"]
    assert llm.skipped == [{"name": "missing", "reason": "missing api key env RAGM_TEST_MISSING_KEY"}]
    assert "RAGM_TEST_MISSING_KEY" in caplog.text


def test_build_llm_from_settings_order_and_per_provider_timeout(monkeypatch) -> None:
    monkeypatch.setenv("RAGM_TEST_PROVIDER_KEY", "secret")
    settings = Settings(
        _env_file=None,
        llm_providers=json.dumps(
            [
                {
                    "name": "p1",
                    "base_url": "http://127.0.0.1:1/v1",
                    "model": "m1",
                    "api_key_env": "RAGM_TEST_PROVIDER_KEY",
                    "first_token_timeout": 7,
                },
                {"name": "p2", "base_url": "http://127.0.0.1:2/v1", "model": "m2", "api_key": "k2"},
            ]
        ),
        llm_chat_timeout=5,
        llm_first_token_timeout=2,
    )

    llm = build_llm(settings)

    assert llm.provider_names == ["p1", "p2"]
    assert llm.chat_timeout == 5
    assert [provider.first_token_timeout for provider in llm.providers] == [7, 2]
    assert llm.clients[0].api_key == "secret"
    assert llm.clients[0].base_url == "http://127.0.0.1:1/v1"
    assert llm.provider_status() == {
        "p1": {"state": "closed", "failures": 0},
        "p2": {"state": "closed", "failures": 0},
    }
    assert llm.last_provider is None


def test_build_llm_skips_default_primary_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    settings = Settings(_env_file=None, llm_providers="")

    llm = build_llm(settings)

    assert llm.provider_names == ["ollama"]


# ---------- health ----------
@pytest.fixture
async def health_app(monkeypatch):
    monkeypatch.setenv("CLASSIFIER_BACKEND", "rules")
    monkeypatch.setenv("RAG_BACKEND", "demo")
    get_settings.cache_clear()

    async def _ok(*args, **kwargs) -> HealthCheck:
        return HealthCheck(ok=True, detail="test")

    monkeypatch.setattr(routes, "_check_milvus", _ok)
    monkeypatch.setattr(routes, "_check_ollama", _ok)
    app = create_app()
    app.state.kg = SimpleNamespace(names=["entity"])
    llm = FallbackLLM([FakeClient("primary"), FakeClient("backup")])
    app.state.llm = llm

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, llm
    get_settings.cache_clear()


async def test_health_backup_circuit_open_does_not_degrade(health_app) -> None:
    client, llm = health_app
    for _ in range(3):
        llm.breakers["backup"].record_failure()

    body = (await client.get("/api/health")).json()

    assert body["status"] == "ok"
    providers = body["checks"]["providers"]
    assert providers["ok"] is True
    detail = json.loads(providers["detail"])
    assert [item["name"] for item in detail] == ["primary", "backup"]
    assert detail[0] == {
        "name": "primary",
        "ok": True,
        "circuit_open": False,
        "state": "closed",
        "failures": 0,
    }
    assert detail[1]["ok"] is False
    assert detail[1]["circuit_open"] is True
    assert detail[1]["state"] == "open"


async def test_health_primary_circuit_open_degrades(health_app) -> None:
    client, llm = health_app
    for _ in range(3):
        llm.breakers["primary"].record_failure()

    body = (await client.get("/api/health")).json()

    assert body["status"] == "degraded"
    providers = body["checks"]["providers"]
    assert providers["ok"] is False
    detail = json.loads(providers["detail"])
    assert detail[0]["circuit_open"] is True
    assert detail[1]["circuit_open"] is False
