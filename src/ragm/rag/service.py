"""RAG 服务层：定义契约 + 演示实现。

后续接入 LangGraph 时，新增 LangGraphRagService 实现同一 Protocol，
在 main.py 的 lifespan 里按 settings.rag_backend 选择即可，API 层不用改。
"""
from __future__ import annotations

import asyncio
import copy
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Protocol

from ..api.schemas import (
    DoneEvent,
    QueryRequest,
    QueryResponse,
    ScenarioSummary,
    StageEvent,
    TokenEvent,
)
from .demo import DEMO_SCENARIOS

StreamEvent = StageEvent | TokenEvent | DoneEvent


class RagService(Protocol):
    """API 层只依赖这个契约。"""

    def scenarios(self) -> list[ScenarioSummary]: ...

    async def query(self, req: QueryRequest) -> QueryResponse: ...

    def stream(self, req: QueryRequest) -> AsyncIterator[StreamEvent]: ...


class DemoRagService:
    """演示实现：返回内置案例，用于在真实索引就绪前跑通前后端契约。"""

    def __init__(self, delay_scale: float = 0.55) -> None:
        self.delay_scale = delay_scale

    # ---------- 选择案例 ----------
    def _pick(self, req: QueryRequest) -> dict:
        if req.route_override != "auto":
            for s in DEMO_SCENARIOS:
                if s["route"] == req.route_override:
                    return s
            return DEMO_SCENARIOS[0]

        q = req.question.lower()
        if any(k in q for k in ("对比", "为什么", "推理", "分析", "选型", "建议", "怎么变化")):
            route = "agentic"
        elif any(k in q for k in ("哪个机构", "哪个城市", "之后", "还发布", "关系", "作者")):
            route = "multihop"
        else:
            route = "simple"
        for s in DEMO_SCENARIOS:
            if s["route"] == route and not s.get("refusal"):
                return s
        return DEMO_SCENARIOS[0]

    def _build(self, req: QueryRequest, scenario: dict) -> QueryResponse:
        payload = copy.deepcopy(scenario)
        payload.pop("title", None)
        payload["question"] = req.question
        payload["id"] = "q_" + uuid.uuid4().hex[:12]
        payload["served_by"] = "demo"
        payload["generated_at"] = datetime.now(UTC).isoformat()
        return QueryResponse.model_validate(payload)

    # ---------- 契约实现 ----------
    def scenarios(self) -> list[ScenarioSummary]:
        return [
            ScenarioSummary(
                id=s["id"],
                question=s["question"],
                route=s["route"],
                confidence=s["confidence"],
                title=s.get("title", s["id"]),
            )
            for s in DEMO_SCENARIOS
        ]

    async def query(self, req: QueryRequest) -> QueryResponse:
        return self._build(req, self._pick(req))

    async def stream(self, req: QueryRequest) -> AsyncIterator[StreamEvent]:
        scenario = self._pick(req)
        response = self._build(req, scenario)

        for index, step in enumerate(response.trace):
            await asyncio.sleep(min(0.35, step.ms / 1000 * self.delay_scale))
            yield StageEvent(index=index, step=step)

        if response.refusal is None:
            text = response.answer.lead + ("\n\n" + response.answer.body if response.answer.body else "")
            for i in range(0, len(text), 10):
                await asyncio.sleep(0.012)
                yield TokenEvent(text=text[i : i + 10])

        yield DoneEvent(response=response)
