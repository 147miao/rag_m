"""LangGraphRagService：真实检索 + LLM 生成，实现 RagService 契约。"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from ..agent.agent_graph import AgentGraph
from ..agent.tools import ToolExecutor
from ..api.schemas import (
    Answer,
    ClassifierInfo,
    DoneEvent,
    Evidence,
    QueryRequest,
    QueryResponse,
    Refusal,
    ScenarioSummary,
    StageEvent,
    Stats,
    TokenEvent,
    ToolCall,
    TraceStep,
)
from ..classifier.base import ROUTE_TO_LABEL
from ..config import Settings
from ..graph.kg import KGStore
from .demo import DEMO_SCENARIOS
from .graph import RagGraph
from .llm import LLMClient

DATASET_NAMES = {"hotpotqa": "HotpotQA", "cmrc": "CMRC2018", "pubmedqa": "PubMedQA"}


class LangGraphRagService:
    def __init__(
        self,
        classifier: Any,
        search: Any,
        llm: LLMClient,
        settings: Settings,
        kg: KGStore | None = None,
        agent: AgentGraph | None = None,
        executor: ToolExecutor | None = None,
    ) -> None:
        self.settings = settings
        self.kg = kg
        self.agent = agent
        self.executor = executor
        self.rag = RagGraph(
            classifier,
            search,
            llm,
            kg=kg,
            max_tokens=settings.rag_max_tokens,
            top_k=settings.rag_top_k,
            multihop_k=settings.rag_multihop_k,
            agentic_k=settings.rag_agentic_k,
        )

    # ---------- scenarios ----------
    def scenarios(self) -> list[ScenarioSummary]:
        return [
            ScenarioSummary(id=s["id"], question=s["question"], route=s["route"], confidence=s["confidence"], title=s.get("title", s["id"]))
            for s in DEMO_SCENARIOS
        ]

    # ---------- response ----------
    def _evidence(self, hits: list[dict]) -> list[Evidence]:
        out = []
        for i, h in enumerate(hits):
            source = str(h.get("source") or "")
            out.append(
                Evidence(
                    id=str(h.get("chunk_id") or f"hit-{i}"),
                    rank=i + 1,
                    title=str(h.get("title") or "")[:120],
                    source=f"{source}://{h.get('doc_id', '')}",
                    dataset=DATASET_NAMES.get(source, source or "corpus"),
                    score=round(float(h.get("score") or 0.0), 4),
                    dense=round(float(h.get("dense") or 0.0), 4),
                    sparse=round(float(h.get("sparse") or 0.0), 4),
                    snippet=" ".join(str(h.get("text") or "").split())[:260],
                )
            )
        return out

    def _trace(self, state: dict) -> list[TraceStep]:
        hits = state.get("hits") or []
        route = state.get("route", "simple")
        if state.get("tool_log"):
            names = ", ".join(t.get("name", "") for t in state["tool_log"]) or "无"
            return [
                TraceStep(key="route", label="路由分类", ms=int(state.get("classify_ms", 0)), detail=f"{state.get('classifier_backend', '')} · {route} {state.get('confidence', 0):.2f}"),
                TraceStep(key="plan", label="Agent 规划", ms=int(state.get("plan_ms", 0)), detail=f"{state.get('iterations', 0)} 轮 · {len(state['tool_log'])} 次工具调用"),
                TraceStep(key="tools", label="工具调用", ms=int(state.get("tools_ms", 0)), detail=names),
                TraceStep(key="generate", label="生成", ms=int(state.get("generate_ms", 0)), detail=f"{state.get('tokens_in', 0):,} in / {state.get('tokens_out', 0):,} out"),
                TraceStep(key="verify", label="引用校验", ms=int(state.get("verify_ms", 0)), detail=state.get("citation_check", "")),
            ]
        return [
            TraceStep(key="route", label="路由分类", ms=int(state.get("classify_ms", 0)), detail=f"{state.get('classifier_backend', '')} · {route} {state.get('confidence', 0):.2f}"),
            TraceStep(key="retrieve", label="混合检索", ms=int(state.get("retrieve_ms", 0)), detail=state.get("retrieve_detail") or f"dense + BM25 · RRF · top {len(hits)}"),
            TraceStep(key="generate", label="生成", ms=int(state.get("generate_ms", 0)), detail=f"{state.get('tokens_in', 0):,} in / {state.get('tokens_out', 0):,} out"),
            TraceStep(key="verify", label="引用校验", ms=int(state.get("verify_ms", 0)), detail=state.get("citation_check", "")),
        ]

    def _build(self, req: QueryRequest, state: dict, latency_ms: int) -> QueryResponse:
        hits = state.get("hits") or []
        route = state.get("route", "simple")
        trace = self._trace(state)
        tools = None
        if state.get("tool_log"):
            tools = [
                ToolCall(name=t.get("name", ""), args=t.get("args", ""), result=t.get("result", ""), ms=int(t.get("ms", 0)), status=t.get("status", "ok"))
                for t in state["tool_log"]
            ]
        refusal = state.get("refusal")
        return QueryResponse(
            id="q_" + uuid.uuid4().hex[:12],
            question=req.question,
            route=route,
            confidence=round(float(state.get("confidence", 0.0)), 4),
            reason=state.get("reason", ""),
            answer=Answer(lead=state.get("answer_lead", ""), body=state.get("answer_body", "")),
            citations=state.get("citations", []),
            evidence=self._evidence(hits),
            trace=trace,
            tools=tools,
            graph=state.get("kg_payload"),
            stats=Stats(
                latency_ms=latency_ms,
                tokens_in=int(state.get("tokens_in", 0)),
                tokens_out=int(state.get("tokens_out", 0)),
                cost_cny=0.0,
                citation_check=state.get("citation_check", "0/0 通过（无引用）"),
            ),
            refusal=Refusal(**refusal) if refusal else None,
            classifier=ClassifierInfo(
                backend=state.get("classifier_backend", ""),
                label=state.get("label") or ROUTE_TO_LABEL.get(route, "simple"),
                route=route,
                confidence=round(float(state.get("confidence", 0.0)), 4),
                latency_ms=float(state.get("classify_ms", 0.0)),
                reason=state.get("reason", ""),
            ),
            served_by="langgraph",
            generated_at=datetime.now(UTC),
        )

    # ---------- contract ----------
    def _agent_state(self, decision: Any, route: str, astate: dict) -> dict:
        return {
            "route": route,
            "label": decision.label,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "classifier_backend": decision.backend,
            "classify_ms": decision.latency_ms,
            "hits": astate.get("hits") or [],
            "kg_paths": astate.get("kg_paths") or [],
            "kg_payload": None,
            "retrieve_detail": f"agent · {len(astate.get('tool_log') or [])} 次工具调用",
            "answer_lead": astate.get("answer_lead", ""),
            "answer_body": astate.get("answer_body", ""),
            "citations": astate.get("citations") or [],
            "citation_check": astate.get("citation_check", ""),
            "tokens_in": astate.get("tokens_in", 0),
            "tokens_out": astate.get("tokens_out", 0),
            "plan_ms": astate.get("plan_ms", 0),
            "tools_ms": astate.get("tools_ms", 0),
            "generate_ms": astate.get("generate_ms", 0),
            "verify_ms": astate.get("verify_ms", 0),
            "iterations": astate.get("iterations", 0),
            "tool_log": astate.get("tool_log") or [],
            "refusal": astate.get("refusal"),
        }

    async def query(self, req: QueryRequest) -> QueryResponse:
        t0 = time.perf_counter()
        if self.agent is not None:
            decision = await asyncio.to_thread(self.rag.classifier.classify, req.question)
            route = decision.route if req.route_override == "auto" else req.route_override
            if route == "agentic":
                astate = await asyncio.to_thread(self.agent.invoke, req.question)
                state = self._agent_state(decision, route, astate)
                return self._build(req, state, int((time.perf_counter() - t0) * 1000))
        state = await asyncio.to_thread(self.rag.invoke, req.question, req.route_override)
        return self._build(req, state, int((time.perf_counter() - t0) * 1000))

    async def stream(self, req: QueryRequest) -> AsyncIterator[StageEvent | TokenEvent | DoneEvent]:
        t0 = time.perf_counter()
        if self.agent is not None:
            decision = await asyncio.to_thread(self.rag.classifier.classify, req.question)
            route = decision.route if req.route_override == "auto" else req.route_override
            if route == "agentic":
                astate = await asyncio.to_thread(self.agent.invoke, req.question)
                state = self._agent_state(decision, route, astate)
                for index, step in enumerate(self._trace(state)):
                    yield StageEvent(index=index, step=step)
                text = (state.get("answer_lead", "") + "\n\n" + state.get("answer_body", "")).strip()
                for i in range(0, len(text), 16):
                    yield TokenEvent(text=text[i : i + 16])
                    await asyncio.sleep(0.01)
                yield DoneEvent(response=self._build(req, state, int((time.perf_counter() - t0) * 1000)))
                return
        state = await asyncio.to_thread(self.rag.prepare, req.question, req.route_override)
        steps = [
            ("route", "路由分类", state.get("classify_ms", 0), f"{state.get('classifier_backend', '')} · {state.get('route')} {state.get('confidence', 0):.2f}"),
            ("retrieve", "混合检索", state.get("retrieve_ms", 0), state.get("retrieve_detail") or f"dense + BM25 · RRF · top {len(state.get('hits') or [])}"),
        ]
        for index, (key, label, ms, detail) in enumerate(steps):
            yield StageEvent(index=index, step=TraceStep(key=key, label=label, ms=int(ms), detail=detail))

        hits = state.get("hits") or []
        text_parts: list[str] = []
        if hits:
            messages = self.rag.stream_messages(state)
            async for token in self._aiter_tokens(messages):
                text_parts.append(token)
                yield TokenEvent(text=token)
        text = "".join(text_parts)
        prompt_chars = sum(len(m["content"]) for m in self.rag.stream_messages(state)) if hits else 0
        state = self.rag.finish_stream(state, text, tokens_in=prompt_chars // 3, tokens_out=len(text) // 2)
        yield DoneEvent(response=self._build(req, state, int((time.perf_counter() - t0) * 1000)))

    async def _aiter_tokens(self, messages: list[dict]) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()

        def worker() -> None:
            try:
                for token in self.rag.llm.stream_chat(messages, self.rag.max_tokens):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield str(item)