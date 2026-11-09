"""LangGraph 生成链路：classify → retrieve → generate → verify。

- query()（非流式）：走完整 StateGraph；
- stream()：复用 prepare/finish 节点做 SSE 流式（先生成阶段事件，再逐 token 推流）。
"""
from __future__ import annotations

import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..classifier.base import override_decision
from ..embeddings.reranker import BaseReranker
from ..graph.kg import KGStore
from .citations import verify_citations
from .llm import LLMClient
from .prompts import build_messages, build_messages_plain

ROUTE_TOP_K = {"simple": 8, "multihop": 12, "agentic": 12}


class RagState(TypedDict, total=False):
    question: str
    route_override: str
    classifier_decision: Any
    route: str
    label: str
    confidence: float
    reason: str
    classifier_backend: str
    classify_ms: float
    hits: list[dict]
    retrieve_ms: float
    kg_paths: list[dict]
    kg_payload: Any
    kg_seeds: list[str]
    graph_added: int
    retrieve_detail: str
    rerank_ms: float | None
    rerank_model: str | None
    answer_lead: str
    answer_body: str
    generate_ms: float
    tokens_in: int
    tokens_out: int
    citations: list[int]
    invalid_citations: list[int]
    citation_check: str
    verify_ms: float
    refusal: dict | None


def parse_structured_answer(text: str) -> tuple[str, str]:
    """解析"第一行结论 + 空行 + 展开"格式，兼容单行/JSON 残留。"""
    raw = (text or "").strip()
    fence = chr(96) * 3
    if raw.startswith(fence):
        raw = re.sub(r"^" + fence + r"[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?" + fence + r"$", "", raw).strip()
    lines = raw.splitlines()
    lead, rest_start = "", 0
    for i, line in enumerate(lines):
        if line.strip():
            lead = line.strip()
            rest_start = i + 1
            break
    lead = re.sub(r"^(结论|答案|一句话结论)[：:]\s*", "", lead).strip()
    body = "\n".join(lines[rest_start:]).strip()
    if body:
        return lead, body
    parts = re.split(r"(?<=[。！？.!?])\s*", lead, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return lead, ""


def split_plain_answer(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    parts = re.split(r"(?<=[。！？.!?])\s*", raw, maxsplit=1)
    if len(parts) == 2 and len(parts[0]) <= 120:
        return parts[0].strip(), parts[1].strip()
    return "", raw


class RagGraph:
    def __init__(
        self,
        classifier: Any,
        search: Any,
        llm: LLMClient,
        *,
        kg: KGStore | None = None,
        reranker: BaseReranker | None = None,
        rerank_candidates: int = 20,
        max_tokens: int = 1024,
        top_k: int = 8,
        multihop_k: int = 12,
        agentic_k: int = 12,
        kg_hops: int = 2,
        kg_limit: int = 40,
        kg_iterative: bool = True,
    ) -> None:
        self.classifier = classifier
        self.search = search
        self.llm = llm
        self.kg = kg
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.multihop_k = multihop_k
        self.agentic_k = agentic_k
        self.kg_hops = kg_hops
        self.kg_limit = kg_limit
        self.kg_iterative = kg_iterative
        self._graph = self._build()

    # ---------- nodes ----------
    def _classify_node(self, state: RagState) -> dict:
        """复用上游分类结果；显式路由时合成决策，只有 auto 且无缓存时才真正分类。"""
        decision = state.get("classifier_decision")
        if decision is None:
            route_override = state.get("route_override", "auto")
            if route_override == "auto":
                t0 = time.perf_counter()
                decision = self.classifier.classify(state["question"])
                classify_ms = round((time.perf_counter() - t0) * 1000, 1)
            else:
                decision = override_decision(route_override)
                classify_ms = 0.0
        else:
            classify_ms = round(float(getattr(decision, "latency_ms", 0.0)), 1)
        route = decision.route if state.get("route_override", "auto") == "auto" else state["route_override"]
        return {
            "label": decision.label,
            "route": route,
            "confidence": round(float(decision.confidence), 4),
            "reason": decision.reason,
            "classifier_backend": decision.backend,
            "classify_ms": classify_ms,
        }

    def _retrieve_node(self, state: RagState) -> dict:
        t0 = time.perf_counter()
        route = state.get("route", "simple")
        k = {"simple": self.top_k, "multihop": self.multihop_k, "agentic": self.agentic_k}.get(route, self.top_k)
        res = self.search.search_with_scores(state["question"], k=k)
        hits = res["hits"]
        kg_paths: list[dict] = []
        kg_payload = None
        kg_seeds: list[str] = []
        graph_added = 0
        iter_added = 0
        if self.kg is not None and route == "multihop":
            kg_seeds = self.kg.link(state["question"], [h.get("title") or "" for h in hits])
            edges, neighbors = self.kg.expand_hops(kg_seeds, hops=self.kg_hops, limit=self.kg_limit)
            titles = [n for n in neighbors if n not in kg_seeds][:12]
            graph_chunks = self.search.chunks_by_titles(titles, limit=8)
            existing = {h["chunk_id"] for h in hits}
            extra = []
            for gc in graph_chunks:
                if gc["chunk_id"] in existing:
                    continue
                extra.append({**gc, "score": 0.5, "dense": 0.0, "sparse": 0.0, "graph": True})
                existing.add(gc["chunk_id"])
            graph_added = len(extra)
            if self.kg_iterative and not extra and neighbors:
                res2 = self.search.search_with_scores(state["question"] + " " + " ".join(neighbors[:3]), k=k)
                existing2 = {h["chunk_id"] for h in hits}
                more = [h for h in res2["hits"] if h["chunk_id"] not in existing2]
                if more:
                    iter_added = len(more[:4])
                    hits = (more[:4] + hits)[: k + 4]
            hits = (extra + hits)[: k + 4]
            kg_paths = edges[:20]
            if kg_seeds:
                kg_payload = self.kg.payload(state["question"], kg_seeds, kg_paths, titles[:6])
        rerank_ms = None
        if self.reranker is not None and hits:
            t_r = time.perf_counter()
            hits = self.reranker.rerank(state["question"], hits[: self.rerank_candidates], top_n=max(k, 8))
            rerank_ms = round((time.perf_counter() - t_r) * 1000, 1)
        detail = f"dense + BM25 · RRF · top {len(hits)}"
        if graph_added:
            detail += f" · 图谱 +{graph_added}"
        if iter_added:
            detail += f" · 迭代 +{iter_added}"
        if rerank_ms is not None:
            detail += f" · 重排 {self.reranker.name}"
        return {
            "hits": hits,
            "rerank_ms": rerank_ms,
            "rerank_model": self.reranker.name if self.reranker is not None else None,
            "retrieve_ms": round((time.perf_counter() - t0) * 1000, 1),
            "kg_paths": kg_paths,
            "kg_payload": kg_payload,
            "kg_seeds": kg_seeds,
            "graph_added": graph_added,
            "retrieve_detail": detail,
        }

    def _generate_node(self, state: RagState) -> dict:
        t0 = time.perf_counter()
        hits = state.get("hits") or []
        if not hits:
            return {"answer_lead": "", "answer_body": "", "generate_ms": 0.0, "tokens_in": 0, "tokens_out": 0}
        messages = build_messages(state["question"], hits, state.get("route", "simple"), state.get("kg_paths") or [])
        out = self.llm.chat(messages, self.max_tokens)
        lead, body = parse_structured_answer(out["text"])
        return {
            "answer_lead": lead,
            "answer_body": body,
            "generate_ms": round((time.perf_counter() - t0) * 1000, 1),
            "tokens_in": out.get("tokens_in") or 0,
            "tokens_out": out.get("tokens_out") or 0,
        }

    def _verify_node(self, state: RagState) -> dict:
        t0 = time.perf_counter()
        hits = state.get("hits") or []
        text = f"{state.get('answer_lead', '')} {state.get('answer_body', '')}"
        valid, invalid, check = verify_citations(text, len(hits))
        refusal = None
        if not hits:
            refusal = {
                "title": "未检索到相关资料",
                "reason": "当前索引没有命中与问题相关的片段，可能是问题超出语料范围。",
                "suggestion": "换一种问法，或补充相关文档后再试。",
            }
        elif "资料中没有足够信息" in text or "无法回答" in text:
            refusal = {
                "title": "资料不足以回答",
                "reason": "检索到的片段没有覆盖问题所需的关键事实。",
                "suggestion": "补充相关文档，或把问题拆成更小的子问题。",
            }
        return {
            "citations": valid,
            "invalid_citations": invalid,
            "citation_check": check,
            "verify_ms": round((time.perf_counter() - t0) * 1000, 1),
            "refusal": refusal,
        }

    def _build(self) -> Any:
        g = StateGraph(RagState)
        g.add_node("classify", self._classify_node)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("generate", self._generate_node)
        g.add_node("verify", self._verify_node)
        g.add_edge(START, "classify")
        g.add_edge("classify", "retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", "verify")
        g.add_edge("verify", END)
        return g.compile()

    # ---------- public ----------
    def invoke(self, question: str, route_override: str = "auto", decision: Any | None = None) -> RagState:
        state: RagState = {"question": question, "route_override": route_override}
        if decision is not None:
            state["classifier_decision"] = decision
        return self._graph.invoke(state)

    def prepare(self, question: str, route_override: str = "auto", decision: Any | None = None) -> RagState:
        state: RagState = {"question": question, "route_override": route_override}
        if decision is not None:
            state["classifier_decision"] = decision
        state.update(self._classify_node(state))
        state.update(self._retrieve_node(state))
        return state

    def stream_messages(self, state: RagState) -> list[dict]:
        return build_messages_plain(
            state["question"], state.get("hits") or [], state.get("route", "simple"), state.get("kg_paths") or []
        )

    def finish_stream(self, state: RagState, text: str, tokens_in: int = 0, tokens_out: int = 0) -> RagState:
        lead, body = split_plain_answer(text)
        state.update({"answer_lead": lead, "answer_body": body, "tokens_in": tokens_in, "tokens_out": tokens_out})
        state.update(self._verify_node(state))
        return state