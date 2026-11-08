"""Agentic 工具链：LangGraph 循环（plan → tools → … → synthesize → verify）。"""
from __future__ import annotations

import json
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..observability.tracing import traceable
from ..rag.citations import verify_citations
from ..rag.graph import parse_structured_answer
from ..rag.llm import LLMClient
from ..rag.prompts import build_messages
from .tools import TOOL_SCHEMAS, ToolExecutor, tool_result_text

AGENT_SYSTEM = """你是 RAG-M 的 agentic 检索助手。你可以调用以下工具：
- vector_search：本地语料库混合检索（事实查询优先）
- kg_query：实体知识图谱查询（关系型/多跳问题）
- web_search：联网搜索（语料库之外的最新信息）

规则：
1. 先判断是否需要工具；需要就调用工具收集证据，可多轮，最多 3 轮。
2. 每次工具调用只解决一个明确子问题，query 要具体（不要一次问一大段）。
3. 证据足够后立即停止调用工具，直接给出最终答案。
4. 最终答案只能基于工具返回的证据；关键结论后用 [n] 标注资料编号；证据不足时明确说"资料中没有足够信息"。
5. 用与问题相同的语言回答，不要输出 JSON。"""


class AgentState(TypedDict, total=False):
    question: str
    messages: list[dict]
    iterations: int
    hits: list[dict]
    kg_paths: list[dict]
    web_results: list[dict]
    tool_log: list[dict]
    answer_lead: str
    answer_body: str
    citations: list[int]
    citation_check: str
    refusal: dict | None
    tokens_in: int
    tokens_out: int
    plan_ms: float
    tools_ms: float
    generate_ms: float
    verify_ms: float
    tool_names: list[str]


class AgentGraph:
    def __init__(
        self,
        llm: LLMClient,
        executor: ToolExecutor,
        *,
        max_iterations: int = 3,
        max_tokens: int = 1024,
        max_calls: int = 4,
        total_tokens: int = 8000,
        retry: int = 1,
    ) -> None:
        self.llm = llm
        self.executor = executor
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.max_calls = max_calls
        self.total_tokens = total_tokens
        self.retry = retry
        self._graph = self._build()

    # ---------- nodes ----------
    @traceable(run_type="chain", name="agent_plan")
    def _plan_node(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        messages = list(state.get("messages") or [])
        out = self.llm.chat_tools(messages, TOOL_SCHEMAS, max_tokens=600)
        calls = out.get("tool_calls") or []
        assistant: dict[str, Any] = {"role": "assistant", "content": out.get("text") or ""}
        if calls:
            assistant["tool_calls"] = [
                {
                    "id": c.get("id") or f"call_{i}",
                    "type": "function",
                    "function": {"name": c["function"]["name"], "arguments": c["function"].get("arguments") or "{}"},
                }
                for i, c in enumerate(calls)
            ]
        return {
            "messages": messages + [assistant],
            "iterations": int(state.get("iterations") or 0) + (1 if calls else 0),
            "tokens_in": int(state.get("tokens_in") or 0) + out.get("tokens_in", 0),
            "tokens_out": int(state.get("tokens_out") or 0) + out.get("tokens_out", 0),
            "plan_ms": round(float(state.get("plan_ms") or 0) + (time.perf_counter() - t0) * 1000, 1),
        }

    def _tools_node(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        messages = list(state.get("messages") or [])
        last = messages[-1] if messages else {}
        hits = list(state.get("hits") or [])
        kg_paths = list(state.get("kg_paths") or [])
        web_results = list(state.get("web_results") or [])
        tool_log = list(state.get("tool_log") or [])
        names = list(state.get("tool_names") or [])
        existing = {h.get("chunk_id") for h in hits}
        for call in last.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = None
            attempts = 0
            for attempt in range(self.retry + 1):
                attempts = attempt + 1
                result = self.executor.run(name, args)
                if result.get("ok") or attempt >= self.retry:
                    break
                time.sleep(0.3)
            assert result is not None
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": tool_result_text(result)})
            tool_log.append(
                {
                    "name": name,
                    "args": json.dumps(args, ensure_ascii=False)[:300],
                    "result": (result.get("note") or f"{len(result.get('hits') or result.get('results') or result.get('paths') or [])} 条结果"),
                    "ms": int(result.get("latency_ms") or 0),
                    "status": "ok" if result.get("ok") else "error",
                    "attempts": attempts,
                }
            )
            names.append(name)
            if name == "vector_search":
                for h in result.get("hits") or []:
                    if h["chunk_id"] in existing:
                        continue
                    existing.add(h["chunk_id"])
                    hits.append({**h, "score": h.get("score", 0.5), "dense": 0.0, "sparse": 0.0})
            elif name == "web_search":
                for i, w in enumerate(result.get("results") or []):
                    cid = "web:" + str(w.get("url") or i)
                    if cid in existing:
                        continue
                    existing.add(cid)
                    hits.append({"chunk_id": cid, "title": w.get("title", ""), "source": w.get("url", ""), "lang": "", "score": 0.5, "dense": 0.0, "sparse": 0.0, "text": w.get("snippet", "")})
                    web_results.append(w)
            elif name == "kg_query":
                kg_paths.extend(result.get("paths") or [])
        return {
            "messages": messages,
            "hits": hits,
            "kg_paths": kg_paths,
            "web_results": web_results,
            "tool_log": tool_log,
            "tool_names": names,
            "tools_ms": round(float(state.get("tools_ms") or 0) + (time.perf_counter() - t0) * 1000, 1),
        }

    @traceable(run_type="chain", name="agent_synthesize")
    def _synthesize_node(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        hits = list(state.get("hits") or [])
        if not hits:
            res = self.executor.vector_search(state["question"], 8)
            hits = [
                {**h, "score": h.get("score", 0.5), "dense": 0.0, "sparse": 0.0}
                for h in (res.get("hits") or [])
            ]
        hits = hits[:12]
        messages = build_messages(
            state["question"], hits, "agentic", state.get("kg_paths") or [], state.get("web_results") or []
        )
        out = self.llm.chat(messages, self.max_tokens)
        lead, body = parse_structured_answer(out["text"])
        return {
            "hits": hits,
            "answer_lead": lead,
            "answer_body": body,
            "generate_ms": round((time.perf_counter() - t0) * 1000, 1),
            "tokens_in": int(state.get("tokens_in") or 0) + out.get("tokens_in", 0),
            "tokens_out": int(state.get("tokens_out") or 0) + out.get("tokens_out", 0),
        }

    def _verify_node(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        hits = state.get("hits") or []
        text = f"{state.get('answer_lead', '')} {state.get('answer_body', '')}"
        valid, _invalid, check = verify_citations(text, len(hits))
        refusal = None
        if not hits:
            refusal = {"title": "未检索到相关资料", "reason": "工具没有返回可用证据。", "suggestion": "换个问法或补充文档。"}
        elif "资料中没有足够信息" in text or "无法回答" in text:
            refusal = {"title": "资料不足以回答", "reason": "工具返回的证据没有覆盖问题所需事实。", "suggestion": "补充文档，或把问题拆成子问题。"}
        return {
            "citations": valid,
            "citation_check": check,
            "verify_ms": round((time.perf_counter() - t0) * 1000, 1),
            "refusal": refusal,
        }

    # ---------- routing ----------
    def _route_after_plan(self, state: AgentState) -> str:
        messages = state.get("messages") or []
        last = messages[-1] if messages else {}
        calls = len(state.get("tool_names") or [])
        used = int(state.get("tokens_in") or 0) + int(state.get("tokens_out") or 0)
        if (
            last.get("tool_calls")
            and int(state.get("iterations") or 0) < self.max_iterations
            and calls < self.max_calls
            and used < self.total_tokens
        ):
            return "tools"
        return "synthesize"

    def _build(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("plan", self._plan_node)
        g.add_node("tools", self._tools_node)
        g.add_node("synthesize", self._synthesize_node)
        g.add_node("verify", self._verify_node)
        g.add_edge(START, "plan")
        g.add_conditional_edges("plan", self._route_after_plan, {"tools": "tools", "synthesize": "synthesize"})
        g.add_edge("tools", "plan")
        g.add_edge("synthesize", "verify")
        g.add_edge("verify", END)
        return g.compile()

    def invoke(self, question: str) -> AgentState:
        messages = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": f"【问题】{question}\n\n请判断是否需要调用工具；需要的话先调用工具收集证据。"},
        ]
        return self._graph.invoke({"question": question, "messages": messages})