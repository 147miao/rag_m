"""Agentic 工具：vector_search / kg_query / web_search。"""
from __future__ import annotations

import json
import time
import urllib.parse
import xml.etree.ElementTree as ET

import httpx

from ..config import Settings
from ..graph.kg import KGStore
from ..observability.tracing import traceable
from ..store.search_service import SearchService

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "在本地 Milvus 语料库做混合检索（dense + BM25），返回最相关的片段，适合事实查询。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索语句"},
                    "k": {"type": "integer", "description": "返回条数，默认 6，最大 20"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kg_query",
            "description": "查询实体知识图谱：给定实体名，返回与其相连的实体及关系路径，适合多跳/关系型问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "实体名，如 Milvus / Sam Bankman-Fried"},
                    "hops": {"type": "integer", "description": "扩展跳数，1 或 2"},
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "联网搜索（Bing RSS / SearXNG），用于语料库之外的最新信息；返回标题、URL、摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "k": {"type": "integer", "description": "返回条数，默认 5"},
                },
                "required": ["query"],
            },
        },
    },
]


class ToolExecutor:
    def __init__(self, search: SearchService, kg: KGStore | None, settings: Settings) -> None:
        self.search = search
        self.kg = kg
        self.settings = settings

    @traceable(run_type="tool", name="vector_search")
    def vector_search(self, query: str, k: int = 6) -> dict:
        k = max(1, min(int(k or 6), 20))
        res = self.search.search_with_scores(query, k=k)
        hits = [
            {
                "chunk_id": h["chunk_id"],
                "title": h.get("title", ""),
                "source": h.get("source", ""),
                "lang": h.get("lang", ""),
                "score": h.get("score", 0.0),
                "text": h.get("text", "")[:400],
            }
            for h in res["hits"]
        ]
        return {"ok": True, "tool": "vector_search", "query": query, "hits": hits, "latency_ms": res["latency_ms"]}

    @traceable(run_type="tool", name="kg_query")
    def kg_query(self, entity: str, hops: int = 1) -> dict:
        if self.kg is None:
            return {"ok": False, "tool": "kg_query", "error": "knowledge graph unavailable"}
        names = [n for n in self.kg.names if entity and entity.lower() in n.lower()][:3]
        if not names:
            return {"ok": True, "tool": "kg_query", "entity": entity, "paths": [], "neighbors": [], "note": "实体未命中图谱"}
        hops = max(1, min(int(hops or 1), 2))
        edges, neighbors = self.kg.expand_hops(names, hops=hops, limit=30)
        paths = [
            {"a": e["a"], "b": e["b"], "type": e.get("qtype") or "co_evidence", "question": (e.get("question") or "")[:120]}
            for e in edges[:12]
        ]
        return {"ok": True, "tool": "kg_query", "entity": entity, "matched": names, "paths": paths, "neighbors": neighbors[:10]}

    @traceable(run_type="tool", name="web_search")
    def web_search(self, query: str, k: int = 5) -> dict:
        k = max(1, min(int(k or 5), 10))
        results = self._bing_rss(query, k)
        source = "bing"
        if not results:
            results = self._searxng(query, k)
            source = "searxng"
        return {"ok": bool(results), "tool": "web_search", "query": query, "source": source, "results": results}

    def _bing_rss(self, query: str, k: int) -> list[dict]:
        url = "https://www.bing.com/search?" + urllib.parse.urlencode(
            {"q": query, "format": "rss", "mkt": "en-US", "cc": "US"}
        )
        try:
            with httpx.Client(timeout=8, follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"})
            if resp.status_code != 200:
                return []
            root = ET.fromstring(resp.text)
            out = []
            for item in root.findall(".//item")[:k]:
                out.append(
                    {
                        "title": (item.findtext("title") or "").strip(),
                        "url": (item.findtext("link") or "").strip(),
                        "snippet": (item.findtext("description") or "").strip()[:300],
                    }
                )
            return out
        except Exception:  # noqa: BLE001
            return []

    def _searxng(self, query: str, k: int) -> list[dict]:
        try:
            with httpx.Client(timeout=2) as client:
                resp = client.get(self.settings.web_search_url, params={"q": query, "format": "json"})
            if resp.status_code != 200:
                return []
            return [
                {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": (r.get("content") or "")[:300]}
                for r in resp.json().get("results", [])[:k]
            ]
        except Exception:  # noqa: BLE001
            return []

    def run(self, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        try:
            if name == "vector_search":
                out = self.vector_search(args.get("query", ""), args.get("k", 6))
            elif name == "kg_query":
                out = self.kg_query(args.get("entity", ""), args.get("hops", 1))
            elif name == "web_search":
                out = self.web_search(args.get("query", ""), args.get("k", 5))
            else:
                out = {"ok": False, "error": f"unknown tool: {name}"}
        except Exception as exc:  # noqa: BLE001
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out


def tool_result_text(result: dict, limit: int = 1800) -> str:
    return json.dumps(result, ensure_ascii=False)[:limit]