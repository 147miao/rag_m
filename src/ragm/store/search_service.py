"""检索服务：dense / sparse / hybrid / small-to-big。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import Settings
from ..embeddings.embedder import Embedder
from .milvus_store import MilvusStore

CHILD_COLLECTION = "rag_children"


class SearchService:
    def __init__(self, settings: Settings) -> None:
        if not settings.embed_model_path:
            raise RuntimeError("EMBED_MODEL_PATH 未配置")
        self.settings = settings
        self.store = MilvusStore(settings.milvus_uri, settings.milvus_collection, settings.embed_dim)
        self.child_store = MilvusStore(settings.milvus_uri, CHILD_COLLECTION, settings.embed_dim, with_parent=True)
        self.parents: dict[str, dict] = {}
        parents_path = Path(settings.data_root) / "data" / "rag" / "parents.jsonl"
        if parents_path.exists():
            for line in parents_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    p = json.loads(line)
                    self.parents[p["parent_id"]] = p
        self.embedder = Embedder(settings.embed_model_path, settings.embed_model, settings.embed_max_len)

    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> dict:
        t0 = time.perf_counter()
        qvec = self.embedder.encode_queries([query])[0].tolist()
        if mode == "dense":
            hits = self.store.dense_search(qvec, k)
        elif mode == "sparse":
            hits = self.store.sparse_search(query, k)
        else:
            hits = self.store.hybrid_search(query, qvec, k)
        return {"query": query, "mode": mode, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "hits": hits}

    def search_with_scores(self, query: str, k: int = 8, pool: int = 40) -> dict:
        """dense + sparse 分别检索后在本地做 RRF，保留 dense/sparse 原始分数用于前端证据展示。"""
        t0 = time.perf_counter()
        qvec = self.embedder.encode_queries([query])[0].tolist()
        dense_hits = self.store.dense_search(qvec, pool)
        sparse_hits = self.store.sparse_search(query, pool)
        merged: dict[str, dict] = {}
        for rank, h in enumerate(dense_hits):
            item = merged.setdefault(h["chunk_id"], {**h, "rrf": 0.0, "dense": 0.0, "sparse": 0.0})
            item["rrf"] += 1.0 / (60 + rank + 1)
            item["dense"] = max(0.0, min(1.0, (h["score"] + 1.0) / 2.0))
        max_sparse = max((h["score"] for h in sparse_hits), default=1.0) or 1.0
        for rank, h in enumerate(sparse_hits):
            item = merged.setdefault(h["chunk_id"], {**h, "rrf": 0.0, "dense": 0.0, "sparse": 0.0})
            item["rrf"] += 1.0 / (60 + rank + 1)
            item["sparse"] = max(0.0, min(1.0, h["score"] / max_sparse))
        ranked = sorted(merged.values(), key=lambda x: -x["rrf"])[:k]
        max_rrf = ranked[0]["rrf"] if ranked else 1.0
        for i, item in enumerate(ranked):
            item["score"] = round(item["rrf"] / max_rrf, 4)
            item["rank"] = i + 1
        return {
            "query": query,
            "mode": "hybrid",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "hits": ranked,
        }

    def search_small2big(self, query: str, k: int = 8, k_children: int = 50) -> dict:
        """小块检索（精确性）→ 父块上下文（丰富性），按 parent 去重后返回 top-k。"""
        t0 = time.perf_counter()
        qvec = self.embedder.encode_queries([query])[0].tolist()
        children = self.child_store.hybrid_search(query, qvec, k_children)
        hits: list[dict] = []
        seen: set[str] = set()
        for ch in children:
            pid = ch.get("parent_id")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            parent = self.parents.get(pid)
            if parent is None:
                continue
            hits.append(
                {
                    "parent_id": pid,
                    "doc_id": parent["doc_id"],
                    "title": parent["title"],
                    "source": parent["source"],
                    "lang": parent["lang"],
                    "score": ch["score"],
                    "text": parent["text"],
                    "child_text": ch["text"],
                }
            )
            if len(hits) >= k:
                break
        return {
            "query": query,
            "mode": "small2big",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "hits": hits,
        }

    def info(self) -> dict:
        return {
            "collection": self.store.collection,
            "rows": self.store.count(),
            "child_collection": CHILD_COLLECTION,
            "child_rows": self.child_store.count(),
            "parents_loaded": len(self.parents),
            "model": self.embedder.path,
            "dim": self.embedder.dim,
        }