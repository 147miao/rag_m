"""检索服务：dense / sparse / hybrid / small-to-big。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import Settings
from ..embeddings.embedder import Embedder
from ..observability.tracing import traceable
from .milvus_store import MilvusStore

CHILD_COLLECTION = "rag_children"  # 兼容旧默认值


class SearchService:
    def __init__(self, settings: Settings) -> None:
        if not settings.embed_model_path:
            raise RuntimeError("EMBED_MODEL_PATH 未配置")
        self.settings = settings
        self.store = MilvusStore(settings.milvus_uri, settings.milvus_collection, settings.embed_dim)
        child_collection = settings.milvus_child_collection or CHILD_COLLECTION
        self.child_store = MilvusStore(settings.milvus_uri, child_collection, settings.embed_dim, with_parent=True)
        self.parents: dict[str, dict] = {}
        parents_path = settings.parents_path or (Path(settings.data_root) / "data" / "rag" / "parents.jsonl")
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

    @traceable(run_type="retriever", name="hybrid_search")
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

    def chunks_by_titles(self, titles: list[str], limit: int = 8) -> list[dict]:
        """按标题批量取 chunk（图谱扩展后的证据补齐）。"""
        if not titles:
            return []
        escaped = [t.replace('\\', '\\\\').replace('"', '\\"') for t in titles[:20]]
        expr = "title in [" + ", ".join('"' + t + '"' for t in escaped) + "]"
        rows = self.store.client.query(
            collection_name=self.store.collection,
            filter=expr,
            output_fields=["chunk_id", "doc_id", "title", "source", "lang", "chunk_index", "text"],
            limit=limit,
        )
        return [dict(r) for r in rows]

    def _child_status(self) -> tuple[bool, str | None]:
        """子集合是否可用；集合缺失时不应让整个 SearchService 初始化失败。"""
        collection = self.child_store.collection
        try:
            exists = bool(self.child_store.client.has_collection(collection))
        except Exception as exc:  # noqa: BLE001 - 连接/权限异常类型由 pymilvus 决定
            return False, f"{type(exc).__name__}: {str(exc)[:200]}"
        if not exists:
            return False, f"child collection not found: {collection}"
        return True, None

    def search_small2big(self, query: str, k: int = 8, k_children: int = 50) -> dict:
        """小块检索（精确性）→ 父块上下文（丰富性），按 parent 去重后返回 top-k。"""
        child_ready, child_error = self._child_status()
        if not child_ready:
            raise RuntimeError(
                f"small2big unavailable: child collection '{self.child_store.collection}' not ready ({child_error})"
            )
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
        child_ready, child_error = self._child_status()
        child_rows = None
        if child_ready:
            try:
                child_rows = self.child_store.count()
            except Exception as exc:  # noqa: BLE001 - 子集合统计失败不应拖垮主检索服务
                child_ready = False
                child_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        return {
            "collection": self.store.collection,
            "rows": self.store.count(),
            "child_collection": self.child_store.collection,
            "child_rows": child_rows,
            "child_ready": child_ready,
            "child_error": child_error,
            "parents_loaded": len(self.parents),
            "model": self.embedder.path,
            "dim": self.embedder.dim,
        }