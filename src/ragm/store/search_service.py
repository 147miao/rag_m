"""检索服务：query → embedding → Milvus hybrid search。"""
from __future__ import annotations

import time

from ..config import Settings
from ..embeddings.embedder import Embedder
from .milvus_store import MilvusStore


class SearchService:
    def __init__(self, settings: Settings) -> None:
        if not settings.embed_model_path:
            raise RuntimeError("EMBED_MODEL_PATH 未配置")
        self.store = MilvusStore(settings.milvus_uri, settings.milvus_collection, settings.embed_dim)
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

    def info(self) -> dict:
        return {
            "collection": self.store.collection,
            "rows": self.store.count(),
            "model": self.embedder.path,
            "dim": self.embedder.dim,
        }
