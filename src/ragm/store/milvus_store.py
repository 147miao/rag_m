"""Milvus 3.0 存储：dense + BM25 sparse 混合检索。"""
from __future__ import annotations

from typing import Any

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

OUTPUT_FIELDS = ["chunk_id", "doc_id", "title", "source", "lang", "chunk_index", "text"]


class MilvusStore:
    def __init__(self, uri: str, collection: str, dim: int) -> None:
        self.uri = uri
        self.collection = collection
        self.dim = dim
        self.client = MilvusClient(uri=uri)

    # ---------- schema ----------
    def recreate(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=128)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("source", DataType.VARCHAR, max_length=64)
        schema.add_field("lang", DataType.VARCHAR, max_length=8)
        schema.add_field("chunk_index", DataType.INT32)
        schema.add_field("text", DataType.VARCHAR, max_length=16384, enable_analyzer=True, analyzer_params={"type": "chinese"})
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(name="bm25_fn", function_type=FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"])
        )
        self.client.create_collection(collection_name=self.collection, schema=schema)
        idx = self.client.prepare_index_params()
        idx.add_index(field_name="dense", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
        idx.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        self.client.create_index(self.collection, idx)

    def load(self) -> None:
        self.client.load_collection(self.collection)

    # ---------- write ----------
    def insert(self, rows: list[dict], batch_size: int = 256) -> int:
        total = 0
        for i in range(0, len(rows), batch_size):
            self.client.insert(collection_name=self.collection, data=rows[i : i + batch_size])
            total += len(rows[i : i + batch_size])
        return total

    def flush(self) -> None:
        self.client.flush(self.collection)

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))

    # ---------- read ----------
    def hybrid_search(self, query: str, dense_vec: list[float], k: int = 20, rrf_k: int = 60) -> list[dict[str, Any]]:
        req_dense = AnnSearchRequest(data=[dense_vec], anns_field="dense", param={"metric_type": "COSINE", "params": {}}, limit=k)
        req_sparse = AnnSearchRequest(data=[query], anns_field="sparse", param={"metric_type": "BM25", "params": {}}, limit=k)
        res = self.client.hybrid_search(
            self.collection, [req_dense, req_sparse], ranker=RRFRanker(rrf_k), limit=k, output_fields=OUTPUT_FIELDS
        )
        return [self._hit(h) for h in res[0]]

    def dense_search(self, dense_vec: list[float], k: int = 20) -> list[dict[str, Any]]:
        res = self.client.search(
            self.collection, data=[dense_vec], anns_field="dense",
            search_params={"metric_type": "COSINE", "params": {}}, limit=k, output_fields=OUTPUT_FIELDS,
        )
        return [self._hit(h) for h in res[0]]

    def sparse_search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        res = self.client.search(
            self.collection, data=[query], anns_field="sparse",
            search_params={"metric_type": "BM25", "params": {}}, limit=k, output_fields=OUTPUT_FIELDS,
        )
        return [self._hit(h) for h in res[0]]

    @staticmethod
    def _hit(h: dict) -> dict[str, Any]:
        entity = h.get("entity", {})
        return {"score": float(h.get("distance", 0.0)), **{k: entity.get(k) for k in OUTPUT_FIELDS}}