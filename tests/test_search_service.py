"""SearchService 子集合容错测试（fake store，不依赖 Milvus / GPU）。"""
from __future__ import annotations

import pytest

from ragm.store.search_service import SearchService


class FakeClient:
    def __init__(self, exists: bool) -> None:
        self.exists = exists

    def has_collection(self, name: str) -> bool:
        return self.exists


class FakeStore:
    def __init__(
        self,
        collection: str,
        *,
        exists: bool = True,
        rows: int = 3,
        count_error: Exception | None = None,
    ) -> None:
        self.collection = collection
        self.client = FakeClient(exists)
        self.rows = rows
        self.count_error = count_error

    def count(self) -> int:
        if self.count_error is not None:
            raise self.count_error
        return self.rows


class ExplodingEmbedder:
    path = "fake"
    dim = 8

    def encode_queries(self, _queries: list[str]):
        raise AssertionError("child collection 缺失时不应加载/调用 embedding")


def _service(child: FakeStore) -> SearchService:
    svc = SearchService.__new__(SearchService)
    svc.store = FakeStore("rag_chunks", rows=11)
    svc.child_store = child
    svc.parents = {}
    svc.embedder = ExplodingEmbedder()
    return svc


def test_info_tolerates_missing_child_collection() -> None:
    info = _service(FakeStore("rag_children", exists=False)).info()
    assert info["child_ready"] is False
    assert info["child_rows"] is None
    assert "not found" in info["child_error"]
    assert info["rows"] == 11


def test_info_tolerates_child_count_failure() -> None:
    child = FakeStore("rag_children", count_error=RuntimeError("DescribeCollectionException"))
    info = _service(child).info()
    assert info["child_ready"] is False
    assert info["child_rows"] is None
    assert "DescribeCollectionException" in info["child_error"]


def test_info_reports_child_rows_when_ready() -> None:
    info = _service(FakeStore("rag_children", rows=7)).info()
    assert info["child_ready"] is True
    assert info["child_rows"] == 7
    assert info["child_error"] is None


def test_small2big_raises_clear_error_when_child_missing() -> None:
    svc = _service(FakeStore("rag_children", exists=False))
    with pytest.raises(RuntimeError, match="child collection"):
        svc.search_small2big("测试问题")
