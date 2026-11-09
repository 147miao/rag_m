"""摄取流水线测试：解析、切块、幂等、增量编码（fake Milvus，不依赖 GPU）。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from docx import Document

from ragm.ingest.pipeline import build

ROOT = Path(__file__).resolve().parents[1]


def _load_ingest_script():
    spec = importlib.util.spec_from_file_location("ingest_docs_for_tests", ROOT / "scripts" / "ingest_docs.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ingest_docs = _load_ingest_script()


class FakeMilvusStore:
    """记录 delete/insert 调用的最小 MilvusStore 替身。"""

    def __init__(self, exists: bool = True) -> None:
        self.client = self
        self.exists = exists
        self.recreated = False
        self.deleted: list[str] = []
        self.inserted: list[dict] = []
        self.flush_count = 0
        self.load_count = 0

    def has_collection(self, collection: str) -> bool:
        return self.exists

    def recreate(self) -> None:
        self.recreated = True
        self.exists = True
        self.inserted.clear()

    def delete_by_filter(self, expr: str) -> int:
        self.deleted.append(expr)
        return 0

    def insert(self, rows: list[dict], batch_size: int = 256) -> int:
        self.inserted.extend(rows)
        return len(rows)

    def flush(self) -> None:
        self.flush_count += 1

    def load(self) -> None:
        self.load_count += 1

    def count(self) -> int:
        return len(self.inserted)


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], _threads: int = 32) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]


def _make_two_md(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.md").write_text("# A\n\n" + "甲" * 300, encoding="utf-8")
    (root / "b.md").write_text("# B\n\n" + "乙" * 300, encoding="utf-8")


def _manifest(out: Path) -> dict:
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def _make_docs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.md").write_text("# 第一章\n\n这是第一段，讲的是背景。\n\n## 1.1 小节\n\n这是小节内容，包含更多细节。\n", encoding="utf-8")
    (root / "b.txt").write_text("纯文本文件，用于测试解析与切块。" * 20, encoding="utf-8")
    doc = Document()
    doc.add_heading("标题一", level=1)
    doc.add_paragraph("这是 Word 文档的正文段落。" * 10)
    doc.add_heading("标题二", level=2)
    doc.add_paragraph("第二段内容，测试章节切分。" * 10)
    doc.save(str(root / "c.docx"))


def test_build_parses_and_is_idempotent(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_docs(src)
    r1 = build(src, out)
    assert r1["parsed"] == 3
    assert r1["chunks"] > 0 and r1["children"] > 0 and r1["parents"] > 0
    r2 = build(src, out)
    assert r2["parsed"] == 0 and r2["skipped"] == 3
    assert r2["chunks"] == r1["chunks"]


def test_changed_file_is_reparsed(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_docs(src)
    build(src, out)
    (src / "a.md").write_text("# 新章节\n\n内容变了，需要重新解析。" * 5, encoding="utf-8")
    r = build(src, out)
    assert r["parsed"] == 1 and r["skipped"] == 2


def test_deleted_file_removed(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_docs(src)
    build(src, out)
    (src / "b.txt").unlink()
    r = build(src, out)
    assert r["files_total"] == 2


def test_build_cleans_rows_when_file_becomes_empty(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    a_md = src / "a.md"
    a_md.write_text("# 标题\n\n这是初始内容。" * 80, encoding="utf-8")
    first = build(src, out)
    assert first["chunks"] > 0
    old_doc_id = _manifest(out)["files"]["a.md"]["doc_id"]
    rows_file = out / "rows" / f"{old_doc_id}.json"
    assert rows_file.exists()

    a_md.write_text("   \n\n", encoding="utf-8")
    report = build(src, out)

    assert report["parsed"] == 0 and report["skipped"] == 1
    assert report["chunks"] == 0
    assert old_doc_id in report["stale_doc_ids"]
    assert "a.md" not in _manifest(out)["files"]
    assert not rows_file.exists()
    assert (out / "chunks.jsonl").read_text(encoding="utf-8") == ""


def test_embed_incremental_only_reencodes_changed_doc(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_two_md(src)
    build(src, out)
    files = _manifest(out)["files"]
    doc_a, doc_b = files["a.md"]["doc_id"], files["b.md"]["doc_id"]

    store = FakeMilvusStore()
    embedder = FakeEmbedder()
    report = ingest_docs.embed_chunks(
        out, "rag_test", False, changed_files=["a.md"], store=store, embedder=embedder
    )

    inserted_docs = {row["doc_id"] for row in store.inserted}
    assert inserted_docs == {doc_a}
    assert doc_b not in inserted_docs
    assert store.deleted == [f'doc_id == "{doc_a}"']
    assert report["mode"] == "incremental"
    assert report["inserted"] == len(store.inserted) > 0
    assert len(embedder.calls) == 1


def test_embed_incremental_skips_unchanged_docs_without_touching_store(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_two_md(src)
    build(src, out)

    store = FakeMilvusStore()
    embedder = FakeEmbedder()
    report = ingest_docs.embed_chunks(out, "rag_test", False, changed_files=[], store=store, embedder=embedder)

    assert report["inserted"] == 0
    assert store.deleted == [] and store.inserted == []
    assert embedder.calls == []
    assert store.flush_count == 0 and store.load_count == 0


def test_embed_incremental_deletes_old_doc_id_after_change(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_two_md(src)
    build(src, out)
    old_a = _manifest(out)["files"]["a.md"]["doc_id"]

    (src / "a.md").write_text("# A\n\n" + "丙" * 300, encoding="utf-8")
    build_report = build(src, out)
    new_a = _manifest(out)["files"]["a.md"]["doc_id"]
    assert build_report["changed_files"] == ["a.md"]
    assert old_a in build_report["stale_doc_ids"]
    assert new_a != old_a

    store = FakeMilvusStore()
    embedder = FakeEmbedder()
    ingest_docs.embed_chunks(
        out,
        "rag_test",
        False,
        changed_files=build_report["changed_files"],
        stale_doc_ids=build_report["stale_doc_ids"],
        store=store,
        embedder=embedder,
    )

    assert store.deleted == [f'doc_id == "{new_a}"', f'doc_id == "{old_a}"']
    assert {row["doc_id"] for row in store.inserted} == {new_a}


def test_embed_creates_missing_collection_and_reindexes_all(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_two_md(src)
    build(src, out)
    all_docs = {meta["doc_id"] for meta in _manifest(out)["files"].values()}

    store = FakeMilvusStore(exists=False)
    embedder = FakeEmbedder()
    report = ingest_docs.embed_chunks(out, "rag_missing", False, changed_files=[], store=store, embedder=embedder)

    assert store.recreated is True
    assert store.deleted == []
    assert {row["doc_id"] for row in store.inserted} == all_docs
    assert report["mode"] == "recreate-missing"


def test_embed_recreate_rebuilds_all_docs(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "out"
    _make_two_md(src)
    build(src, out)
    all_docs = {meta["doc_id"] for meta in _manifest(out)["files"].values()}

    store = FakeMilvusStore(exists=True)
    embedder = FakeEmbedder()
    report = ingest_docs.embed_chunks(
        out, "rag_test", True, changed_files=["a.md"], stale_doc_ids=["old_doc"], store=store, embedder=embedder
    )

    assert store.recreated is True
    assert store.deleted == []
    assert {row["doc_id"] for row in store.inserted} == all_docs
    assert report["mode"] == "recreate"
