"""摄取流水线测试：解析、切块、幂等。"""
from __future__ import annotations

from pathlib import Path

from docx import Document

from ragm.ingest.pipeline import build


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
