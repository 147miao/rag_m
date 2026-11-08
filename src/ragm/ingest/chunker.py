"""章节感知切块：parent（上下文）/ child（精确检索）/ flat chunk（hybrid 索引）。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .readers import Section

PARENT_MAX = 1600
CHILD = 200
CHILD_OVERLAP = 40
FLAT = 800
FLAT_OVERLAP = 100


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def chunks_of(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, i = [], 0
    while i < len(text):
        piece = text[i:i + size].strip()
        if piece:
            out.append(piece)
        i += size - overlap
    return out


def doc_id_for(file_hash: str, rel_path: str) -> str:
    return "doc_" + sha1(file_hash + "|" + rel_path)[:12]


def build_rows(
    rel_path: str,
    file_hash: str,
    sections: list[Section],
    lang: str = "zh",
    source: str = "docs",
) -> tuple[list[dict], list[dict], list[dict]]:
    """返回 (parents, children, chunks)。parent = 章节（≤1600 字切片），child = 200 字。"""
    doc_id = doc_id_for(file_hash, rel_path)
    parents: list[dict] = []
    children: list[dict] = []
    chunks: list[dict] = []
    title = Path(rel_path).stem
    for sec in sections:
        section_title = sec.section or "正文"
        page_tag = f" · p{sec.page}" if sec.page else ""
        full_title = f"{rel_path}{page_tag} · {section_title}"[:200]
        for pi, ptext in enumerate(chunks_of(sec.text, PARENT_MAX, 0)):
            parent_id = f"{doc_id}::p{len(parents)}"
            parents.append({
                "parent_id": parent_id, "doc_id": doc_id, "title": full_title, "source": source,
                "lang": lang, "text": ptext, "page": sec.page, "section": section_title, "chunk_hash": sha1(ptext),
            })
            for ci, ctext in enumerate(chunks_of(ptext, CHILD, CHILD_OVERLAP)):
                children.append({
                    "chunk_id": f"{parent_id}::c{ci}", "parent_id": parent_id, "doc_id": doc_id,
                    "title": full_title, "source": source, "lang": lang, "chunk_index": ci,
                    "text": ctext, "chunk_hash": sha1(ctext),
                })
        for fi, ftext in enumerate(chunks_of(sec.text, FLAT, FLAT_OVERLAP)):
            chunks.append({
                "chunk_id": f"{doc_id}::f{len(chunks)}", "doc_id": doc_id, "title": full_title,
                "source": source, "lang": lang, "chunk_index": fi, "text": ftext, "chunk_hash": sha1(ftext),
            })
    return parents, children, chunks