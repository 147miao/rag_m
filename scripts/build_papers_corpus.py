"""构建 arXiv 论文语料（真实长文档）：pypdf 抽文本 → small-to-big 切块。"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader

PDF_DIR = Path("/run/media/myj/D/rag_m/data/papers/pdf")
OUT = Path("/run/media/myj/D/rag_m/data/papers")
PARENT_MAX, CHILD, CHILD_OVERLAP, FLAT, FLAT_OVERLAP = 1600, 200, 40, 800, 100


def chunks_of(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    docs = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 3000:
            continue
        docs.append({"doc_id": "paper_" + pdf.stem.replace(".", "_"), "title": pdf.stem, "text": text, "chars": len(text)})
    parents, children, chunks = [], [], []
    for d in docs:
        for pi, ptext in enumerate(chunks_of(d["text"], PARENT_MAX, 0)):
            pid = d["doc_id"] + "::p" + str(pi)
            parents.append({"parent_id": pid, "doc_id": d["doc_id"], "title": d["title"], "source": "arxiv", "lang": "en", "text": ptext})
            for ci, ctext in enumerate(chunks_of(ptext, CHILD, CHILD_OVERLAP)):
                children.append({"chunk_id": pid + "::c" + str(ci), "parent_id": pid, "doc_id": d["doc_id"], "title": d["title"], "source": "arxiv", "lang": "en", "chunk_index": ci, "text": ctext})
        for fi, ftext in enumerate(chunks_of(d["text"], FLAT, FLAT_OVERLAP)):
            chunks.append({"chunk_id": d["doc_id"] + "::f" + str(fi), "doc_id": d["doc_id"], "title": d["title"], "source": "arxiv", "lang": "en", "chunk_index": fi, "text": ftext})
    for name, data in (("parents", parents), ("children", children), ("chunks", chunks)):
        (OUT / (name + ".jsonl")).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data), encoding="utf-8")
    print(json.dumps({"docs": len(docs), "parents": len(parents), "children": len(children), "chunks": len(chunks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
