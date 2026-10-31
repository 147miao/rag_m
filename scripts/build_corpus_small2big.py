"""构建 small-to-big 语料：小块（child）用于检索精确性，大块（parent）用于上下文丰富性。

- ZH（CMRC 文章）：child = 200 字 / overlap 40；parent = 整篇文章（≤1600 字，超出按 1600 切）。
- EN（HotpotQA 段落）：段落本身就是自然段（~550 字），child = parent = 段落（1:1）。
输出：
  data/rag/children.jsonl  {chunk_id, parent_id, doc_id, title, source, lang, chunk_index, text}
  data/rag/parents.jsonl   {parent_id, doc_id, title, source, lang, text}
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RET = Path("/run/media/myj/D/rag_m/data/retrieval")
OUT = Path("/run/media/myj/D/rag_m/data/rag")
CHILD = 200
OVERLAP = 40
PARENT_MAX = 1600


def chunks_of(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + size])
        i += size - overlap
    return out


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    children, parents = [], []

    # EN: 段落级 parent = child
    for r in load(RET / "corpus_en.jsonl"):
        title = r["id"]
        doc_id = "en_" + hashlib.sha1(title.encode()).hexdigest()[:12]
        text = f"{title}. {r['text']}".strip()
        parent_id = f"{doc_id}::p0"
        parents.append({"parent_id": parent_id, "doc_id": doc_id, "title": title, "source": "hotpotqa", "lang": "en", "text": text})
        children.append({"chunk_id": f"{parent_id}::c0", "parent_id": parent_id, "doc_id": doc_id, "title": title,
                         "source": "hotpotqa", "lang": "en", "chunk_index": 0, "text": text})

    # ZH: 文章按 1600 切 parent，每个 parent 再切 200 字 child
    for r in load(RET / "corpus_zh.jsonl"):
        doc_id = r["id"]
        title = r["text"][:30].replace("\n", " ")
        for pi, ptext in enumerate(chunks_of(r["text"], PARENT_MAX, 0)):
            parent_id = f"{doc_id}::p{pi}"
            parents.append({"parent_id": parent_id, "doc_id": doc_id, "title": title, "source": "cmrc", "lang": "zh", "text": ptext})
            for ci, ctext in enumerate(chunks_of(ptext, CHILD, OVERLAP)):
                children.append({"chunk_id": f"{parent_id}::c{ci}", "parent_id": parent_id, "doc_id": doc_id, "title": title,
                                 "source": "cmrc", "lang": "zh", "chunk_index": ci, "text": ctext})

    for name, rows in (("children", children), ("parents", parents)):
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {
        "children": len(children), "parents": len(parents),
        "children_by_lang": {k: sum(1 for r in children if r["lang"] == k) for k in ("en", "zh")},
        "avg_child_chars": round(sum(len(r["text"]) for r in children) / len(children), 1),
        "avg_parent_chars": round(sum(len(r["text"]) for r in parents) / len(parents), 1),
    }
    (OUT / "small2big_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
