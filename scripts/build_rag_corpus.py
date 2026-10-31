"""构建 RAG 语料：EN HotpotQA 段落 + ZH CMRC 文章，切块并写 jsonl。

输出：/run/media/myj/D/rag_m/data/rag/corpus.jsonl
字段：chunk_id / doc_id / title / source / lang / chunk_index / text
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SRC = Path("/run/media/myj/D/rag_m/data/retrieval")
OUT = Path("/run/media/myj/D/rag_m/data/rag")
MAX_CHARS = 800
MAX_TEXT_CHARS = 5000  # Milvus VARCHAR 16384 字节上限，中文 3 字节/字
OVERLAP = 100


def chunks_of(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= MAX_CHARS:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + MAX_CHARS])
        i += MAX_CHARS - OVERLAP
    return out


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    # EN: 段落级（HotpotQA paragraph 已是自然段）
    for r in load(SRC / "corpus_en.jsonl"):
        title = r["id"]
        doc_id = "en_" + hashlib.sha1(title.encode()).hexdigest()[:12]
        text = f"{title}. {r['text']}".strip()[:MAX_TEXT_CHARS]
        rows.append(
            {"chunk_id": f"{doc_id}::c0", "doc_id": doc_id, "title": title, "source": "hotpotqa",
             "lang": "en", "chunk_index": 0, "text": text}
        )
    # ZH: 文章切块
    for r in load(SRC / "corpus_zh.jsonl"):
        doc_id = r["id"]
        title = r["text"][:30].replace("\n", " ")
        for i, ch in enumerate(chunks_of(r["text"])):
            rows.append(
                {"chunk_id": f"{doc_id}::c{i}", "doc_id": doc_id, "title": title, "source": "cmrc",
                 "lang": "zh", "chunk_index": i, "text": ch}
            )
    with (OUT / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {
        "total": len(rows),
        "by_lang": {k: sum(1 for r in rows if r["lang"] == k) for k in ("en", "zh")},
        "by_source": {k: sum(1 for r in rows if r["source"] == k) for k in ("hotpotqa", "cmrc")},
        "avg_chars": round(sum(len(r["text"]) for r in rows) / len(rows), 1),
    }
    (OUT / "corpus_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()