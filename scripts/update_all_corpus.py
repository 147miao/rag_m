"""把清洗后的火影语料同步到全量索引（删除旧 moegirl 行 + 插入新行 + 更新合并文件）。"""
from __future__ import annotations

import json
from pathlib import Path

from ragm.config import get_settings
from ragm.embeddings.embedder import Embedder
from ragm.store.milvus_store import MilvusStore

D = Path("/run/media/myj/D/rag_m/data")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    s = get_settings()
    naruto_chunks = load(D / "naruto" / "chunks.jsonl")
    naruto_parents = load(D / "naruto" / "parents.jsonl")
    default_chunks = load(D / "rag" / "corpus.jsonl")
    default_parents = load(D / "rag" / "parents.jsonl")
    papers_chunks = load(D / "papers" / "chunks.jsonl")
    papers_parents = load(D / "papers" / "parents.jsonl")
    merged_chunks = default_chunks + naruto_chunks + papers_chunks
    merged_parents = default_parents + naruto_parents + papers_parents
    (D / "all").mkdir(parents=True, exist_ok=True)
    (D / "all" / "chunks.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in merged_chunks), encoding="utf-8")
    (D / "all" / "parents.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in merged_parents), encoding="utf-8")

    store = MilvusStore(s.milvus_uri, "rag_all_chunks", s.embed_dim)
    deleted = store.delete_by_filter('source == "moegirl"')
    emb = Embedder(s.embed_model_path, s.embed_model, s.embed_max_len)
    total = 0
    for i in range(0, len(naruto_chunks), 256):
        batch = naruto_chunks[i:i + 256]
        vecs = emb.encode([c["text"] for c in batch], 32)
        rows = [{**{k: c[k] for k in ("chunk_id", "doc_id", "title", "source", "lang", "chunk_index", "text")}, "dense": vecs[j].tolist()} for j, c in enumerate(batch)]
        store.insert(rows)
        total += len(rows)
    store.flush()
    print(json.dumps({"deleted": deleted, "inserted": total, "merged_chunks": len(merged_chunks), "rows": store.count()}, ensure_ascii=False))


if __name__ == "__main__":
    main()