"""把 corpus.jsonl 编码并写入 Milvus（dense + BM25 sparse + 标量元数据）。

用法：
  .venv/bin/python scripts/ingest_milvus.py [--recreate] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ragm.config import get_settings
from ragm.embeddings.embedder import Embedder
from ragm.store.milvus_store import MilvusStore

DEFAULT_CORPUS = Path("/run/media/myj/D/rag_m/data/rag/corpus.jsonl")


def load_corpus(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--collection", default="")
    ap.add_argument("--with-parent", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    if not s.embed_model_path:
        raise SystemExit("EMBED_MODEL_PATH 未配置")
    rows = load_corpus(Path(args.corpus), args.limit or None)
    print(f"corpus: {len(rows)} chunks | model={s.embed_model} dim={s.embed_dim}")

    collection = args.collection or s.milvus_collection
    store = MilvusStore(s.milvus_uri, collection, s.embed_dim, with_parent=args.with_parent)
    if args.recreate:
        print("recreating collection ...", flush=True)
        store.recreate()
    embedder = Embedder(s.embed_model_path, s.embed_model, s.embed_max_len)
    print(f"embedder ready: dim={embedder.dim}", flush=True)

    t0 = time.perf_counter()
    inserted = 0
    for i in range(0, len(rows), 256):
        batch = rows[i : i + 256]
        vecs = embedder.encode([r["text"] for r in batch], args.batch)
        payload = []
        for j, r in enumerate(batch):
            row = {
                "chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "title": r["title"],
                "source": r["source"], "lang": r["lang"], "chunk_index": r["chunk_index"],
                "text": r["text"][:5000], "dense": vecs[j].tolist(),
            }
            if args.with_parent:
                row["parent_id"] = r["parent_id"]
            payload.append(row)
        inserted += store.insert(payload)
        done = inserted
        if done % 2048 < 256:
            elapsed = time.perf_counter() - t0
            print(f"  {done}/{len(rows)} | {done/max(elapsed,1e-6):.1f} chunks/s | {elapsed:.0f}s", flush=True)
    store.flush()
    store.load()
    print(f"done: inserted={inserted}, collection rows={store.count()}, elapsed={time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()