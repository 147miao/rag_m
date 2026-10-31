"""Chunk size 消融实验（中文语料，CMRC 文章）。

对同一批文章按 200/400/800/1600 字符切块（overlap=20%），分别建 Milvus collection，
用同一批 100 条中文 query 评测 dense / hybrid 检索，比较 chunk size 的影响。

用法：.venv/bin/python scripts/bench_chunk_size.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ragm.config import get_settings
from ragm.embeddings.embedder import Embedder
from ragm.store.milvus_store import MilvusStore

RET = Path("/run/media/myj/D/rag_m/data/retrieval")
SIZES = [200, 400, 800, 1600]
OVERLAP_RATIO = 0.2


def chunks_of(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + size])
        i += size - overlap
    return out


def metrics(hits: list[list[dict]], golds: list[list[str]], ks=(5, 10, 20)) -> dict:
    out = {}
    for k in ks:
        recalls, ndcgs, mrrs = [], [], []
        for h, g in zip(hits, golds):
            gs = set(g)
            top = list(dict.fromkeys(x["doc_id"] for x in h[:k]))[:k]
            recalls.append(sum(1 for d in top if d in gs) / max(1, len(gs)))
            dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(top) if d in gs)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gs), k)))
            ndcgs.append(dcg / idcg if idcg else 0.0)
            mrrs.append(next((1 / (i + 1) for i, d in enumerate(top) if d in gs), 0.0))
        out[f"recall@{k}"] = round(float(np.mean(recalls)), 4)
        if k == 10:
            out["ndcg@10"] = round(float(np.mean(ndcgs)), 4)
            out["mrr@10"] = round(float(np.mean(mrrs)), 4)
    return out


def main() -> None:
    s = get_settings()
    articles = [json.loads(line) for line in (RET / "corpus_zh.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    queries = [json.loads(line) for line in (RET / "queries_zh.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    golds = [q["gold_ids"] for q in queries]
    embedder = Embedder(s.embed_model_path, s.embed_model, s.embed_max_len)
    results = []
    for size in SIZES:
        overlap = max(1, int(size * OVERLAP_RATIO))
        rows = []
        for a in articles:
            doc_id = a["id"]
            title = a["text"][:30].replace("\n", " ")
            for i, ch in enumerate(chunks_of(a["text"], size, overlap)):
                rows.append(
                    {"chunk_id": f"{doc_id}::s{size}::{i}", "doc_id": doc_id, "title": title,
                     "source": "cmrc", "lang": "zh", "chunk_index": i, "text": ch}
                )
        coll = f"rag_zh_chunk{size}"
        store = MilvusStore(s.milvus_uri, coll, s.embed_dim)
        store.recreate()
        t0 = time.perf_counter()
        for i in range(0, len(rows), 256):
            batch = rows[i : i + 256]
            vecs = embedder.encode([r["text"] for r in batch], 32)
            store.insert([{**{k: r[k] for k in ("chunk_id", "doc_id", "title", "source", "lang", "chunk_index", "text")}, "dense": vecs[j].tolist()} for j, r in enumerate(batch)])
        store.flush()
        store.load()
        ingest_s = round(time.perf_counter() - t0, 1)
        for mode in ("dense", "hybrid"):
            hits = []
            for q in queries:
                qvec = embedder.encode_queries([q["question"]])[0].tolist()
                h = store.dense_search(qvec, 20) if mode == "dense" else store.hybrid_search(q["question"], qvec, 20)
                hits.append(h)
            m = metrics(hits, golds)
            m.update({"chunk_size": size, "overlap": overlap, "mode": mode, "chunks": len(rows),
                      "avg_chars": round(sum(len(r["text"]) for r in rows) / len(rows), 1), "ingest_s": ingest_s})
            print(json.dumps(m, ensure_ascii=False), flush=True)
            results.append(m)
        store.client.drop_collection(coll)
    Path("reports/chunk_size_benchmark.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved reports/chunk_size_benchmark.json")


if __name__ == "__main__":
    main()
