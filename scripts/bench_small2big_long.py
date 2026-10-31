"""Small-to-big 在长文档上的对照实验（PubMedQA labeled，1,000 篇医学摘要，~1.2k 字符/篇）。

对比：
- baseline：800 字符 chunk 直接检索 top-8
- small2big：200 字符 child 检索 top-50 → 去重到 parent（整篇文档）top-8

指标：doc hit@8、evidence sentence coverage@8（返回上下文是否包含 gold 文档中的某句证据）、平均上下文字符数。
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ragm.config import get_settings
from ragm.embeddings.embedder import Embedder
from ragm.store.milvus_store import MilvusStore

SRC = Path("/run/media/myj/D/rag_m/data/retrieval/long/pqa_labeled.parquet")
CHUNK_COLL = "rag_long_chunks"
CHILD_COLL = "rag_long_children"
CHILD, OVERLAP = 200, 40
BASE_CHUNK, BASE_OVERLAP = 800, 100
N_QUERIES = 100
SEED = 42


def chunks_of(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + size])
        i += size - overlap
    return out


def metrics(rows: list[dict], golds: list[str], evidences: list[str]) -> dict:
    n = len(rows)
    doc_hits = [1.0 if r["doc_id"] == g else 0.0 for r, g in zip(rows, golds)]
    cover = [1.0 if e and any(e in t for t in r["texts"]) else 0.0 for r, e in zip(rows, evidences)]
    return {
        "n": n,
        "doc_hit@8": round(float(np.mean(doc_hits)), 4),
        "evidence_coverage@8": round(float(np.mean(cover)), 4),
        "avg_context_chars": round(float(np.mean([sum(len(t) for t in r["texts"]) for r in rows])), 1),
    }


def main() -> None:
    s = get_settings()
    rows = pq.read_table(SRC).to_pylist()
    rng = random.Random(SEED)
    docs = []
    for r in rows:
        sentences = (r["context"] or {}).get("contexts") or []
        text = " ".join(sentences).strip()
        if len(text) < 400:
            continue
        docs.append({"doc_id": f"pmid_{r['pubid']}", "question": r["question"], "text": text, "sentences": sentences})
    print(f"docs: {len(docs)} | avg chars {sum(len(d['text']) for d in docs)/len(docs):.0f}", flush=True)

    # queries + evidence sentence（随机取一句作为"答案所在证据"）
    queries = []
    for d in rng.sample(docs, N_QUERIES):
        ev = rng.choice([x for x in d["sentences"] if len(x) > 40] or d["sentences"])
        queries.append({"question": d["question"], "gold": d["doc_id"], "evidence": ev})

    embedder = Embedder(s.embed_model_path, s.embed_model, s.embed_max_len)
    results = {}

    # ---------- baseline: 800 chunk ----------
    base_rows = []
    for d in docs:
        for i, ch in enumerate(chunks_of(d["text"], BASE_CHUNK, BASE_OVERLAP)):
            base_rows.append({"chunk_id": f"{d['doc_id']}::b{i}", "doc_id": d["doc_id"], "title": d["doc_id"],
                              "source": "pubmedqa", "lang": "en", "chunk_index": i, "text": ch})
    base = MilvusStore(s.milvus_uri, CHUNK_COLL, s.embed_dim)
    base.recreate()
    t0 = time.perf_counter()
    for i in range(0, len(base_rows), 256):
        batch = base_rows[i : i + 256]
        vecs = embedder.encode([r["text"] for r in batch], 32)
        base.insert([{**{k: r[k] for k in ("chunk_id", "doc_id", "title", "source", "lang", "chunk_index", "text")}, "dense": vecs[j].tolist()} for j, r in enumerate(batch)])
    base.flush(); base.load()
    print(f"baseline indexed {len(base_rows)} chunks in {time.perf_counter()-t0:.0f}s", flush=True)

    # ---------- small2big: child 200 -> parent doc ----------
    child_rows = []
    for d in docs:
        for i, ch in enumerate(chunks_of(d["text"], CHILD, OVERLAP)):
            child_rows.append({"chunk_id": f"{d['doc_id']}::c{i}", "parent_id": d["doc_id"], "doc_id": d["doc_id"],
                               "title": d["doc_id"], "source": "pubmedqa", "lang": "en", "chunk_index": i, "text": ch})
    child = MilvusStore(s.milvus_uri, CHILD_COLL, s.embed_dim, with_parent=True)
    child.recreate()
    t0 = time.perf_counter()
    for i in range(0, len(child_rows), 256):
        batch = child_rows[i : i + 256]
        vecs = embedder.encode([r["text"] for r in batch], 32)
        child.insert([{**{k: r[k] for k in ("chunk_id", "parent_id", "doc_id", "title", "source", "lang", "chunk_index", "text")}, "dense": vecs[j].tolist()} for j, r in enumerate(batch)])
    child.flush(); child.load()
    print(f"child indexed {len(child_rows)} chunks in {time.perf_counter()-t0:.0f}s", flush=True)
    parent_text = {d["doc_id"]: d["text"] for d in docs}

    # ---------- eval ----------
    for mode in ("baseline", "small2big"):
        out = []
        lat = []
        for q in queries:
            t0 = time.perf_counter()
            qv = embedder.encode_queries([q["question"]])[0].tolist()
            if mode == "baseline":
                hits = base.hybrid_search(q["question"], qv, 8)
                out.append({"doc_id": hits[0]["doc_id"] if hits else "", "texts": [h["text"] for h in hits]})
            else:
                children = child.hybrid_search(q["question"], qv, 50)
                seen, texts, top_doc = set(), [], ""
                for ch in children:
                    pid = ch["parent_id"]
                    if pid in seen:
                        continue
                    seen.add(pid)
                    if not top_doc:
                        top_doc = pid
                    texts.append(parent_text.get(pid, ""))
                    if len(texts) >= 8:
                        break
                out.append({"doc_id": top_doc, "texts": texts})
            lat.append((time.perf_counter() - t0) * 1000)
        m = metrics(out, [q["gold"] for q in queries], [q["evidence"] for q in queries])
        m.update({"mode": mode, "p50_ms": round(sorted(lat)[len(lat) // 2], 1)})
        results[mode] = m
        print(json.dumps(m, ensure_ascii=False), flush=True)

    Path("reports/small2big_long.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved reports/small2big_long.json")


if __name__ == "__main__":
    main()
