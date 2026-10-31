"""small-to-big 的 k_parents 权衡曲线（长文档 PubMedQA）。"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ragm.config import get_settings
from ragm.embeddings.embedder import Embedder
from ragm.store.milvus_store import MilvusStore

SRC = Path("/run/media/myj/D/rag_m/data/retrieval/long/pqa_labeled.parquet")
CHILD_COLL, CHUNK_COLL = "rag_long_children", "rag_long_chunks"


def main() -> None:
    s = get_settings()
    rows = pq.read_table(SRC).to_pylist()
    rng = random.Random(42)
    docs = []
    for r in rows:
        text = " ".join((r["context"] or {}).get("contexts") or []).strip()
        if len(text) >= 400:
            docs.append({"doc_id": f"pmid_{r['pubid']}", "question": r["question"], "text": text, "sentences": (r["context"] or {}).get("contexts") or []})
    queries = []
    for d in rng.sample(docs, 100):
        ev = rng.choice([x for x in d["sentences"] if len(x) > 40] or d["sentences"])
        queries.append({"question": d["question"], "gold": d["doc_id"], "evidence": ev})
    parent_text = {d["doc_id"]: d["text"] for d in docs}
    emb = Embedder(s.embed_model_path, s.embed_model, s.embed_max_len)
    child = MilvusStore(s.milvus_uri, CHILD_COLL, s.embed_dim, with_parent=True)
    base = MilvusStore(s.milvus_uri, CHUNK_COLL, s.embed_dim)
    out = []
    for k in (2, 4, 6, 8):
        cov, chars, hit = [], [], []
        for q in queries:
            qv = emb.encode_queries([q["question"]])[0].tolist()
            children = child.hybrid_search(q["question"], qv, 50)
            seen, texts, top = set(), [], ""
            for ch in children:
                pid = ch["parent_id"]
                if pid in seen:
                    continue
                seen.add(pid)
                if not top:
                    top = pid
                texts.append(parent_text.get(pid, ""))
                if len(texts) >= k:
                    break
            hit.append(1.0 if top == q["gold"] else 0.0)
            cov.append(1.0 if any(q["evidence"] in t for t in texts) else 0.0)
            chars.append(sum(len(t) for t in texts))
        out.append({"k_parents": k, "doc_hit@1": round(float(np.mean(hit)), 3), "evidence_coverage": round(float(np.mean(cov)), 3), "avg_chars": round(float(np.mean(chars)), 0)})
        print(json.dumps(out[-1]), flush=True)
    # baseline for reference
    cov, chars, hit = [], [], []
    for q in queries:
        qv = emb.encode_queries([q["question"]])[0].tolist()
        hits = base.hybrid_search(q["question"], qv, 8)
        hit.append(1.0 if hits and hits[0]["doc_id"] == q["gold"] else 0.0)
        cov.append(1.0 if any(q["evidence"] in h["text"] for h in hits) else 0.0)
        chars.append(sum(len(h["text"]) for h in hits))
    out.append({"mode": "baseline_800_top8", "doc_hit@1": round(float(np.mean(hit)), 3), "evidence_coverage": round(float(np.mean(cov)), 3), "avg_chars": round(float(np.mean(chars)), 0)})
    print(json.dumps(out[-1]), flush=True)
    Path("reports/small2big_sweep.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
