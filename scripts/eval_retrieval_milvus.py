"""用检索评测集评估 Milvus 三种检索模式（dense / sparse / hybrid）。"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ragm.config import get_settings
from ragm.store.search_service import SearchService

D = Path("/run/media/myj/D/rag_m/data/retrieval")


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (D / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def gold_doc_ids(lang: str, queries: list[dict]) -> list[list[str]]:
    if lang == "en":
        return [["en_" + hashlib.sha1(t.encode()).hexdigest()[:12] for t in q["gold_ids"]] for q in queries]
    return [q["gold_ids"] for q in queries]


def metrics(hits: list[list[dict]], golds: list[list[str]], ks=(5, 10, 20)) -> dict:
    out = {}
    for k in ks:
        recalls, ndcgs, mrrs = [], [], []
        for h, g in zip(hits, golds):
            gs = set(g)
            # 同一 doc 的多个 chunk 去重，避免 recall 超过 1
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--modes", default="dense,sparse,hybrid")
    args = ap.parse_args()
    svc = SearchService(get_settings())
    results = []
    for lang in ("en", "zh"):
        queries = load(f"queries_{lang}.jsonl")
        golds = gold_doc_ids(lang, queries)
        for mode in args.modes.split(","):
            t0 = time.perf_counter()
            hits = [svc.search(q["question"], args.k, mode)["hits"] for q in queries]
            m = metrics(hits, golds)
            m.update({"lang": lang, "mode": mode, "queries": len(queries), "wall_s": round(time.perf_counter() - t0, 1)})
            print(json.dumps(m, ensure_ascii=False), flush=True)
            results.append(m)
    Path("reports/retrieval_milvus.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved reports/retrieval_milvus.json")


if __name__ == "__main__":
    main()