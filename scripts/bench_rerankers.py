"""Reranker 横向对比：hybrid top-20 候选 → 各 reranker 重排 → 指标。

候选：无重排（baseline）、bge-reranker-base、bge-reranker-v2-m3、Qwen3-Reranker-0.6B。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ragm.config import get_settings
from ragm.embeddings.reranker import build_reranker
from ragm.store.search_service import SearchService

RET = Path("/run/media/myj/D/rag_m/data/retrieval")
MODELS = Path("/run/media/myj/D/rag_m/models")
CANDIDATES = [
    ("bge-reranker-base", str(MODELS / "rerankers" / "bge-reranker-base")),
    ("bge-reranker-v2-m3", str(MODELS / "rerankers" / "bge-reranker-v2-m3")),
    ("qwen3-reranker-0.6b", str(MODELS / "embeddings" / "qwen3-reranker-0.6b")),
]
TOP_CANDIDATES = 20


def gold_doc_ids(lang: str, gold_ids: list[str]) -> list[str]:
    if lang == "en":
        return ["en_" + hashlib.sha1(t.encode()).hexdigest()[:12] for t in gold_ids]
    return list(gold_ids)


def metrics(ranked: list[list[str]], golds: list[list[str]], ks=(5, 10, 20)) -> dict:
    out = {}
    for k in ks:
        recalls, ndcgs, mrrs = [], [], []
        for ranking, gold in zip(ranked, golds):
            gs = set(gold)
            top = list(dict.fromkeys(ranking))[:k]
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
    ap.add_argument("--langs", default="en,zh")
    ap.add_argument("--out", default="reports/reranker_benchmark.json")
    ap.add_argument("--models", default="all", help="逗号分隔的模型名，all 表示全部")
    ap.add_argument("--candidates", default="reports/rerank_candidates.json", help="候选缓存路径")
    ap.add_argument("--reuse", action="store_true", help="复用已有候选缓存（避免重复加载 embedder）")
    args = ap.parse_args()
    s = get_settings()
    cand_path = Path(args.candidates)
    if args.reuse and cand_path.exists():
        data = json.loads(cand_path.read_text(encoding="utf-8"))
        print(f"reused candidates from {cand_path}", flush=True)
    else:
        svc = SearchService(s)
        data = {}
        for lang in args.langs.split(","):
            qs = [json.loads(line) for line in (RET / f"queries_{lang}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            cands: list[dict] = []
            t0 = time.perf_counter()
            for q in qs:
                res = svc.search_with_scores(q["question"], k=TOP_CANDIDATES)
                cands.append({"question": q["question"], "gold": q["gold_ids"], "hits": res["hits"]})
            print(f"[{lang}] candidates for {len(qs)} queries in {time.perf_counter()-t0:.1f}s", flush=True)
            data[lang] = {"queries": cands, "golds": [gold_doc_ids(lang, c["gold"]) for c in cands]}
        del svc
        import torch as _torch
        _torch.cuda.empty_cache()
        cand_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    results = []
    # baseline: hybrid 顺序
    for lang in data:
        ranked = [[h["doc_id"] for h in c["hits"]] for c in data[lang]["queries"]]
        m = metrics(ranked, data[lang]["golds"])
        m.update({"model": "no_rerank(hybrid)", "lang": lang, "p50_ms": 0.0, "peak_vram_gib": 0.0})
        print(json.dumps(m, ensure_ascii=False), flush=True)
        results.append(m)
    wanted = None if args.models == "all" else {x.strip() for x in args.models.split(",")}
    for name, path in CANDIDATES:
        if wanted is not None and name not in wanted:
            continue
        if not Path(path).exists():
            print(f"skip {name}: missing {path}", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            reranker = build_reranker(name, path)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {name}: load failed {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            continue
        load_s = time.perf_counter() - t0
        for lang in data:
            lat = []
            ranked = []
            t0 = time.perf_counter()
            for c in data[lang]["queries"]:
                t1 = time.perf_counter()
                reranked = reranker.rerank(c["question"], c["hits"], top_n=TOP_CANDIDATES)
                lat.append((time.perf_counter() - t1) * 1000)
                ranked.append([h["doc_id"] for h in reranked])
            m = metrics(ranked, data[lang]["golds"])
            m.update(
                {
                    "model": name,
                    "lang": lang,
                    "load_s": round(load_s, 1),
                    "p50_ms": round(float(np.median(lat)), 1),
                    "total_s": round(time.perf_counter() - t0, 1),
                    "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0,
                }
            )
            print(json.dumps(m, ensure_ascii=False), flush=True)
            results.append(m)
        del reranker
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    import torch

    main()