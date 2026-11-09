"""对比 multihop 路由：hybrid-only vs hybrid + Graph RAG 扩展。

指标：gold 文档命中率（top-k 去重 doc）、gold 文档召回、证据数、延迟；并统计图谱额外带回的 gold。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ragm.config import get_settings
from ragm.graph.kg import KGStore
from ragm.store.search_service import SearchService

RET = Path("/run/media/myj/D/rag_m/data/retrieval")
N_QUERIES = 40


def doc_id(title: str) -> str:
    return "en_" + hashlib.sha1(title.encode()).hexdigest()[:12]


def main() -> None:
    s = get_settings()
    svc = SearchService(s)
    kg = KGStore(s.neo4j_uri, s.neo4j_user, s.neo4j_password)
    kg.load()
    qs = [json.loads(line) for line in (RET / "queries_en.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    qs = [q for q in qs if q.get("type") in ("bridge", "comparison")][:N_QUERIES]
    rows = []
    for q in qs:
        gold = {doc_id(t) for t in q["gold_ids"]}
        t0 = time.perf_counter()
        base = svc.search_with_scores(q["question"], k=12)
        base_ms = (time.perf_counter() - t0) * 1000
        base_docs = list(dict.fromkeys(h["doc_id"] for h in base["hits"]))
        t0 = time.perf_counter()
        seeds = kg.link(q["question"], [h.get("title") or "" for h in base["hits"]])
        _edges, neighbors = kg.expand(seeds, limit=40)
        titles = [n for n in neighbors if n not in seeds][:12]
        graph_chunks = svc.chunks_by_titles(titles, limit=8)
        graph_docs = list(dict.fromkeys(c["doc_id"] for c in graph_chunks))
        merged_docs = list(dict.fromkeys(graph_docs + base_docs))
        graph_ms = (time.perf_counter() - t0) * 1000
        rows.append(
            {
                "question": q["question"][:70],
                "gold": len(gold),
                "base_hit": bool(gold & set(base_docs)),
                "graph_hit": bool(gold & set(merged_docs)),
                "base_recall": len(gold & set(base_docs)) / len(gold),
                "graph_recall": len(gold & set(merged_docs)) / len(gold),
                "graph_only_gold": len(gold & set(graph_docs) - set(base_docs)),
                "seeds": seeds[:3],
                "graph_docs": len(graph_docs),
                "base_ms": round(base_ms, 1),
                "graph_ms": round(graph_ms, 1),
            }
        )
    def agg(key: str) -> float:
        return round(float(np.mean([r[key] for r in rows])), 4)
    summary = {
        "n": len(rows),
        "base_hit_rate": agg("base_hit"),
        "graph_hit_rate": agg("graph_hit"),
        "base_recall": agg("base_recall"),
        "graph_recall": agg("graph_recall"),
        "graph_only_gold_avg": agg("graph_only_gold"),
        "base_ms": agg("base_ms"),
        "graph_ms": agg("graph_ms"),
    }
    Path("reports/graph_rag_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    kg.close()


if __name__ == "__main__":
    main()
