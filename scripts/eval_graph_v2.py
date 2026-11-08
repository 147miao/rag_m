"""Graph RAG v1（1 跳）vs v2（2 跳 + 迭代检索）对照评测。

需要 GPU + Milvus + Neo4j（会加载 embedding/reranker）：
  python scripts/eval_graph_v2.py --hops 1 2 --limit 12
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragm.classifier.factory import build_classifier
from ragm.config import get_settings
from ragm.embeddings.embedder import Embedder
from ragm.embeddings.reranker import build_reranker
from ragm.graph.kg import KGStore
from ragm.rag.graph import RagGraph
from ragm.rag.llm import LLMClient
from ragm.store.milvus_store import MilvusStore
from ragm.store.search_service import SearchService

QUESTIONS = [
    ("自来也的弟子中谁成为了四代火影？", ["水门", "波风水门"]),
    ("晓组织里与宇智波鼬搭档的是谁？", ["鬼鲛"]),
    ("千手柱间的弟弟是谁？", ["扉间"]),
    ("谁把九尾封印在鸣人体内？", ["水门", "波风水门"]),
    ("宇智波鼬的弟弟是谁？", ["佐助"]),
    ("木叶三忍除了自来也和纲手还有谁？", ["大蛇丸"]),
    ("谁教鸣人螺旋丸？", ["自来也"]),
    ("佩恩的本体是谁？", ["长门"]),
    ("四代火影的儿子是谁？", ["鸣人"]),
    ("谁和宇智波斑一起建立了木叶？", ["柱间"]),
    ("九尾人柱力中谁成为了七代火影？", ["鸣人"]),
    ("大蛇丸的弟子有哪些？", ["兜", "水月"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()
    s = get_settings()
    emb = Embedder(s.embed_model_path, s.embed_model, s.embed_max_len)
    store = MilvusStore(s.milvus_uri, s.milvus_collection, emb.dim)
    search = SearchService(store, emb)
    kg = KGStore(s.neo4j_uri, s.neo4j_user, s.neo4j_password, s.kg_label, s.kg_relation)
    kg.load()
    reranker = build_reranker(s.rerank_model, s.rerank_model_path) if s.rerank_enabled else None
    llm = LLMClient(s.llm_base_url, s.llm_model, s.llm_api_key)
    cls = build_classifier()
    results = {}
    for hops in args.hops:
        rag = RagGraph(cls, search, llm, kg=kg, reranker=reranker, max_tokens=s.rag_max_tokens, top_k=s.rag_top_k, multihop_k=s.rag_multihop_k, kg_hops=hops, kg_iterative=hops > 1)
        rows = []
        for q, gold in QUESTIONS[: args.limit]:
            out = rag.invoke(q)
            ans = f"{out.get('answer_lead', '')} {out.get('answer_body', '')}"
            rows.append({"q": q, "ok": any(g in ans for g in gold), "graph_added": out.get("graph_added", 0), "lat": out.get("generate_ms", 0)})
        results[f"hops={hops}"] = {"n": len(rows), "correct": sum(1 for r in rows if r["ok"]) / len(rows), "avg_graph_added": round(statistics.mean(r["graph_added"] for r in rows), 2)}
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()