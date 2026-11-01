"""arXiv 论文语料问答测试（真实长文档）。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ragm.classifier.factory import build_classifier
from ragm.config import Settings
from ragm.embeddings.reranker import build_reranker
from ragm.rag.graph import RagGraph
from ragm.rag.llm import LLMClient
from ragm.store.search_service import SearchService

DATA = Path("/run/media/myj/D/rag_m/data/papers")
Q = [
    ("What two components does RAG combine?", ["retriev", "generat"]),
    ("What problem does RAG address for language models?", ["hallucin", "knowledge", "factual"]),
    ("What metrics does RAGAS use?", ["faithfulness", "answer relevance", "context relevance"]),
    ("How does Self-RAG decide when to retrieve?", ["retrieval token", "reflection", "self"]),
    ("What is the GraphRAG approach?", ["entity", "graph", "community", "summar"]),
    ("What does Corrective RAG do when retrieval is poor?", ["web", "search", "correct", "evaluat"]),
    ("What are the stages of a RAG system?", ["index", "retriev", "generat"]),
    ("What taxonomy does the RAG survey propose?", ["naive", "advanced", "modular"]),
    ("How does RAGAS evaluate without human references?", ["llm", "judge", "reference", "generat"]),
    ("What is the main advantage of GraphRAG over baseline RAG?", ["global", "summar", "entity", "community"]),
]


def main() -> None:
    base = Settings()
    settings = Settings(
        milvus_collection="rag_papers_chunks",
        milvus_child_collection="rag_papers_children",
        parents_path=DATA / "parents.jsonl",
        rerank_enabled=True,
        rerank_model_path=Path("/run/media/myj/D/rag_m/models/rerankers/bge-reranker-base"),
    )
    classifier = build_classifier()
    search = SearchService(settings)
    reranker = build_reranker("bge-reranker-base", str(settings.rerank_model_path))
    llm = LLMClient(base.llm_base_url, base.llm_model, base.llm_api_key, base.llm_temperature)
    rag = RagGraph(classifier, search, llm, kg=None, reranker=reranker)
    rows = []
    for q, gold in Q:
        t0 = time.perf_counter()
        st = rag.invoke(q)
        lat = round((time.perf_counter() - t0) * 1000)
        ans = (st.get("answer_lead", "") + " " + st.get("answer_body", "")).strip()
        ok = any(g in ans.lower() for g in gold)
        rows.append({"q": q, "ok": ok, "ans": ans[:200], "lat": lat, "cites": st.get("citations"), "check": st.get("citation_check"), "ev": [h.get("title", "") for h in (st.get("hits") or [])][:4]})
        print(json.dumps({"q": q[:50], "ok": ok, "lat": lat, "ans": ans[:80]}, ensure_ascii=False), flush=True)
    n = len(rows)
    summary = {"n": n, "correct": round(sum(1 for r in rows if r["ok"]) / n, 3), "p50_ms": sorted(r["lat"] for r in rows)[n // 2]}
    Path("reports/papers_test.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()