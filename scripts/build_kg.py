"""构建实体图：HotpotQA 段落标题为实体，同一问题的 supporting_facts 标题之间建 CO_EVIDENCE 边。

输出：
  data/rag/kg_nodes.jsonl  {name, lang}
  data/rag/kg_edges.jsonl  {a, b, qid, qtype, question}
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

CACHE = Path("/run/media/myj/D/rag_m/data/classifier/_cache")
OUT = Path("/run/media/myj/D/rag_m/data/rag")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for shard in ("hotpot_train0.parquet", "hotpot_train1.parquet"):
        p = CACHE / shard
        if p.exists():
            rows.extend(pq.read_table(p).to_pylist())
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for r in rows:
        sf = r["supporting_facts"]
        titles = list(dict.fromkeys(sf["title"] if isinstance(sf, dict) else [x[0] for x in sf]))
        for t in titles:
            nodes.setdefault(t, {"name": t, "lang": "en"})
        if len(titles) < 2:
            continue
        q = " ".join(str(r.get("question") or "").split())[:300]
        qtype = r.get("type") or ""
        for i, a in enumerate(titles):
            for b in titles[i + 1 :]:
                for x, y in ((a, b), (b, a)):
                    key = (x, y, r["id"])
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    edges.append({"a": x, "b": y, "qid": r["id"], "qtype": qtype, "question": q})
    with (OUT / "kg_nodes.jsonl").open("w", encoding="utf-8") as fh:
        for n in nodes.values():
            fh.write(json.dumps(n, ensure_ascii=False) + "\n")
    with (OUT / "kg_edges.jsonl").open("w", encoding="utf-8") as fh:
        for e in edges:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    stats = {"nodes": len(nodes), "edges": len(edges), "questions": len(rows)}
    (OUT / "kg_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
