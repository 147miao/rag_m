"""用数据集自带的问题类型元数据重标 weak_label（v2）。

动机：按"数据集来源"打标会把 FRAMES（多跳事实）误标为 reasoning、把 GraphRAG-Bench 的
Fact Retrieval 误标为 reasoning（与 rubric 标注一致率仅 10%~20%）。这里改用元数据：

- squad/cmrc/drcd            -> simple
- hotpot/wiki2/musique       -> multihop
- frames                     -> multihop（多跳事实题，不是开放式推理）
- graphrag_bench             -> Fact Retrieval:simple / Complex Reasoning|Contextual Summarize|Creative:reasoning
- multihop_rag               -> comparison:multihop / inference|temporal:reasoning / null:丢弃
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

CACHE = Path("/run/media/myj/D/rag_m/data/classifier/_cache")
RAW = Path("/run/media/myj/D/rag_m/data/classifier/raw")
OUT = Path("/run/media/myj/D/rag_m/data/classifier/raw_v2")

GRAPH_MAP = {
    "Fact Retrieval": "simple",
    "Complex Reasoning": "reasoning",
    "Contextual Summarize": "reasoning",
    "Creative": "reasoning",
}
MHR_MAP = {
    "comparison_query": "multihop",
    "inference_query": "reasoning",
    "temporal_query": "reasoning",
}


def load_graph_types() -> dict[str, str]:
    types: dict[str, str] = {}
    for name in ("graphrag_medical.json", "graphrag_novel.json"):
        path = CACHE / name
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            types[row["question"].strip()] = row.get("question_type", "")
    return types


def load_mhr_types() -> dict[str, str]:
    path = CACHE / "multihoprag.json"
    if not path.exists():
        return {}
    return {row["query"].strip(): row.get("question_type", "") for row in json.loads(path.read_text(encoding="utf-8"))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    graph_types = load_graph_types()
    mhr_types = load_mhr_types()
    print(f"metadata: graphrag={len(graph_types)} multihop_rag={len(mhr_types)}")

    total = Counter()
    changed = Counter()
    dropped = Counter()
    for path in sorted(RAW.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        out = []
        for r in rows:
            src, q = r["source"], r["question"].strip()
            label = r["weak_label"]
            if src == "graphrag_bench":
                qt = graph_types.get(q, "")
                new = GRAPH_MAP.get(qt)
                if new:
                    if new != label:
                        changed[f"{src}:{qt}"] += 1
                    label = new
            elif src == "multihop_rag":
                qt = mhr_types.get(q, "")
                if qt == "null_query":
                    dropped[src] += 1
                    continue
                new = MHR_MAP.get(qt)
                if new:
                    if new != label:
                        changed[f"{src}:{qt}"] += 1
                    label = new
            elif src == "frames":
                label = "multihop"
                changed[src] += 1
            total[label] += 1
            out.append({**r, "weak_label": label, "question_type": graph_types.get(q) or mhr_types.get(q) or ""})
        (OUT / path.name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
        print(f"  {path.name:24} {len(rows):5} -> {len(out):5}")

    print("label distribution:", dict(total))
    print("changed:", dict(changed))
    print("dropped:", dict(dropped))
    print("total kept:", sum(total.values()))


if __name__ == "__main__":
    main()
