"""Agentic v2 评测：15 题，统计准确率/延迟/工具次数/预算超限。"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8000"

CASES = [
    ("鸣人父亲是谁？", ["波风水门"]),
    ("自来也的弟子中谁成为了四代火影？", ["水门"]),
    ("Which US documentary was made first: Scared Straight! or Soldier Girls?", ["Scared Straight"]),
    ("What metrics does RAGAS use?", ["faithfulness"]),
    ("宇智波鼬为什么灭族？", ["保护", "村子"]),
    ("佩恩的本体是谁？", ["长门"]),
    ("What is the GraphRAG approach?", ["graph"]),
    ("谁教鸣人螺旋丸？", ["自来也"]),
    ("大蛇丸的弟子有哪些？", ["兜"]),
    ("What problem does RAG address?", ["hallucination"]),
    ("千手柱间的弟弟是谁？", ["扉间"]),
    ("What is the main advantage of GraphRAG?", ["global"]),
    ("九尾人柱力中谁成为了七代火影？", ["鸣人"]),
    ("How does Self-RAG decide when to retrieve?", ["reflection"]),
    ("谁和宇智波斑一起建立了木叶？", ["柱间"]),
]


def main() -> None:
    rows = []
    with httpx.Client(timeout=240, trust_env=False) as client:
        for q, gold in CASES:
            r = client.post(f"{API}/api/query", json={"question": q, "route_override": "agentic"})
            r.raise_for_status()
            d = r.json()
            ans = f"{d['answer']['lead']} {d['answer']['body']}"
            tools = d.get("tools") or []
            rows.append({"q": q, "ok": any(g in ans for g in gold), "lat": d["stats"]["latencyMs"], "tools": len(tools), "over_budget": len(tools) > 4})
    n = len(rows)
    print(json.dumps({
        "n": n,
        "accuracy": round(sum(1 for r in rows if r["ok"]) / n, 3),
        "p50_latency_ms": int(statistics.median(r["lat"] for r in rows)),
        "avg_tools": round(statistics.mean(r["tools"] for r in rows), 2),
        "over_budget": sum(1 for r in rows if r["over_budget"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()