"""把火影检索/生成矩阵结果渲染成 markdown。"""
from __future__ import annotations

import json
from pathlib import Path

R = Path("reports")


def table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    data = json.loads((R / "naruto_retrieval_matrix.json").read_text(encoding="utf-8"))
    out = ["# 火影语料横向测试矩阵", "", "## 一、检索（28 题，gold=条目名）", ""]
    for axis, title in (("embedding", "Embedding 模型"), ("mode", "检索模式"), ("reranker", "Reranker"), ("chunk_size", "Chunk size")):
        rows = [r for r in data if r["axis"] == axis]
        if not rows:
            continue
        out += ["### " + title, "", table(rows, ["name", "recall@5", "recall@10", "ndcg@10", "mrr@10", "p50_ms", "chunks"]), ""]
    gen_files = sorted(R.glob("naruto_generation_*.json"))
    if gen_files:
        out += ["## 二、生成质量（28 题）", ""]
        rows = []
        for f in gen_files:
            s = json.loads(f.read_text(encoding="utf-8"))["summary"]
            rows.append({"config": s["config"], "correct": s["correct"], "faithfulness": s.get("faithfulness"),
                         "context_precision": s.get("context_precision"), "context_recall": s.get("context_recall"),
                         "citation_valid": s["citation_valid"], "p50_ms": s["p50_latency_ms"]})
        out += [table(rows, ["config", "correct", "faithfulness", "context_precision", "context_recall", "citation_valid", "p50_ms"]), ""]
    (R / "naruto_matrix.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()