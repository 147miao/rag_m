"""构建评测 golden 集与冻结检索快照（一次性脚本，结果提交到 tests/fixtures）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragm.eval.metrics import summarize

DATA = Path("/run/media/myj/D/rag_m/data")
FIX = ROOT / "tests" / "fixtures"
API = "http://127.0.0.1:8000/api/search"

NARUTO = [
    ("漩涡鸣人的父亲是谁？", ["波风水门"], "波风水门"),
    ("九尾是什么？", ["九尾", "尾兽"], "尾兽"),
    ("宇智波鼬为什么灭族？", ["宇智波鼬"], "保护村子"),
    ("晓组织的目的是什么？", ["晓", "晓(火影忍者)"], "收集尾兽"),
    ("旗木卡卡西的写轮眼是从谁那里得到的？", ["旗木卡卡西", "宇智波带土"], "带土"),
    ("千手柱间和宇智波斑是什么关系？", ["千手柱间", "宇智波斑"], "共同建立木叶"),
    ("尾兽一共有几只？", ["尾兽"], "九只"),
    ("自来也有哪些弟子？", ["自来也(火影忍者)"], "水门、长门"),
    ("第四次忍界大战的起因是什么？", ["第四次忍界大战"], "月之眼"),
    ("大筒木辉夜是什么人？", ["大筒木辉夜"], "查克拉之祖"),
]
ARXIV = [
    ("What metrics does RAGAS use?", ["2309.15217"], "faithfulness"),
    ("What problem does RAG address for language models?", ["2005.11401"], "hallucination"),
    ("How does Self-RAG decide when to retrieve?", ["2310.11511"], "reflection"),
    ("What is the GraphRAG approach?", ["2404.16130"], "entity graph"),
    ("What is the main advantage of GraphRAG over baseline RAG?", ["2404.16130"], "global"),
]


def answers_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for shard in ("hotpot_train0.parquet", "hotpot_train1.parquet"):
        p = DATA / "classifier" / "_cache" / shard
        if p.exists():
            for r in pq.read_table(p, columns=["id", "answer"]).to_pylist():
                out.setdefault(r["id"], (r["answer"] or "").strip())
    return out


def main() -> None:
    FIX.mkdir(parents=True, exist_ok=True)
    answers = answers_map()
    golden: list[dict] = []
    seen: set[str] = set()
    for i, q in enumerate([json.loads(line) for line in (DATA / "retrieval" / "queries_en.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][:20]):
        if q["id"] in seen:
            continue
        seen.add(q["id"])
        golden.append({"id": f"hp-{i}", "question": q["question"], "gold": q["gold_ids"], "gold_answers": [answers.get(q["id"], "")], "suite": "hotpotqa", "lang": "en"})
    for i, (question, gold, answer) in enumerate(NARUTO):
        golden.append({"id": f"nr-{i}", "question": question, "gold": gold, "gold_answers": [answer], "suite": "naruto", "lang": "zh"})
    for i, (question, gold, answer) in enumerate(ARXIV):
        golden.append({"id": f"ax-{i}", "question": question, "gold": gold, "gold_answers": [answer], "suite": "arxiv", "lang": "en"})

    hits: list[dict] = []
    with httpx.Client(timeout=60, trust_env=False) as client:
        for row in golden:
            resp = client.post(API, json={"query": row["question"], "k": 10, "mode": "hybrid"})
            resp.raise_for_status()
            data = resp.json()
            hits.append({"id": row["id"], "ranked": [h["title"] for h in data.get("hits", [])]})

    (FIX / "golden.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in golden), encoding="utf-8")
    (FIX / "hits.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in hits), encoding="utf-8")

    hit_map = {h["id"]: h["ranked"] for h in hits}
    baseline = {"all": summarize([{"ranked": hit_map[g["id"]], "gold": g["gold"]} for g in golden])}
    for suite in ("hotpotqa", "naruto", "arxiv"):
        rows = [{"ranked": hit_map[g["id"]], "gold": g["gold"]} for g in golden if g["suite"] == suite]
        baseline[suite] = summarize(rows)
    (FIX / "eval_baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"golden": len(golden), "hits": len(hits), "baseline": baseline["all"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()