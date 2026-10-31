"""对比：800 字 chunk 直接检索 vs small-to-big（200 字 child → 父块上下文）。

指标：
- doc recall@k：返回的 parent/chunk 是否命中 gold 文章；
- answer coverage@k：返回的上下文是否包含 gold 答案原文（CMRC 有答案文本）；
- 平均上下文字符数 / 延迟。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from ragm.config import get_settings
from ragm.store.search_service import SearchService

RET = Path("/run/media/myj/D/rag_m/data/retrieval")
CACHE = Path("/run/media/myj/D/rag_m/data/classifier/_cache")


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (RET / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def answers_map() -> dict[str, str]:
    out: dict[str, str] = {}
    rows = pq.read_table(CACHE / "cmrc2018_train.parquet", columns=["id", "answers"]).to_pylist()
    for r in rows:
        try:
            a = json.loads(r["answers"].replace("'", '"')) if isinstance(r["answers"], str) else r["answers"]
            texts = a.get("text") or []
            out[r["id"]] = texts[0].strip() if texts else ""
        except Exception:  # noqa: BLE001
            out[r["id"]] = ""
    # HotpotQA 答案（EN 覆盖用）
    for shard in ("hotpot_train0.parquet", "hotpot_train1.parquet"):
        pth = CACHE / shard
        if not pth.exists():
            continue
        for r in pq.read_table(pth, columns=["id", "answer"]).to_pylist():
            out.setdefault(r["id"], (r["answer"] or "").strip())
    return out


def run(name: str, queries: list[dict], ans: dict[str, str], svc: SearchService, mode: str, k: int = 8) -> dict:
    recalls, coverages, chars, lat = [], [], [], []
    for q in queries:
        t0 = time.perf_counter()
        if mode == "small2big":
            res = svc.search_small2big(q["question"], k=k, k_children=50)
        else:
            res = svc.search(q["question"], k=k, mode="hybrid")
        lat.append((time.perf_counter() - t0) * 1000)
        docs = [h["doc_id"] for h in res["hits"]]
        texts = [h["text"] for h in res["hits"]]
        gold = set(q["gold_ids"])
        recalls.append(1.0 if gold & set(docs) else 0.0)
        a = ans.get(q["id"], "")
        coverages.append(1.0 if a and any(a in t for t in texts) else 0.0)
        chars.append(sum(len(t) for t in texts))
    n = len(queries)
    return {
        "variant": name,
        "n": n,
        "doc_hit@8": round(sum(recalls) / n, 4),
        "answer_coverage@8": round(sum(coverages) / n, 4),
        "avg_context_chars": round(sum(chars) / n, 1),
        "p50_ms": round(sorted(lat)[n // 2], 1),
    }


def main() -> None:
    s = get_settings()
    svc = SearchService(s)
    print(json.dumps(svc.info(), ensure_ascii=False, indent=2))
    ans = answers_map()
    results = []
    for lang, qfile in (("zh", "queries_zh.jsonl"), ("en", "queries_en.jsonl")):
        queries = load(qfile)
        if lang == "zh":
            results.append(run("baseline_800_hybrid", queries, ans, svc, "hybrid"))
            results.append(run("small2big_200_to_parent", queries, ans, svc, "small2big"))
        else:
            results.append(run("baseline_paragraph_hybrid", queries, ans, svc, "hybrid"))
            results.append(run("small2big_paragraph", queries, ans, svc, "small2big"))
        for r in results:
            r["lang"] = lang
    Path("reports/small2big_benchmark.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in results:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()