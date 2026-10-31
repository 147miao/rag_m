"""构建 embedding 检索评测集（中英各一套）。

EN: HotpotQA 段落池（约 3.5~4 万段）+ 300 条 query，gold = supporting_facts 标题。
ZH: CMRC2018 文章池 + 100 条 query，gold = 所属文章。

输出：/run/media/myj/D/rag_m/data/retrieval/
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

CACHE = Path("/run/media/myj/D/rag_m/data/classifier/_cache")
OUT = Path("/run/media/myj/D/rag_m/data/retrieval")
SEED = 42
POOL_N = 4000
QUERY_N = 300
ZH_QUERY_N = 100


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def build_en() -> dict:
    rows = []
    for shard in ("hotpot_train0.parquet", "hotpot_train1.parquet"):
        p = CACHE / shard
        if p.exists():
            rows.extend(pq.read_table(p).to_pylist())
    rng = random.Random(SEED)
    pool = rng.sample(rows, min(POOL_N, len(rows)))

    corpus: dict[str, str] = {}
    for r in pool:
        for title, sentences in zip(r["context"]["title"], r["context"]["sentences"]):
            corpus.setdefault(title, " ".join(sentences).strip())

    queries = rng.sample(pool, min(QUERY_N, len(pool)))
    qrows = []
    for r in queries:
        sf = r["supporting_facts"]
        titles = sf["title"] if isinstance(sf, dict) else [x[0] for x in sf]
        gold = [t for t in dict.fromkeys(titles) if t in corpus]
        if not gold:
            continue
        qrows.append({"id": r["id"], "question": r["question"], "gold_ids": gold, "type": r.get("type")})

    write_jsonl(OUT / "corpus_en.jsonl", [{"id": k, "text": v} for k, v in corpus.items()])
    write_jsonl(OUT / "queries_en.jsonl", qrows)
    return {"corpus": len(corpus), "queries": len(qrows), "avg_gold": round(sum(len(q["gold_ids"]) for q in qrows) / len(qrows), 2)}


def build_zh() -> dict:
    p = CACHE / "cmrc2018_train.parquet"
    rows = pq.read_table(p).to_pylist()
    rng = random.Random(SEED)
    corpus: dict[str, str] = {}
    for r in rows:
        cid = "zh_" + hashlib.sha1(r["context"].encode("utf-8")).hexdigest()[:12]
        corpus.setdefault(cid, r["context"].strip())
    qrows = []
    for r in rng.sample(rows, min(ZH_QUERY_N, len(rows))):
        cid = "zh_" + hashlib.sha1(r["context"].encode("utf-8")).hexdigest()[:12]
        qrows.append({"id": r["id"], "question": r["question"], "gold_ids": [cid], "type": "fact"})
    write_jsonl(OUT / "corpus_zh.jsonl", [{"id": k, "text": v} for k, v in corpus.items()])
    write_jsonl(OUT / "queries_zh.jsonl", qrows)
    return {"corpus": len(corpus), "queries": len(qrows)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"en": build_en(), "zh": build_zh()}
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()