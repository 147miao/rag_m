"""生成质量评测 v2：修正 v1 的两个口径问题。

v1 问题：
1. context 用的是 Evidence.snippet（260 字符截断）→ context_recall 被低估；
2. answer_relevancy 用 LLM 直接打分 → 全部 1.0，指标饱和。

v2：
1. 用 Evidence.full_text（与送进 LLM 的上下文一致，1200 字符）重算 faithfulness/precision/recall；
2. answer_relevancy 改为 ragas 原定义：由答案反推 3 个问题 → embedding 相似度均值（Ollama bge-m3）。
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pyarrow.parquet as pq

from ragm.config import get_settings
from ragm.rag.llm import LLMClient

RET = Path("/run/media/myj/D/rag_m/data/retrieval")
CACHE = Path("/run/media/myj/D/rag_m/data/classifier/_cache")
API = "http://127.0.0.1:8000/api/query"
EMBED_URL = "http://127.0.0.1:11434/v1/embeddings"
EMBED_MODEL = "bge-m3"


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
    for shard in ("hotpot_train0.parquet", "hotpot_train1.parquet"):
        p = CACHE / shard
        if p.exists():
            for r in pq.read_table(p, columns=["id", "answer"]).to_pylist():
                out.setdefault(r["id"], (r["answer"] or "").strip())
    return out


_llm: LLMClient | None = None


def llm() -> LLMClient:
    global _llm
    if _llm is None:
        s = get_settings()
        _llm = LLMClient(s.llm_base_url, s.llm_model, s.llm_api_key, 0.0)
    return _llm


def ask_json(prompt: str) -> dict:
    for _ in range(2):
        try:
            out = llm().chat([{"role": "user", "content": prompt}], max_tokens=900)
            raw = out["text"].strip()
            fence = chr(96) * 3
            raw = re.sub(r"^" + fence + r"[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?" + fence + r"$", "", raw).strip()
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return {}


def embed(texts: list[str]) -> np.ndarray:
    with httpx.Client(timeout=60) as client:
        resp = client.post(EMBED_URL, json={"model": EMBED_MODEL, "input": texts})
        resp.raise_for_status()
        vecs = [d["embedding"] for d in resp.json()["data"]]
    arr = np.array(vecs, dtype=np.float32)
    return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)


def judge_quality(question: str, answer: str, contexts: list[str]) -> dict:
    ctx = "\n".join(f"[{i}] {c[:800]}" for i, c in enumerate(contexts, start=1))
    prompt = f"""你是严格的 RAG 评测员。请评估下面这条问答。

【问题】{question}
【答案】{answer}
【检索上下文】
{ctx}

请完成三项判断，只输出 JSON：
1. claims：把答案拆成 1~6 个原子陈述，判断每个是否被上下文支持（supported: true/false）；
2. context_relevant：逐条判断上下文是否与问题相关（与编号一一对应的 0/1 数组）。
输出：{{"claims":[{{"text":"...","supported":true}}],"context_relevant":[1,0,1]}}"""
    out = ask_json(prompt)
    claims = out.get("claims") or []
    rel = out.get("context_relevant") or []
    return {
        "faithfulness": sum(1 for c in claims if c.get("supported")) / len(claims) if claims else 0.0,
        "context_precision": sum(rel) / len(rel) if rel else 0.0,
    }


def answer_relevancy(question: str, answer: str) -> float:
    """ragas 定义：由答案反推 3 个问题，与原问题 embedding 相似度均值。"""
    if not answer.strip():
        return 0.0
    out = ask_json(
        f"""根据下面这段答案，生成 3 个它能回答的问题（只输出 JSON：{{"questions":["...","...","..."]}}）。
【答案】{answer}"""
    )
    gen = [q for q in (out.get("questions") or []) if isinstance(q, str) and q.strip()][:3]
    if not gen:
        return 0.0
    vecs = embed([question] + gen)
    return float(np.mean([float(np.dot(vecs[0], vecs[i])) for i in range(1, len(vecs))]))


def query_api(question: str) -> dict:
    body = json.dumps({"question": question}).encode()
    last: Exception | None = None
    for _ in range(3):
        try:
            req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2)
    raise last  # type: ignore[misc]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def eval_one(item: dict) -> dict:
    q, ref = item["question"], item["reference"]
    t0 = time.perf_counter()
    resp = query_api(q)
    latency = (time.perf_counter() - t0) * 1000
    answer = (resp["answer"]["lead"] + " " + resp["answer"]["body"]).strip()
    contexts = [(e.get("full_text") or e.get("snippet") or "") for e in resp["evidence"][:8]]
    judge = judge_quality(q, answer, contexts) if answer else {"faithfulness": 0.0, "context_precision": 0.0}
    ar = answer_relevancy(q, answer) if answer else 0.0
    recall = 1.0 if ref and normalize(ref) in normalize(" ".join(contexts)) else 0.0
    check = resp["stats"]["citationCheck"]
    num, den = (check.split("通过")[0].strip().split("/") + ["0", "0"])[:2]
    return {
        "lang": item["lang"],
        "question": q,
        "route": resp["route"],
        "latency_ms": round(latency),
        "answer": answer[:300],
        "n_contexts": len(contexts),
        **judge,
        "answer_relevancy": round(ar, 4),
        "context_recall": recall,
        "citation_valid": 1.0 if int(den) > 0 and num == den else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="reports/generation_eval_v2.json")
    args = ap.parse_args()
    ans = answers_map()
    items = []
    for lang, name in (("en", "queries_en.jsonl"), ("zh", "queries_zh.jsonl")):
        picked = [q for q in load(name) if ans.get(q["id"])][: args.n // 2]
        items.extend({"lang": lang, "question": q["question"], "reference": ans[q["id"]]} for q in picked)
    print(f"evaluating {len(items)} questions (v2) ...", flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(eval_one, items))
    keys = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "citation_valid")
    agg = {}
    for lang in ("en", "zh", "all"):
        subset = rows if lang == "all" else [r for r in rows if r["lang"] == lang]
        agg[lang] = {"n": len(subset), **{k: round(sum(r[k] for r in subset) / len(subset), 4) for k in keys},
                     "p50_latency_ms": round(sorted(r["latency_ms"] for r in subset)[len(subset) // 2])}
    result = {"summary": agg, "rows": rows, "elapsed_s": round(time.perf_counter() - t0, 1)}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
