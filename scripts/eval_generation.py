"""生成质量评测（ragas 风格指标，LLM-as-judge 实现）。

指标：
- faithfulness：答案中的每个 claim 是否能被检索上下文支持；
- answer_relevancy：答案是否直接回答了问题；
- context_precision：检索到的上下文与问题的相关性（按条平均）；
- context_recall：gold 答案是否出现在上下文中（客观匹配）；
- citation_valid：API 返回的引用校验是否全通过。

原因：ragas 0.4.3 在本环境 import 时依赖 langchain_community.chat_models.vertexai（langchain 1.x 已移除），
故用同等定义的 LLM-judge 实现，指标定义与 ragas 对齐，结果可比。
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from ragm.config import get_settings
from ragm.rag.llm import LLMClient

RET = Path("/run/media/myj/D/rag_m/data/retrieval")
CACHE = Path("/run/media/myj/D/rag_m/data/classifier/_cache")
API = "http://127.0.0.1:8000/api/query"


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


def ask_llm(question: str, answer: str, contexts: list[str], lang: str) -> dict:
    """一次调用同时判定 faithfulness / answer_relevancy / context_precision。"""
    ctx_lines = []
    for i, c in enumerate(contexts, start=1):
        ctx_lines.append(f"[{i}] {c[:600]}")
    prompt = f"""你是严格的 RAG 评测员。请评估下面这条问答。

【问题】{question}
【答案】{answer}
【检索上下文】
{chr(10).join(ctx_lines)}

请完成三项判断，只输出 JSON：
1. claims：把答案拆成 1~6 个原子陈述，判断每个陈述是否被上下文支持（supported: true/false）；
2. answer_relevancy：答案是否直接回答了问题（0~1，1 表示完全切题、无答非所问）；
3. context_relevant：逐条判断上下文是否与问题相关（与上下文编号一一对应的 0/1 数组）。

输出格式：{{"claims":[{{"text":"...","supported":true}}],"answer_relevancy":0.9,"context_relevant":[1,0,1]}}"""
    out = ask_json(prompt)
    claims = out.get("claims") or []
    supported = sum(1 for c in claims if c.get("supported"))
    return {
        "faithfulness": supported / len(claims) if claims else 0.0,
        "answer_relevancy": float(out.get("answer_relevancy") or 0.0),
        "context_precision": sum(out.get("context_relevant") or []) / max(1, len(out.get("context_relevant") or [])),
    }


_llm: LLMClient | None = None


def ask_json(prompt: str) -> dict:
    global _llm
    if _llm is None:
        s = get_settings()
        _llm = LLMClient(s.llm_base_url, s.llm_model, s.llm_api_key, 0.0)
    for _ in range(2):
        try:
            out = _llm.chat([{"role": "user", "content": prompt}], max_tokens=900)
            raw = out["text"].strip()
            fence = chr(96) * 3
            raw = re.sub(r"^" + fence + r"[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?" + fence + r"$", "", raw).strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return {}


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
    contexts = [e["snippet"] for e in resp["evidence"][:8]]
    judge = ask_llm(q, answer, contexts, item["lang"]) if answer else {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}
    recall = 1.0 if ref and normalize(ref) in normalize(" ".join(contexts)) else 0.0
    return {
        "lang": item["lang"],
        "question": q,
        "route": resp["route"],
        "latency_ms": round(latency),
        "answer": answer[:300],
        "n_contexts": len(contexts),
        **judge,
        "context_recall": recall,
        "citation_valid": 1.0 if resp["stats"]["citationCheck"].split("通过")[0].strip().split("/")[0] == resp["stats"]["citationCheck"].split("通过")[0].strip().split("/")[1] and int(resp["stats"]["citationCheck"].split("通过")[0].strip().split("/")[1]) > 0 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="reports/generation_eval.json")
    args = ap.parse_args()
    ans = answers_map()
    items = []
    for lang, name in (("en", "queries_en.jsonl"), ("zh", "queries_zh.jsonl")):
        qs = load(name)
        picked = [q for q in qs if ans.get(q["id"])][: args.n // 2]
        items.extend({"lang": lang, "question": q["question"], "reference": ans[q["id"]]} for q in picked)
    print(f"evaluating {len(items)} questions ...", flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(eval_one, items))
    agg = {}
    for lang in ("en", "zh", "all"):
        subset = rows if lang == "all" else [r for r in rows if r["lang"] == lang]
        agg[lang] = {
            "n": len(subset),
            **{k: round(sum(r[k] for r in subset) / len(subset), 4) for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "citation_valid")},
            "p50_latency_ms": round(sorted(r["latency_ms"] for r in subset)[len(subset) // 2]),
        }
    result = {"summary": agg, "rows": rows, "elapsed_s": round(time.perf_counter() - t0, 1)}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()