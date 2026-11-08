"""评测执行：检索（快照/在线）与生成（调用 API）。"""
from __future__ import annotations

import statistics
import time

import httpx

from .metrics import summarize


def retrieval_by_suite(golden: list[dict], hit_map: dict[str, list[str]], k: int = 10) -> dict:
    out: dict[str, dict] = {}
    for suite in ("hotpotqa", "naruto", "arxiv"):
        rows = [{"ranked": hit_map.get(g["id"], []), "gold": g["gold"]} for g in golden if g["suite"] == suite]
        if rows:
            out[suite] = summarize(rows, k=k)
    out["all"] = summarize([{"ranked": hit_map.get(g["id"], []), "gold": g["gold"]} for g in golden], k=k)
    return out


def live_hits(golden: list[dict], api: str, k: int = 10) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with httpx.Client(timeout=60, trust_env=False) as client:
        for g in golden:
            resp = client.post(f"{api}/api/search", json={"query": g["question"], "k": k, "mode": "hybrid"})
            resp.raise_for_status()
            out[g["id"]] = [h["title"] for h in resp.json().get("hits", [])]
    return out


def generation_eval(golden: list[dict], api: str) -> dict:
    rows = []
    with httpx.Client(timeout=180, trust_env=False) as client:
        for g in golden:
            t0 = time.perf_counter()
            try:
                resp = client.post(f"{api}/api/query", json={"question": g["question"]})
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                rows.append({"id": g["id"], "ok": False, "error": f"{type(exc).__name__}: {exc}", "latency_ms": 0})
                continue
            latency = round((time.perf_counter() - t0) * 1000)
            answer = (data["answer"]["lead"] + " " + data["answer"]["body"]).strip()
            golds = [x for x in g.get("gold_answers", []) if x]
            hit = any(x.lower() in answer.lower() for x in golds) if golds else False
            check = data["stats"]["citationCheck"]
            num, den = (check.split("通过")[0].strip().split("/") + ["0", "0"])[:2]
            rows.append({
                "id": g["id"], "ok": hit, "latency_ms": latency,
                "citation_valid": int(den) > 0 and num == den,
                "refusal": data.get("refusal") is not None,
            })
    n = len(rows)
    return {
        "n": n,
        "answer_correct": round(sum(1 for r in rows if r["ok"]) / n, 4) if n else 0.0,
        "citation_valid": round(sum(1 for r in rows if r.get("citation_valid")) / n, 4) if n else 0.0,
        "refusal_rate": round(sum(1 for r in rows if r.get("refusal")) / n, 4) if n else 0.0,
        "p50_latency_ms": int(statistics.median([r["latency_ms"] for r in rows])) if n else 0,
        "rows": rows,
    }


def render_markdown(retrieval: dict | None, generation: dict | None) -> str:
    lines = ["# 评测报告", ""]
    if retrieval:
        lines += ["## 检索（k=10）", "", "| suite | n | recall@10 | ndcg@10 | mrr@10 |", "|---|---|---|---|---|"]
        for name, m in retrieval.items():
            lines.append(f"| {name} | {m['n']} | {m['recall@10']} | {m['ndcg@10']} | {m['mrr@10']} |")
        lines.append("")
    if generation:
        lines += ["## 生成", "", f"- 题数：{generation['n']}",
                  f"- 答案命中：{generation['answer_correct']}",
                  f"- 引用校验通过率：{generation['citation_valid']}",
                  f"- 拒答率：{generation['refusal_rate']}",
                  f"- p50 延迟：{generation['p50_latency_ms']} ms", ""]
    return "\n".join(lines) + "\n"