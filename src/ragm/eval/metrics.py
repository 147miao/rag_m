"""检索/生成评测指标（确定性、无外部依赖）。"""
from __future__ import annotations

import math


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    gs = set(gold)
    if not gs:
        return 0.0
    top = _dedup(ranked)[:k]
    return sum(1 for x in top if x in gs) / len(gs)


def ndcg_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    gs = set(gold)
    if not gs:
        return 0.0
    top = _dedup(ranked)[:k]
    dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(top) if x in gs)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gs), k)))
    return dcg / idcg if idcg else 0.0


def mrr_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    gs = set(gold)
    for i, x in enumerate(_dedup(ranked)[:k]):
        if x in gs:
            return 1 / (i + 1)
    return 0.0


def summarize(rows: list[dict], k: int = 10) -> dict:
    """rows: [{ranked, gold}] → {recall@k, ndcg@k, mrr@k, n}。"""
    n = len(rows)
    if n == 0:
        return {"n": 0, f"recall@{k}": 0.0, f"ndcg@{k}": 0.0, f"mrr@{k}": 0.0}
    return {
        "n": n,
        f"recall@{k}": round(sum(recall_at_k(r["ranked"], r["gold"], k) for r in rows) / n, 4),
        f"ndcg@{k}": round(sum(ndcg_at_k(r["ranked"], r["gold"], k) for r in rows) / n, 4),
        f"mrr@{k}": round(sum(mrr_at_k(r["ranked"], r["gold"], k) for r in rows) / n, 4),
    }


def gate(current: dict, baseline: dict, metric: str = "ndcg@10", threshold: float = 0.02) -> tuple[bool, dict]:
    """当前指标相对基线下降超过阈值则失败。"""
    cur = float(current.get(metric, 0.0))
    base = float(baseline.get(metric, 0.0))
    delta = round(cur - base, 4)
    return delta >= -abs(threshold), {"metric": metric, "current": cur, "baseline": base, "delta": delta, "threshold": threshold}