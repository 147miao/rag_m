"""评测指标与门禁单测（不依赖 GPU/网络）。"""
from __future__ import annotations

from ragm.eval.dataset import load_baseline, load_golden, load_hits
from ragm.eval.metrics import gate, mrr_at_k, ndcg_at_k, recall_at_k, summarize


def test_retrieval_metrics() -> None:
    ranked = ["a", "b", "c", "a"]
    gold = ["b", "c"]
    assert recall_at_k(ranked, gold, 3) == 1.0
    assert recall_at_k(ranked, gold, 2) == 0.5
    assert ndcg_at_k(ranked, ["c"], 2) == 0.0
    assert ndcg_at_k(ranked, ["c"], 3) == 0.5
    assert mrr_at_k(ranked, gold, 10) == 0.5
    m = summarize([{"ranked": ranked, "gold": gold}])
    assert m["n"] == 1 and m["recall@10"] == 1.0


def test_gate_threshold() -> None:
    ok, detail = gate({"ndcg@10": 0.60}, {"ndcg@10": 0.58}, "ndcg@10", 0.02)
    assert ok and detail["delta"] == 0.02
    ok, _ = gate({"ndcg@10": 0.50}, {"ndcg@10": 0.58}, "ndcg@10", 0.02)
    assert not ok


def test_fixtures_load() -> None:
    golden = load_golden()
    hits = load_hits()
    baseline = load_baseline()
    assert len(golden) >= 30
    assert len(hits) == len(golden)
    assert "all" in baseline and baseline["all"]["ndcg@10"] > 0