"""路由分类复用测试：避免 /api/query 与 service/graph 重复分类。"""
from __future__ import annotations

from ragm.classifier.base import RouteDecision
from ragm.rag.graph import RagGraph


class CountingClassifier:
    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision
        self.calls = 0

    def classify(self, question: str) -> RouteDecision:
        self.calls += 1
        return self.decision


def _graph(decision: RouteDecision) -> tuple[RagGraph, CountingClassifier]:
    classifier = CountingClassifier(decision)
    return RagGraph(classifier, search=None, llm=None), classifier


def test_classify_node_reuses_precomputed_decision() -> None:
    decision = RouteDecision(label="multihop", confidence=0.91, backend="test", latency_ms=7.0, reason="test")
    graph, classifier = _graph(decision)
    state = graph._classify_node({"question": "q", "route_override": "auto", "classifier_decision": decision})
    assert state["route"] == "multihop"
    assert state["classifier_backend"] == "test"
    assert state["classify_ms"] == 7.0
    assert classifier.calls == 0


def test_classify_node_override_skips_classifier() -> None:
    decision = RouteDecision(label="simple", confidence=0.9, backend="test", latency_ms=1.0)
    graph, classifier = _graph(decision)
    state = graph._classify_node({"question": "q", "route_override": "agentic"})
    assert state["route"] == "agentic"
    assert state["label"] == "reasoning"
    assert state["classifier_backend"] == "override"
    assert classifier.calls == 0
