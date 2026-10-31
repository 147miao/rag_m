"""分类器抽象：路由决策的数据结构与协议。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

Label = Literal["simple", "multihop", "reasoning"]
Route = Literal["simple", "multihop", "agentic"]

LABEL_TO_ROUTE: dict[str, Route] = {"simple": "simple", "multihop": "multihop", "reasoning": "agentic"}
ROUTE_TO_LABEL: dict[str, Label] = {"simple": "simple", "multihop": "multihop", "agentic": "reasoning"}


@dataclass
class RouteDecision:
    label: Label
    confidence: float
    backend: str
    latency_ms: float
    reason: str = ""

    @property
    def route(self) -> Route:
        return LABEL_TO_ROUTE[self.label]

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "route": self.route,
            "confidence": round(self.confidence, 4),
            "backend": self.backend,
            "latency_ms": round(self.latency_ms, 1),
            "reason": self.reason,
        }


class Classifier(Protocol):
    name: str

    def classify(self, question: str) -> RouteDecision: ...

    def info(self) -> dict: ...
