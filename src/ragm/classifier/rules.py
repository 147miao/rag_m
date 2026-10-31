"""规则基线：关键词 + 问句模式。无依赖，作为兜底与对照。"""
from __future__ import annotations

import time

from .base import Label, RouteDecision

REASONING_KW = (
    "对比", "比较", "为什么", "分析", "权衡", "建议", "优劣", "估算", "步骤", "影响", "原因",
    "推理", "方案", "选型", "区别", "怎么变化", "是否值得", "取舍",
)
MULTIHOP_KW = (
    "哪个机构", "哪个城市", "之后", "还发布", "的作者的", "的创始人", "关系", "谁的学生",
    "哪位", "以及", "并且", "分别", "还", "创办", "收购",
)


class RulesClassifier:
    name = "rules"

    def classify(self, question: str) -> RouteDecision:
        t0 = time.perf_counter()
        q = question.lower()
        if any(k in q for k in REASONING_KW):
            label: Label = "reasoning"
            conf = 0.7
            reason = "命中推理关键词"
        elif any(k in q for k in MULTIHOP_KW):
            label = "multihop"
            conf = 0.65
            reason = "命中多跳关系词"
        else:
            label = "simple"
            conf = 0.6
            reason = "默认单跳"
        return RouteDecision(label=label, confidence=conf, backend=self.name, latency_ms=(time.perf_counter() - t0) * 1000, reason=reason)

    def info(self) -> dict:
        return {"name": self.name, "ready": True}
