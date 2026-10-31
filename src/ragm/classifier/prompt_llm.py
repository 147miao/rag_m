"""gemma4 提示词分类器（Ollama，OpenAI/原生 API 均可）。"""
from __future__ import annotations

import json
import re
import time

import httpx

from .base import RouteDecision

RUBRIC = """你是 RAG 查询路由标注员。把问题分为三类，只输出 JSON：{"label":"simple|multihop|reasoning","confidence":0.0-1.0}
simple=单段落可答的事实/数值/定义；multihop=需串联两个及以上实体或文档；reasoning=需多源比较/计算/规划/给建议。拿不准按更贵的标。问题："""


class PromptLLMClassifier:
    name = "prompt-llm"

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "gemma4:31b-cloud") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def classify(self, question: str) -> RouteDecision:
        t0 = time.perf_counter()
        body = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 80},
            "messages": [{"role": "user", "content": RUBRIC + question}],
        }
        with httpx.Client(timeout=60) as client:
            resp = client.post(self.base_url + "/api/chat", json=body)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
        m = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        label = data.get("label")
        if label not in ("simple", "multihop", "reasoning"):
            label = "simple"
        conf = float(data.get("confidence", 0.6) or 0.6)
        return RouteDecision(
            label=label, confidence=min(max(conf, 0.0), 1.0), backend=self.name,
            latency_ms=(time.perf_counter() - t0) * 1000, reason="LLM 结构化输出",
        )

    def info(self) -> dict:
        return {"name": self.name, "ready": True, "model": self.model}
