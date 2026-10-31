"""按配置构建分类器；失败时降级到规则。"""
from __future__ import annotations

import logging

from ..config import get_settings
from .base import Classifier, RouteDecision
from .rules import RulesClassifier

logger = logging.getLogger("ragm.classifier")


class FallbackClassifier:
    """包装器：主分类器异常时降级到规则。"""

    def __init__(self, primary: Classifier | None, fallback: Classifier | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or RulesClassifier()
        self.name = getattr(primary, "name", "rules")

    def classify(self, question: str) -> RouteDecision:
        if self.primary is not None:
            try:
                return self.primary.classify(question)
            except Exception:  # noqa: BLE001
                logger.warning("primary classifier failed, fallback to rules", exc_info=True)
        return self.fallback.classify(question)

    def info(self) -> dict:
        info = {"backend": self.name}
        if self.primary is not None:
            info["primary"] = self.primary.info()
        info["fallback"] = self.fallback.info()
        return info


def build_classifier() -> FallbackClassifier:
    settings = get_settings()
    backend = (settings.classifier_backend or "rules").lower()
    primary: Classifier | None = None
    try:
        if backend == "lora" and settings.classifier_run_dir:
            from .lora import LoRAClassifier

            primary = LoRAClassifier(settings.classifier_run_dir)
        elif backend == "prompt":
            from .prompt_llm import PromptLLMClassifier

            primary = PromptLLMClassifier(base_url=settings.llm_base_url.removesuffix("/v1"), model=settings.llm_model)
        elif backend == "rules":
            primary = RulesClassifier()
        else:
            logger.warning("unknown classifier backend %s, using rules", backend)
            primary = RulesClassifier()
    except Exception:  # noqa: BLE001
        logger.exception("failed to build classifier %s, fallback to rules", backend)
        primary = None
    return FallbackClassifier(primary)
