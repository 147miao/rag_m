"""查询改写 / 子问题拆解（v1.1 W1）。

一次 LLM 调用同时完成两件事：
1. 指代消解：把「他 / 这个 / 继续」等替换成历史中明确的实体；
2. 子问题拆解：比较、因果、多跳类问题拆成 1-3 个可独立检索的子问题。

任何异常或非法 JSON 都回退到原文 + 空子问题，保证主链路可用。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MAX_SUBQUESTIONS = 3
MAX_REWRITTEN_CHARS = 2000
MAX_SUBQUESTION_CHARS = 200
MAX_HISTORY_CHARS = 4000
DEFAULT_MAX_TOKENS = 512

REWRITE_SYSTEM_PROMPT = (
    "你是多轮对话的查询改写器。给定对话历史与用户的新问题，请：\n"
    "1) 消解代词与省略（如「他」「这个」「继续」），把新问题改写成可独立检索的完整问题；\n"
    "2) 如果问题涉及比较、因果或多跳，把它拆成 1-3 个可独立检索的子问题；"
    "简单问题只给 1 个子问题或空数组；\n"
    "3) 只做改写，不要回答问题，也不要引入历史中没有依据的新事实。\n"
    '只输出 JSON：{"rewritten": "...", "subquestions": ["..."]}'
)

ROLE_NAMES = {"user": "用户", "assistant": "助手", "system": "系统"}


@dataclass(frozen=True)
class SessionContext:
    """API 层传给 RAG service 的会话上下文（只读）。"""

    session_id: str
    question: str
    rewritten_question: str
    subquestions: tuple[str, ...] = field(default_factory=tuple)


def _fallback(question: str) -> dict[str, Any]:
    return {"rewritten": question, "subquestions": []}


def _history_block(history: list[dict[str, Any]]) -> str:
    lines = []
    for message in history:
        role = ROLE_NAMES.get(str(message.get("role") or ""), str(message.get("role") or ""))
        content = " ".join(str(message.get("content") or "").split())
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)[-MAX_HISTORY_CHARS:]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出中提取 JSON 对象，兼容 ```json 围栏与前后噪音。"""
    raw = (text or "").strip()
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        raw = raw[first_newline + 1 :] if first_newline >= 0 else raw
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize(data: dict[str, Any]) -> dict[str, Any] | None:
    rewritten = data.get("rewritten")
    if not isinstance(rewritten, str) or not rewritten.strip():
        return None
    rewritten = rewritten.strip()[:MAX_REWRITTEN_CHARS]
    raw_subquestions = data.get("subquestions")
    subquestions: list[str] = []
    if isinstance(raw_subquestions, list):
        for item in raw_subquestions:
            if not isinstance(item, str):
                continue
            sub = " ".join(item.split()).strip()[:MAX_SUBQUESTION_CHARS]
            if not sub or sub in subquestions:
                continue
            subquestions.append(sub)
    return {"rewritten": rewritten, "subquestions": subquestions[:MAX_SUBQUESTIONS]}


class Rewriter:
    """基于任意 LLM 客户端（需实现 chat(messages, max_tokens, json_mode)）的改写器。"""

    def __init__(self, llm: Any, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def rewrite(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        if not history:
            return _fallback(question)
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"对话历史：\n{_history_block(history)}\n\n新的问题：{question}",
            },
        ]
        try:
            result = self.llm.chat(messages, max_tokens=self.max_tokens, json_mode=True)
            text = result.get("text") if isinstance(result, dict) else None
            data = _extract_json_object(str(text or ""))
            normalized = _normalize(data) if data is not None else None
            if normalized is not None:
                return normalized
            logger.warning("rewrite returned invalid JSON; falling back to original question")
        except Exception as exc:  # noqa: BLE001 - 改写失败不能拖垮主查询
            logger.warning("rewrite failed (%s); falling back to original question", type(exc).__name__)
        return _fallback(question)
