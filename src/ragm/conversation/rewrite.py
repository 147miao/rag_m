"""查询改写 / 子问题拆解（v1.1 W1）。

一次 LLM 调用同时完成两件事：
1. 指代消解：把「他 / 这个 / 继续」等替换成历史中明确的实体；
2. 子问题拆解：比较、因果、多跳类问题拆成 1-3 个可独立检索的子问题。

任何异常或非法 JSON 都回退到原文 + 空子问题，保证主链路可用。
"""
from __future__ import annotations

import json
import logging
import re
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
    "1) 消解代词与省略（如「他」「这个」「继续」），把新问题改写成可独立检索的完整问题；"
    "指代优先解析为上一轮**用户问题**中的主语/实体，而不是助手回答里的实体；\n"
    "2) 如果问题涉及比较、因果或多跳，把它拆成 1-3 个可独立检索的子问题；"
    "简单问题只给 1 个子问题或空数组；\n"
    "3) 只做改写，不要回答问题，也不要引入历史中没有依据的新事实。\n"
    '只输出 JSON：{"rewritten": "...", "subquestions": ["..."]}\n\n'
    "示例：\n"
    "历史：用户：鸣人的父亲是波风水门。助手：是的，鸣人的父亲是波风水门。\n"
    "新问题：他的师父是谁？\n"
    "正确输出："
    '{"rewritten": "鸣人的师父是谁？", "subquestions": ["鸣人的师父是谁？"]}\n'
    "注意：「他」指代上一轮用户问题的主语「鸣人」，不是助手回答里的「波风水门」。"
)

ROLE_NAMES = {"user": "用户", "assistant": "助手", "system": "系统"}
# 句首代词/指示词：改写结果可能丢失主语，需要确定性兜底。
PRONOUN_PREFIXES = ("他", "她", "它", "这个", "那个", "上述")
PRONOUN_ENTITIES = frozenset({"他", "她", "它", "这个", "那个", "上述", "其", "该"})
GENERIC_ENTITIES = frozenset({"谁", "什么", "怎么", "为什么", "哪些", "哪个", "如何", "多少", "是否", "请问"})
CLAUSE_SPLIT_RE = re.compile(r"[，。！？、；：,.!?;:]")
LEADING_NOISE_RE = re.compile(r"^(那么|所以|另外|请问|关于)")


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


def _last_user_question(history: list[dict[str, Any]]) -> str | None:
    for message in reversed(history):
        if str(message.get("role") or "") != "user":
            continue
        content = " ".join(str(message.get("content") or "").split()).strip()
        if content:
            return content
    return None


def _main_entity(question: str) -> str | None:
    """启发式抽取上一轮用户问题的主实体：取首个分句「的/是」前的主语。"""
    clause = CLAUSE_SPLIT_RE.split(question.strip(), maxsplit=1)[0].strip()
    if "的" in clause:
        candidate = clause.split("的", 1)[0]
    elif "是" in clause:
        candidate = clause.split("是", 1)[0]
    else:
        candidate = clause
    candidate = LEADING_NOISE_RE.sub("", candidate).strip(" \t“”\"'《》()（）")
    if len(candidate) < 2 or candidate in PRONOUN_ENTITIES or candidate in GENERIC_ENTITIES:
        return None
    return candidate


def _apply_pronoun_guard(
    question: str, data: dict[str, Any], history: list[dict[str, Any]]
) -> dict[str, Any]:
    """确定性兜底：句首代词未解析出上一轮用户问题主语时，把该主语拼到改写结果前。"""
    if not any(question.lstrip().startswith(prefix) for prefix in PRONOUN_PREFIXES):
        return data
    previous = _last_user_question(history)
    if not previous:
        return data
    entity = _main_entity(previous)
    rewritten = str(data.get("rewritten") or "")
    if not entity or entity in rewritten:
        return data
    logger.warning("rewrite lost pronoun subject %r; prepending it deterministically", entity)
    return {**data, "rewritten": f"{entity}，{rewritten}"}


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
                return _apply_pronoun_guard(question, normalized, history)
            logger.warning("rewrite returned invalid JSON; falling back to original question")
        except Exception as exc:  # noqa: BLE001 - 改写失败不能拖垮主查询
            logger.warning("rewrite failed (%s); falling back to original question", type(exc).__name__)
        return _fallback(question)
