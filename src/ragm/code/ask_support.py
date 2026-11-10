"""代码问答的查询扩展 / 合并 / 抽取式兜底纯函数。

从 service.py 拆出，避免单文件超过 700 行；这些函数不依赖 CodeService，
输入都是 service 已归一化的 chunk/context dict。
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

MAX_QUERY_TOKENS = 8
RRF_K = 60
MAX_FALLBACK_CONTEXTS = 2
DEFAULT_SEARCH_K = 8
MAX_SEARCH_K = 50
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
REFUSAL_MARKERS = (
    "无法基于现有代码回答",
    "无法回答",
    "无法提供",
    "cannot answer",
    "unable to answer",
    "not enough information",
    "insufficient information",
)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def extract_identifier_tokens(text: str, limit: int = MAX_QUERY_TOKENS) -> list[str]:
    """提取问题中的 ASCII 标识符词元（至少 3 字符），保持出现顺序并去重。

    中文词不参与 FTS（unicode61 分词效果差），因此只把 FallbackLLM / build_llm /
    hybrid_search 这类标识符拼成补充查询。
    """
    tokens: list[str] = []
    for match in IDENTIFIER_RE.finditer(text or ""):
        token = match.group(0)
        if token not in tokens:
            tokens.append(token)
        if len(tokens) >= max(1, limit):
            break
    return tokens


def hit_key(hit: dict[str, Any]) -> str:
    chunk_id = _to_text(hit.get("chunk_id")).strip()
    if chunk_id:
        return chunk_id
    return "|".join(
        (
            _to_text(hit.get("repo_id")),
            _to_text(hit.get("path")),
            str(_to_int(hit.get("start_line"))),
            str(_to_int(hit.get("end_line"))),
            _to_text(hit.get("qualname")),
        )
    )


def merge_search_hits(
    result_sets: Sequence[Sequence[dict[str, Any]]], k: int
) -> list[dict[str, Any]]:
    """按 chunk_id 去重合并多次检索结果，用 RRF 排名后取 top k。"""
    limit = max(1, min(_to_int(k, DEFAULT_SEARCH_K), MAX_SEARCH_K))
    merged: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, int] = {}
    for hits in result_sets:
        for rank, hit in enumerate(hits or []):
            if not isinstance(hit, dict):
                continue
            key = hit_key(hit)
            item = merged.get(key)
            if item is None:
                item = {**hit, "_rrf": 0.0}
                merged[key] = item
                first_seen[key] = len(first_seen)
            elif _to_float(hit.get("score")) > _to_float(item.get("score")):
                item["score"] = hit.get("score")
            item["_rrf"] = _to_float(item.get("_rrf")) + 1.0 / (RRF_K + rank + 1)
    ranked = sorted(
        merged.values(),
        key=lambda item: (-_to_float(item.get("_rrf")), first_seen.get(hit_key(item), 0)),
    )
    return [{key: value for key, value in item.items() if not key.startswith("_")} for item in ranked[:limit]]


def looks_like_refusal(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker.lower() in lowered for marker in REFUSAL_MARKERS)


def first_code_line(text: Any) -> str:
    for line in _to_text(text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return ""


def build_extractive_fallback(
    contexts: Sequence[dict[str, Any]], provider: str
) -> dict[str, Any] | None:
    """按 top 2 上下文生成确定性答案；无可用 context 时返回 None。"""
    citations: list[dict[str, Any]] = []
    lines: list[str] = []
    for context in contexts[:MAX_FALLBACK_CONTEXTS]:
        number = _to_int(context.get("n"))
        path = _to_text(context.get("path"))
        if number < 1 or not path:
            continue
        start_line = max(1, _to_int(context.get("start_line"), 1))
        end_line = max(start_line, _to_int(context.get("end_line"), start_line))
        qualname = _to_text(context.get("qualname"))
        first_line = first_code_line(context.get("text"))
        location = f"{path}:{start_line}-{end_line}"
        if qualname:
            location = f"{location}（{qualname}）"
        detail = f"{location}：{first_line}" if first_line else location
        lines.append(f"- {detail} [{number}]")
        citations.append(
            {
                "n": number,
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
                "qualname": qualname,
            }
        )
    if not citations:
        return None
    return {
        "answer": {
            "lead": f"已定位到 {len(citations)} 处相关代码（抽取式摘要）",
            "body": "LLM 未给出可验证引用，以下为确定性抽取的关键位置：\n" + "\n".join(lines),
        },
        "citations": citations,
        "provider": provider,
        "refusal": None,
    }
