"""代码库检索 / 符号 / 概览 / 问答服务（W2）。

本模块只依赖 plan-code.md 冻结的 CodeStore / CodeIndexer 契约（用 typing.Protocol
描述），不 import W1 实现；测试注入内存 fake 即可运行。

范围说明：
- Python-only，检索走 SQLite FTS5（W1）；
- impact 仅文件级反向 import，启发式，不承诺精确调用链；
- 问答只使用检索到的代码上下文，引用经 verify_code_citations 校验。
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from ..rag.citations import parse_citations
from ..rag.graph import parse_structured_answer
from .ask_support import (
    build_extractive_fallback,
    extract_identifier_tokens,
    looks_like_refusal,
    merge_search_hits,
)

logger = logging.getLogger(__name__)

# ---- 可调常量（集中在此，避免散落魔法数） ----
DEFAULT_SEARCH_K = 8
DEFAULT_ASK_K = 6
MAX_SEARCH_K = 50
MAX_CONTEXT_CHARS = 12000
MAX_SNIPPET_CHARS = 1600
ASK_MAX_TOKENS = 1024
PROVIDER_UNKNOWN = ""


IMPACT_NOTE = "文件级反向 import（启发式，非精确调用链）"
REFUSAL_TITLE = "无法基于现有代码回答"
REFUSAL_SUGGESTION = "请补充更具体的符号名、文件路径或问题范围后重试。"

CODE_ANSWER_SYSTEM_PROMPT = (
    "你是代码库分析助手。只基于上下文中的代码回答，禁止使用上下文之外的知识或猜测。\n"
    "引用规则：每个结论后用 [n] 标注来源，n 必须对应上下文条目编号；不要编造编号。\n"
    "如果上下文不足以回答，直接说明“无法基于现有代码回答”，不要猜测。\n"
    "输出格式：第一行是一句话结论；空一行后给出 2-6 条展开说明。\n"
    "保留代码标识符与路径原文，不要改写。"
)

# 第一次输出拒答/无引用时的重试提示：不允许整体拒答，信息不完整也要给出可确定部分。
RETRY_SYSTEM_PROMPT = (
    CODE_ANSWER_SYSTEM_PROMPT
    + "\n\n补充要求：上下文包含相关代码，必须基于可见代码回答；"
    "信息不完整就说明能确定的部分，不要整体拒答。必须给出至少一个 [n] 引用。"
)


class CodeStoreProtocol(Protocol):
    """W1 CodeStore 的冻结契约（本模块只依赖这些方法）。"""

    def add_repo(self, root: str, name: str) -> dict[str, Any]:
        ...

    def list_repos(self) -> list[dict[str, Any]]:
        ...

    def get_repo(self, repo_id: str) -> dict[str, Any] | None:
        ...

    def delete_repo(self, repo_id: str) -> bool:
        ...

    def search(
        self,
        query: str,
        k: int = 8,
        repo_id: str | None = None,
        path_prefix: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def get_symbol(self, repo_id: str, qualname: str, path: str | None = None) -> dict[str, Any] | None:
        ...

    def reverse_imports(self, repo_id: str, path: str) -> list[str]:
        ...

    def overview(self, repo_id: str) -> dict[str, Any]:
        ...


class CodeIndexerProtocol(Protocol):
    """W1 CodeIndexer 的冻结契约。"""

    def index(self, root: str, name: str, reindex: bool = False) -> dict[str, Any]:
        ...


class CodeServiceError(RuntimeError):
    """代码服务相关错误的基类。"""


class CodeRepoNotFound(CodeServiceError):
    """repo_id 不存在。"""


class CodeValidationError(CodeServiceError, ValueError):
    """调用参数不合法。"""


class CodeUnavailableError(CodeServiceError):
    """依赖（如 LLM）未装配。"""


# ---------- 归一化工具 ----------

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


def _normalize_repo(raw: dict[str, Any] | None) -> dict[str, Any]:
    """把 W1 返回的 repo dict 归一化为 API 冻结字段。"""
    data = raw or {}
    return {
        "repo_id": _to_text(data.get("repo_id") or data.get("id")),
        "root": _to_text(data.get("root")),
        "name": _to_text(data.get("name")),
        "created_at": _to_text(data.get("created_at")),
        "updated_at": _to_text(data.get("updated_at")),
        "file_count": _to_int(data.get("file_count")),
        "symbol_count": _to_int(data.get("symbol_count")),
        "status": _to_text(data.get("status")) or "ready",
    }


def _normalize_hit(raw: dict[str, Any], repo_id: str | None = None) -> dict[str, Any]:
    """归一化检索命中；兼容 W1 的 chunk 字段与 code 字段。"""
    start_line = _to_int(raw.get("start_line"))
    end_line = _to_int(raw.get("end_line"), start_line)
    start_line = max(start_line, 1)
    end_line = max(end_line, start_line)
    return {
        "repo_id": _to_text(raw.get("repo_id") or repo_id),
        "chunk_id": _to_text(raw.get("chunk_id")),
        "path": _to_text(raw.get("path")),
        "qualname": _to_text(raw.get("qualname")),
        "kind": _to_text(raw.get("kind")),
        "start_line": start_line,
        "end_line": end_line,
        "score": _to_float(raw.get("score")),
        "text": _to_text(raw.get("text") or raw.get("code")),
    }


def _normalize_symbol(
    raw: dict[str, Any], repo_id: str, qualname: str, path: str | None = None
) -> dict[str, Any]:
    start_line = _to_int(raw.get("start_line"))
    end_line = _to_int(raw.get("end_line"), start_line)
    start_line = max(start_line, 1)
    end_line = max(end_line, start_line)
    return {
        "repo_id": _to_text(raw.get("repo_id") or repo_id),
        "path": _to_text(raw.get("path") or path),
        "qualname": _to_text(raw.get("qualname") or qualname),
        "kind": _to_text(raw.get("kind")),
        "start_line": start_line,
        "end_line": end_line,
        "docstring": _to_text(raw.get("docstring")),
        "code": _to_text(raw.get("code") or raw.get("text")),
    }


def _overview_metric(raw: dict[str, Any], keys: Sequence[str], default: int = 0) -> int:
    """overview 的计数键兼容：file_count / files / top_files 等。"""
    for key in keys:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, (list, tuple, set, dict)):
            return len(value)
        return _to_int(value, default)
    return default


def _top_paths(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    paths: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = _to_text(item.get("path") or item.get("name"))
        else:
            text = _to_text(item)
        if text:
            paths.append(text)
    return paths[:20]


def _kind_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _to_int(count) for key, count in value.items() if _to_int(count) >= 0}


def _normalize_overview(
    repo_id: str, repo: dict[str, Any], raw: dict[str, Any] | None
) -> dict[str, Any]:
    data = raw or {}
    return {
        "repo_id": repo_id,
        "name": _to_text(data.get("name") or repo.get("name")),
        "root": _to_text(data.get("root") or repo.get("root")),
        "file_count": _overview_metric(
            data, ("file_count", "files", "top_files"), _to_int(repo.get("file_count"))
        ),
        "symbol_count": _overview_metric(
            data, ("symbol_count", "symbols"), _to_int(repo.get("symbol_count"))
        ),
        "kinds": _kind_counts(data.get("kinds") or data.get("kind_counts")),
        "top_paths": _top_paths(data.get("top_paths") or data.get("top_files") or data.get("paths")),
        "status": _to_text(data.get("status") or repo.get("status")) or "ready",
        "note": _to_text(data.get("note")),
    }


# ---------- 引用校验 ----------

def _default_file_exists(path: str) -> bool:
    return Path(path).is_file()


def _default_read_lines(path: str) -> int:
    with Path(path).open("r", encoding="utf-8", errors="ignore") as fh:
        return sum(1 for _ in fh)


def _citation_numbers(answer: str | Sequence[int]) -> list[int]:
    if isinstance(answer, str):
        return parse_citations(answer)
    if isinstance(answer, Sequence) and not isinstance(answer, (bytes, bytearray)):
        numbers: list[int] = []
        for item in answer:
            number = _to_int(item, 0)
            if number and number not in numbers:
                numbers.append(number)
        return numbers
    return []


def verify_code_citations(
    answer: str | Sequence[int],
    contexts: Sequence[dict[str, Any]],
    *,
    file_exists: Callable[[str], bool] | None = None,
    read_lines: Callable[[str], int] | None = None,
) -> tuple[list[dict[str, Any]], list[int], str]:
    """校验回答中的 [n] 引用。

    params:
        answer: LLM 回答文本（解析 [n]）或已解析的引用编号序列；
        contexts: 每条含 path / abs_path / start_line / end_line / qualname；
        file_exists: 可注入（默认 Path.is_file），参数为解析后的绝对路径；
        read_lines: 可注入（默认读取文件计行数），参数同上。

    returns:
        (有效引用列表, 无效引用编号列表, 校验文案)

    引用有效的条件：编号在范围内、path 非空、start_line >= 1、end_line >= start_line、
    文件存在且 end_line 不超过文件总行数。无效引用不会出现在返回的 citations 中。
    """
    numbers = _citation_numbers(answer)
    exists = file_exists or _default_file_exists
    line_count = read_lines or _default_read_lines
    citations: list[dict[str, Any]] = []
    invalid: list[int] = []

    for number in numbers:
        if number < 1 or number > len(contexts):
            invalid.append(number)
            continue
        context = contexts[number - 1]
        path = _to_text(context.get("path"))
        abs_path = _to_text(context.get("abs_path")) or path
        start_line = _to_int(context.get("start_line"))
        end_line = _to_int(context.get("end_line"), start_line)
        if not path or not abs_path or start_line < 1 or end_line < start_line:
            invalid.append(number)
            continue
        try:
            if not exists(abs_path):
                invalid.append(number)
                continue
            total_lines = _to_int(line_count(abs_path))
        except Exception:
            logger.debug("citation file check failed: %s", abs_path, exc_info=True)
            invalid.append(number)
            continue
        if total_lines < end_line:
            invalid.append(number)
            continue
        citations.append(
            {
                "n": number,
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
                "qualname": _to_text(context.get("qualname")),
            }
        )

    if not numbers:
        check = "0/0 通过（无引用）"
    else:
        check = f"{len(citations)}/{len(numbers)} 通过"
    return citations, invalid, check


# ---------- 服务 ----------

class CodeService:
    """代码检索 / 符号 / 概览 / 问答的门面。"""

    def __init__(
        self,
        store: CodeStoreProtocol,
        llm: Any | None = None,
        provider_getter: Callable[[], str | None] | None = None,
        *,
        file_exists: Callable[[str], bool] | None = None,
        read_lines: Callable[[str], int] | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.provider_getter = provider_getter
        # 测试注入点：生产默认读真实文件系统。
        self.file_exists = file_exists
        self.read_lines = read_lines

    # ---- repos ----
    def list_repos(self) -> list[dict[str, Any]]:
        return [_normalize_repo(item) for item in (self.store.list_repos() or []) if isinstance(item, dict)]

    def get_repo(self, repo_id: str) -> dict[str, Any] | None:
        repo_id = _to_text(repo_id).strip()
        if not repo_id:
            return None
        raw = self.store.get_repo(repo_id)
        return _normalize_repo(raw) if raw else None

    def delete_repo(self, repo_id: str) -> bool:
        repo_id = _to_text(repo_id).strip()
        if not repo_id:
            return False
        return bool(self.store.delete_repo(repo_id))

    def _require_repo(self, repo_id: str) -> dict[str, Any]:
        repo = self.get_repo(repo_id)
        if repo is None:
            raise CodeRepoNotFound(f"repo not found: {repo_id}")
        return repo

    # ---- search / symbol / overview ----
    def search_code(
        self,
        query: str,
        k: int = DEFAULT_SEARCH_K,
        repo_id: str | None = None,
        path_prefix: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        query = _to_text(query).strip()
        if not query:
            raise CodeValidationError("query must not be empty")
        k = max(1, min(_to_int(k, DEFAULT_SEARCH_K), MAX_SEARCH_K))
        repo_id = _to_text(repo_id).strip() or None
        if repo_id is not None:
            self._require_repo(repo_id)
        path_prefix = _to_text(path_prefix).strip() or None
        kind = _to_text(kind).strip().lower() or None
        raw_hits = self.store.search(
            query,
            k=k,
            repo_id=repo_id,
            path_prefix=path_prefix,
            kind=kind,
        )
        return [_normalize_hit(item, repo_id) for item in (raw_hits or []) if isinstance(item, dict)]

    def _search_with_expansion(
        self, question: str, k: int, repo_id: str | None = None
    ) -> list[dict[str, Any]]:
        """原问题 + ASCII 标识符词元各检索一次，按 chunk_id 去重合并后取 top k。"""
        primary = self.search_code(question, k=k, repo_id=repo_id)
        token_query = " ".join(extract_identifier_tokens(question))
        if not token_query:
            return primary
        # 即使词元串与原文相同也按契约各检索一次（结果会按 chunk_id 去重）。
        expanded = self.search_code(token_query, k=k, repo_id=repo_id)
        return merge_search_hits([primary, expanded], k)

    def get_symbol(
        self, repo_id: str, qualname: str, path: str | None = None
    ) -> dict[str, Any] | None:
        repo_id = _to_text(repo_id).strip()
        qualname = _to_text(qualname).strip()
        path = _to_text(path).strip() or None
        if not repo_id or not qualname:
            raise CodeValidationError("repo_id and qualname must not be empty")
        self._require_repo(repo_id)
        raw = self.store.get_symbol(repo_id, qualname, path=path)
        if not raw:
            return None
        return _normalize_symbol(raw, repo_id, qualname, path)

    def overview(self, repo_id: str) -> dict[str, Any]:
        repo_id = _to_text(repo_id).strip()
        if not repo_id:
            raise CodeValidationError("repo_id must not be empty")
        repo = self._require_repo(repo_id)
        raw = self.store.overview(repo_id) or {}
        if not isinstance(raw, dict):
            raw = {}
        return _normalize_overview(repo_id, repo, raw)

    def reverse_imports(self, repo_id: str, path: str) -> list[str]:
        repo_id = _to_text(repo_id).strip()
        path = _to_text(path).strip()
        if not repo_id or not path:
            raise CodeValidationError("repo_id and path must not be empty")
        self._require_repo(repo_id)
        raw = self.store.reverse_imports(repo_id, path) or []
        return [_to_text(item) for item in raw if _to_text(item)]

    def impact(self, repo_id: str, path: str) -> dict[str, Any]:
        repo_id = _to_text(repo_id).strip()
        path = _to_text(path).strip()
        imports = self.reverse_imports(repo_id, path)
        return {
            "repo_id": repo_id,
            "path": path,
            "imports": imports,
            "heuristic": True,
            "note": IMPACT_NOTE,
        }

    # ---- ask ----
    def ask(
        self,
        question: str,
        repo_id: str | None = None,
        k: int = DEFAULT_ASK_K,
        *,
        llm: Any | None = None,
    ) -> dict[str, Any]:
        """基于代码检索上下文回答。

        流程：问题/标识符双查询合并 → 首次生成 → 拒答或无有效引用时用强化提示重试
        → 仍无有效引用则返回 top 2 上下文的确定性抽取式兜底；完全无命中才拒答。
        """
        question = _to_text(question).strip()
        if not question:
            raise CodeValidationError("question must not be empty")
        # 兼容 plan 冻结契约 ask(question, repo_id=None, llm=None) 的第三位参数写法。
        if llm is None and hasattr(k, "chat"):
            llm = k
            k = DEFAULT_ASK_K
        repo_id = _to_text(repo_id).strip() or None
        if repo_id is not None:
            self._require_repo(repo_id)
        llm = llm if llm is not None else self.llm
        if llm is None or not hasattr(llm, "chat"):
            raise CodeUnavailableError("code ask requires an LLM with chat()")

        hits = self._search_with_expansion(question, k, repo_id)
        provider = self._provider_name(llm)
        if not hits:
            return self._refusal("未检索到相关代码片段", provider)

        contexts = self._build_contexts(hits, repo_id)
        if not contexts:
            return self._refusal("检索结果缺少可用路径与行号", provider)

        messages = self._ask_messages(question, contexts)
        text, error = self._chat(llm, messages)
        provider = self._provider_name(llm)
        if error is not None:
            logger.warning("code ask llm failed: %s", error)
            return self._refusal(f"LLM 调用失败（{error}）", provider)

        answer = self._answer_from_text(text, contexts, provider, require_non_refusal=True)
        if answer is not None:
            return answer

        # 第一次拒答/无引用：用更强的提示重试一次；仍无引用则走确定性抽取式兜底。
        logger.info("code ask retry after refusal or missing citations")
        retry_text, retry_error = self._chat(llm, self._ask_messages(question, contexts, retry=True))
        provider = self._provider_name(llm)
        if retry_error is None:
            answer = self._answer_from_text(retry_text, contexts, provider, require_non_refusal=False)
            if answer is not None:
                return answer
        else:
            logger.warning("code ask retry failed: %s", retry_error)
        fallback = build_extractive_fallback(contexts, provider)
        if fallback is not None:
            return fallback
        return self._refusal("检索结果缺少可用路径与行号", provider)

    # ---- ask helpers ----
    def _ask_messages(
        self, question: str, contexts: Sequence[dict[str, Any]], *, retry: bool = False
    ) -> list[dict[str, str]]:
        system_prompt = RETRY_SYSTEM_PROMPT if retry else CODE_ANSWER_SYSTEM_PROMPT
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._build_prompt(question, contexts)},
        ]

    def _chat(self, llm: Any, messages: list[dict[str, str]]) -> tuple[str, str | None]:
        try:
            raw = llm.chat(messages, max_tokens=ASK_MAX_TOKENS)
        except Exception as exc:
            logger.debug("code ask chat error", exc_info=True)
            return "", type(exc).__name__
        text = _to_text(raw.get("text")) if isinstance(raw, dict) else _to_text(raw)
        return text, None

    def _answer_from_text(
        self,
        text: str,
        contexts: Sequence[dict[str, Any]],
        provider: str,
        *,
        require_non_refusal: bool,
    ) -> dict[str, Any] | None:
        if not text.strip():
            return None
        if require_non_refusal and looks_like_refusal(text):
            return None
        citations, invalid, check = verify_code_citations(
            text,
            contexts,
            file_exists=self.file_exists,
            read_lines=self.read_lines,
        )
        if invalid:
            logger.info("code ask dropped invalid citations: %s (%s)", invalid, check)
        if not citations:
            return None
        lead, body = parse_structured_answer(text)
        return {
            "answer": {"lead": lead or "已基于代码上下文生成回答", "body": body},
            "citations": citations,
            "provider": provider,
            "refusal": None,
        }

    def _provider_name(self, llm: Any) -> str:
        if self.provider_getter is not None:
            try:
                name = self.provider_getter()
            except TypeError:
                # 兼容 provider_getter(llm) 写法。
                try:
                    name = self.provider_getter(llm)  # type: ignore[call-arg]
                except Exception:  # noqa: BLE001
                    name = None
            except Exception:
                logger.debug("provider_getter failed", exc_info=True)
                name = None
            if name:
                return _to_text(name)
        for attr in ("last_provider", "active_provider", "provider", "provider_name", "model"):
            value = getattr(llm, attr, None)
            if isinstance(value, str) and value:
                return value
        return PROVIDER_UNKNOWN

    def _repo_root(self, repo_id: str) -> str:
        if not repo_id:
            return ""
        try:
            repo = self.get_repo(repo_id)
        except Exception:
            logger.debug("repo lookup failed: %s", repo_id, exc_info=True)
            return ""
        return _to_text((repo or {}).get("root"))

    def _resolve_path(self, path: str, root: str) -> str:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return str(candidate)
        if root:
            return str(Path(root).expanduser() / candidate)
        return str(candidate)

    def _build_contexts(
        self, hits: Sequence[dict[str, Any]], repo_id: str | None = None
    ) -> list[dict[str, Any]]:
        roots: dict[str, str] = {}
        contexts: list[dict[str, Any]] = []
        remaining = MAX_CONTEXT_CHARS
        for hit in hits:
            path = _to_text(hit.get("path")).strip()
            if not path:
                continue
            rid = _to_text(hit.get("repo_id") or repo_id)
            if rid not in roots:
                roots[rid] = self._repo_root(rid)
            start_line = max(1, _to_int(hit.get("start_line"), 1))
            end_line = max(start_line, _to_int(hit.get("end_line"), start_line))
            qualname = _to_text(hit.get("qualname"))
            kind = _to_text(hit.get("kind"))
            text = _to_text(hit.get("text"))[:MAX_SNIPPET_CHARS]
            number = len(contexts) + 1
            header = f"[{number}] {path}:{start_line}-{end_line} ({qualname or '-'}, {kind or '-'})"
            block = f"{header}\n{text}" if text else header
            if len(block) > remaining:
                if contexts:
                    break
                block = block[:remaining]
            contexts.append(
                {
                    "n": number,
                    "repo_id": rid,
                    "path": path,
                    "abs_path": self._resolve_path(path, roots[rid]),
                    "start_line": start_line,
                    "end_line": end_line,
                    "qualname": qualname,
                    "kind": kind,
                    "text": text,
                    "block": block,
                }
            )
            remaining -= len(block)
            if remaining <= 0:
                break
        return contexts

    def _build_prompt(self, question: str, contexts: Sequence[dict[str, Any]]) -> str:
        blocks = "\n\n".join(_to_text(context.get("block")) for context in contexts)
        return f"问题：{question}\n\n代码上下文：\n{blocks}"

    def _refusal(self, reason: str, provider: str) -> dict[str, Any]:
        return {
            "answer": {"lead": REFUSAL_TITLE, "body": f"{reason}。{REFUSAL_SUGGESTION}"},
            "citations": [],
            "provider": provider,
            "refusal": {
                "title": REFUSAL_TITLE,
                "reason": reason,
                "suggestion": REFUSAL_SUGGESTION,
            },
        }
