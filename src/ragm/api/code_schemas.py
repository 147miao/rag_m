"""代码库分析 API 契约（/api/code）。

字段与默认值在此冻结；W1 的 store/indexer 返回值先经 CodeService 归一化再构造响应，
避免 DB 字段变化直接泄漏到 HTTP 层。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CodeModel(BaseModel):
    """代码 API 基类：严格字段（拒绝未声明字段），便于契约测试。"""

    model_config = ConfigDict(extra="forbid")


# ---------- repos ----------

class RepoCreateRequest(CodeModel):
    root: str = Field(min_length=1, max_length=4096, description="本地目录绝对路径（需在 allowed_roots 内）")
    name: str = Field(min_length=1, max_length=200, description="仓库显示名")

    @field_validator("root", "name")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class RepoInfo(CodeModel):
    repo_id: str
    root: str = ""
    name: str = ""
    created_at: str = ""
    updated_at: str = ""
    file_count: int = Field(default=0, ge=0)
    symbol_count: int = Field(default=0, ge=0)
    status: str = "ready"


class RepoListResponse(CodeModel):
    repos: list[RepoInfo] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)


class RepoIndexResponse(CodeModel):
    """POST /repos 与 POST /repos/{id}/reindex 的统一响应。"""

    repo_id: str
    stats: dict[str, Any] = Field(default_factory=dict)


class DeleteRepoResponse(CodeModel):
    repo_id: str
    deleted: bool = True


# ---------- search / symbol ----------

class SearchHit(CodeModel):
    repo_id: str = ""
    chunk_id: str = ""
    path: str = ""
    qualname: str = ""
    kind: str = ""
    start_line: int = Field(default=0, ge=0)
    end_line: int = Field(default=0, ge=0)
    score: float = 0.0
    text: str = ""


class SymbolInfo(CodeModel):
    repo_id: str = ""
    path: str = ""
    qualname: str = ""
    kind: str = ""
    start_line: int = Field(default=0, ge=0)
    end_line: int = Field(default=0, ge=0)
    docstring: str = ""
    code: str = ""


# ---------- overview / impact ----------

class OverviewResponse(CodeModel):
    repo_id: str
    name: str = ""
    root: str = ""
    file_count: int = Field(default=0, ge=0)
    symbol_count: int = Field(default=0, ge=0)
    kinds: dict[str, int] = Field(default_factory=dict)
    top_paths: list[str] = Field(default_factory=list)
    status: str = "ready"
    note: str = ""


class ImpactResponse(CodeModel):
    repo_id: str
    path: str
    imports: list[str] = Field(default_factory=list)
    heuristic: bool = True
    note: str = ""


# ---------- ask ----------

class Citation(CodeModel):
    n: int = Field(ge=1)
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    qualname: str = ""


class Answer(CodeModel):
    lead: str = ""
    body: str = ""


class Refusal(CodeModel):
    title: str = ""
    reason: str = ""
    suggestion: str = ""


class AskRequest(CodeModel):
    question: str = Field(min_length=1, max_length=2000, description="代码相关问题")
    repo_id: str | None = Field(default=None, max_length=200, description="限定仓库；空 = 全部仓库")

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("repo_id")
    @classmethod
    def _blank_repo_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class AskResponse(CodeModel):
    answer: Answer = Field(default_factory=Answer)
    citations: list[Citation] = Field(default_factory=list)
    provider: str = ""
    refusal: Refusal | None = None
