"""代码库分析 HTTP 路由（/api/code）。

装配点（主 agent 合并时接线）：
- app.state.code_service = CodeService(store, llm=...)
- app.state.code_indexer = CodeIndexer(db_path, allowed_roots)
- app.state.code_allowed_roots 可选，覆盖 settings.code_allowed_roots（测试注入用）
- main.py include_router(code_router)

未装配时所有端点返回 503；根目录越权 403、不存在 404、参数非法 400/422。
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from ..code.service import (
    CodeRepoNotFound,
    CodeService,
    CodeServiceError,
    CodeUnavailableError,
    CodeValidationError,
)
from ..config import get_settings
from .code_schemas import (
    Answer,
    AskRequest,
    AskResponse,
    Citation,
    DeleteRepoResponse,
    ImpactResponse,
    OverviewResponse,
    Refusal,
    RepoCreateRequest,
    RepoIndexResponse,
    RepoInfo,
    RepoListResponse,
    SearchHit,
    SymbolInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/code", tags=["code"])

SERVICE_NOT_READY = "code service not ready"
INDEXER_NOT_READY = "code indexer not ready"


# ---------- dependencies ----------

def _service(request: Request) -> CodeService:
    service = getattr(request.app.state, "code_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail=SERVICE_NOT_READY)
    return service


def _indexer(request: Request) -> Any:
    indexer = getattr(request.app.state, "code_indexer", None)
    if indexer is None:
        raise HTTPException(status_code=503, detail=INDEXER_NOT_READY)
    return indexer


def _allowed_roots(request: Request) -> list[Path]:
    override = getattr(request.app.state, "code_allowed_roots", None)
    if override is not None:
        raw: Any = override
    else:
        settings = get_settings()
        raw = getattr(settings, "code_allowed_roots", None)
        if raw is None:
            raw = [str(Path.home())]
    if isinstance(raw, (str, Path)):
        raw = [raw]
    roots: list[Path] = []
    for item in raw or []:
        text = str(item).strip()
        if not text:
            continue
        try:
            roots.append(Path(text).expanduser().resolve())
        except OSError:
            continue
    # 显式空列表表示禁止所有根目录；只有 settings 缺失字段时才回退 home。
    return roots


def _validate_root(raw_root: str, request: Request) -> Path:
    """resolve 后必须在 allowed_roots 内（消除符号链接逃逸），且必须是已存在目录。"""
    try:
        resolved = Path(raw_root).expanduser().resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"invalid repo root: {raw_root}") from exc
    allowed = _allowed_roots(request)
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed):
        raise HTTPException(status_code=403, detail=f"repo root not allowed: {resolved}")
    if not resolved.is_dir():
        raise HTTPException(status_code=404, detail=f"repo root not found: {resolved}")
    return resolved


# ---------- error mapping ----------

def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except CodeRepoNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CodeValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CodeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except CodeServiceError as exc:
        logger.exception("code service error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _run_index(indexer: Any, root: Path, name: str, reindex: bool) -> dict[str, Any]:
    try:
        result = indexer.index(str(root), name, reindex=reindex)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"repo root not found: {root}") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"repo root not readable: {root}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"index failed: {exc}") from exc
    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="indexer returned invalid result")

    repo_id = str(result.get("repo_id") or result.get("id") or "")
    if not repo_id:
        raise HTTPException(status_code=500, detail="indexer returned no repo_id")
    stats = result.get("stats")
    if not isinstance(stats, dict):
        stats = {
            key: value
            for key, value in result.items()
            if key not in {"repo_id", "id", "stats", "repo"}
        }
    return {"repo_id": repo_id, "stats": stats}


def _repo_info(raw: dict[str, Any]) -> RepoInfo:
    return RepoInfo(**raw)


# ---------- repos ----------

@router.post("/repos", response_model=RepoIndexResponse)
def create_repo(payload: RepoCreateRequest, request: Request) -> RepoIndexResponse:
    indexer = _indexer(request)
    root = _validate_root(payload.root, request)
    out = _run_index(indexer, root, payload.name, reindex=False)
    return RepoIndexResponse(**out)


@router.get("/repos", response_model=RepoListResponse)
def list_repos(request: Request) -> RepoListResponse:
    service = _service(request)
    repos = _call(service.list_repos)
    return RepoListResponse(repos=[_repo_info(item) for item in repos], total=len(repos))


@router.get("/repos/{repo_id}", response_model=RepoInfo)
def get_repo(repo_id: str, request: Request) -> RepoInfo:
    service = _service(request)
    repo = _call(service.get_repo, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail=f"repo not found: {repo_id}")
    return _repo_info(repo)


@router.delete("/repos/{repo_id}", response_model=DeleteRepoResponse)
def delete_repo(repo_id: str, request: Request) -> DeleteRepoResponse:
    service = _service(request)
    repo = _call(service.get_repo, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail=f"repo not found: {repo_id}")
    deleted = _call(service.delete_repo, repo_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"repo not found: {repo_id}")
    return DeleteRepoResponse(repo_id=repo_id, deleted=True)


@router.post("/repos/{repo_id}/reindex", response_model=RepoIndexResponse)
def reindex_repo(repo_id: str, request: Request) -> RepoIndexResponse:
    indexer = _indexer(request)
    service = _service(request)
    repo = _call(service.get_repo, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail=f"repo not found: {repo_id}")
    root = str(repo.get("root") or "").strip()
    if not root:
        raise HTTPException(status_code=400, detail=f"repo has no root: {repo_id}")
    name = str(repo.get("name") or "").strip() or Path(root).name
    resolved = _validate_root(root, request)
    out = _run_index(indexer, resolved, name, reindex=True)
    return RepoIndexResponse(**out)


# ---------- search / symbol / overview / impact ----------

@router.get("/search", response_model=list[SearchHit])
def search_code(
    request: Request,
    q: str = Query(min_length=1, max_length=1000, description="查询词"),
    repo_id: str | None = Query(default=None, description="限定仓库"),
    k: int = Query(default=8, ge=1, le=50, description="返回条数"),
    path_prefix: str | None = Query(default=None, description="路径前缀过滤"),
    kind: str | None = Query(default=None, description="符号类型过滤"),
) -> list[SearchHit]:
    service = _service(request)
    hits = _call(
        service.search_code,
        q,
        k=k,
        repo_id=repo_id,
        path_prefix=path_prefix,
        kind=kind,
    )
    return [SearchHit(**hit) for hit in hits]


@router.get("/symbol", response_model=SymbolInfo)
def get_symbol(
    request: Request,
    repo_id: str = Query(min_length=1),
    qualname: str = Query(min_length=1),
    path: str | None = Query(default=None),
) -> SymbolInfo:
    service = _service(request)
    symbol = _call(service.get_symbol, repo_id, qualname, path)
    if symbol is None:
        raise HTTPException(status_code=404, detail=f"symbol not found: {qualname}")
    return SymbolInfo(**symbol)


@router.get("/overview", response_model=OverviewResponse)
def overview(
    request: Request,
    repo_id: str = Query(min_length=1),
) -> OverviewResponse:
    service = _service(request)
    data = _call(service.overview, repo_id)
    return OverviewResponse(**data)


@router.get("/impact", response_model=ImpactResponse)
def impact(
    request: Request,
    repo_id: str = Query(min_length=1),
    path: str = Query(min_length=1),
) -> ImpactResponse:
    service = _service(request)
    data = _call(service.impact, repo_id, path)
    return ImpactResponse(**data)


# ---------- ask ----------

@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    service = _service(request)
    data = _call(service.ask, payload.question, payload.repo_id)
    refusal = data.get("refusal")
    return AskResponse(
        answer=Answer(**data.get("answer", {})),
        citations=[Citation(**item) for item in data.get("citations", [])],
        provider=str(data.get("provider") or ""),
        refusal=Refusal(**refusal) if isinstance(refusal, dict) else None,
    )
