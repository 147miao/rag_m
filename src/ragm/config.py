"""统一配置入口：所有路径与连接串集中在这里，禁止在业务代码里硬编码。"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# v1.1 内置 LLM provider 链：opencode-go 主 → ollama gemma4 备 → 本地 ollama qwen2.5 兜底
# （gemma4 免费额度用尽时上游返回 402，自动落到本地小模型；可用 LLM_PROVIDERS JSON 覆盖）。
# model 必须是对应服务实际返回的 id（relay 看 GET /v1/models），否则 404 后降级。
DEFAULT_LLM_PROVIDERS: tuple[dict[str, Any], ...] = (
    {
        "name": "opencode-go",
        "base_url": "http://127.0.0.1:8788/inference/go/v1",
        "model": "deepseek-v4.1-flash",
        "api_key_env": "OPENCODE_GO_API_KEY",
        # relay 要求每次请求带会话头；$UUID 由 LLMClient 每次请求替换为新的 uuid4。
        "extra_headers": {"x-opencode-session": "$UUID"},
        "first_token_timeout": 15,
    },
    {
        "name": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "gemma4:31b-cloud",
        "api_key": "ollama",
        "first_token_timeout": 20,
    },
    {
        # 末级兜底：gemma4 云模型额度用尽（402）时用本地 1.5B 小模型；只保可用，不保证质量。
        "name": "ollama-qwen",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:1.5b-instruct",
        "api_key": "ollama",
        "first_token_timeout": 20,
    },
)


def load_env_file(path: str | Path | None = None) -> bool:
    """把 .env 补进 os.environ（override=False）：只补缺失变量，不覆盖已 export 的值。

    - 默认读当前工作目录的 .env；systemd 的 WorkingDirectory=仓库根时即命中；
    - api_key_env 只认 os.environ，所以 API 启动（lifespan）要先调用本函数，Settings 本身已读 .env；
    - 文件缺失 / 解析失败静默返回 False；不写日志，避免把 key 打进日志。
    """
    env_path = Path(path) if path is not None else Path(".env")
    if not env_path.is_file():
        return False
    try:
        load_dotenv(env_path, override=False)
    except Exception:  # noqa: BLE001 - .env 损坏不应阻止服务启动
        return False
    return True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- storage ----
    data_root: Path = Path("/run/media/myj/D/rag_m")
    require_mount: bool = True

    # ---- vector db ----
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "rag_chunks"
    milvus_child_collection: str = "rag_children"
    parents_path: Path | None = None

    # ---- graph db ----
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "ragm-local-dev"
    graph_max_hops: int = 2
    graph_hop_limit: int = 40
    graph_iterative: bool = True
    kg_label: str = "Entity"
    kg_relation: str = "CO_EVIDENCE"

    # ---- llm (OpenAI-compatible: Ollama / DeepSeek) ----
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "gemma4:31b-cloud"
    llm_api_key: str = "ollama"
    llm_temperature: float = 0.0
    # v1.1 LLM 降级：JSON 数组（格式见 plan.md 3.1）；空 = 内置 DEFAULT_LLM_PROVIDERS。
    # api_key 不写入仓库，provider 通过 api_key_env 从环境变量读取。
    llm_providers: str = ""
    llm_chat_timeout: float = 60.0
    llm_first_token_timeout: float = 15.0

    # ---- embeddings ----
    embed_model: str = "google/embeddinggemma-300m"
    embed_dim: int = 768
    embed_batch_size: int = 32
    embed_model_path: Path | None = None
    embed_max_len: int = 512
    rerank_model: str = "bge-reranker-v2-m3"
    rerank_enabled: bool = False
    rerank_model_path: Path | None = None
    rerank_candidates: int = 20
    rerank_top_n: int = 8

    # ---- classifier ----
    classifier_backend: str = "rules"  # rules | prompt | lora
    classifier_run_dir: Path | None = None
    classifier_threshold: float = 0.6

    # ---- observability (LangSmith) ----
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "rag-m"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # ---- agentic ----
    agent_max_iterations: int = 2
    agent_max_calls: int = 4
    agent_max_tokens: int = 8000
    agent_retry: int = 1
    web_search_url: str = "http://127.0.0.1:8080/search"
    web_search_k: int = 5

    # ---- generation (langgraph) ----
    rag_backend: str = "langgraph"  # langgraph | demo
    # langgraph 依赖不可用时是否允许显式回退 demo；默认 False，/api/query 返回 503
    rag_allow_demo_fallback: bool = False
    rag_max_tokens: int = 1024
    rag_top_k: int = 8
    rag_multihop_k: int = 12
    rag_agentic_k: int = 12

    # ---- retrieval ----
    top_k_dense: int = 50
    top_k_sparse: int = 50
    top_k_final: int = 8
    rrf_k: int = 60

    # ---- multi-turn sessions (v1.1 W1) ----
    # 空 = data_root/data/sessions.sqlite，由 session_store_path 解析。
    session_db_path: str = ""
    session_max_turns: int = 6
    session_max_chars: int = 4000
    session_rewrite_enabled: bool = True

    # ---- MCP (v1.1 W2) ----
    mcp_enabled: bool = False
    mcp_sandbox_root: str = ""
    mcp_confirm_ttl_s: int = 60
    # 确认 token 的 HMAC secret；空 = 进程内随机（重启后旧 token 全部失效）。
    mcp_confirm_secret: str = ""

    # ---- async ingest jobs (v1.1 W4) ----
    # 空 = data_root/data/jobs.sqlite，由 jobs_store_path 解析。
    jobs_db_path: str = ""
    jobs_worker: bool = True
    # 总尝试次数（1 次初始 + N 次重试）：默认 4 = 1 次初始 + 3 次重试。
    jobs_max_attempts: int = 4

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "data" / "processed"

    @property
    def corpus_dir(self) -> Path:
        return self.data_root / "data" / "corpus"

    @property
    def models_dir(self) -> Path:
        return self.data_root / "models"

    @property
    def session_store_path(self) -> Path:
        """会话 SQLite 实际路径；SESSION_DB_PATH 为空时落到 data_root/data/sessions.sqlite。"""
        if self.session_db_path:
            return Path(self.session_db_path)
        return self.data_root / "data" / "sessions.sqlite"

    @property
    def jobs_store_path(self) -> Path:
        """任务 SQLite 实际路径；JOBS_DB_PATH 为空时落到 data_root/data/jobs.sqlite。"""
        if self.jobs_db_path:
            return Path(self.jobs_db_path)
        return self.data_root / "data" / "jobs.sqlite"

    @property
    def resolved_llm_providers(self) -> list[dict[str, Any]]:
        """解析 LLM_PROVIDERS；空字符串返回内置 provider 链（opencode-go → ollama）。"""
        raw = self.llm_providers.strip()
        if not raw:
            return [dict(item) for item in DEFAULT_LLM_PROVIDERS]
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise TypeError("LLM_PROVIDERS 必须是 JSON 数组")
        return [dict(item) for item in parsed]

    def ensure_dirs(self) -> None:
        """创建目录，并校验 DATA_ROOT 确实是挂载点（防止 D 盘未挂载时写进 tmpfs）。"""
        root = self.data_root
        if self.require_mount and not os.path.ismount(root.parent):
            raise RuntimeError(
                f"DATA_ROOT {root} 所在磁盘未挂载（{root.parent} 不是挂载点）。"
                "请先挂载 D 盘，或设置 RAGM_REQUIRE_MOUNT=false 使用本地目录。"
            )
        for p in (self.raw_dir, self.processed_dir, self.corpus_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()