"""统一配置入口：所有路径与连接串集中在这里，禁止在业务代码里硬编码。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- storage ----
    data_root: Path = Path("/run/media/myj/D/rag_m")
    require_mount: bool = True

    # ---- vector db ----
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "rag_chunks"

    # ---- graph db ----
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "ragm-local-dev"

    # ---- llm (OpenAI-compatible: Ollama / DeepSeek) ----
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "gemma4:31b-cloud"
    llm_api_key: str = "ollama"
    llm_temperature: float = 0.0

    # ---- embeddings ----
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    embed_batch_size: int = 16
    rerank_model: str = "Qwen/Qwen3-Reranker-0.6B"

    # ---- classifier ----
    classifier_backend: str = "rules"  # rules | prompt | lora
    classifier_run_dir: Path | None = None
    classifier_threshold: float = 0.6

    # ---- retrieval ----
    top_k_dense: int = 50
    top_k_sparse: int = 50
    top_k_final: int = 8
    rrf_k: int = 60

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
