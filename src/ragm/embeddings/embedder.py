"""可插拔 Embedder：transformers 直接加载 + 手动 pooling（绕开 sentence-transformers 6.x 的 Pooling 兼容问题）。

模型规格来自 docs/05-embedding-benchmark.md 的实测结论：
- google/embeddinggemma-300m：mean pooling，768 维，默认
- BAAI/bge-m3：CLS pooling，1024 维，备选
- BAAI/bge-large-zh-v1.5：CLS pooling，1024 维，纯中文
- Qwen/Qwen3-Embedding-0.6B：last-token pooling，1024 维，需 query instruction
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_SPECS: dict[str, dict] = {
    "google/embeddinggemma-300m": {
        "kind": "mean",
        "dim": 768,
        "max_len": 2048,
        "query_instruction": "task: search result | query: ",
    },
    "BAAI/bge-m3": {"kind": "cls", "dim": 1024, "max_len": 8192},
    "BAAI/bge-large-zh-v1.5": {"kind": "cls", "dim": 1024, "max_len": 512},
    "Qwen/Qwen3-Embedding-0.6B": {
        "kind": "last",
        "dim": 1024,
        "max_len": 32768,
        "query_instruction": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    },
}


class Embedder:
    def __init__(self, model_path: str | Path, model_name: str | None = None, max_len: int | None = None) -> None:
        self.path = str(model_path)
        spec = MODEL_SPECS.get(model_name or "", {})
        self.kind = spec.get("kind", "mean")
        self.dim = spec.get("dim", 0)
        self.max_len = max_len or spec.get("max_len", 512)
        self.query_instruction = spec.get("query_instruction", "")
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModel.from_pretrained(self.path, dtype=torch.bfloat16, device_map={"": 0}).eval()
        self.dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(self, texts: list[str], batch: int = 32) -> np.ndarray:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to("cuda")
            h = self.model(**enc).last_hidden_state
            if self.kind == "cls":
                emb = h[:, 0]
            elif self.kind == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                idx = enc["attention_mask"].sum(1) - 1
                emb = h[torch.arange(h.size(0), device=h.device), idx]
            out.append(F.normalize(emb.float(), p=2, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def encode_queries(self, texts: list[str], batch: int = 16) -> np.ndarray:
        return self.encode([self.query_instruction + t for t in texts], batch)