"""Reranker：bge 系列（分类头）与 Qwen3-Reranker（生成式 yes/no 打分）。"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


class BaseReranker:
    name = "base"

    def rerank(self, query: str, hits: list[dict], top_n: int = 8) -> list[dict]:
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    """bge-reranker-base / bge-reranker-v2-m3 等分类头模型。"""

    def __init__(self, model_path: str, name: str | None = None, max_len: int = 512, batch: int = 16) -> None:
        self.name = name or model_path.rstrip("/").split("/")[-1]
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map={"": 0}
        ).eval()
        self.max_len = max_len
        self.batch = batch

    @torch.inference_mode()
    def score(self, query: str, passages: list[str]) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(passages), self.batch):
            batch = passages[i : i + self.batch]
            enc = self.tok(
                [[query, p] for p in batch], padding=True, truncation="only_second", max_length=self.max_len, return_tensors="pt"
            ).to("cuda")
            logits = self.model(**enc).logits
            scores.extend(logits[:, 0].float().cpu().tolist())
        return scores

    def rerank(self, query: str, hits: list[dict], top_n: int = 8) -> list[dict]:
        if not hits:
            return []
        scores = self.score(query, [h.get("text", "")[:2000] for h in hits])
        order = np.argsort(-np.array(scores))
        return [{**hits[i], "rerank_score": float(scores[i])} for i in order[:top_n]]


class Qwen3Reranker(BaseReranker):
    """Qwen3-Reranker：生成 yes/no，取 yes 概率作为相关性分数。"""

    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"

    def __init__(self, model_path: str, name: str | None = None, max_len: int = 1024, batch: int = 4) -> None:
        self.name = name or model_path.rstrip("/").split("/")[-1]
        self.tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map={"": 0}).eval()
        self.yes_id = self.tok.encode("yes", add_special_tokens=False)[0]
        self.no_id = self.tok.encode("no", add_special_tokens=False)[0]
        self.max_len = max_len
        self.batch = batch

    @torch.inference_mode()
    def score(self, query: str, passages: list[str]) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(passages), self.batch):
            batch = passages[i : i + self.batch]
            texts = [
                self.PREFIX
                + f"<Instruct>: {self.INSTRUCT}\n<Query>: {query}\n<Document>: {p[:2000]}<|im_end|>\n<|im_start|>assistant\n"
                for p in batch
            ]
            enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to("cuda")
            logits = self.model(**enc).logits[:, -1, :]
            pair = torch.stack([logits[:, self.yes_id], logits[:, self.no_id]], dim=-1)
            scores.extend(torch.softmax(pair, dim=-1)[:, 0].float().cpu().tolist())
        return scores

    def rerank(self, query: str, hits: list[dict], top_n: int = 8) -> list[dict]:
        if not hits:
            return []
        scores = self.score(query, [h.get("text", "")[:2000] for h in hits])
        order = np.argsort(-np.array(scores))
        return [{**hits[i], "rerank_score": float(scores[i])} for i in order[:top_n]]


def build_reranker(name: str, model_path: str) -> BaseReranker:
    if "qwen3-reranker" in name.lower():
        return Qwen3Reranker(model_path, name=name)
    return CrossEncoderReranker(model_path, name=name)
