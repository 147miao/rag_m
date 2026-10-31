"""LoRA 分类器：加载 seq_cls / gen adapter，单条前向出 3 类概率。

依赖训练产物目录：meta.json + adapter/（由 scripts/train_route_cls.py 生成）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForSequenceClassification

from .base import Label, RouteDecision

LABELS: list[Label] = ["simple", "multihop", "reasoning"]
ABC = ["A", "B", "C"]
SYSTEM = (
    "你是 RAG 查询路由器。根据问题判断回答所需策略，只输出一个字母："
    "A=简单（单段落可答），B=多跳（需串联两个及以上实体），C=推理（需多源比较/计算/建议）。"
)


class LoRAClassifier:
    def __init__(self, run_dir: str | Path, device: str = "cuda") -> None:
        self.run_dir = Path(run_dir)
        meta = json.loads((self.run_dir / "meta.json").read_text(encoding="utf-8"))
        self.arch = meta["arch"]
        self.model_path = meta["model"]
        self.name = f"lora-{Path(self.model_path).name}-{meta['arch']}-{meta['variant']}"
        self.device = device if torch.cuda.is_available() else "cpu"

        self.tok = AutoTokenizer.from_pretrained(self.run_dir / "adapter")
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        if self.arch == "seq_cls":
            base = Qwen3ForSequenceClassification.from_pretrained(
                self.model_path, num_labels=3, dtype=torch.bfloat16, device_map={"": 0} if self.device == "cuda" else "cpu"
            )
            base.config.pad_token_id = self.tok.pad_token_id
        else:
            base = AutoModelForCausalLM.from_pretrained(
                self.model_path, dtype=torch.bfloat16, device_map={"": 0} if self.device == "cuda" else "cpu"
            )
        self.model = PeftModel.from_pretrained(base, str(self.run_dir / "adapter")).eval()
        self.label_ids = [self.tok.encode(lab, add_special_tokens=False)[0] for lab in ABC]

    def _prompt(self, question: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def classify(self, question: str) -> RouteDecision:
        t0 = time.perf_counter()
        enc = self.tok(self._prompt(question), truncation=True, max_length=192, add_special_tokens=False, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            if self.arch == "seq_cls":
                logits = self.model(**enc).logits.float()[0]
            else:
                logits = self.model(**enc).logits[0, -1].float()
                logits = torch.tensor([logits[i] for i in self.label_ids], device=self.device)
        probs = torch.softmax(logits, dim=-1).cpu()
        idx = int(probs.argmax())
        return RouteDecision(
            label=LABELS[idx], confidence=float(probs[idx]), backend=self.name,
            latency_ms=(time.perf_counter() - t0) * 1000, reason="LoRA 单前向",
        )

    def info(self) -> dict:
        return {"name": self.name, "ready": True, "arch": self.arch, "run_dir": str(self.run_dir)}
