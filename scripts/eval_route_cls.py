"""评测路由分类器：加载 adapter，在指定数据上输出指标 + 延迟 + 混淆矩阵。

用法：
  .venv/bin/python scripts/eval_route_cls.py --run /run/media/myj/D/rag_m/models/classifier/s1_06b_rslora \
      --data /run/media/myj/D/rag_m/data/classifier/test_curated.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForSequenceClassification

LABELS = ["simple", "multihop", "reasoning"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ABC = ["A", "B", "C"]
SYSTEM = (
    "你是 RAG 查询路由器。根据问题判断回答所需策略，只输出一个字母："
    "A=简单（单段落可答），B=多跳（需串联两个及以上实体），C=推理（需多源比较/计算/建议）。"
)


def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(preds: np.ndarray, golds: np.ndarray, probs: np.ndarray | None = None) -> dict:
    acc = float((preds == golds).mean())
    recalls = {LABELS[i]: float(((preds == i) & (golds == i)).sum() / max(1, (golds == i).sum())) for i in range(3)}
    f1s = []
    for i in range(3):
        tp = int(((preds == i) & (golds == i)).sum())
        fp = int(((preds == i) & (golds != i)).sum())
        fn = int(((preds != i) & (golds == i)).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1s.append(2 * p * r / max(1e-9, p + r))
    out = {
        "n": int(len(golds)),
        "accuracy": acc,
        "macro_f1": float(sum(f1s) / 3),
        "per_class_recall": recalls,
        "confusion": [[int(((golds == i) & (preds == j)).sum()) for j in range(3)] for i in range(3)],
    }
    if probs is not None:
        conf = probs.max(axis=1)
        correct = (preds == golds).astype(float)
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for b in range(10):
            m = (conf >= bins[b]) & (conf < bins[b + 1])
            if m.sum() == 0:
                continue
            ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
        out["ece"] = float(ece)
    return out


def load_model(run_dir: Path):
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    model_path = meta["model"]
    tok = AutoTokenizer.from_pretrained(run_dir / "adapter")
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if meta["arch"] == "seq_cls":
        base = Qwen3ForSequenceClassification.from_pretrained(model_path, num_labels=3, dtype=torch.bfloat16, device_map={"": 0})
        base.config.pad_token_id = tok.pad_token_id
        model = PeftModel.from_pretrained(base, str(run_dir / "adapter"))
    else:
        base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map={"": 0})
        model = PeftModel.from_pretrained(base, str(run_dir / "adapter"))
    model.eval()
    return model, tok, meta


def predict_seq_cls(model, tok, rows, max_len: int, batch: int = 16) -> tuple[np.ndarray, np.ndarray, list[float]]:
    preds, probs, lat = [], [], []
    for r in rows:
        text = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tok(text, truncation=True, max_length=max_len, add_special_tokens=False, return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(**enc).logits.float()
        lat.append((time.perf_counter() - t0) * 1000)
        p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        probs.append(p)
        preds.append(int(p.argmax()))
    return np.array(preds), np.stack(probs), lat


def predict_gen(model, tok, rows, max_len: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    label_ids = [tok.encode(lab, add_special_tokens=False)[0] for lab in ABC]
    preds, probs, lat = [], [], []
    for r in rows:
        text = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tok(text, truncation=True, max_length=max_len, add_special_tokens=False, return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(**enc).logits[0, -1].float()
        lat.append((time.perf_counter() - t0) * 1000)
        p = torch.softmax(torch.tensor([logits[i] for i in label_ids]), dim=0).cpu().numpy()
        probs.append(p)
        preds.append(int(p.argmax()))
    return np.array(preds), np.stack(probs), lat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-len", type=int, default=192)
    args = ap.parse_args()

    run_dir = Path(args.run)
    rows = load_rows(args.data)
    golds = np.array([LABEL2ID[r["label"]] for r in rows])
    model, tok, meta = load_model(run_dir)
    t0 = time.perf_counter()
    if meta["arch"] == "seq_cls":
        preds, probs, lat = predict_seq_cls(model, tok, rows, args.max_len)
    else:
        preds, probs, lat = predict_gen(model, tok, rows, args.max_len)
    m = metrics(preds, golds, probs)
    m.update(
        {
            "arch": meta["arch"],
            "variant": meta["variant"],
            "model": meta["model"],
            "run": str(run_dir),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "wall_s": round(time.perf_counter() - t0, 1),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "test_data": str(args.data),
        }
    )
    print(json.dumps(m, ensure_ascii=False, indent=2))
    out = Path(args.out) if args.out else run_dir / ("eval_" + Path(args.data).stem + ".json")
    out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
