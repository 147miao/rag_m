"""训练路由分类器（三分类）：分类头（SEQ_CLS）或生成式（A/B/C）。

示例：
  # 分类头 + rsLoRA
  .venv/bin/python scripts/train_route_cls.py --arch seq_cls --model /run/media/myj/D/rag_m/models/Qwen3-0.6B \
      --variant rslora --out /run/media/myj/D/rag_m/models/classifier/s1_06b_rslora

  # 生成式
  .venv/bin/python scripts/train_route_cls.py --arch gen --model ... --out ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Qwen3ForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABELS = ["simple", "multihop", "reasoning"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ABC = ["A", "B", "C"]
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

SYSTEM = (
    "你是 RAG 查询路由器。根据问题判断回答所需策略，只输出一个字母："
    "A=简单（单段落可答），B=多跳（需串联两个及以上实体），C=推理（需多源比较/计算/建议）。"
)


def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def lora_kwargs(variant: str) -> dict:
    if variant == "rslora":
        return {"use_rslora": True}
    if variant == "dora":
        return {"use_dora": True}
    if variant == "pissa":
        return {"init_lora_weights": "pissa"}
    return {}


class SeqClsDataset(Dataset):
    def __init__(self, rows: list[dict], tok, max_len: int):
        self.items = []
        for r in rows:
            text = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["question"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tok(text, truncation=True, max_length=max_len, add_special_tokens=False)
            self.items.append(
                {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": LABEL2ID[r["label"]]}
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


class GenDataset(Dataset):
    def __init__(self, rows: list[dict], tok, max_len: int):
        self.items = []
        self.label_ids = {lab: tok.encode(lab, add_special_tokens=False)[0] for lab in ABC}
        for r in rows:
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["question"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            p_ids = tok(prompt, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]
            a_id = self.label_ids[ABC[LABEL2ID[r["label"]]]]
            input_ids = p_ids + [a_id]
            labels = [-100] * len(p_ids) + [a_id]
            self.items.append({"input_ids": input_ids, "labels": labels, "answer_start": len(p_ids)})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


class GenCollator:
    def __init__(self, tok):
        self.pad = tok.pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append([self.pad] * pad + b["input_ids"])
            labels.append([-100] * pad + b["labels"])
            attn.append([0] * pad + [1] * len(b["input_ids"]))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weight: torch.Tensor | None = None, label_smoothing: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        loss = F.cross_entropy(logits, labels, weight=self.class_weight, label_smoothing=self.label_smoothing)
        return (loss, outputs) if return_outputs else loss


def metrics_from_preds(preds: np.ndarray, golds: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(golds, preds)),
        "macro_f1": float(f1_score(golds, preds, average="macro")),
        "per_class_recall": {
            LABELS[i]: float(((preds == i) & (golds == i)).sum() / max(1, (golds == i).sum())) for i in range(3)
        },
        "confusion": [[int(((golds == i) & (preds == j)).sum()) for j in range(3)] for i in range(3)],
    }


def predict_seq_cls(model, tok, rows, max_len: int, batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ds = SeqClsDataset(rows, tok, max_len)
    coll = DataCollatorWithPadding(tok, padding=True)
    preds, probs = [], []
    with torch.inference_mode():
        for i in range(0, len(ds), batch):
            items = [ds[j] for j in range(i, min(i + batch, len(ds)))]
            batch_in = coll(items)
            batch_in = {k: v.to("cuda") for k, v in batch_in.items()}
            logits = model(**batch_in).logits.float()
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def predict_gen(model, tok, rows, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    label_ids = [tok.encode(lab, add_special_tokens=False)[0] for lab in ABC]
    preds, probs = [], []
    with torch.inference_mode():
        for r in rows:
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["question"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tok(prompt, truncation=True, max_length=max_len, add_special_tokens=False, return_tensors="pt").to("cuda")
            logits = model(**enc).logits[0, -1].float()
            abc = torch.tensor([logits[i] for i in label_ids])
            p = torch.softmax(abc, dim=0).cpu().numpy()
            probs.append(p)
            preds.append(int(p.argmax()))
    return np.array(preds), np.stack(probs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["seq_cls", "gen"], default="seq_cls")
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", choices=["plain", "rslora", "dora", "pissa"], default="plain")
    ap.add_argument("--data-dir", default="/run/media/myj/D/rag_m/data/classifier")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--grad-checkpoint", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    train_rows = load_rows(str(data_dir / "train.jsonl"))
    dev_rows = load_rows(str(data_dir / "dev.jsonl"))
    print(f"train={len(train_rows)} dev={len(dev_rows)} arch={args.arch} variant={args.variant}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    variant_kw = lora_kwargs(args.variant)
    if args.arch == "seq_cls":
        model = Qwen3ForSequenceClassification.from_pretrained(
            args.model, num_labels=3, dtype=torch.bfloat16, device_map={"": 0}
        )
        model.config.pad_token_id = tok.pad_token_id
        lora = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=TARGETS,
            modules_to_save=["score"],
            **variant_kw,
        )
        model = get_peft_model(model, lora)
        train_ds = SeqClsDataset(train_rows, tok, args.max_len)
        collator = DataCollatorWithPadding(tok, padding=True)
        compute_metrics = lambda p: metrics_from_preds(p.predictions.argmax(-1), p.label_ids)
        trainer_cls = WeightedTrainer
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=TARGETS,
            **variant_kw,
        )
        model = get_peft_model(model, lora)
        train_ds = GenDataset(train_rows, tok, args.max_len)
        collator = GenCollator(tok)
        compute_metrics = None
        trainer_cls = Trainer

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("gradient checkpointing enabled")
    model.print_trainable_parameters()

    # class weights
    class_weight = None
    if not args.no_class_weight:
        counts = np.array([sum(1 for r in train_rows if r["label"] == l) for l in LABELS], dtype=np.float32)
        w = counts.sum() / (3 * np.maximum(counts, 1))
        class_weight = torch.tensor(w, device="cuda")

    # param groups：LoRA 与分类头不同 lr
    lora_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if "score" in name or "modules_to_save" in name else lora_params).append(p)
    groups = [{"params": lora_params, "lr": args.lr}]
    if head_params:
        groups.append({"params": head_params, "lr": args.head_lr})

    is_seq = args.arch == "seq_cls"
    targs = TrainingArguments(
        output_dir=str(out / "hf"),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch" if is_seq else "no",
        save_strategy="epoch" if is_seq else "no",
        save_total_limit=1,
        load_best_model_at_end=is_seq,
        metric_for_best_model="macro_f1" if is_seq else None,
        greater_is_better=is_seq,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    if trainer_cls is WeightedTrainer:
        trainer = trainer_cls(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=SeqClsDataset(dev_rows, tok, args.max_len),
            data_collator=collator,
            compute_metrics=compute_metrics,
            class_weight=class_weight,
            optimizers=(torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8), None),
        )
        trainer.train()
    else:
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=None,
            data_collator=collator,
            optimizers=(torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8), None),
        )
        trainer.train()

    # 用 dev 选出最佳后，保存 adapter
    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))

    dev_preds, _dev_probs = (
        predict_seq_cls(model, tok, dev_rows, args.max_len)
        if args.arch == "seq_cls"
        else predict_gen(model, tok, dev_rows, args.max_len)
    )
    dev_golds = np.array([LABEL2ID[r["label"]] for r in dev_rows])
    dev_metrics = metrics_from_preds(dev_preds, dev_golds)
    meta = {
        "arch": args.arch,
        "variant": args.variant,
        "model": args.model,
        "epochs": args.epochs,
        "lr": args.lr,
        "head_lr": args.head_lr,
        "max_len": args.max_len,
        "seed": args.seed,
        "dev_metrics": dev_metrics,
        "train_size": len(train_rows),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "elapsed_s": round(trainer.state.log_history[-1].get("train_runtime", 0), 1) if trainer.state.log_history else None,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dev_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()