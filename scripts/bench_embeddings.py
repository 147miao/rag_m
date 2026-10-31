"""Embedding 横向对比：同一 corpus / query / 索引协议下比较 4 个模型 + reranker 消融。

用法：
  .venv/bin/python scripts/bench_embeddings.py --out reports/embedding_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DATA = Path(os.environ.get("RAGM_RETRIEVAL_DIR", "/run/media/myj/D/rag_m/data/retrieval"))
MODELS_DIR = Path("/run/media/myj/D/rag_m/models/embeddings")
MODELS = [
    ("bge-m3", MODELS_DIR / "bge-m3", "", "cls"),
    ("qwen3-embedding-0.6b", MODELS_DIR / "qwen3-embedding-0.6b",
     "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", "last"),
    ("bge-large-zh-v1.5", MODELS_DIR / "bge-large-zh-v1.5", "", "cls"),
    ("embeddinggemma-300m", MODELS_DIR / "embeddinggemma-300m",
     "task: search result | query: ", "mean"),
]
RERANKER_DIR = MODELS_DIR / "qwen3-reranker-0.6b"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(ranked: list[list[str]], golds: list[list[str]], ks=(5, 10, 20)) -> dict:
    out = {}
    for k in ks:
        recalls, ndcgs, mrrs = [], [], []
        for ranking, gold in zip(ranked, golds):
            gold_set = set(gold)
            topk = ranking[:k]
            hit = sum(1 for d in topk if d in gold_set)
            recalls.append(hit / max(1, len(gold_set)))
            dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(topk) if d in gold_set)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_set), k)))
            ndcgs.append(dcg / idcg if idcg else 0.0)
            mrr = 0.0
            for i, d in enumerate(ranking):
                if d in gold_set:
                    mrr = 1.0 / (i + 1)
                    break
            mrrs.append(mrr)
        out[f"recall@{k}"] = round(float(np.mean(recalls)), 4)
        if k == 10:
            out["ndcg@10"] = round(float(np.mean(ndcgs)), 4)
            out["mrr@10"] = round(float(np.mean(mrrs)), 4)
    return out


class Embedder:
    """用 transformers 直接加载 + 手动 pooling，绕开 sentence-transformers 6.x 的 Pooling 兼容问题。

    kind: cls（bge 系列）/ last（Qwen3-Embedding）/ mean（embeddinggemma）
    """

    def __init__(self, path: Path, kind: str, max_len: int = 512) -> None:
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModel.from_pretrained(str(path), dtype=torch.bfloat16, device_map={"": 0}).eval()
        self.kind = kind
        self.max_len = max_len

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
            else:  # last non-pad token
                idx = enc["attention_mask"].sum(1) - 1
                emb = h[torch.arange(h.size(0), device=h.device), idx]
            out.append(F.normalize(emb.float(), p=2, dim=-1).cpu().numpy())
        return np.concatenate(out)


def run_one(name: str, path: Path, instruction: str, kind: str, lang: str, batch: int, max_len: int) -> dict:
    corpus = load_jsonl(DATA / f"corpus_{lang}.jsonl")
    queries = load_jsonl(DATA / f"queries_{lang}.jsonl")
    corpus_texts = [c["text"][: max_len * 4] for c in corpus]
    query_texts = [instruction + q["question"] for q in queries]

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = Embedder(path, kind, max_len)
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    doc_emb = model.encode(corpus_texts, batch)
    encode_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    q_emb = model.encode(query_texts, 32)
    query_encode_s = time.perf_counter() - t0

    doc_t = torch.from_numpy(doc_emb).to("cuda")
    q_t = torch.from_numpy(q_emb).to("cuda")
    t0 = time.perf_counter()
    scores = q_t @ doc_t.T
    topk = torch.topk(scores, k=min(50, doc_t.shape[0]), dim=1).indices.cpu().numpy()
    search_s = time.perf_counter() - t0
    ranked = [[corpus[i]["id"] for i in row] for row in topk]
    golds = [q["gold_ids"] for q in queries]
    m = metrics(ranked, golds)
    m.update(
        {
            "model": name,
            "lang": lang,
            "dim": int(doc_emb.shape[1]),
            "corpus": len(corpus),
            "queries": len(queries),
            "load_s": round(load_s, 2),
            "encode_s": round(encode_s, 1),
            "docs_per_s": round(len(corpus) / max(encode_s, 1e-6), 1),
            "query_encode_s": round(query_encode_s, 2),
            "search_s": round(search_s, 2),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
    )
    del model, doc_t, q_t
    torch.cuda.empty_cache()
    return m


# ---------- reranker（Qwen3-Reranker 生成式打分） ----------
def rerank(name: str, lang: str, top_n: int = 50, batch: int = 4) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    corpus = load_jsonl(DATA / f"corpus_{lang}.jsonl")
    queries = load_jsonl(DATA / f"queries_{lang}.jsonl")
    doc_by_id = {c["id"]: c["text"] for c in corpus}
    tok = AutoTokenizer.from_pretrained(RERANKER_DIR, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(RERANKER_DIR, dtype=torch.bfloat16, device_map={"": 0}).eval()
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    instruct = "Given a web search query, retrieve relevant passages that answer the query"

    # 用 bge-m3 先召回 top_n（与主对比一致的检索器）
    bge = Embedder(MODELS_DIR / "bge-m3", "cls", 512)
    doc_emb = bge.encode([doc_by_id[c["id"]][:2048] for c in corpus], 32)
    q_emb = bge.encode([q["question"] for q in queries], 32)
    scores = torch.from_numpy(q_emb).to("cuda") @ torch.from_numpy(doc_emb).to("cuda").T
    topk = torch.topk(scores, k=top_n, dim=1).indices.cpu().numpy()
    del bge, doc_emb, q_emb
    torch.cuda.empty_cache()

    ranked = []
    t0 = time.perf_counter()
    for qi, q in enumerate(queries):
        cand_ids = [corpus[i]["id"] for i in topk[qi]]
        texts = [
            prefix + f"<Instruct>: {instruct}\n<Query>: {q['question']}\n<Document>: {doc_by_id[cid][:2048]}<|im_end|>\n<|im_start|>assistant\n"
            for cid in cand_ids
        ]
        scores_list = []
        for i in range(0, len(texts), batch):
            enc = tok(texts[i : i + batch], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                logits = model(**enc).logits[:, -1, :]
            pair = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=-1)
            scores_list.extend(torch.softmax(pair, dim=-1)[:, 0].float().cpu().tolist())
        order = np.argsort(-np.array(scores_list))
        ranked.append([cand_ids[i] for i in order])
    m = metrics(ranked, [q["gold_ids"] for q in queries])
    m.update({"model": "bge-m3 + qwen3-reranker-0.6b", "lang": lang, "rerank_s": round(time.perf_counter() - t0, 1), "top_n": top_n})
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out", default="reports/embedding_benchmark.json")
    ap.add_argument("--langs", default="en,zh")
    ap.add_argument("--models", default="all", help="逗号分隔的模型名，all 表示全部")
    ap.add_argument("--skip-rerank", action="store_true")
    ap.add_argument("--rerank-only", action="store_true")
    ap.add_argument("--rerank-batch", type=int, default=4)
    args = ap.parse_args()
    results = []
    wanted = None if args.models == "all" else {x.strip() for x in args.models.split(",")}
    for name, path, instruction, kind in MODELS:
        if wanted is not None and name not in wanted:
            continue
        if args.rerank_only:
            break
        if not path.exists():
            print(f"skip {name}: {path} missing")
            continue
        for lang in args.langs.split(","):
            print(f"== {name} / {lang} ==", flush=True)
            try:
                m = run_one(name, path, instruction, kind, lang, args.batch, args.max_len)
                print(json.dumps(m, ensure_ascii=False), flush=True)
                results.append(m)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {name}/{lang}: {type(exc).__name__}: {exc}", flush=True)
    if RERANKER_DIR.exists() and not args.skip_rerank:
        for lang in args.langs.split(","):
            print(f"== reranker / {lang} ==", flush=True)
            try:
                results.append(rerank("bge-m3 + reranker", lang, batch=args.rerank_batch))
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED reranker/{lang}: {type(exc).__name__}: {exc}", flush=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()