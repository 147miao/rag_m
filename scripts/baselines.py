"""三条非 LoRA 基线：规则 / gemma4 提示词 / bge-m3 + 逻辑回归。

用法：
  .venv/bin/python scripts/baselines.py --data .../test_curated.jsonl --out reports/baselines.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

LABELS = ["simple", "multihop", "reasoning"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
OLLAMA = "http://127.0.0.1:11434"

REASONING_KW = ("对比", "比较", "为什么", "分析", "权衡", "建议", "优劣", "估算", "步骤", "影响", "原因", "推理", "方案", "选型", "区别", "怎么变化", "是否值得")
MULTIHOP_KW = ("哪个机构", "哪个城市", "之后", "还发布", "的作者的", "的创始人", "关系", "谁的学生", "哪位", "还", "以及", "并且", "分别")

RUBRIC = """你是 RAG 查询路由标注员。把问题分为三类，只输出 JSON：{"label":"simple|multihop|reasoning"}。
simple=单段落可答的事实/数值/定义；multihop=需串联两个及以上实体或文档；reasoning=需多源比较/计算/规划/给建议。拿不准按更贵的标。问题："""


def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(preds: list[int], golds: list[int]) -> dict:
    preds_a, golds_a = np.array(preds), np.array(golds)
    f1s = []
    for i in range(3):
        tp = int(((preds_a == i) & (golds_a == i)).sum())
        fp = int(((preds_a == i) & (golds_a != i)).sum())
        fn = int(((preds_a != i) & (golds_a == i)).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1s.append(2 * p * r / max(1e-9, p + r))
    return {
        "n": len(golds),
        "accuracy": float((preds_a == golds_a).mean()),
        "macro_f1": float(sum(f1s) / 3),
        "per_class_recall": {LABELS[i]: float(((preds_a == i) & (golds_a == i)).sum() / max(1, (golds_a == i).sum())) for i in range(3)},
        "confusion": [[int(((golds_a == i) & (preds_a == j)).sum()) for j in range(3)] for i in range(3)],
    }


# ---------- 规则 ----------
def rule_predict(q: str) -> int:
    if any(k in q for k in REASONING_KW):
        return 2
    if any(k in q for k in MULTIHOP_KW):
        return 1
    return 0


# ---------- gemma4 提示词 ----------
def llm_predict(question: str) -> int:
    body = {
        "model": "gemma4:31b-cloud",
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 60},
        "messages": [{"role": "user", "content": RUBRIC + question}],
    }
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        content = json.load(resp)["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        return 0
    label = json.loads(m.group(0)).get("label")
    return LABEL2ID.get(label, 0)


# ---------- bge-m3 + 逻辑回归 ----------
def embed_batch(texts: list[str], batch: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        body = {"model": "bge-m3", "input": chunk}
        req = urllib.request.Request(OLLAMA + "/api/embed", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            out.extend(json.load(resp)["embeddings"])
    return np.array(out, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train", default="/run/media/myj/D/rag_m/data/classifier/train.jsonl")
    ap.add_argument("--out", default="reports/baselines.json")
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.data)
    golds = [LABEL2ID[r["label"]] for r in rows]
    results: dict[str, dict] = {}

    # 规则
    t0 = time.perf_counter()
    preds = [rule_predict(r["question"]) for r in rows]
    results["rules"] = {**metrics(preds, golds), "latency_s": round(time.perf_counter() - t0, 3)}

    # 提示词
    if not args.skip_llm:
        t0 = time.perf_counter()
        preds = []
        for i, r in enumerate(rows):
            try:
                preds.append(llm_predict(r["question"]))
            except Exception as exc:  # noqa: BLE001
                print("llm fail", i, exc)
                preds.append(0)
        results["gemma4_prompt"] = {**metrics(preds, golds), "latency_s": round(time.perf_counter() - t0, 1)}

    # 逻辑回归
    from sklearn.linear_model import LogisticRegression

    train_rows = load_rows(args.train)
    t0 = time.perf_counter()
    Xtr = embed_batch([r["question"] for r in train_rows])
    Xte = embed_batch([r["question"] for r in rows])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr, [LABEL2ID[r["label"]] for r in train_rows])
    preds = clf.predict(Xte).tolist()
    results["bge_m3_logreg"] = {**metrics(preds, golds), "latency_s": round(time.perf_counter() - t0, 1)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {"acc": v["accuracy"], "f1": v["macro_f1"]} for k, v in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
