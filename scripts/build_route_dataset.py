"""构建路由分类数据集：原始 jsonl → train/dev/test_weak。

用法：
    .venv/bin/python scripts/build_route_dataset.py \
        --raw /run/media/myj/D/rag_m/data/classifier/raw \
        --out /run/media/myj/D/rag_m/data/classifier

设计要点：
- 归一化 + 精确去重 + 近重复（前缀）标记；
- 类别平衡（每类上限 cap），分层采样；
- 按问题哈希切分，报告来源分布；
- 不生成模板合成数据（避免标签噪声），合成/增强留到后续版本。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("simple", "multihop", "reasoning")
PUNCT = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)


def normalize(q: str) -> str:
    return PUNCT.sub(" ", q.lower()).strip()


def qhash(q: str) -> str:
    return hashlib.sha1(normalize(q).encode("utf-8")).hexdigest()[:16]


def prefix_key(q: str, n: int = 40) -> str:
    return normalize(q)[:n]


def load_raw(raw_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = (r.get("question") or "").strip()
            label = r.get("weak_label")
            if not q or label not in LABELS:
                continue
            if r.get("answerable") is False:
                continue
            rows.append(
                {
                    "id": r.get("id") or qhash(q),
                    "question": q,
                    "label": label,
                    "source": r.get("source", path.stem),
                    "lang": r.get("lang", "en"),
                }
            )
    return rows


def dedupe(rows: list[dict]) -> tuple[list[dict], dict]:
    seen_hash: set[str] = set()
    seen_prefix: set[str] = set()
    out: list[dict] = []
    dup_exact = dup_prefix = 0
    for r in rows:
        h, p = qhash(r["question"]), prefix_key(r["question"])
        if h in seen_hash:
            dup_exact += 1
            continue
        if p in seen_prefix:
            dup_prefix += 1
            continue
        seen_hash.add(h)
        seen_prefix.add(p)
        out.append(r)
    return out, {"exact_duplicates": dup_exact, "prefix_duplicates": dup_prefix}


def balance(rows: list[dict], cap: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    out: list[dict] = []
    for label in LABELS:
        group = by_label[label]
        rng.shuffle(group)
        out.extend(group[:cap])
    rng.shuffle(out)
    return out


def split(rows: list[dict], dev_n: int, test_n: int, seed: int) -> tuple[list, list, list]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    dev, test, train = [], [], []
    for label in LABELS:
        group = by_label[label][:]
        rng.shuffle(group)
        n = len(group)
        n_dev = max(1, round(dev_n * n / len(rows)))
        n_test = max(1, round(test_n * n / len(rows)))
        dev.extend(group[:n_dev])
        test.extend(group[n_dev : n_dev + n_test])
        train.extend(group[n_dev + n_test :])
    for part in (dev, test, train):
        rng.shuffle(part)
    return train, dev, test


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(name: str, rows: list[dict]) -> dict:
    return {
        "name": name,
        "n": len(rows),
        "label": dict(Counter(r["label"] for r in rows)),
        "source": dict(Counter(r["source"] for r in rows)),
        "lang": dict(Counter(r["lang"] for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/run/media/myj/D/rag_m/data/classifier/raw")
    ap.add_argument("--out", default="/run/media/myj/D/rag_m/data/classifier")
    ap.add_argument("--cap-per-class", type=int, default=2500)
    ap.add_argument("--dev", type=int, default=400)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir, out_dir = Path(args.raw), Path(args.out)
    rows = load_raw(raw_dir)
    print(f"loaded {len(rows)} rows from {raw_dir}")
    rows, dup_stats = dedupe(rows)
    print(f"after dedupe {len(rows)} | {dup_stats}")
    rows = balance(rows, args.cap_per_class, args.seed)
    print(f"after balance {len(rows)} | cap={args.cap_per_class}")
    train, dev, test = split(rows, args.dev, args.test, args.seed)

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "dev.jsonl", dev)
    write_jsonl(out_dir / "test_weak.jsonl", test)
    stats = {
        "dup": dup_stats,
        "cap_per_class": args.cap_per_class,
        "splits": [summarize("train", train), summarize("dev", dev), summarize("test_weak", test)],
    }
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
