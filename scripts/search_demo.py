"""检索演示：dense / sparse / hybrid 三种模式对比。

用法：.venv/bin/python scripts/search_demo.py "Milvus 的 HNSW 参数" [--k 5]
"""
from __future__ import annotations

import argparse
import json

from ragm.config import get_settings
from ragm.store.search_service import SearchService


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="What magazine was started first, Arthur's Magazine or First for Women?")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    svc = SearchService(get_settings())
    print(json.dumps(svc.info(), ensure_ascii=False, indent=2))
    for mode in ("dense", "sparse", "hybrid"):
        res = svc.search(args.query, args.k, mode)
        print(f"\n== {mode} ({res['latency_ms']} ms) ==")
        for h in res["hits"]:
            print(f"  {h['score']:.4f} [{h['lang']}/{h['source']}] {str(h['title'])[:50]} | {str(h['text'])[:90]}")


if __name__ == "__main__":
    main()
