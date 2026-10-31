"""从 ModelScope 下载 embedding / reranker 权重（多线程分文件）。

用法：python scripts/fetch_models.py
输出：/run/media/myj/D/rag_m/models/embeddings/<slug>/
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://modelscope.cn/api/v1/models"
DEST = Path("/run/media/myj/D/rag_m/models/embeddings")
MODELS = [
    ("BAAI/bge-m3", "bge-m3"),
    ("Qwen/Qwen3-Embedding-0.6B", "qwen3-embedding-0.6b"),
    ("BAAI/bge-large-zh-v1.5", "bge-large-zh-v1.5"),
    ("google/embeddinggemma-300m", "embeddinggemma-300m"),
    ("Qwen/Qwen3-Reranker-0.6B", "qwen3-reranker-0.6b"),
]
SKIP_EXT = (".jpg", ".png", ".gif", ".md", ".gitattributes")


def list_files(model_id: str) -> list[tuple[str, int]]:
    url = f"{BASE}/{model_id}/repo/files?Revision=master&Root="
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    out = []
    for f in data["Data"]["Files"]:
        if f["Type"] != "blob" or f["Name"].endswith(SKIP_EXT) or f["Name"].startswith("."):
            continue
        out.append((f["Name"], int(f["Size"])))
    return out


def download(model_id: str, name: str, dest: Path) -> str:
    files = list_files(model_id)
    dest.mkdir(parents=True, exist_ok=True)
    todo = [(n, s) for n, s in files if not (dest / n).exists() or (dest / n).stat().st_size != s]
    print(f"[{name}] {len(files)} files, {len(todo)} to fetch, {sum(s for _, s in files)/1e6:.0f} MB", flush=True)

    def fetch(item: tuple[str, int]) -> str:
        fname, size = item
        url = f"{BASE}/{model_id}/repo?Revision=master&FilePath={urllib.parse.quote(fname)}"
        target = dest / fname
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=300) as resp, target.open("wb") as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                if target.stat().st_size == size:
                    return f"OK {fname}"
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
                if attempt == 3:
                    return f"FAIL {fname} {last}"
        return f"FAIL {fname}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, item) for item in todo]
        for fut in as_completed(futures):
            print(f"  {fut.result()}", flush=True)
    return f"[{name}] done"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for model_id, name in MODELS:
        try:
            print(download(model_id, name, DEST / name), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    for p in sorted(DEST.iterdir()):
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        print(f"{p.name}: {total/1e6:.0f} MB")


if __name__ == "__main__":
    main()
