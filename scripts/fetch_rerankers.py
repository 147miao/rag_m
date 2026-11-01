"""下载 reranker 候选模型（ModelScope）。"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://modelscope.cn/api/v1/models"
DEST = Path("/run/media/myj/D/rag_m/models/rerankers")
MODELS = [
    ("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3"),
    ("BAAI/bge-reranker-base", "bge-reranker-base"),
]
SKIP_EXT = (".jpg", ".png", ".gif", ".md", ".gitattributes")


def list_files(model_id: str) -> list[tuple[str, int]]:
    with urllib.request.urlopen(f"{BASE}/{model_id}/repo/files?Revision=master&Root=", timeout=30) as resp:
        data = json.load(resp)
    return [
        (f["Name"], int(f["Size"]))
        for f in data["Data"]["Files"]
        if f["Type"] == "blob" and not f["Name"].endswith(SKIP_EXT) and not f["Name"].startswith(".")
    ]


def download(model_id: str, name: str) -> None:
    files = list_files(model_id)
    dest = DEST / name
    dest.mkdir(parents=True, exist_ok=True)
    todo = [(n, s) for n, s in files if not (dest / n).exists() or (dest / n).stat().st_size != s]
    print(f"[{name}] {len(todo)} files, {sum(s for _, s in files)/1e6:.0f} MB", flush=True)

    def fetch(item: tuple[str, int]) -> str:
        fname, size = item
        url = f"{BASE}/{model_id}/repo?Revision=master&FilePath={urllib.parse.quote(fname)}"
        target = dest / fname
        for _ in range(4):
            try:
                with urllib.request.urlopen(url, timeout=600) as resp, target.open("wb") as fh:
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
                if target.stat().st_size == size:
                    return f"OK {fname}"
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
        return f"FAIL {fname} {err}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in as_completed([pool.submit(fetch, item) for item in todo]):
            print("  " + fut.result(), flush=True)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for model_id, name in MODELS:
        try:
            download(model_id, name)
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] FAILED: {exc}", file=sys.stderr, flush=True)
    for p in sorted(DEST.iterdir()):
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        print(f"{p.name}: {total/1e6:.0f} MB")


if __name__ == "__main__":
    main()
