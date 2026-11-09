"""真实文档摄取 CLI：解析 + 切块 + 增量缓存。

用法：
  python scripts/ingest_docs.py --src ~/docs --out data/docs
  python scripts/ingest_docs.py --src ~/docs --out data/docs --recreate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragm.ingest.pipeline import build, load_manifest

EMBED_BATCH = 128
EMBED_THREADS = 32
TITLE_LIMIT = 500
TEXT_LIMIT = 16000


def _escape_filter(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _has_collection(store: Any, collection: str) -> bool:
    try:
        return bool(store.client.has_collection(collection))
    except Exception:  # noqa: BLE001 - pymilvus 在集合缺失/连接异常时抛出的异常类型不稳定
        return False


def _read_chunks(out: Path) -> list[dict]:
    path = out / "chunks.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _encode_and_insert(store: Any, embedder: Any, chunks: list[dict]) -> int:
    inserted = 0
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        vecs = embedder.encode([c["text"] for c in batch], EMBED_THREADS)
        rows = []
        for c, v in zip(batch, vecs):
            dense = v.tolist() if hasattr(v, "tolist") else list(v)
            rows.append({
                "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": c["title"][:TITLE_LIMIT],
                "source": c["source"], "lang": c["lang"], "chunk_index": int(c["chunk_index"]),
                "text": c["text"][:TEXT_LIMIT], "dense": dense,
            })
        store.insert(rows)
        inserted += len(rows)
    return inserted


def embed_chunks(
    out: Path,
    collection: str,
    recreate: bool = False,
    changed_files: list[str] | None = None,
    *,
    stale_doc_ids: list[str] | None = None,
    source: str = "docs",
    store: Any = None,
    embedder: Any = None,
) -> dict:
    """增量编码到 Milvus：只重编码 changed_files 对应的 doc_id。

    - changed_files=None 表示调用方没有增量信息，按 source 全量刷新（兼容旧行为）；
    - stale_doc_ids 是 build() 返回的旧 doc_id（文件修改/清空/删除后需要清理）；
    - --recreate 或集合不存在时自动全量重建。
    """
    manifest = load_manifest(out)
    chunks = _read_chunks(out)
    manifest_files = manifest.get("files", {})
    current_doc_ids = {meta["doc_id"] for meta in manifest_files.values() if meta.get("doc_id")}

    incremental = changed_files is not None and not recreate
    changed_doc_ids: list[str] = []
    if incremental:
        for rel in changed_files:
            meta = manifest_files.get(rel)
            if meta and meta.get("doc_id"):
                changed_doc_ids.append(meta["doc_id"])

    delete_doc_ids: list[str] = []
    if incremental:
        for doc_id in [*changed_doc_ids, *(stale_doc_ids or [])]:
            if doc_id and doc_id not in delete_doc_ids:
                delete_doc_ids.append(doc_id)

    if not chunks and not delete_doc_ids:
        return {"inserted": 0, "deleted_docs": 0, "docs": 0, "rows": 0, "collection": collection, "mode": "noop"}

    if store is None:
        from ragm.config import get_settings
        from ragm.store.milvus_store import MilvusStore

        settings = get_settings()
        store = MilvusStore(settings.milvus_uri, collection, settings.embed_dim)

    collection_exists = _has_collection(store, collection)
    if not recreate and not collection_exists and not chunks:
        return {"inserted": 0, "deleted_docs": 0, "docs": 0, "rows": 0, "collection": collection, "mode": "noop"}
    if incremental and not changed_doc_ids and not delete_doc_ids and collection_exists:
        return {"inserted": 0, "deleted_docs": 0, "docs": 0, "rows": 0, "collection": collection, "mode": "incremental"}

    selected_doc_ids: set[str] | None
    if recreate or not collection_exists:
        store.recreate()
        mode = "recreate" if recreate else "recreate-missing"
        selected_doc_ids = None
        delete_doc_ids = []
    elif not incremental:
        store.delete_by_filter(f'source == "{_escape_filter(source)}"')
        mode = "full"
        selected_doc_ids = None
    else:
        for doc_id in delete_doc_ids:
            store.delete_by_filter(f'doc_id == "{_escape_filter(doc_id)}"')
        mode = "incremental"
        selected_doc_ids = set(changed_doc_ids)

    selected = chunks if selected_doc_ids is None else [c for c in chunks if c["doc_id"] in selected_doc_ids]
    inserted = 0
    if selected:
        if embedder is None:
            from ragm.config import get_settings
            from ragm.embeddings.embedder import Embedder

            settings = get_settings()
            embedder = Embedder(settings.embed_model_path, settings.embed_model, settings.embed_max_len)
        inserted = _encode_and_insert(store, embedder, selected)
    store.flush()
    store.load()
    return {
        "inserted": inserted,
        "deleted_docs": len(delete_doc_ids),
        "docs": len(current_doc_ids) if selected_doc_ids is None else len(selected_doc_ids),
        "rows": store.count(),
        "collection": collection,
        "mode": mode,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="文档目录（递归）")
    ap.add_argument("--out", default="data/docs", help="输出目录（相对仓库或绝对路径）")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--source", default="docs")
    ap.add_argument("--recreate", action="store_true", help="忽略缓存，全部重新解析并重建 Milvus 集合")
    ap.add_argument("--embed", action="store_true", help="编码并写入 Milvus（需要 GPU）")
    ap.add_argument("--collection", default="rag_docs_chunks")
    args = ap.parse_args()
    src = Path(args.src).expanduser()
    if not src.exists():
        raise SystemExit(f"src not found: {src}")
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    report = build(src, out, lang=args.lang, source=args.source, recreate=args.recreate)
    if args.embed:
        report["embed"] = embed_chunks(
            out,
            args.collection,
            args.recreate,
            changed_files=report.get("changed_files", []),
            stale_doc_ids=report.get("stale_doc_ids", []),
            source=args.source,
        )
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "ingest_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 文档摄取报告",
        "",
        f"- 文件总数：{report['files_total']}（解析 {report['parsed']} / 跳过 {report['skipped']} / 失败 {report['failed']}）",
        f"- parents：{report['parents']}，children：{report['children']}，chunks：{report['chunks']}",
        f"- 输出目录：{report['out']}",
    ]
    if report["failed_files"]:
        lines += ["", "## 失败文件", ""]
        lines += [f"- {x['file']}: {x['error']}" for x in report["failed_files"]]
    (reports / "ingest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("files_total", "parsed", "skipped", "failed", "parents", "children", "chunks")}, ensure_ascii=False))


if __name__ == "__main__":
    main()