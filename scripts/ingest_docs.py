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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragm.ingest.pipeline import build


def embed_chunks(out: Path, collection: str, recreate: bool) -> dict:
    """把 out/chunks.jsonl 编码写入 Milvus（source=docs）。非 --recreate 时先删同 source 旧行。"""
    from ragm.config import get_settings
    from ragm.embeddings.embedder import Embedder
    from ragm.store.milvus_store import MilvusStore

    s = get_settings()
    chunks = [json.loads(line) for line in (out / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not chunks:
        return {"inserted": 0, "rows": 0}
    store = MilvusStore(s.milvus_uri, collection, s.embed_dim)
    if recreate:
        store.recreate()
    elif store.client.has_collection(collection):
        store.delete_by_filter('source == "docs"')
    embedder = Embedder(s.embed_model_path, s.embed_model, s.embed_max_len)
    inserted = 0
    for i in range(0, len(chunks), 128):
        batch = chunks[i : i + 128]
        vecs = embedder.encode([c["text"] for c in batch], 32)
        rows = []
        for c, v in zip(batch, vecs):
            rows.append({
                "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": c["title"][:500],
                "source": c["source"], "lang": c["lang"], "chunk_index": int(c["chunk_index"]),
                "text": c["text"][:16000], "dense": v.tolist(),
            })
        store.insert(rows)
        inserted += len(rows)
    store.flush()
    store.load()
    return {"inserted": inserted, "rows": store.count(), "collection": collection}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="文档目录（递归）")
    ap.add_argument("--out", default="data/docs", help="输出目录（相对仓库或绝对路径）")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--source", default="docs")
    ap.add_argument("--recreate", action="store_true", help="忽略缓存，全部重新解析")
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
        report["embed"] = embed_chunks(out, args.collection, args.recreate)
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