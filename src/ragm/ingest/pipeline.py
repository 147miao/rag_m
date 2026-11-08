"""增量摄取流水线：扫描 → 解析 → 切块 → 按文件缓存 rows → 汇总 JSONL。"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .chunker import build_rows, doc_id_for
from .readers import SUPPORTED, read_file

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
ROWS_DIR = "rows"


def file_hash(path: Path) -> str:
    h = hashlib_sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hashlib_sha1():
    import hashlib

    return hashlib.sha1()


def scan(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


def load_manifest(out: Path) -> dict:
    path = out / MANIFEST
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            logger.warning("manifest unreadable, rebuilding: %s", path, exc_info=True)
    return {"version": 1, "files": {}}


def save_manifest(out: Path, manifest: dict) -> None:
    (out / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_rows(out: Path, doc_id: str, payload: dict) -> None:
    rows_dir = out / ROWS_DIR
    rows_dir.mkdir(parents=True, exist_ok=True)
    (rows_dir / f"{doc_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def build(
    src: Path,
    out: Path,
    lang: str = "zh",
    source: str = "docs",
    recreate: bool = False,
) -> dict:
    """增量构建，返回报告。行级缓存保证幂等；文件修改/删除会同步更新。"""
    src, out = src.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "files": {}} if recreate else load_manifest(out)
    if recreate:
        rows_dir = out / ROWS_DIR
        if rows_dir.exists():
            for p in rows_dir.glob("*.json"):
                p.unlink()

    files = scan(src)
    seen: set[str] = set()
    parsed, skipped, failed = [], [], []
    for path in files:
        rel = str(path.relative_to(src))
        seen.add(rel)
        try:
            digest = file_hash(path)
        except Exception as exc:  # noqa: BLE001
            failed.append({"file": rel, "error": f"hash: {exc}"})
            continue
        old = manifest["files"].get(rel)
        if old and old.get("file_hash") == digest and (out / ROWS_DIR / f"{old.get('doc_id')}.json").exists():
            skipped.append(rel)
            continue
        try:
            sections = read_file(path)
            if not sections:
                skipped.append(rel)
                continue
            doc_id = doc_id_for(digest, rel)
            parents, children, chunks = build_rows(rel, digest, sections, lang=lang, source=source)
            if not chunks:
                skipped.append(rel)
                continue
            _write_rows(out, doc_id, {
                "rel_path": rel, "file_hash": digest, "doc_id": doc_id,
                "parents": parents, "children": children, "chunks": chunks,
            })
            manifest["files"][rel] = {
                "file_hash": digest, "doc_id": doc_id, "updated_at": int(time.time()),
                "counts": {"parents": len(parents), "children": len(children), "chunks": len(chunks)},
                "size": path.stat().st_size,
            }
            parsed.append(rel)
        except Exception as exc:  # noqa: BLE001
            failed.append({"file": rel, "error": f"{type(exc).__name__}: {exc}"})

    # 删除已不存在的文件
    for rel in list(manifest["files"].keys()):
        if rel not in seen:
            doc_id = manifest["files"][rel].get("doc_id", "")
            p = out / ROWS_DIR / f"{doc_id}.json"
            if p.exists():
                p.unlink()
            del manifest["files"][rel]

    # 汇总 JSONL
    parents, children, chunks = [], [], []
    for rel, meta in manifest["files"].items():
        p = out / ROWS_DIR / f"{meta['doc_id']}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        parents.extend(data["parents"])
        children.extend(data["children"])
        chunks.extend(data["chunks"])
    for name, rows in (("parents", parents), ("children", children), ("chunks", chunks)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    save_manifest(out, manifest)
    return {
        "files_total": len(files), "parsed": len(parsed), "skipped": len(skipped), "failed": len(failed),
        "changed_files": parsed, "failed_files": failed,
        "parents": len(parents), "children": len(children), "chunks": len(chunks),
        "out": str(out),
    }