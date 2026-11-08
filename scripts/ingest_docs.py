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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="文档目录（递归）")
    ap.add_argument("--out", default="data/docs", help="输出目录（相对仓库或绝对路径）")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--source", default="docs")
    ap.add_argument("--recreate", action="store_true", help="忽略缓存，全部重新解析")
    args = ap.parse_args()
    src = Path(args.src).expanduser()
    if not src.exists():
        raise SystemExit(f"src not found: {src}")
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    report = build(src, out, lang=args.lang, source=args.source, recreate=args.recreate)
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