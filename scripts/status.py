"""打印实验进度看板，并写入 reports/status.json + reports/STATUS.md。

用法：python scripts/status.py [--watch 10]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ragm.ops.status import collect_status, render_markdown

ROOT = Path(__file__).resolve().parents[1]


def once() -> dict:
    status = collect_status()
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_markdown(status)
    (ROOT / "reports" / "STATUS.md").write_text(md, encoding="utf-8")
    print(md)
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="每 N 秒刷新一次；0 表示只输出一次")
    args = ap.parse_args()
    while True:
        once()
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
