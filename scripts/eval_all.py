"""统一评测入口：检索门禁 + 生成评测。

用法：
  python scripts/eval_all.py --suite retrieval --fixture-only   # CI 门禁（无 GPU/LLM）
  python scripts/eval_all.py --suite retrieval --live           # 调 API 实时检索
  python scripts/eval_all.py --suite generation                 # 调 API 生成评测
  python scripts/eval_all.py --suite all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragm.eval.dataset import load_baseline, load_golden, load_hits
from ragm.eval.metrics import gate
from ragm.eval.runner import generation_eval, live_hits, render_markdown, retrieval_by_suite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["retrieval", "generation", "all"], default="retrieval")
    ap.add_argument("--fixture-only", action="store_true", help="只用冻结快照（CI 模式）")
    ap.add_argument("--live", action="store_true", help="调用 API 实时检索")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="reports/eval_report.md")
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()

    golden = load_golden()
    report: dict = {"golden": len(golden)}
    retrieval = None
    if args.suite in ("retrieval", "all"):
        if args.live:
            hit_map = live_hits(golden, args.api, k=args.k)
            mode = "live"
        else:
            hit_map = load_hits()
            mode = "fixture"
        retrieval = retrieval_by_suite(golden, hit_map, k=args.k)
        report["retrieval"] = retrieval
        report["retrieval_mode"] = mode
        baseline = load_baseline()
        ok, detail = gate(retrieval["all"], baseline.get("all", {}), f"ndcg@{args.k}")
        report["gate"] = {"ok": ok, **detail}
        print(json.dumps({"mode": mode, "all": retrieval["all"], "gate": report["gate"]}, ensure_ascii=False))

    generation = None
    if args.suite in ("generation", "all"):
        generation = generation_eval(golden, args.api)
        report["generation"] = {k: v for k, v in generation.items() if k != "rows"}
        print(json.dumps(report["generation"], ensure_ascii=False))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(retrieval, generation), encoding="utf-8")
    (ROOT / "reports" / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if retrieval and not args.no_gate and not report["gate"]["ok"]:
        print("GATE FAILED:", json.dumps(report["gate"], ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()