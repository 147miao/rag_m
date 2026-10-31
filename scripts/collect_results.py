"""汇总实验结果为一张 markdown 表 + JSON。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(name: str):
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    rows = []
    for p in sorted(REPORTS.glob("eval_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": p.stem.removeprefix("eval_"),
                "arch": d.get("arch", "-"),
                "variant": d.get("variant", "-"),
                "model": Path(d.get("model", "")).name,
                "acc": d.get("accuracy"),
                "macro_f1": d.get("macro_f1"),
                "ece": d.get("ece"),
                "p50_ms": d.get("p50_ms"),
                "p95_ms": d.get("p95_ms"),
                "vram_gib": d.get("peak_vram_gib"),
                "recall_simple": d.get("per_class_recall", {}).get("simple"),
                "recall_multihop": d.get("per_class_recall", {}).get("multihop"),
                "recall_reasoning": d.get("per_class_recall", {}).get("reasoning"),
            }
        )

    baselines = load("baselines.json") or {}
    for name, d in baselines.items():
        rows.append(
            {
                "name": name,
                "arch": "baseline",
                "variant": "-",
                "model": "-",
                "acc": d.get("accuracy"),
                "macro_f1": d.get("macro_f1"),
                "ece": None,
                "p50_ms": None,
                "p95_ms": None,
                "vram_gib": None,
                "recall_simple": d.get("per_class_recall", {}).get("simple"),
                "recall_multihop": d.get("per_class_recall", {}).get("multihop"),
                "recall_reasoning": d.get("per_class_recall", {}).get("reasoning"),
            }
        )

    rows.sort(key=lambda r: (r["macro_f1"] is None, -(r["macro_f1"] or 0)))
    (REPORTS / "classifier_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(v, nd=3):
        return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    lines = [
        "# 路由分类器实验结果",
        "",
        "| 排名 | 实验 | 架构 | 变体 | 模型 | macro-F1 | Acc | ECE | p50(ms) | p95(ms) | 显存(GB) | R(simple/multi/reason) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        rec = "/".join(fmt(r[k], 2) for k in ("recall_simple", "recall_multihop", "recall_reasoning"))
        lines.append(
            f"| {i} | {r['name']} | {r['arch']} | {r['variant']} | {r['model']} | {fmt(r['macro_f1'])} | "
            f"{fmt(r['acc'])} | {fmt(r['ece'])} | {fmt(r['p50_ms'], 1)} | {fmt(r['p95_ms'], 1)} | {fmt(r['vram_gib'], 2)} | {rec} |"
        )
    (REPORTS / "classifier_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
