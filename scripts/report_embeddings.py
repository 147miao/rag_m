"""把 embedding_benchmark.json 渲染成 markdown 报告。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "embedding_benchmark.json"
OUT = ROOT / "reports" / "embedding_benchmark.md"


def main() -> None:
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    lines = ["# Embedding 模型横向对比", ""]
    lines.append("评测协议：同一 corpus / 同一 query / 同一 HNSW 检索（精确 top-50），L2 归一化 + COSINE；")
    lines.append("max_seq_length=512，batch=32，bf16；指标为段落级（EN: HotpotQA 37,368 段 / 300 query；ZH: CMRC 2,403 篇 / 100 query）。")
    lines.append("")
    for lang, title in (("en", "英文（HotpotQA）"), ("zh", "中文（CMRC2018）")):
        subset = [r for r in rows if r.get("lang") == lang]
        if not subset:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| 模型 | 维度 | R@5 | R@10 | R@20 | nDCG@10 | MRR@10 | docs/s | 峰值显存 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(subset, key=lambda x: -(x.get("ndcg@10") or 0)):
            lines.append(
                f"| {r['model']} | {r.get('dim','-')} | {r.get('recall@5','-')} | {r.get('recall@10','-')} | "
                f"{r.get('recall@20','-')} | {r.get('ndcg@10','-')} | {r.get('mrr@10','-')} | "
                f"{r.get('docs_per_s','-')} | {r.get('peak_vram_gib','-')} GiB |"
            )
        lines.append("")
        best = max(subset, key=lambda x: (x.get("ndcg@10") or 0))
        lines.append(f"**{title} 最佳**：{best['model']}（nDCG@10 {best.get('ndcg@10')}，R@10 {best.get('recall@10')}）")
        lines.append("")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
