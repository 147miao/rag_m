"""用 rubric 让 gemma4 重新标注 test_weak，产出 test_curated。

输出字段：id/question/label（rubric 标签）/weak_label/llm_label/agree/confidence。
用法：
    .venv/bin/python scripts/curate_test.py --in .../test_weak.jsonl --out .../test_curated.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "gemma4:31b-cloud"

RUBRIC = """你是 RAG 查询路由标注员。把问题分为三类，只输出 JSON：
{"label":"simple|multihop|reasoning","confidence":0.0-1.0,"why":"一句话"}

判定标准（按回答所需证据形态，不是按领域）：
- simple：单个段落即可回答，答案是事实/数值/定义；问题里通常只有一个实体，不需要串联。
- multihop：需要连接两个及以上实体或文档（bridge 或 comparison）；例如"A 的作者的出生地""A 和 B 有什么关系"。
- reasoning：需要多源聚合、比较、计算、规划或给建议，单次检索无法回答；例如"对比 X 与 Y 的优劣并给建议""为什么会出现 Z"。

边界规则：
1) 能用一个段落直接回答 → simple，即使需要一点常识；
2) 需要两个段落且关系明确 → multihop；
3) 需要 3 个以上来源，或需要比较/计算/权衡/给建议 → reasoning；
4) 拿不准时按更贵的策略标注（reasoning > multihop > simple）。

问题："""


def ask(question: str) -> dict:
    body = {
        "model": MODEL,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 120},
        "messages": [{"role": "user", "content": RUBRIC + question}],
    }
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    content = data["message"]["content"]
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        raise ValueError("no json in response: " + content[:200])
    out = json.loads(m.group(0))
    label = out.get("label")
    if label not in ("simple", "multihop", "reasoning"):
        raise ValueError("bad label: " + str(label))
    return {"llm_label": label, "llm_confidence": float(out.get("confidence", 0.0)), "why": out.get("why", "")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/run/media/myj/D/rag_m/data/classifier/test_weak.jsonl")
    ap.add_argument("--out", default="/run/media/myj/D/rag_m/data/classifier/test_curated.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.inp).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.out)
    done: list[dict] = []
    if out_path.exists():
        done = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    done_ids = {r["id"] for r in done}
    with out_path.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            if r["id"] in done_ids:
                continue
            try:
                res = ask(r["question"])
            except Exception as exc:  # noqa: BLE001
                print(f"[{i}] FAIL {r['id']}: {type(exc).__name__} {exc}")
                continue
            rec = {
                **r,
                "weak_label": r.get("label", ""),
                "label": res["llm_label"],
                "llm_label": res["llm_label"],
                "llm_confidence": res["llm_confidence"],
                "why": res["why"],
                "agree": res["llm_label"] == r["label"],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(rows)}] {r['label']} -> {rec['label']} ({res['llm_confidence']:.2f})")
            if args.sleep:
                time.sleep(args.sleep)
    total = sum(1 for _ in out_path.read_text(encoding="utf-8").splitlines())
    agree = sum(1 for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("agree"))
    print(f"done: {total} rows, agreement {agree}/{total} = {agree/max(1,total):.3f}")


if __name__ == "__main__":
    main()