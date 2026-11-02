"""全量语料（demo+火影+论文）单服务测试：三路问题 + 检索稀释评估。"""
from __future__ import annotations

import json
import time

import httpx

API = "http://127.0.0.1:8000/api/query"
CASES = [
    ("鸣人父亲是谁", ["波风水门"]),
    ("Which US documentary was made first: Scared Straight! or Soldier Girls?", ["Scared Straight"]),
    ("What metrics does RAGAS use?", ["faithfulness", "Faithfulness"]),
    ("自来也的弟子中谁成为了四代火影？", ["水门"]),
    ("What is the GraphRAG approach?", ["graph", "entity", "community"]),
]


def main() -> None:
    for q, gold in CASES:
        t0 = time.perf_counter()
        with httpx.Client(timeout=180) as c:
            r = c.post(API, json={"question": q})
            r.raise_for_status()
            d = r.json()
        lat = round((time.perf_counter() - t0) * 1000)
        ans = (d["answer"]["lead"] + " " + d["answer"]["body"]).strip()
        ok = any(g in ans for g in gold)
        srcs = sorted({e["dataset"] for e in d["evidence"][:8]})
        print(json.dumps({"q": q[:40], "ok": ok, "lat": lat, "route": d["route"], "cites": d["citations"], "check": d["stats"]["citationCheck"], "srcs": srcs}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()