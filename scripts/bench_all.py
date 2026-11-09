"""并行基准编排：GPU 任务与 LLM/网络任务重叠执行，监控 CPU/GPU 利用率。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import UsageMonitor, start_api, stop_api

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w")
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)


def wait(p: subprocess.Popen, name: str) -> int:
    code = p.wait()
    print(f"[{name}] exit={code}", flush=True)
    return code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-span", action="store_true")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--gen-tag", default="rerank_off")
    args = ap.parse_args()
    t0 = time.perf_counter()
    stop_api()
    time.sleep(2)
    env_api = dict(os.environ)
    env_api.update({
        "MILVUS_COLLECTION": "rag_naruto_chunks",
        "MILVUS_CHILD_COLLECTION": "rag_naruto_children",
        "PARENTS_PATH": "/run/media/myj/D/rag_m/data/naruto/parents.jsonl",
        "CLASSIFIER_BACKEND": "rules",
        "RERANK_ENABLED": "false",
    })
    api = subprocess.Popen(
        [str(ROOT / ".venv/bin/python"), "-m", "uvicorn", "ragm.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT), env=env_api, stdout=(ROOT / "logs/bench_api.log").open("w"), stderr=subprocess.STDOUT,
    )
    time.sleep(20)
    mon = UsageMonitor().start()
    procs = []
    if not args.skip_span:
        procs.append(("span", run([str(ROOT / ".venv/bin/python"), "-u", "scripts/naruto_span_matrix.py"], ROOT / "reports/bench_span.log")))
    if not args.skip_gen:
        procs.append(("gen", run([str(ROOT / ".venv/bin/python"), "-u", "scripts/naruto_generation_matrix.py", args.gen_tag], ROOT / f"reports/bench_gen_{args.gen_tag}.log")))
    for name, p in procs:
        wait(p, name)
    usage = mon.stop()
    api.terminate()
    try:
        api.wait(timeout=10)
    except subprocess.TimeoutExpired:
        api.kill()
    start_api()
    summary = {"wall_s": round(time.perf_counter() - t0, 1), "usage": usage}
    (ROOT / "reports/bench_usage.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()