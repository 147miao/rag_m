"""实验/任务进度汇总：CLI 与 API 共用。"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = Path("/run/media/myj/D/rag_m/models/classifier")
DATA_DIR = Path("/run/media/myj/D/rag_m/data/classifier")
LOG_DIR = ROOT / "reports" / "train_logs"
EVAL_DIR = ROOT / "reports"
TQDM_RE = re.compile(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]")


def _pgrep(pattern: str) -> list[str]:
    try:
        out = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True, timeout=5, check=False).stdout
    except Exception:  # noqa: BLE001
        return []
    return [line for line in out.splitlines() if "pgrep -af" not in line and "status.py" not in line]


def _gpu() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip().splitlines()[0]
        used, total, util = [int(x.strip()) for x in out.split(",")]
        return {"used_mib": used, "total_mib": total, "util_pct": util}
    except Exception:  # noqa: BLE001
        return {"used_mib": None, "total_mib": None, "util_pct": None}


def _log_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    tail = path.read_bytes()[-8000:].decode("utf-8", "ignore")
    matches = TQDM_RE.findall(tail)
    if not matches:
        return {"raw_tail": tail.strip().splitlines()[-1][:120] if tail.strip() else ""}
    pct, step, total, timing = matches[-1]
    parts = timing.split("<")
    return {
        "percent": int(pct),
        "step": int(step),
        "total_steps": int(total),
        "elapsed": parts[0].strip() if parts else "",
        "eta": parts[1].strip() if len(parts) > 1 else "",
    }


def _eval_summary(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return {
        "accuracy": d.get("accuracy"),
        "macro_f1": d.get("macro_f1"),
        "p50_ms": d.get("p50_ms"),
        "test_data": Path(d.get("test_data", "")).name,
    }


def collect_status() -> dict:
    running = {
        "training": _pgrep("train_route_cls.py"),
        "curation": _pgrep("curate_test.py"),
        "baselines": _pgrep("baselines.py"),
        "matrix": _pgrep("run_matrix.sh"),
    }
    runs = []
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir()):
            if not d.is_dir():
                continue
            meta = {}
            if (d / "meta.json").exists():
                try:
                    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    meta = {}
            evals = {}
            for f in EVAL_DIR.glob(f"eval_{d.name}*.json"):
                evals[f.stem.replace(f"eval_{d.name}", "") or "default"] = _eval_summary(f)
            runs.append(
                {
                    "name": d.name,
                    "arch": meta.get("arch"),
                    "variant": meta.get("variant"),
                    "model": Path(meta.get("model", "")).name if meta.get("model") else None,
                    "dev_macro_f1": (meta.get("dev_metrics") or {}).get("macro_f1"),
                    "adapter": (d / "adapter").exists(),
                    "evals": evals,
                    "progress": _log_progress(LOG_DIR / f"{d.name}.log"),
                }
            )
    data = {}
    for name in ("train", "dev", "test_weak", "test_human", "test_curated"):
        p = DATA_DIR / f"{name}.jsonl"
        if p.exists():
            data[name] = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": _gpu(),
        "running": {k: v for k, v in running.items() if v},
        "runs": runs,
        "data": data,
        "reports": {
            "baselines": (EVAL_DIR / "baselines_human.json").exists(),
            "matrix": bool(_pgrep("run_matrix.sh")),
        },
    }


def render_markdown(status: dict) -> str:
    lines = [f"# 实验进度 · {status['generated_at']}", ""]
    gpu = status["gpu"]
    lines.append(f"GPU: {gpu['used_mib']} / {gpu['total_mib']} MiB, util {gpu['util_pct']}%")
    if status["running"]:
        for kind, procs in status["running"].items():
            lines.append(f"- running **{kind}**: {len(procs)} 个进程")
    else:
        lines.append("- 当前无运行中的任务")
    lines += ["", "| 实验 | 架构 | 变体 | dev F1 | 进度 | 评测(test) |", "|---|---|---|---|---|---|"]
    for r in status["runs"]:
        prog = r["progress"]
        p = f"{prog.get('percent','?')}% ({prog.get('step','?')}/{prog.get('total_steps','?')}, ETA {prog.get('eta','?')})" if prog.get("percent") is not None else "-"
        ev = "; ".join(f"{k}: F1 {v.get('macro_f1',0):.3f}" for k, v in r["evals"].items()) if r["evals"] else "-"
        lines.append(f"| {r['name']} | {r['arch'] or '-'} | {r['variant'] or '-'} | {r['dev_macro_f1']:.3f} | {p} | {ev} |" if r["dev_macro_f1"] else f"| {r['name']} | {r['arch'] or '-'} | {r['variant'] or '-'} | - | {p} | {ev} |")
    return "\n".join(lines) + "\n"