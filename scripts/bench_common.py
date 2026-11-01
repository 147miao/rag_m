"""基准测试公共设施：GPU 串行锁、候选缓存、资源监控、API 服务管理。"""
from __future__ import annotations

import contextlib
import fcntl
import json
import subprocess
import threading
import time
from pathlib import Path

CACHE = Path("/run/media/myj/D/rag_m/cache/bench")
GPU_LOCK = CACHE / "gpu.lock"


@contextlib.contextmanager
def gpu_lock():
    CACHE.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK.open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def stop_api() -> None:
    subprocess.run(["systemctl", "--user", "stop", "ragm-api.service"], capture_output=True)


def start_api() -> None:
    subprocess.run(["systemctl", "--user", "start", "ragm-api.service"], capture_output=True)


def candidate_path(collection: str, query: str, mode: str, k: int) -> Path:
    import hashlib
    key = hashlib.sha1((collection + "|" + query + "|" + mode + "|" + str(k)).encode()).hexdigest()[:16]
    return CACHE / "candidates" / (key + ".json")


def cached_hits(collection: str, query: str, mode: str, k: int, compute):
    path = candidate_path(collection, query, mode, k)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    hits = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hits, ensure_ascii=False), encoding="utf-8")
    return hits


class UsageMonitor:
    def __init__(self) -> None:
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "UsageMonitor":
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=2)
        if not self.samples:
            return {}
        cpu = [s["cpu"] for s in self.samples]
        gpu = [s["gpu"] for s in self.samples if s.get("gpu") is not None]
        return {
            "cpu_avg": round(sum(cpu) / len(cpu), 1),
            "cpu_max": round(max(cpu), 1),
            "gpu_avg": round(sum(gpu) / len(gpu), 1) if gpu else None,
            "gpu_max": round(max(gpu), 1) if gpu else None,
            "samples": len(self.samples),
        }

    def _run(self) -> None:
        prev = self._cpu_times()
        while not self._stop.is_set():
            time.sleep(1.0)
            cur = self._cpu_times()
            total = sum(cur) - sum(prev)
            idle = cur[3] - prev[3]
            cpu = 0.0 if total <= 0 else round((1 - idle / total) * 100, 1)
            prev = cur
            gpu = self._gpu_util()
            self.samples.append({"cpu": cpu, "gpu": gpu})

    @staticmethod
    def _cpu_times():
        with open("/proc/stat") as fh:
            parts = fh.readline().split()[1:]
        return [int(x) for x in parts[:8]]

    @staticmethod
    def _gpu_util():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout.strip().splitlines()[0]
            return int(out)
        except Exception:
            return None