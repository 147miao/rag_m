"""评测数据加载：golden 与冻结检索快照。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_golden(path: Path | None = None) -> list[dict]:
    return load_jsonl(path or FIXTURES / "golden.jsonl")


def load_hits(path: Path | None = None) -> dict[str, list[str]]:
    rows = load_jsonl(path or FIXTURES / "hits.jsonl")
    return {r["id"]: r["ranked"] for r in rows}


def load_baseline(path: Path | None = None) -> dict:
    p = path or FIXTURES / "eval_baseline.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}