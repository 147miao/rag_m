"""代码问答 E2E：调用运行中的 /api/code/ask，校验引用 path/行号并输出报告。

用法（需先启动 API，并把 rag_m 索引进代码库）：
    python scripts/code_e2e.py --base-url http://127.0.0.1:8000 --repo-name rag_m

输出：
    reports/code_e2e.md   人读报告
    reports/code_e2e.json 机器可读结果

退出码：0 = 无 refusal 且所有 citation path/行范围有效；1 = 存在 refusal/无效引用。
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests" / "code_e2e_cases.json"
DEFAULT_MD = ROOT / "reports" / "code_e2e.md"
DEFAULT_JSON = ROOT / "reports" / "code_e2e.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="代码问答 E2E（真实 /api/code/ask）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API 根地址")
    parser.add_argument("--repo-id", default="", help="仓库 ID；省略时按 --repo-name 查找")
    parser.add_argument("--repo-name", default="rag_m", help="仓库名（默认 rag_m）")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="fixtures JSON")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD, help="Markdown 输出路径")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON, help="JSON 输出路径")
    parser.add_argument("--timeout", type=float, default=180.0, help="单请求超时秒数")
    return parser


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"fixtures 必须是非空 JSON 数组: {path}")
    return [item for item in data if isinstance(item, dict)]


def find_repo(client: httpx.Client, repo_id: str, repo_name: str) -> dict[str, Any]:
    if repo_id:
        resp = client.get(f"/api/code/repos/{repo_id}")
        resp.raise_for_status()
        return dict(resp.json())
    resp = client.get("/api/code/repos")
    resp.raise_for_status()
    repos = list(resp.json().get("repos") or [])
    for repo in repos:
        if str(repo.get("name")) == repo_name:
            return dict(repo)
    if len(repos) == 1:
        return dict(repos[0])
    names = ", ".join(str(repo.get("name")) for repo in repos) or "<none>"
    raise RuntimeError(f"未找到 repo_name={repo_name!r}；现有仓库: {names}")


def file_line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def validate_citations(citations: list[dict[str, Any]], repo_root: Path) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for citation in citations:
        raw_path = str(citation.get("path") or "")
        candidate = Path(raw_path).expanduser()
        abs_path = candidate if candidate.is_absolute() else repo_root / candidate
        total = file_line_count(abs_path)
        start = int(citation.get("start_line") or 0)
        end = int(citation.get("end_line") or 0)
        valid = abs_path.is_file() and start >= 1 and end >= start and end <= total
        details.append(
            {
                "n": citation.get("n"),
                "path": raw_path,
                "start_line": start,
                "end_line": end,
                "exists": abs_path.is_file(),
                "line_count": total,
                "valid": valid,
            }
        )
    return details


def ranges_overlap(start: int, end: int, expected: list[int]) -> bool:
    if len(expected) != 2:
        return False
    exp_start, exp_end = int(expected[0]), int(expected[1])
    return start <= exp_end and exp_start <= end


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# code-insight E2E 报告",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- base_url: {summary['base_url']}",
        f"- repo: {summary['repo_name']} ({summary['repo_id']})",
        f"- repo_root: {summary['repo_root']}",
        f"- questions: {summary['questions']}",
        f"- refusal_count: {summary['refusal_count']}",
        f"- citation_valid_rate: {summary['citation_valid_rate']}",
        f"- expected_path_hit_rate: {summary['expected_path_hit_rate']}",
        f"- p50_ms: {summary['p50_ms']}",
        f"- avg_ms: {summary['avg_ms']}",
        "",
        "| # | question | refusal | citations | valid | expected path | ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary["results"]:
        question = str(row["question"]).replace("|", "\\|")
        lines.append(
            f"| {row['n']} | {question} | {row['refusal']} | {row['citations']} | "
            f"{row['citations_valid']} | {row['expected_path_hit']} | {row['latency_ms']} |"
        )
    lines.extend(["", "## 引用明细", ""])
    for row in summary["results"]:
        lines.append(f"### {row['n']}. {row['question']}")
        if row["error"]:
            lines.append(f"- error: {row['error']}")
        if row["refusal"]:
            lines.append(f"- refusal: {row['refusal_reason']}")
        if not row["citation_details"]:
            lines.append("- citations: <none>")
        for detail in row["citation_details"]:
            mark = "OK" if detail["valid"] else "INVALID"
            lines.append(
                f"- [{detail['n']}] {detail['path']}:{detail['start_line']}-{detail['end_line']} "
                f"({mark}; file_lines={detail['line_count']})"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    base_url = args.base_url.rstrip("/")
    cases = load_cases(args.cases)
    # 本机 API 不走环境代理（CI/本机可能设置格式异常的 ALL_PROXY）。
    with httpx.Client(base_url=base_url, timeout=args.timeout, trust_env=False) as client:
        repo = find_repo(client, args.repo_id, args.repo_name)
        repo_root = Path(str(repo.get("root") or "")).expanduser()
        if not repo_root:
            raise RuntimeError(f"repo {repo.get('repo_id')} 缺少 root")
        results: list[dict[str, Any]] = []
        for index, case in enumerate(cases, start=1):
            question = str(case.get("question") or "")
            started = time.perf_counter()
            error = ""
            payload: dict[str, Any] = {}
            try:
                resp = client.post("/api/code/ask", json={"question": question, "repo_id": repo["repo_id"]})
                resp.raise_for_status()
                payload = dict(resp.json())
            except Exception as exc:  # noqa: BLE001 - 单题失败记录后继续
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            refusal = payload.get("refusal")
            citations = [item for item in (payload.get("citations") or []) if isinstance(item, dict)]
            details = validate_citations(citations, repo_root)
            citations_valid = bool(details) and all(item["valid"] for item in details)
            expected_paths = [str(item) for item in (case.get("expect_paths") or [])]
            expected_path_hit = any(
                any(expected in detail["path"] for expected in expected_paths) for detail in details
            )
            expected_line_range = case.get("expect_line_range")
            expected_line_covered = None
            if expected_line_range:
                expected_line_covered = any(
                    any(expected in detail["path"] for expected in expected_paths)
                    and ranges_overlap(detail["start_line"], detail["end_line"], expected_line_range)
                    for detail in details
                )
            results.append(
                {
                    "n": index,
                    "question": question,
                    "refusal": bool(refusal),
                    "refusal_reason": (refusal or {}).get("reason", "") if isinstance(refusal, dict) else "",
                    "citations": len(citations),
                    "citation_details": details,
                    "citations_valid": citations_valid,
                    "expected_path_hit": expected_path_hit,
                    "expected_line_range": expected_line_range,
                    "expected_line_covered": expected_line_covered,
                    "provider": payload.get("provider"),
                    "latency_ms": latency_ms,
                    "error": error,
                    "lead": str((payload.get("answer") or {}).get("lead") or "")[:160],
                }
            )
            print(
                f"[{index}/{len(cases)}] {question} | refusal={bool(refusal)} "
                f"citations={len(citations)} valid={citations_valid} "
                f"expected={expected_path_hit} {latency_ms}ms",
                flush=True,
            )

    latencies = [row["latency_ms"] for row in results]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "repo_id": repo["repo_id"],
        "repo_name": repo.get("name", ""),
        "repo_root": str(repo_root),
        "questions": len(results),
        "refusal_count": sum(1 for row in results if row["refusal"]),
        "citation_valid_rate": round(
            sum(1 for row in results if row["citations_valid"]) / max(1, len(results)), 4
        ),
        "expected_path_hit_rate": round(
            sum(1 for row in results if row["expected_path_hit"]) / max(1, len(results)), 4
        ),
        "p50_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "avg_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
        "results": results,
    }
    write_markdown(args.out_md, summary)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("repo_id", "questions", "refusal_count", "citation_valid_rate", "expected_path_hit_rate", "p50_ms", "avg_ms")}, ensure_ascii=False), flush=True)
    print(f"reports: {args.out_md}, {args.out_json}", flush=True)
    return 0 if summary["refusal_count"] == 0 and summary["citation_valid_rate"] == 1.0 else 1


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
