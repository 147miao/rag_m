"""引用解析与校验。"""
from __future__ import annotations

import re

CITE_RE = re.compile(r"\[(\d{1,2})\]")


def parse_citations(text: str) -> list[int]:
    seen: list[int] = []
    for m in CITE_RE.finditer(text or ""):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def verify_citations(text: str, n_evidence: int) -> tuple[list[int], list[int], str]:
    """返回 (有效引用, 无效引用, 校验文案)。"""
    cited = parse_citations(text)
    valid = [n for n in cited if 1 <= n <= n_evidence]
    invalid = [n for n in cited if n < 1 or n > n_evidence]
    if not cited:
        return [], [], "0/0 通过（无引用）"
    return valid, invalid, f"{len(valid)}/{len(cited)} 通过"
