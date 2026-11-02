"""清洗萌娘百科文本：去掉导航/信息框/引用噪声，只保留正文散文。"""
from __future__ import annotations

import json
import re
from pathlib import Path

IN = Path("/run/media/myj/D/rag_m/data/naruto/moegirl.jsonl")
OUT = Path("/run/media/myj/D/rag_m/data/naruto/moegirl_clean.jsonl")

JUNK = re.compile(
    r"^(跳转到正文|Skip to content|简体（中国）|清除页面缓存|贡献者：|欢迎正在阅读这个条目的您协助编辑本条目|编辑前请阅读Wiki入门或条目编辑规范|萌娘百科带您|目录|展开|收起|编辑|添加义项|词条图册|参考资料|外部链接|分类：|__NOTOC__)"
)
STOP = re.compile(r"^(注释|参考资料|外部链接|词条图册|此页面最后编辑|分类|评论|查看全部|回到顶部|展开|折叠|目录|导航|萌娘百科)")
QUOTE_LINE = re.compile(r"^[“”「」『』\-—\s]*[“「]")
NAV_WORDS = re.compile(r"(编辑本条目|协助编辑|消歧义|萌娘百科|Wiki入门|条目编辑规范|跳转到正文|此页面最后编辑)")


def clean(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    kept: list[str] = []
    for ln in lines:
        if STOP.match(ln):
            break
        if JUNK.match(ln) or QUOTE_LINE.match(ln) or NAV_WORDS.search(ln):
            continue
        kept.append(ln)
    start = 0
    for i, ln in enumerate(kept):
        if len(ln) >= 40 and re.search(r"[。！？]$", ln):
            start = i
            break
    kept = kept[start:]
    kept = [ln for ln in kept if len(ln) >= 18 or re.search(r"[。！？…；：]$", ln)]
    body = re.sub(r"\s{2,}", " ", " ".join(kept))
    sentences = re.split(r"(?<=[。！？…])", body)
    out = [s.strip() for s in sentences if len(s.strip()) >= 18 and not NAV_WORDS.search(s)]
    return "".join(out).strip()[:20000]


def main() -> None:
    rows = [json.loads(line) for line in IN.read_text(encoding="utf-8").splitlines() if line.strip()]
    cleaned = []
    kept_chars = 0
    for r in rows:
        body = clean(r["text"])
        if len(body) < 300:
            continue
        cleaned.append({"title": r["title"], "url": r.get("url", ""), "text": body, "chars": len(body)})
        kept_chars += len(body)
    OUT.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cleaned), encoding="utf-8")
    print(json.dumps({"docs_in": len(rows), "docs_out": len(cleaned), "chars": kept_chars}, ensure_ascii=False))


if __name__ == "__main__":
    main()