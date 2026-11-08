"""文档读取：PDF / DOCX / EPUB / Markdown / HTML / TXT → 章节列表。"""
from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".docx", ".epub", ".md", ".markdown", ".txt", ".html", ".htm"}


@dataclass
class Section:
    text: str
    page: int | None = None
    section: str = ""
    kind: str = "text"
    meta: dict = field(default_factory=dict)


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf(path: Path) -> list[Section]:
    reader = PdfReader(str(path))
    out: list[Section] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = _clean(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            text = ""
        if text:
            out.append(Section(text=text, page=i, section=f"第 {i} 页"))
    return out


def read_docx(path: Path) -> list[Section]:
    doc = Document(str(path))
    out: list[Section] = []
    stack: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            text = _clean("\n".join(buf))
            if text:
                out.append(Section(text=text, section=" / ".join(stack) or "正文"))
            buf.clear()

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "") if para.style else ""
        m = re.match(r"Heading (\d)", style)
        if m:
            level = int(m.group(1))
            flush()
            stack = stack[: level - 1] + [text]
        else:
            buf.append(text)
    flush()
    return out


def _html_sections(html: str, fallback_title: str) -> list[Section]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    out: list[Section] = []
    current = fallback_title
    buf: list[str] = []

    def flush() -> None:
        text = _clean("\n".join(buf))
        if text:
            out.append(Section(text=text, section=current))
        buf.clear()

    for node in soup.find_all(["h1", "h2", "h3", "p", "li", "div"]):
        if node.name in {"h1", "h2", "h3"}:
            flush()
            current = node.get_text(" ", strip=True)[:80] or fallback_title
        elif node.name in {"p", "li"}:
            text = node.get_text(" ", strip=True)
            if text:
                buf.append(text)
    flush()
    return out


def read_epub(path: Path) -> list[Section]:
    out: list[Section] = []
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
        names.sort()
        for name in names:
            try:
                html = zf.read(name).decode("utf-8", "ignore")
            except Exception:
                logger.warning("epub chapter decode failed: %s", name, exc_info=True)
                continue
            out.extend(_html_sections(html, Path(name).stem))
    return out


def read_markdown(path: Path) -> list[Section]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[Section] = []
    stack: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        text = _clean("\n".join(buf))
        if text:
            out.append(Section(text=text, section=" / ".join(stack) or "正文"))
        buf.clear()

    for line in lines:
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            flush()
            level = len(m.group(1))
            stack = stack[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return out


def read_text(path: Path) -> list[Section]:
    return [Section(text=_clean(path.read_text(encoding="utf-8", errors="ignore")), section="正文")]


def read_html(path: Path) -> list[Section]:
    return _html_sections(path.read_text(encoding="utf-8", errors="ignore"), path.stem)


def read_file(path: Path) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".epub":
        return read_epub(path)
    if suffix in {".md", ".markdown"}:
        return read_markdown(path)
    if suffix in {".html", ".htm"}:
        return read_html(path)
    if suffix == ".txt":
        return read_text(path)
    raise ValueError(f"unsupported suffix: {suffix}")