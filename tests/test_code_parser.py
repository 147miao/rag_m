"""解析器测试：真实 graph.py 结构、嵌套/async/decorator、import/reference、语法错误。"""
from __future__ import annotations

from pathlib import Path

from ragm.code.parser import MODULE_QUALNAME, ModuleInfo, parse_file, parse_source

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "src" / "ragm" / "rag" / "graph.py"

NESTED_SOURCE = '''"""module doc."""

import os
from . import sibling
from ..pkg import mod as m


def outer(x):
    """outer doc."""
    @decorator
    async def inner():
        class Local:
            def method(self):
                return os.path

        return Local

    return inner


@staticmethod
class Meta:
    @classmethod
    async def make(cls):
        return m
'''


def _by_qualname(info: ModuleInfo) -> dict:
    return {symbol.qualname: symbol for symbol in info.symbols}


def test_parse_real_graph_structure_and_line_numbers() -> None:
    info = parse_file(GRAPH_PATH, path_label="src/ragm/rag/graph.py")
    assert info.error is None
    assert info.path == "src/ragm/rag/graph.py"
    source_lines = GRAPH_PATH.read_text(encoding="utf-8").splitlines()
    symbols = _by_qualname(info)

    module = symbols[MODULE_QUALNAME]
    assert module.kind == "module"
    assert module.parent is None
    assert (module.start_line, module.end_line) == (1, len(source_lines))
    assert module.docstring is not None and module.docstring.startswith("LangGraph 生成链路")

    rag_graph = symbols["RagGraph"]
    assert rag_graph.kind == "class"
    assert rag_graph.parent == MODULE_QUALNAME
    assert (rag_graph.start_line, rag_graph.end_line) == (121, 352)
    assert source_lines[rag_graph.start_line - 1].startswith("class RagGraph:")

    invoke = symbols["RagGraph.invoke"]
    assert invoke.kind == "method"
    assert invoke.parent == "RagGraph"
    assert (invoke.start_line, invoke.end_line) == (311, 324)
    assert source_lines[invoke.start_line - 1].strip().startswith("def invoke(")

    parse_fn = symbols["parse_structured_answer"]
    assert parse_fn.kind == "function"
    assert (parse_fn.start_line, parse_fn.end_line) == (60, 81)

    imports = {(item.module, item.level, item.lineno) for item in info.imports}
    assert ("logging", 0, 8) in imports
    assert ("langgraph.graph", 0, 13) in imports
    assert ("citations", 1, 18) in imports

    references = {(item.name, item.lineno) for item in info.references}
    assert ("verify_citations", 275) in references


def test_parse_nested_async_decorated_symbols() -> None:
    info = parse_source("pkg/nested.py", NESTED_SOURCE)
    assert info.error is None
    symbols = _by_qualname(info)

    outer = symbols["outer"]
    assert outer.kind == "function" and outer.parent == MODULE_QUALNAME
    assert outer.docstring == "outer doc."
    assert (outer.start_line, outer.end_line) == (8, 18)

    inner = symbols["outer.inner"]
    assert inner.kind == "function" and inner.is_async is True
    assert inner.parent == "outer"
    assert inner.decorators == ["decorator"]
    assert inner.start_line == 10  # decorator 行
    assert inner.code.startswith("    @decorator")

    local = symbols["outer.inner.Local"]
    assert local.kind == "class" and local.parent == "outer.inner"
    method = symbols["outer.inner.Local.method"]
    assert method.kind == "method" and method.parent == "outer.inner.Local"

    meta = symbols["Meta"]
    assert meta.kind == "class" and meta.decorators == ["staticmethod"]
    assert meta.start_line == 21
    make = symbols["Meta.make"]
    assert make.kind == "method" and make.is_async is True
    assert make.decorators == ["classmethod"]
    assert make.parent == "Meta"

    imports = {(item.module, item.level, item.lineno) for item in info.imports}
    assert ("os", 0, 3) in imports
    assert ("", 1, 4) in imports
    assert ("pkg", 2, 5) in imports

    references = {(item.name, item.lineno) for item in info.references}
    assert ("os", 14) in references
    assert ("m", 25) in references
    assert ("decorator", 10) in references


def test_syntax_error_returns_error_and_skips_symbols(tmp_path: Path) -> None:
    info = parse_source("bad.py", "def broken(:\n")
    assert info.error is not None and "SyntaxError" in info.error
    assert info.symbols == [] and info.imports == []

    bad_file = tmp_path / "bad.py"
    bad_file.write_text("class (:\n", encoding="utf-8")
    from_file = parse_file(bad_file, path_label="bad.py")
    assert from_file.error is not None and "SyntaxError" in from_file.error
    assert from_file.symbols == []


def test_empty_module_still_has_module_symbol() -> None:
    info = parse_source("empty.py", "")
    assert info.error is None
    assert len(info.symbols) == 1
    module = info.symbols[0]
    assert (module.qualname, module.kind) == (MODULE_QUALNAME, "module")
    assert (module.start_line, module.end_line) == (1, 1)
    assert module.code == ""
    assert module.docstring is None
