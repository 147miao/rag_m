"""代码库分析 E2E（hermetic）：自举索引 rag_m，验证检索命中与符号行号。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragm.code.indexer import CodeIndexer
from ragm.code.store import CodeStore

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((Path(__file__).parent / "code_e2e_cases.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    db = tmp_path_factory.mktemp("code") / "index.sqlite"
    report = CodeIndexer(db, allowed_roots=[str(ROOT.parent)]).index(str(ROOT), "rag_m")
    store = CodeStore(db)
    return store, report["repo_id"], report


def test_self_index_has_symbols(indexed):
    store, repo_id, report = indexed
    assert report["files"] > 50
    assert report["symbols"] > 200
    assert report["chunks"] > 200
    assert store.get_repo(repo_id) is not None


@pytest.mark.parametrize("case", CASES, ids=[c["question"][:18] for c in CASES])
def test_search_hits_expected_path(indexed, case):
    store, repo_id, _ = indexed
    hits = store.search(case["question"], k=5, repo_id=repo_id)
    paths = [h["path"] for h in hits]
    assert any(expected in p for p in paths for expected in case["expect_paths"]), paths
    quals = [h["qualname"] for h in hits]
    assert any(q in qual or qual.startswith(q) for q in case.get("expect_qualnames", []) for qual in quals), quals


@pytest.mark.parametrize("case", CASES, ids=[c["question"][:18] for c in CASES])
def test_symbol_lookup_has_lines(indexed, case):
    store, repo_id, _ = indexed
    for qualname in case["expect_qualnames"]:
        row = store.get_symbol(repo_id, qualname)
        assert row is not None, qualname
        assert row["start_line"] >= 1 and row["end_line"] >= row["start_line"]
        assert row["path"].endswith(".py")
