"""默认入库执行器：复用 scripts/ingest_docs.py 的 build + embed_chunks。

队列只负责调度，解析/切块/编码逻辑仍留在既有脚本；这里按文件路径加载脚本，
避免把 scripts 变成包或复制一份实现。测试通过 tick(ingest_fn=...) 注入 fake，
不会触发 Milvus / GPU；真正执行时才惰性导入 Embedder/MilvusStore。
"""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from ..config import get_settings

INGEST_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "ingest_docs.py"
DOCS_SUBDIR = "docs"
DEFAULT_SOURCE = "docs"
DEFAULT_LANG = "zh"
MODULE_NAME = "ragm_jobs_ingest_script"


@lru_cache(maxsize=1)
def _load_ingest_script() -> ModuleType:
    if not INGEST_SCRIPT.exists():
        raise FileNotFoundError(f"ingest script not found: {INGEST_SCRIPT}")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, INGEST_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load ingest script: {INGEST_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    # 注册后再 exec，脚本内部若使用 dataclass/typing 自引用可正常解析。
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def default_ingest(payload: dict[str, Any]) -> dict[str, Any]:
    """执行一个 ingest payload：{src, collection, recreate}。"""
    src_value = str(payload.get("src") or "").strip()
    if not src_value:
        raise ValueError("ingest payload 缺少 src")
    src = Path(src_value).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"src not found: {src}")
    settings = get_settings()
    collection = str(payload.get("collection") or settings.milvus_collection)
    recreate = bool(payload.get("recreate", False))
    out = settings.data_root / "data" / DOCS_SUBDIR
    script = _load_ingest_script()
    build_report = script.build(src, out, lang=DEFAULT_LANG, source=DEFAULT_SOURCE, recreate=recreate)
    embed_report = script.embed_chunks(
        out,
        collection,
        recreate,
        changed_files=build_report.get("changed_files", []),
        stale_doc_ids=build_report.get("stale_doc_ids", []),
        source=DEFAULT_SOURCE,
    )
    return {"build": build_report, "embed": embed_report}
