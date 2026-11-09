#!/usr/bin/env bash
# 本地等价 CI：lint + 单测 + 前端构建（不依赖 GPU/Milvus/LLM）
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m pytest tests -q ${COV_ARGS:---cov=ragm --cov-report=term-missing}
(cd web && pnpm build >/dev/null)
echo "ci ok"
