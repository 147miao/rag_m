# 代码库智能分析（code-insight v1，2026-11-10）

## 能力
- 索引本地 Python 仓库：`ast` 解析模块/类/函数/方法/嵌套/async/decorator，记录 docstring、行号、imports、名称引用；
- 函数/类级 chunk（超长按 1500 字/120 行切分），SQLite **FTS5**（bm25）检索 + path/kind 过滤，天然幂等、无 GPU 依赖；
- 符号精确查找、仓库概览、**文件级反向 import**（启发式，非精确调用链）；
- `/api/code/ask`：FTS 检索 top-k 代码上下文 → LLM 基于上下文回答，引用 `[n]` 映射到 `path:start-end`，经 `verify_code_citations` 校验；无证据/引用无效时拒答或抽取式兜底；
- 与文档 RAG 双模式共存，互不影响。

## 快速开始
```bash
# 索引一个仓库（默认 data/code_index.sqlite）
python scripts/index_code.py --root /path/to/repo --name myrepo --reindex

# 服务
curl -s localhost:8000/api/code/repos
curl -s "localhost:8000/api/code/search?q=ToolExecutor&repo_id=repo_xxx&k=5"
curl -s "localhost:8000/api/code/symbol?repo_id=repo_xxx&qualname=ToolExecutor"
curl -s "localhost:8000/api/code/impact?repo_id=repo_xxx&path=src/ragm/agent/tools.py"
curl -s -X POST localhost:8000/api/code/ask -H "Content-Type: application/json" \
  -d '{"question":"ToolExecutor run 如何分发 vector_search","repo_id":"repo_xxx"}'
```

## 设计
| 模块 | 职责 |
|---|---|
| `code/scanner.py` | 递归扫描 .py、排除目录、allowed_roots/符号链接校验、file_hash |
| `code/parser.py` | ast → symbols/imports/references；语法错误跳过 |
| `code/chunker.py` | 符号级 chunk、行号、确定性 chunk_id |
| `code/store.py` | SQLite（repos/files/symbols/imports/references/chunks + FTS5） |
| `code/indexer.py` | 增量索引（改/删文件同步）、repo 统计 |
| `code/service.py` | search/get_symbol/overview/impact/ask + 引用校验 |
| `api/code_routes.py` | `/api/code/*` 端点 |
| `agent/code_tools.py` | search_code/get_symbol/impact_analysis/repo_overview 工具定义 |

## 实测（rag_m 自举）
- 索引：132–143 文件、1.1–1.3k 符号、1.3k chunks、约 1.5s（增量约 11ms）；
- 检索：10 个标识符类问题 hit@5 = 10/10（`tests/test_code_e2e.py`）；
- 问答：10 题真实 LLM，引用 path 全部存在（refusal 情况已通过查询扩展 + 抽取式兜底收敛）。

## 限制
- 仅 Python（TS/Go 及树莓派语言的 tree-sitter 解析留 v2）；
- FTS5 unicode61 不切分中文：中文自然语言问题检索弱，建议问题中包含标识符/路径；v2 接 dense embedding；
- impact 只做文件级反向 import，不承诺调用链/数据流；
- 单文件 >1MB 或 >5000 行跳过；>5000 文件暂同步索引。
