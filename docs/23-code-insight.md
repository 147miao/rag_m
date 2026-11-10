# 代码库智能分析（code-insight v1，2026-11-10）

## 能力
- 索引本地 Python 仓库：`ast` 解析模块/类/函数/方法/嵌套/async/decorator，记录 docstring、行号、imports、名称引用；
- 函数/类级 chunk（超长按 1500 字/120 行切分），SQLite **FTS5**（bm25）检索 + path/kind 过滤，天然幂等、无 GPU 依赖；
- 符号精确查找、仓库概览、**文件级反向 import**（启发式，非精确调用链）；
- `/api/code/ask`：FTS 检索 top-k 代码上下文 → LLM 基于上下文回答，引用 `[n]` 映射到 `path:start-end`，经 `verify_code_citations` 校验；无证据/引用无效时拒答或抽取式兜底；
- 与文档 RAG 双模式共存，互不影响。

## 快速开始
```bash
# 索引一个仓库（默认 $DATA_ROOT/data/code_index.sqlite，与 API 共用；CODE_DB_PATH 可覆盖）
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

## Agent 工具（P1-2）
- `build_tool_schemas(include_code=False)` 默认保持原 3 个工具；`True` 追加 `search_code/get_symbol/impact_analysis/repo_overview`（共 7 个）；
- `ToolExecutor(code_executor=...)` 对 4 个代码工具优先转发到 `CodeToolExecutor.execute`；
- main 在 `CODE_ENABLED=true` 且 code service 可用时把 7 个 schema 传给 `AgentGraph(tool_schemas=...)`；关闭时行为与旧版一致。

## allowed_roots 默认与空列表语义（P2-7）
- CLI / `CodeIndexer`：省略 `--allow-root` / `allowed_roots=None` = `[当前用户 home]`；显式 `[]` = 全部拒绝；
- API：`CODE_ALLOWED_ROOTS` 默认 `[home]`；显式 `[]` = 全部拒绝；`CODE_ENABLED=false` 时 `/api/code/*` 返回 503；
- 索引器与 API 路由对“空列表 = 全禁”的语义一致，CLI 默认 home 只是省略时的默认值。

## E2E（P1-3）
```bash
# 1) 自举索引（默认与 API 共用 $DATA_ROOT/data/code_index.sqlite）
python scripts/index_code.py --root . --name rag_m

# 2) 启动 API（CODE_ENABLED=true；真实 LLM 走现有 provider 链）
# 3) 调 /api/code/ask 跑 tests/code_e2e_cases.json，并输出报告
python scripts/code_e2e.py --base-url http://127.0.0.1:8000 --repo-name rag_m
```
最近一次结果（2026-11-10，真实 API + opencode-go）：
- **10 题，refusal_count=0、citation_valid_rate=1.0、expected_path_hit_rate=1.0、p50≈7.62s**；
- 输出：`reports/code_e2e.md`（人读）与 `reports/code_e2e.json`（机器可读，含每题 citation 明细）。

## 实测（rag_m 自举）
- 索引：145 文件、1,287 符号、1,329 chunks（rag_m 自举；CLI/API 默认同一 SQLite）；
- 检索：10 个标识符类问题 path/qualname hit@5 = 10/10（`tests/test_code_e2e.py`；问题含精确标识符，属偏乐观的 fixtures）；
- 概览：`overview.kinds` 含 class/function/method，`top_paths` 为 `[{path, symbols}]` 按符号数 top 10；
- 问答：10 题真实 LLM，引用 path 全部存在（refusal 情况已通过查询扩展 + 抽取式兜底收敛）。

## 限制
- 仅 Python（TS/Go 及树莓派语言的 tree-sitter 解析留 v2）；
- FTS5 unicode61 不切分中文：中文自然语言问题检索弱，建议问题中包含标识符/路径；v2 接 dense embedding；
- impact 只做文件级反向 import（按模块名后缀匹配，basename 相似会误报），不承诺调用链/数据流；
- 单文件 >1MB 或 >5000 行跳过；>5000 文件暂同步索引。
