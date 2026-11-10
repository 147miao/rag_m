# 代码库智能分析系统（code-insight）计划 v2（吸收审核）

## 0. 需求与默认值（用户提问窗口超时，按推荐项执行）
| 维度 | 选择 |
|---|---|
| 场景 | 代码问答/理解（path:line 引用）+ 符号查找 + 仓库概览；影响面仅文件级反向 import（启发式，明确标注非精确） |
| 输入/语言 | 本地目录，Python only（stdlib ast） |
| 索引 | **v1 用 SQLite FTS5（无 Milvus/GPU）**；dense 留 v2 |
| 与现有系统 | 双模式：保留文档 RAG；代码模式独立 SQLite，不影响现有 145 tests |
| 验收 | 自举索引 rag_m 自身；10 题 fixtures（期望 path/qualname/行范围），hermetic + 真实服务各验一次 |

## 1. 验收标准（可证伪）
| # | 标准 | 测法 |
|---|---|---|
| A1 | 扫描：排除 `.git/.venv*/__pycache__/site-packages/data/.volumes`；按 file_hash 增量；root 必须在 allowed_roots 内且 resolve 后无符号链接逃逸 | 临时仓库 + `tests/test_code_scanner.py` |
| A2 | AST：模块/类/函数/方法/async/decorator/嵌套函数，start/end 行、docstring、imports、references（名称引用行）；语法错误文件跳过并记录 | 解析真实 `src/ragm/rag/graph.py` 断言 |
| A3 | 切块：函数/类级，超长按 1500 字/120 行切；metadata 含 repo/path/qualname/kind/lines；确定性 chunk_id | 单测 + 行号 |
| A4 | SQLite：repos/files/symbols/imports/references 表；FTS5（unicode61）索引 `path qualname code`；重复索引幂等（先删后插）；改文件只重索引该文件；删文件清理 | 索引 rag_m 两次计数相同；改/删单文件断言 |
| A5 | 检索：`search_code(query,k,repo,path_prefix,kind)` 用 FTS5 bm25 + SQL 过滤（过滤值转义）；`get_symbol(repo,qualname,path=None)` 精确返回 code/lines | 单测 + 真实查询 |
| A6 | 影响面（降级）：`reverse_imports(repo,path)` 返回直接反向 import 文件列表（基于 imports 表）；不承诺符号级/传递闭包，文档标注 heuristic | 用 rag_m 内已知文件验证 |
| A7 | API：`POST /api/code/repos {root,name}` → 200 `{repo_id,stats}`（同步索引，>5000 文件返回 202+job 可后续）；`GET /api/code/repos`、`GET/DELETE /api/code/repos/{id}`、`POST /api/code/repos/{id}/reindex`、`GET /api/code/search`、`GET /api/code/symbol`、`GET /api/code/impact`、`GET /api/code/overview`、`POST /api/code/ask`；错误码 400/403/404/422 明确 | TestClient + 真实服务 |
| A8 | 问答：`/api/code/ask {question,repo_id?}` → answer + citations `[{n,path,start_line,end_line,qualname}]` + provider；引用 verifier 检查每个 n 的 path 存在且行范围非空；无证据返回明确拒答 | 10 题 fixtures：hit@5 中包含期望 path/qualname，引用存在率 100% |
| A9 | 双模式：现有 145 tests 全过；代码模式不打开时 `/api/code/*` 404/503 明确；代码 DB 默认 `data/code_index.sqlite` | 全量 pytest + smoke |
| A10 | 文档/CLI：`scripts/index_code.py`（--root/--name/--db/--reindex/--json）；README + docs/23 + .env.example；标注 Python-only、impact heuristic、FTS5 中文局限 | diff + CLI 实跑 |

## 2. 冻结接口

### 模块
| 文件 | Owner | 职责 |
|---|---|---|
| `src/ragm/code/{scanner,parser,chunker,store,indexer,__init__}.py` | W1 | 扫描/解析/切块/SQLite FTS5/索引器 |
| `src/ragm/code/service.py`、`api/code_schemas.py`、`api/code_routes.py`、`agent/code_tools.py` | W2 | 检索/符号/概览/问答 + API + agent 工具 |
| `scripts/index_code.py` | W1 | CLI |
| `main.py`/`config.py`/`agent/tools.py` 装配 | 主 agent | 合并时接线 |

### 契约（W2 用 fake，不依赖 W1 实现）
```python
class CodeStore:
    def __init__(self, db_path: Path): ...
    def add_repo(self, root: str, name: str) -> dict  # {repo_id,file_count,symbol_count,status}
    def list_repos(self) -> list[dict]
    def get_repo(self, repo_id: str) -> dict | None
    def delete_repo(self, repo_id: str) -> bool
    def search(self, query: str, k: int = 8, repo_id: str | None = None, path_prefix: str | None = None, kind: str | None = None) -> list[dict]
    def get_symbol(self, repo_id: str, qualname: str, path: str | None = None) -> dict | None
    def reverse_imports(self, repo_id: str, path: str) -> list[str]
    def overview(self, repo_id: str) -> dict
class CodeIndexer:
    def __init__(self, db_path: Path, allowed_roots: list[str]): ...
    def index(self, root: str, name: str, reindex: bool = False) -> dict
class CodeService:  # W2 实现/包装
    def search_code(...); def get_symbol(...); def overview(...); def ask(question, repo_id=None, llm=None) -> dict
```

### DB schema（W1 冻结）
- `repos(id TEXT PK, root TEXT, name TEXT, created_at, updated_at, file_count INT, symbol_count INT, status TEXT)`
- `files(repo_id, path, file_hash, size, mtime, PRIMARY KEY(repo_id,path))`
- `symbols(repo_id, path, qualname, kind, start_line, end_line, docstring, code_hash, code TEXT)`
- `imports(repo_id, src_path, module, level, lineno)`
- `references(repo_id, src_path, name, lineno)`
- `chunks(repo_id, chunk_id PK, path, qualname, kind, start_line, end_line, text)` + `chunks_fts` (content=chunks, tokenize unicode61)

## 3. 执行
- W1 分支 `feat/code-index`（worktree ragm-wt-code-index）：只写 `src/ragm/code/{scanner,parser,chunker,store,indexer}.py`、`scripts/index_code.py`、`tests/test_code_*.py`；
- W2 分支 `feat/code-api`（worktree ragm-wt-code-api）：只写 `code/service.py`、`api/code_{schemas,routes}.py`、`agent/code_tools.py`、`tests/test_code_api.py`；用 fake CodeStore/CodeIndexer，不 import W1 实现；
- 主 agent：合并、main/config/tools 装配、自举索引、10 题 E2E、README/docs、acceptance/completeness。

## 4. 约束/风险
- 不新增依赖（sqlite3 自带 FTS5；无 Milvus/GPU）；单文件 ≤700 行；`make ci` 不能依赖网络/GPU；
- 中文查询在 FTS5 unicode61 下效果差：文档标注，代码查询以英文标识符/路径为主；
- 大仓库限制：单文件 >1MB 或 >5000 行跳过；>5000 文件 API 返回 202（v1 可先同步并说明）；
- allowed_roots 默认 `[str(Path.home())]`，测试传 tmp_path；
- impact 仅文件级反向 import，不承诺精确调用链。

## 5. 不在本轮
TypeScript/Go、跨语言调用图、数据流、前端代码浏览器、PR 集成、dense 向量检索。
