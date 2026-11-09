# v1.1 计划：多轮会话 / MCP / LLM 降级 / 异步入库（2026-11-09）

## 0. 需求核对清单（用户原话）

1. **多轮会话 + 查询改写/子问题拆解（1~2 天）**：会话历史存 SQLite/Redis，检索前做指代消解 + 子问题分解。
2. **MCP 工具接入 + 敏感操作确认（1~2 天）**：用 Python MCP SDK 接一个示例 MCP Server，加 requires_confirmation 拦截。
3. **LLM 多 provider + 首包超时/熔断/降级（1 天）**：OpenAI 兼容适配 + 首 token 超时探测 + 候选模型降级。
4. **入库异步队列 + 状态机/重试（1 天）**：Celery/RQ 或 FastAPI BackgroundTasks + 任务表状态机。

## 0.1 默认选择（用户未在提问窗口回答，按推荐项执行；可随时改）

| 维度 | 选择 | 理由 |
|---|---|---|
| 会话存储 | **SQLite**（`data/sessions.sqlite`） | 零依赖、重启不丢；Redis 留接口 `SESSION_BACKEND` |
| 会话处理 | **指代消解 + 子问题拆解** | LLM JSON 输出 + 规则兜底；拆解结果做多查询检索 |
| MCP 示例 | 自写 Python MCP Server（`mcp.server.MCPServer`），工具：`list_dir` / `read_file` / `write_note`；`write_note` 需确认 | 可完全本地跑，无需外部服务；确认流走 API token |
| LLM provider | Ollama `gemma4:31b-cloud`（主）→ Ollama `deepseek-v4.1-flash:cloud`（备）；DeepSeek 官方 key 存在时优先接官方 | 本机已有两个模型，无需外部 key 即可演示降级 |
| 队列 | **FastAPI BackgroundTasks + SQLite 任务表** | 零新依赖、状态可持久化、重启可恢复；不引入 Redis |

## 1. 验收标准

| # | 标准 | 证据 |
|---|---|---|
| A1 | 会话 CRUD 持久化：创建→追加→重启后仍能读回 | `tests/test_conversation.py` + sqlite 文件 |
| A2 | 带 session_id 的 query 会做指代消解（"他/这个"→具体实体） | API 响应 `rewritten_question` |
| A3 | 子问题拆解 ≥2 时多查询检索并去重合并 | 响应 `subquestions` + 命中数 |
| A4 | MCP 客户端能 list_tools 并调用只读工具 | 单测/接口输出 |
| A5 | 敏感工具（write_note）未确认前不执行，返回 requires_confirmation+token | API 响应 + 文件系统断言 |
| A6 | 确认接口消费 token 后执行，token 单次有效 | 二次调用 410/400 |
| A7 | 主 provider 首 token 超时（默认 8s）自动切备 provider | 假 provider 单测 |
| A8 | 熔断：连续 3 次失败后 60s 内不再尝试该 provider | 单测状态机 |
| A9 | /api/health 暴露 provider 状态（ok/circuit_open） | health JSON |
| A10 | 入库任务 enqueue→running→succeeded；失败重试 3 次后 failed | 任务表 + `/api/jobs/{id}` |
| A11 | 重启后 running 任务恢复为 queued/failed，不丢任务 | 单测/实测 |
| A12 | `pytest` 全过、`ruff check src tests scripts` 全绿、`pnpm build` 通过 | CI |

## 2. 设计（文件所有权）

| 工作流 | 新增/修改 | 说明 |
|---|---|---|
| W1 会话 | `src/ragm/conversation/{store,rewrite,__init__}.py`、`api/conversation_routes.py`、`tests/test_conversation.py` | SQLite 表 sessions/messages；Rewriter 输出 {rewritten, subquestions}；`QueryRequest.session_id` |
| W2 MCP | `src/ragm/mcp/{server,client,confirmation}.py`、`agent/mcp_tools.py`、`api/mcp_routes.py`、`tests/test_mcp.py` | 自写 server；Client 用 `mcp.Client(StdioServerParameters(["python","-m","ragm.mcp.server"]))`；确认表存 SQLite |
| W3 LLM | `src/ragm/rag/llm_router.py`、`rag/llm.py`（保持兼容）、`config.py`、`api` health、`tests/test_llm_router.py` | Provider 列表、首 token 超时、CircuitBreaker、降级 |
| W4 队列 | `src/ragm/jobs/{store,runner,__init__}.py`、`api/jobs_routes.py`、`main.py` 启动 worker、`tests/test_jobs.py` | SQLite jobs 表；BackgroundTasks 触发 + 独立 worker 线程；状态机/重试 |
| 集成 | 主 agent | 各路由模块 `include_router`、config 字段、README/docs、合并冲突 |

## 3. 接口草案

- `POST /api/sessions` → `{id,title}`；`GET /api/sessions/{id}` → 历史消息；
- `POST /api/query` 增加 `session_id`；响应增加 `rewritten_question`、`subquestions`、`session_id`；
- `GET /api/mcp/tools` → 工具清单（含 requires_confirmation）；
- `POST /api/confirm` `{token}` → 执行挂起工具；
- `GET /api/llm/providers`（或并入 health）；
- `POST /api/ingest` `{src, collection, recreate}` → `{job_id}`；`GET /api/jobs/{job_id}`；`POST /api/jobs/{job_id}/retry`。

## 4. 约束

- 保持现有 41 tests 全过；新增每个功能至少 3 条测试（含失败路径）；
- 不新增容器（SQLite 方案）；新增依赖仅 `mcp>=2.3.0`；
- 单文件 ≤700 行；配置项集中在 `config.py`；
- 子 agent 在独立 worktree 工作，分支 `feat/<name>`，不 push、不合并；主 agent 负责合并与冲突解决。

## 5. 风险

| 风险 | 缓解 |
|---|---|
| 4 个分支都改 routes/main/config | 各功能独立 router 模块；主 agent 只加 include_router 与 config 字段；合并前跑全量测试 |
| MCP stdio 子进程与 GPU 无关，但依赖 `mcp` 包安装 | 固定 `mcp>=2.3.0`；单测用 in-memory `Client(MCPServer)` 避免子进程 |
| 会话改写增加 LLM 延迟（+0.5~1s） | 只在有 session_id 且历史非空时触发；失败回退原文 |
| LLM 降级导致答案风格不一致 | 统一 prompt；记录 `provider` 字段 |
| BackgroundTasks 进程退出丢任务 | 任务表持久化 + 启动恢复 running→queued |
| 与 fix/p0 分支冲突 | fix/p0 验收通过后先合并 main，再开 feature worktree |

## 6. 不在本轮范围

- 前端会话 UI / MCP 确认弹窗（先把 API 与测试做扎实，前端下一步）；
- Redis 会话/队列；
- 远程 MCP（HTTP transport）与 OAuth；
- 文件 multipart 上传（`/api/ingest` 接受服务器路径）。
