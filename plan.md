# v1.1 计划（rev.2，吸收独立审核意见）

## 0. 需求核对清单（用户原话）
1. 多轮会话 + 查询改写/子问题拆解：会话历史存 SQLite/Redis，检索前指代消解 + 子问题分解。
2. MCP 工具接入 + 敏感操作确认：Python MCP SDK 接示例 MCP Server，requires_confirmation 拦截。
3. LLM 多 provider + 首包超时/熔断/降级：OpenAI 兼容 + 首 token 超时 + 候选模型降级。
4. 入库异步队列 + 状态机/重试：FastAPI BackgroundTasks/Celery + 任务表。

## 0.1 已确认选择
| 维度 | 选择 |
|---|---|
| 会话存储 | SQLite（`data/sessions.sqlite`），WAL + busy_timeout |
| 会话处理 | 指代消解 + 子问题拆解；历史按 6 轮/4000 字截断 |
| MCP | 自写 Python MCP Server（`list_dir`/`read_file`/`write_note`）；`write_note` 需确认；stdio 子进程 |
| LLM | **主：opencode-go `deepseek-v4-flash`（`http://127.0.0.1:8788/inference/go/v1`，key 从 `OPENCODE_GO_API_KEY`）；备：Ollama `gemma4:31b-cloud`**；首 token 超时 15s/20s 可配；仅首 token 前切换 |
| 队列 | SQLite 任务表 + **单 worker 线程 tick()**（不用 BackgroundTasks，避免双触发） |

## 0.2 并行方式（审核修正）
- `/` 盘仅剩 12G，4 个 worktree 不各自建 venv；**共享主 venv**，测试用 `PYTHONPATH=<worktree>/src .venv/bin/python ...`；
- 先由 prep 子 agent 在 main 上落「config 字段 + schemas + 空 router + include_router」接口层（分支 `prep/v1.1`），再开 4 个 feature worktree；
- 每个 feature 分支只改自己模块 + 必要接口实现；`config.py/main.py/schemas.py` 的装配冲突由主 agent 合并解决。

## 1. 验收标准（可测、精确）
| # | 标准 | 测法 |
|---|---|---|
| A1 | 会话 CRUD：create→append→get→delete，重启后可读 | 临时 sqlite 路径；直接调用 store |
| A2 | 指代消解：历史含「鸣人的父亲是波风水门」+ 新问「他的师父是谁？」→ rewritten 含「鸣人」 | fake rewriter 注入，断言 rewritten 与 subquestions |
| A3 | 子问题拆解：含「和」「对比」的问题 → ≥2 子问题；检索对每个子问题各调用一次并按 chunk_id 去重 | fake search 记录调用次数/去重 |
| A4 | MCP list_tools 返回 3 个工具；`read_file` 能读 sandbox 内文件 | in-memory Client(MCPServer) |
| A5 | `write_note` 未确认前不落盘：`POST /api/mcp/call` 返回 pending+token；确认前文件不存在 | 临时 sandbox + API 测试 |
| A6 | 确认：token 绑定 tool+args、TTL 60s、原子单次消费；过期→410、已用→409、未知→404 | 单测三种状态 |
| A7 | 主 provider 首 token 超时/异常 → 切 gemma4；已输出首 token 后不切 | fake provider（sleep/raise）+ FallbackLLM 单测 |
| A8 | 熔断：连续 3 次失败后 60s 内跳过该 provider，冷却后半开 | 单测状态机 |
| A9 | `/api/health` 逐 provider 展示 ok/circuit_open；不把备 provider 故障算成整体 degraded | health JSON 断言 |
| A9b | 真实 opencode-go：key 存在时一次 chat 成功并记录 provider；key 缺失时跳过并告警 | curl/脚本实测（key 未提供则标注未验证） |
| A10 | 任务状态机 queued→running→succeeded；失败重试 3 次后 failed；`tick()` 可注入控制 | fake ingest + tick() 断言状态序列 |
| A11 | 启动恢复：running→queued（保留 attempts），幂等；单 worker join | 直接调用 recovery 函数 |
| A12 | `pytest` 全过、`ruff check src tests scripts` 全绿、`pnpm build` 通过 | CI |

## 2. 设计

### W1 会话（owner: 子 agent A；分支 feat/session）
- `src/ragm/conversation/store.py`：SQLite（WAL, busy_timeout=3000，每操作独立连接）；sessions(id,title,created_at,updated_at)、messages(id,session_id,role,content,meta_json,created_at)；`data_root/data/sessions.sqlite` 可配置；
- `src/ragm/conversation/rewrite.py`：`Rewriter.rewrite(question, history)` → {rewritten, subquestions}；一次 `chat(json_mode=True)`，JSON 解析失败回退原文（subquestions=[]）；只在有 session_id 且历史非空时调用；
- **改 `src/ragm/rag/graph.py::_retrieve_node`**：subquestions ≥2 时逐个子问题 `search_with_scores`，按 chunk_id 去重合并（owner 补上，审核指出遗漏）；
- API：`api/session_routes.py`（POST/GET/DELETE sessions），`QueryRequest.session_id`，响应 `rewritten_question/subquestions/session_id`；SSE 同样支持；
- 测试：store CRUD/重启；fake rewriter；fake search 多查询去重。

### W2 MCP（owner: 子 agent B；分支 feat/mcp）
- `src/ragm/mcp/server.py`：`MCPServer` 三工具，sandbox root 限制，`write_note` 只允许 root 下；
- `src/ragm/mcp/client.py`：**每次调用起子进程**：`StdioServerParameters(command=sys.executable, args=["-m","ragm.mcp.server"], env={...})`（不能传 list，审核实测 TypeError）；in-memory 用于测试；
- `src/ragm/mcp/confirmation.py`：HMAC token（tool+args hash+exp）存内存 dict，原子 pop；TTL 60s；
- API：`GET /api/mcp/tools`；`POST /api/mcp/call {tool,args}` → 只读工具直接执行；敏感工具返回 202 `{pending:true, token}`；`POST /api/mcp/confirm {token}` → 校验后执行，返回结果；
- 测试：in-memory list/call、sandbox 越界拒绝、token 过期/重放/未知、确认前文件不存在。

### W3 LLM 降级（owner: 子 agent C；分支 feat/llm-fallback）
- `src/ragm/rag/llm_router.py`：`FallbackLLM` 包装多个 `LLMClient`；`chat()` 整请求超时（默认 60s）；`stream_chat()` 首 token 超时（15s/20s），**首 token 后不切换**；
- `CircuitBreaker`：失败阈值 3、冷却 60s、半开探测；provider 状态供 health；
- config：`LLM_PROVIDERS` JSON（见 3.1）；key 只从 env 读；
- 测试：fake clients（sleep/raise/流式首 token）断言切换/不切换、熔断与恢复；health provider 状态。

### W4 异步入库（owner: 子 agent D；分支 feat/jobs）
- `src/ragm/jobs/store.py`：SQLite jobs(id,type,payload,status,attempts,max_attempts,error,created_at,updated_at,started_at,finished_at)；WAL+busy_timeout；
- `src/ragm/jobs/runner.py`：单 worker 线程 `tick()`（可注入 ingest 函数）；enqueue 只插表 + event 唤醒；重试退避 5s/30s/120s；启动 `recover()`：running→queued（保留 attempts），幂等；shutdown join；
- API：`POST /api/ingest`（src/collection/recreate）→ job_id；`GET /api/jobs/{id}`；`POST /api/jobs/{id}/retry`；
- 测试：状态序列、重试、恢复、retry 语义（仅 failed 可重试）。

### 接口层 prep（owner: 子 agent E；分支 prep/v1.1）
- config 新增字段：session_db_path/session_max_turns/session_rewrite_enabled；mcp_enabled/mcp_sandbox_root/mcp_confirm_ttl_s；llm_providers/llm_chat_timeout/llm_first_token_timeout；jobs_db_path/jobs_worker/jobs_max_attempts；
- schemas 新增：Session*/SubQuestion/QueryRequest.session_id/QueryResponse 扩展/McpToolInfo/McpCallRequest/McpConfirmRequest/JobInfo；
- 空 router 模块 + main.py include_router（先返回 501）；测试骨架；
- 验收：pytest/ruff 全过，接口 404→501 变化可见。

## 3.1 LLM provider 配置
```bash
LLM_PROVIDERS='[
 {"name":"opencode-go","base_url":"http://127.0.0.1:8788/inference/go/v1","model":"deepseek-v4-flash","api_key_env":"OPENCODE_GO_API_KEY","first_token_timeout":15},
 {"name":"ollama","base_url":"http://127.0.0.1:11434/v1","model":"gemma4:31b-cloud","api_key":"ollama","first_token_timeout":20}
]'
```
- key 只从环境变量读，缺失时跳过该 provider 并告警；不写仓库、不打印。

## 4. 约束
- 保持 41 tests 全过；每个功能 ≥3 条测试（含失败路径）；单文件 ≤700 行；新增依赖仅 `mcp>=2.3.0`；
- 子 agent 在独立 worktree 工作（共享主 venv + PYTHONPATH），分支各自 commit，不 push、不 merge；
- 主 agent 负责 prep 派发、合并冲突、README/docs、最终 acceptance/completeness。

## 5. 风险（审核修正）
| 风险 | 缓解 |
|---|---|
| 磁盘 12G，worktree 不能各自装依赖 | 共享 venv + PYTHONPATH；worktree 仅源码（~7MB） |
| editable .pth 指向主 src，worktree 测试假绿 | 所有测试命令强制 `PYTHONPATH=<worktree>/src`；子 agent 跑一条 `python -c "import ragm; print(ragm.__file__)"` 证明路径 |
| SQLite 多线程写冲突 | WAL+busy_timeout；每操作独立连接；单 worker |
| token 重放/越权 | HMAC+TTL+绑定+原子 pop；sandbox root 校验 |
| 主模型首 token 临界 | 阈值 15s/20s 可配；仅首 token 前切换；记录 provider |
| CI 无 Milvus/GPU | jobs/会话测试全 fake；MCP 用 in-memory；LLM 用 fake provider |

## 6. 不在本轮范围
- 前端会话 UI / MCP 确认弹窗；Redis；远程 MCP；multipart 上传。
