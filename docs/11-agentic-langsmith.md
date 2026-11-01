# Agentic 工具链 + LangSmith 监控（2026-11-01）

## 一、Agentic 链路（仅 agentic 路由）

    question
      → [classify] LoRA 分类器判定 agentic
      → [plan]  LLM 决定是否调用工具（最多 2 轮）
           ↳ 需要 → [tools] 执行 vector_search / kg_query / web_search → 回到 plan
      → [synthesize] 汇总所有工具证据 → 编号上下文 → 结构化答案 + [n] 引用
      → [verify] 引用校验
      → QueryResponse（tools 列表 + 5 段 trace）

- 实现：`src/ragm/agent/agent_graph.py`（LangGraph StateGraph）、`src/ragm/agent/tools.py`（工具定义与执行器）；
- LLM 工具调用：`LLMClient.chat_tools()`（OpenAI 兼容 tools / tool_calls，Ollama 已验证）；
- 兜底：模型不调用工具时，synthesize 节点自动跑一次 hybrid 检索保证有证据；工具失败不影响主链路；
- 响应：`tools: [{name, args, result, ms, status}]`，前端「调试」面板渲染工具时间线。

## 二、三个工具

| 工具 | 实现 | 说明 |
|---|---|---|
| `vector_search` | Milvus hybrid（dense + BM25 + RRF） | 本地语料事实查询，top-k 可调 |
| `kg_query` | Neo4j 实体图（105,570 实体） | 实体名匹配 + 1 跳扩展，返回关系路径 |
| `web_search` | Bing RSS（主）→ SearXNG（备） | 语料库之外的最新信息 |

网络现实（实测）：SearXNG 的 google/brave/duckduckgo 全部超时、baidu 返回 CAPTCHA，
因此 web_search 默认走 **Bing RSS**（`mkt=en-US&cc=US`，约 330ms，可解析 title/link/description），
SearXNG 仅在 Bing 无结果时作为 2 秒超时的兜底。

## 三、实测

| 用例 | 工具调用 | 延迟 | 结果 |
|---|---|---|---|
| 对比 Scared Straight! / Soldier Girls（本地） | vector_search ×2 | 3.5s | 1978 vs 1981，引用 3/3 通过 |
| Milvus BM25 Function（本地无此文档） | vector_search + web_search | 4.6s | 如实拒答（证据不足） |
| CEO of OpenAI as of 2026（联网） | web_search | 3.3s | 如实拒答（Bing 结果不含答案） |

延迟构成（典型）：规划 2.4s + 工具 0.1~0.4s + 生成 0.8~1.3s；工具调用本身很快，大头是 LLM 规划。

## 四、LangSmith 监控

已接入方式：

1. `.env` 配置（**无需改代码**）：

       LANGSMITH_TRACING=true
       LANGSMITH_API_KEY=lsv2_xxx          # 你自己的 key，写进 .env 即可，不要发到对话里
       LANGSMITH_PROJECT=rag-m
       LANGSMITH_ENDPOINT=https://api.smith.langchain.com

2. 启动时 `configure_tracing()` 设置 `LANGCHAIN_TRACING_V2 / LANGSMITH_*` 环境变量；
3. 关键 span 用 `@traceable` 标注（懒加载，启动后开启也能生效）：
   - `llm_chat` / `llm_chat_tools`（run_type=llm）
   - `hybrid_search`（run_type=retriever）
   - `vector_search` / `kg_query` / `web_search`（run_type=tool）
   - `agent_plan` / `agent_synthesize`、Graph RAG 的 classify/retrieve/generate/verify（run_type=chain）
4. `GET /api/health` 新增 `langsmith` 检查：`enabled · rag-m` / `disabled`；
5. 网络已验证：`api.smith.langchain.com` 可达（HTTP 200）。

> 注意：当前没有 API key，所以 tracing 处于 `disabled`；填入 key 并重启即生效。
> 没有 key 时不会发任何网络请求，也不会影响主链路（测试覆盖了 disabled/enabled 两条路径）。

## 五、已知限制

1. Agent 最多 2 轮工具调用（可配 `AGENT_MAX_ITERATIONS`），复杂问题可能不够；
2. web_search 的 Bing RSS 摘要较短（约 200 字），且部分查询会返回无关结果；
3. 工具失败没有重试/换源逻辑（SearXNG 兜底是唯一降级）；
4. LangSmith 追踪链路尚未端到端验证（缺 key）；span 结构已用单测覆盖；
5. agentic 的流式输出是"先跑完 agent 再分块推答案"（工具循环本身无法逐 token 流式）。

## 六、复现

    # 启动 SearXNG（可选，Bing RSS 主路径不依赖它）
    docker compose --profile search up -d searxng
    # 启动 API
    make api
    # 强制走 agentic 路由测试
    curl -s -X POST localhost:8000/api/query -H "Content-Type: application/json" \
         -d '{"question": "Compare Scared Straight! and Soldier Girls", "route_override": "agentic"}'
    # 前端：?live=1&tab=debug&route=agentic&q=<问题>

## 七、测试

- `tests/test_agent_tools.py`：工具 schema、截断、未知工具、无图谱降级、tracing 开关/环境变量；
- 全量 17 条测试通过（含 10 条 API 契约测试）。
