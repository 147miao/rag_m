# RAG-M：路由式混合 RAG

> 查询分类器 → 三条链路：**传统混合检索 RAG**（简单事实）/ **Graph RAG**（多跳关系）/ **Agentic RAG**（复杂推理，按需调用 vector_search / kg_query / web_search）。
> 技术栈：LangGraph 1.2 · Milvus 3.0 · Neo4j · Unstructured · Docker · Python 3.13

## 快速开始

```bash
cp .env.example .env
make check-data     # 校验 D 盘挂载
make install        # uv venv + 依赖
make up             # Milvus standalone
make health         # 期望 {"state":"OK"}
```

## 目录

```
docker/          Milvus / Neo4j / SearXNG compose
src/ragm/        应用代码（ingest / store / llm / embeddings / rag / tools / eval / api）
scripts/         数据下载、建索引、benchmark、评测
data/ -> D 盘    原始数据与语料（不入 git）
reports/         评测报告
docs/            方案与决策记录
```

## 文档

- [方案 v0.2](docs/00-proposal.md)
- 待补：ARCHITECTURE.md / EVAL_REPORT.md / DECISIONS.md

## 前端（web/）

```bash
cd web
pnpm install
pnpm dev          # http://127.0.0.1:5173
pnpm build        # 产物 web/dist
```

- 设计规范：[web/DESIGN.md](web/DESIGN.md)（遵循 OpenDesign 的 DESIGN.md 协议：色板 / 字体 / 间距 / 组件 / 状态 / 动效 / 无障碍）
- 演示深链：`?demo=multihop-1&tab=graph&fast=1`（demo = simple-1 / multihop-1 / agentic-1 / refusal-1；tab = evidence / graph / trace / debug；fast=1 直接呈现终态）
- 当前为**演示数据**，接口契约见 `web/src/lib/types.ts`，后端接入后由 `/api/query` 返回同结构 JSON
- 截图：[docs/screenshots](docs/screenshots)

## 后端（FastAPI）

```bash
make api        # http://127.0.0.1:8000（生产模式）
make api-dev    # 热重载
make test-api   # 契约测试（8 条，不依赖 Milvus/Ollama）
```

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查：Milvus 连接、Ollama 模型就绪、版本、uptime |
| GET | `/api/scenarios` | 演示案例列表（id / question / route / confidence / title） |
| POST | `/api/route` | 只做查询路由分类（label / route / confidence / backend / latency） |
| POST | `/api/query` | 一次性查询；`route_override=auto` 时先调用分类器，响应带 `classifier` 元数据 |
| POST | `/api/query/stream` | SSE 流式：`stage`（阶段进度）→ `token`（增量文本）→ `done`（完整响应） |
| POST | `/api/search` | 检索（dense / BM25 / hybrid / **small2big**），返回 top-k chunk 或 parent 上下文 |
| POST | `/api/feedback` | 答案反馈，落盘 `logs/feedback.jsonl` |

```bash
curl -s http://127.0.0.1:8000/api/health
curl -s -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Milvus 的 BM25 Function 是什么？","route_override":"auto"}'
curl -N -X POST http://127.0.0.1:8000/api/query/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"bge-m3 是哪个机构发布的？"}'
```

- OpenAPI 文档：<http://127.0.0.1:8000/docs>
- 契约定义：`src/ragm/api/schemas.py` ↔ `web/src/lib/types.ts`（字段一一对应；`Stats` 用 camelCase 别名序列化）
- 服务层抽象：`src/ragm/rag/service.py` 定义 `RagService` Protocol；当前默认 `LangGraphRagService`（`served_by=langgraph`：分类 → hybrid 检索 → gemma4 生成 → 引用校验），Milvus/LLM 不可用时回退 `DemoRagService`
- 路由分类器：`src/ragm/classifier/`（rules / prompt-llm / lora 三种后端）；当前默认 `CLASSIFIER_BACKEND=lora` + `CLASSIFIER_RUN_DIR=models/classifier/s1_06b_plain`（Qwen3-0.6B + SEQ_CLS LoRA，test_human macro-F1 0.898，36ms/条）
- 进度监控：`make status`（或 `python scripts/status.py --watch 10`）、`GET /api/status`、前端顶栏训练徽章
- 生成链路：LangGraph StateGraph（classify → retrieve → generate → verify），SSE 流式输出，详见 [docs/09-langgraph-generation.md](docs/09-langgraph-generation.md)
- Graph RAG：Neo4j 实体图（105,570 实体 / 180,894 条 co-evidence 边），multihop 路由做实体链接 + 1 跳扩展，gold 召回 0.875 → 1.000，详见 [docs/10-graph-rag.md](docs/10-graph-rag.md)
- 检索索引：Milvus collection `rag_chunks`（40,019 chunks；embeddinggemma-300m 768 维 + BM25 sparse），详见 [docs/06-milvus-retrieval.md](docs/06-milvus-retrieval.md)
- Small-to-big：`rag_children`（46,142 child）+ `parents.jsonl`（39,771 parent），`mode=small2big`；长文档 coverage 0.70→0.99，详见 [docs/08-small2big.md](docs/08-small2big.md)
- 请求日志：`logs/api.jsonl`（request_id / method / path / status / elapsed_ms）
- 前端 dev 模式经 Vite 代理 `/api` → `127.0.0.1:8000`；`pnpm build` 后由 FastAPI 直接托管 `web/dist`