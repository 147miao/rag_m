# LangGraph 生成链路（M3，2026-11-01）

## 一、链路

    question
      → [classify]  LoRA 分类器（Qwen3-0.6B，36~39ms）
      → [retrieve]  hybrid 检索：dense(embeddinggemma-300m) + BM25 → 本地 RRF，保留 dense/sparse 分数
      → [generate]  gemma4:31b-cloud（Ollama，OpenAI 兼容）带编号上下文生成
      → [verify]    引用解析 + 越界校验
      → QueryResponse（answer / citations / evidence / trace / stats / served_by=langgraph）

- StateGraph 节点：`src/ragm/rag/graph.py`（`classify → retrieve → generate → verify`）；
- 服务实现：`src/ragm/rag/langgraph_service.py`，实现 `RagService` 契约；
- LLM 客户端：`src/ragm/rag/llm.py`（非流式 + SSE 流式）；
- 提示词：`src/ragm/rag/prompts.py`（编号资料、强制 [n] 引用、语言跟随问题、拒答策略）；
- 引用校验：`src/ragm/rag/citations.py`。

## 二、检索策略（按路由）

| 路由 | top-k | 当前实现 | 后续 |
|---|---|---|---|
| simple | 8 | hybrid（dense + BM25 + RRF） | 可加 reranker |
| multihop | 12 | hybrid（近似多跳） | Graph RAG / KG 扩展 |
| agentic | 12 | hybrid（近似） | 工具调用（vector_search / kg_query / web_search） |

## 三、输出格式

- 纯文本结构化：第一行一句话结论，空行后 2~5 句展开，每句带 [n]；
- 解析失败时自动降级：首句作 lead，其余作 body；
- 拒答：检索为空，或模型明确说"资料中没有足够信息" → `refusal` 字段；
- 引用校验：`[n]` 编号必须落在 evidence 范围内，返回 `n/m 通过`。

## 四、实测（2026-11-01，本机）

| 指标 | 结果 |
|---|---|
| 端到端延迟（16 条中英混合） | p50 ≈ 1.01s（分类 30~80ms + 检索 25~100ms + 生成 0.9~1.5s） |
| 引用全部有效 | 16/16 |
| 引用越界 | 0 |
| 非空答案 | 20/20（另一批 20 条） |
| 拒答 | 3/16（问题确实超出语料范围，符合预期） |
| 语言跟随 | 英文问题英文答、中文问题中文答（prompt 约束后） |

示例（EN）：

> Q: What magazine was started first, Arthur's Magazine or First for Women?
> A: **Arthur's Magazine was started first [1].**
> Arthur's Magazine was published from 1844 to 1846 [1]. The provided materials contain no information regarding a magazine named "First for Women" [1-12].
> 引用校验 1/1 通过；延迟 1.00s；1850 in / 54 out。

## 五、配置

    # .env
    RAG_BACKEND=langgraph      # langgraph | demo（回退演示数据）
    RAG_MAX_TOKENS=1024
    RAG_TOP_K=8
    RAG_MULTIHOP_K=12
    RAG_AGENTIC_K=12

服务启动时：Milvus/embedding 可用且 RAG_BACKEND=langgraph → 真实链路；否则自动回退 DemoRagService。

## 六、API

    POST /api/query          {"question": "..."} → 完整响应（served_by=langgraph）
    POST /api/query/stream   SSE：stage(路由/检索) → token(逐字) → done(完整响应)

前端 deep link：`?live=1&q=<问题>&tab=evidence` 可直接截图/分享真实结果。

## 七、已知限制

1. **没有 reranker**：hybrid 直接 top-k，Qwen3-Reranker 消融因 8GB 显存未跑；
2. **multihop / agentic 仍是同一套 hybrid 检索**：图谱扩展与工具调用还没接（当前只靠 prompt 区分）；
3. 短答案时 body 可能为空（模型只给一行）；
4. 引用只校验编号范围，不校验语义支撑（下一步可做 NLI/字符串对齐校验）；
5. 成本按本地 Ollama 记为 0（gemma4:31b-cloud 免费层）。

## 八、复现

    make api                     # 或 .venv/bin/python -m uvicorn ragm.api.main:app --port 8000
    curl -s -X POST localhost:8000/api/query -H "Content-Type: application/json" \
         -d '{"question": "What magazine was started first, Arthur'"'"'s Magazine or First for Women?"}'
    curl -N -X POST localhost:8000/api/query/stream -H "Content-Type: application/json" \
         -d '{"question": "范廷颂是什么时候被任为主教的？"}'
