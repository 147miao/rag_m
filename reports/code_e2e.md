# code-insight E2E 报告

- generated_at: 2026-11-10T14:02:40.326939+00:00
- base_url: http://127.0.0.1:8801
- repo: rag_m (repo_88cf53a4065da92c)
- repo_root: /home/myj/develop_项目/rag_m
- questions: 10
- refusal_count: 0
- citation_valid_rate: 1.0
- expected_path_hit_rate: 1.0
- p50_ms: 7619.1
- avg_ms: 9663.1

| # | question | refusal | citations | valid | expected path | ms |
|---|---|---|---|---|---|---|
| 1 | ToolExecutor run vector_search 如何分发 | False | 2 | True | True | 7601.8 |
| 2 | SessionStore get_history sqlite 历史 | False | 5 | True | True | 5124.5 |
| 3 | ConfirmationStore consume token HMAC | False | 2 | True | True | 17309.5 |
| 4 | FallbackLLM build_llm provider 降级 | False | 5 | True | True | 6920.4 |
| 5 | JobRunner retry tick 重试 | False | 6 | True | True | 5981.9 |
| 6 | build pipeline file_hash 增量 | False | 3 | True | True | 10224.3 |
| 7 | health providers circuit_open 健康检查 | False | 3 | True | True | 10537.6 |
| 8 | build_classifier rules lora 分类器 | False | 6 | True | True | 7266.1 |
| 9 | MilvusStore hybrid_search BM25 混合检索 | False | 6 | True | True | 7636.4 |
| 10 | LangGraphRagService stream | False | 3 | True | True | 18029.0 |

## 引用明细

### 1. ToolExecutor run vector_search 如何分发
- [2] src/ragm/agent/tools.py:149-163 (OK; file_lines=167)
- [1] src/ragm/agent/tools.py:71-86 (OK; file_lines=167)

### 2. SessionStore get_history sqlite 历史
- [3] src/ragm/conversation/store.py:202-238 (OK; file_lines=238)
- [5] src/ragm/conversation/store.py:193-238 (OK; file_lines=238)
- [1] tests/test_conversation.py:135-150 (OK; file_lines=522)
- [2] tests/test_conversation.py:153-156 (OK; file_lines=522)
- [6] src/ragm/conversation/store.py:83-119 (OK; file_lines=238)

### 3. ConfirmationStore consume token HMAC
- [1] src/ragm/mcp/confirmation.py:138-140 (OK; file_lines=173)
- [2] src/ragm/mcp/confirmation.py:117-136 (OK; file_lines=173)

### 4. FallbackLLM build_llm provider 降级
- [1] src/ragm/rag/llm_router.py:464-466 (OK; file_lines=466)
- [3] tests/test_llm_router.py:277-308 (OK; file_lines=603)
- [2] tests/test_llm_router.py:311-319 (OK; file_lines=603)
- [5] src/ragm/rag/llm_router.py:316-318 (OK; file_lines=466)
- [6] src/ragm/rag/llm_router.py:312-314 (OK; file_lines=466)

### 5. JobRunner retry tick 重试
- [3] src/ragm/jobs/runner.py:73-108 (OK; file_lines=126)
- [4] src/ragm/jobs/runner.py:70-94 (OK; file_lines=126)
- [6] tests/test_jobs.py:64-70 (OK; file_lines=400)
- [2] src/ragm/jobs/runner.py:31-72 (OK; file_lines=126)
- [1] tests/test_jobs.py:125-141 (OK; file_lines=400)
- [5] tests/test_jobs.py:73-101 (OK; file_lines=400)

### 6. build pipeline file_hash 增量
- [1] src/ragm/ingest/pipeline.py:18-23 (OK; file_lines=157)
- [3] src/ragm/ingest/pipeline.py:67-111 (OK; file_lines=157)
- [2] src/ragm/ingest/pipeline.py:112-144 (OK; file_lines=157)

### 7. health providers circuit_open 健康检查
- [3] src/ragm/api/routes.py:187-208 (OK; file_lines=399)
- [2] tests/test_llm_router.py:552-573 (OK; file_lines=603)
- [1] tests/test_llm_router.py:591-603 (OK; file_lines=603)

### 8. build_classifier rules lora 分类器
- [1] src/ragm/classifier/factory.py:37-58 (OK; file_lines=58)
- [5] src/ragm/config.py:104-151 (OK; file_lines=228)
- [2] tests/test_prep_interfaces.py:31-41 (OK; file_lines=210)
- [6] src/ragm/classifier/lora.py:73-74 (OK; file_lines=74)
- [3] scripts/train_route_cls.py:48-55 (OK; file_lines=346)
- [4] scripts/train_route_cls.py:214-251 (OK; file_lines=346)

### 9. MilvusStore hybrid_search BM25 混合检索
- [1] src/ragm/store/milvus_store.py:72-78 (OK; file_lines=96)
- [3] src/ragm/store/milvus_store.py:72-96 (OK; file_lines=96)
- [6] src/ragm/store/milvus_store.py:39-71 (OK; file_lines=96)
- [2] src/ragm/store/milvus_store.py:44-45 (OK; file_lines=96)
- [5] scripts/eval_all_dilution.py:66-79 (OK; file_lines=83)
- [4] src/ragm/store/milvus_store.py:87-92 (OK; file_lines=96)

### 10. LangGraphRagService stream
- [2] src/ragm/rag/langgraph_service.py:281-297 (OK; file_lines=319)
- [3] src/ragm/rag/langgraph_service.py:288-319 (OK; file_lines=319)
- [1] src/ragm/rag/langgraph_service.py:303-310 (OK; file_lines=319)
