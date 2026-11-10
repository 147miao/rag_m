# code-insight E2E 报告

- generated_at: 2026-11-10T14:10:08.187015+00:00
- base_url: http://127.0.0.1:8000
- repo: rag_m (repo_88cf53a4065da92c)
- repo_root: /home/myj/develop_项目/rag_m
- questions: 10
- refusal_count: 0
- citation_valid_rate: 1.0
- expected_path_hit_rate: 1.0
- p50_ms: 8192.2
- avg_ms: 8966.8

| # | question | refusal | citations | valid | expected path | ms |
|---|---|---|---|---|---|---|
| 1 | ToolExecutor run vector_search 如何分发 | False | 4 | True | True | 8058.4 |
| 2 | SessionStore get_history sqlite 历史 | False | 2 | True | True | 8552.9 |
| 3 | ConfirmationStore consume token HMAC | False | 5 | True | True | 7790.6 |
| 4 | FallbackLLM build_llm provider 降级 | False | 5 | True | True | 7473.0 |
| 5 | JobRunner retry tick 重试 | False | 6 | True | True | 14967.8 |
| 6 | build pipeline file_hash 增量 | False | 6 | True | True | 10703.7 |
| 7 | health providers circuit_open 健康检查 | False | 6 | True | True | 8326.0 |
| 8 | build_classifier rules lora 分类器 | False | 2 | True | True | 8031.5 |
| 9 | MilvusStore hybrid_search BM25 混合检索 | False | 6 | True | True | 6793.4 |
| 10 | LangGraphRagService stream | False | 4 | True | True | 8970.7 |

## 引用明细

### 1. ToolExecutor run vector_search 如何分发
- [2] src/ragm/agent/tools.py:149-163 (OK; file_lines=190)
- [1] src/ragm/agent/tools.py:71-86 (OK; file_lines=190)
- [3] src/ragm/agent/tools.py:65-97 (OK; file_lines=190)
- [5] src/ragm/agent/code_tools.py:175-177 (OK; file_lines=177)

### 2. SessionStore get_history sqlite 历史
- [3] src/ragm/conversation/store.py:202-238 (OK; file_lines=238)
- [1] tests/test_conversation.py:135-150 (OK; file_lines=522)

### 3. ConfirmationStore consume token HMAC
- [6] src/ragm/mcp/confirmation.py:83-118 (OK; file_lines=173)
- [1] src/ragm/mcp/confirmation.py:138-140 (OK; file_lines=173)
- [2] src/ragm/mcp/confirmation.py:117-136 (OK; file_lines=173)
- [4] tests/test_mcp.py:225-235 (OK; file_lines=360)
- [5] tests/test_mcp.py:198-204 (OK; file_lines=360)

### 4. FallbackLLM build_llm provider 降级
- [1] src/ragm/rag/llm_router.py:464-466 (OK; file_lines=466)
- [2] tests/test_llm_router.py:311-319 (OK; file_lines=603)
- [3] tests/test_llm_router.py:277-308 (OK; file_lines=603)
- [5] src/ragm/rag/llm_router.py:316-318 (OK; file_lines=466)
- [6] src/ragm/rag/llm_router.py:312-314 (OK; file_lines=466)

### 5. JobRunner retry tick 重试
- [4] src/ragm/jobs/runner.py:70-94 (OK; file_lines=126)
- [2] src/ragm/jobs/runner.py:31-72 (OK; file_lines=126)
- [3] src/ragm/jobs/runner.py:73-108 (OK; file_lines=126)
- [1] tests/test_jobs.py:125-141 (OK; file_lines=400)
- [5] tests/test_jobs.py:73-101 (OK; file_lines=400)
- [6] tests/test_jobs.py:64-70 (OK; file_lines=400)

### 6. build pipeline file_hash 增量
- [1] src/ragm/ingest/pipeline.py:18-23 (OK; file_lines=157)
- [3] src/ragm/ingest/pipeline.py:67-111 (OK; file_lines=157)
- [2] src/ragm/ingest/pipeline.py:112-144 (OK; file_lines=157)
- [4] src/ragm/ingest/pipeline.py:89-91 (OK; file_lines=157)
- [5] src/ragm/code/store.py:298-312 (OK; file_lines=557)
- [6] src/ragm/code/store.py:283-288 (OK; file_lines=557)

### 7. health providers circuit_open 健康检查
- [3] src/ragm/api/routes.py:187-208 (OK; file_lines=399)
- [1] tests/test_llm_router.py:591-603 (OK; file_lines=603)
- [2] tests/test_llm_router.py:552-573 (OK; file_lines=603)
- [6] src/ragm/api/routes.py:211-244 (OK; file_lines=399)
- [4] tests/test_llm_router.py:576-588 (OK; file_lines=603)
- [5] tests/test_llm_router.py:579-580 (OK; file_lines=603)

### 8. build_classifier rules lora 分类器
- [1] src/ragm/classifier/factory.py:37-58 (OK; file_lines=58)
- [5] src/ragm/config.py:104-151 (OK; file_lines=228)

### 9. MilvusStore hybrid_search BM25 混合检索
- [1] src/ragm/store/milvus_store.py:72-78 (OK; file_lines=96)
- [3] src/ragm/store/milvus_store.py:72-96 (OK; file_lines=96)
- [6] src/ragm/store/milvus_store.py:39-71 (OK; file_lines=96)
- [2] src/ragm/store/milvus_store.py:44-45 (OK; file_lines=96)
- [4] src/ragm/store/milvus_store.py:87-92 (OK; file_lines=96)
- [5] scripts/eval_all_dilution.py:66-79 (OK; file_lines=83)

### 10. LangGraphRagService stream
- [2] src/ragm/rag/langgraph_service.py:281-297 (OK; file_lines=319)
- [3] src/ragm/rag/langgraph_service.py:288-319 (OK; file_lines=319)
- [1] src/ragm/rag/langgraph_service.py:303-310 (OK; file_lines=319)
- [4] tests/test_llm_router.py:84-96 (OK; file_lines=603)
