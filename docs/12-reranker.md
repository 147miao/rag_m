# Reranker 横向对比 + 接入（2026-11-01）

## 一、评测协议

- 候选池：hybrid 检索（dense + BM25 + RRF）top-20；
- 语料：EN HotpotQA 37,368 段 / 300 query；ZH CMRC 2,403 篇 / 100 query（与 embedding 对比同一套）；
- 指标：段落级 Recall@5/10、nDCG@10、MRR@10；延迟为单条 query 重排 20 个候选的 p50；
- 硬件：RTX 4060 8GB，bf16。

## 二、结果

### 英文（HotpotQA）

| Reranker | R@5 | R@10 | nDCG@10 | MRR@10 | p50 | 显存 |
|---|---|---|---|---|---|---|
| 无重排（hybrid） | 0.805 | 0.885 | 0.8103 | 0.9009 | — | — |
| **bge-reranker-base** | 0.915 | 0.930 | **0.9217** | **0.9867** | **59ms** | **1.22G** |
| bge-reranker-v2-m3 | 0.915 | 0.933 | 0.9147 | 0.9773 | 231ms | 1.23G |
| qwen3-reranker-0.6b | 0.863 | 0.905 | 0.8655 | 0.9502 | 552ms | 2.85G |

### 中文（CMRC2018）

| Reranker | R@5 | R@10 | nDCG@10 | MRR@10 | p50 | 显存 |
|---|---|---|---|---|---|---|
| 无重排（hybrid） | 0.980 | 1.000 | 0.9631 | 0.9516 | — | — |
| bge-reranker-base | 1.000 | 1.000 | 0.9950 | 0.9933 | 107ms | 1.22G |
| bge-reranker-v2-m3 | 1.000 | 1.000 | 0.9950 | 0.9950 | 330ms | 1.23G |
| qwen3-reranker-0.6b | 1.000 | 1.000 | **0.9963** | **0.9950** | 1213ms | 3.21G |

## 三、结论：选 bge-reranker-base

1. **英文 nDCG@10 0.922**，比无重排 +11.1 点，是四个方案里最高的；中文 0.995（只比 qwen3 低 0.0013）；
2. **最快**：p50 59ms（v2-m3 231ms、qwen3 552ms），显存 1.22GB 最小；
3. qwen3-reranker 中文略优但慢 10~20 倍，英文明显更差（0.866），不值得；
4. v2-m3 与 base 接近（0.915 vs 0.922），但慢 4 倍、体积 2 倍。

## 四、接入

    # .env
    RERANK_MODEL=bge-reranker-base
    RERANK_ENABLED=true
    RERANK_MODEL_PATH=/run/media/myj/D/rag_m/models/rerankers/bge-reranker-base
    RERANK_CANDIDATES=20
    RERANK_TOP_N=8

- 接入点：`RagGraph._retrieve_node`（simple/multihop），hybrid + 图谱扩展后 → top-20 → rerank → top-k；
- 响应 trace 增加「重排」步骤（实测 77ms / 12 条候选）；
- agentic 路由的 `vector_search` 工具仍是 hybrid（工具内部可后续加重排）。

实测（multihop，带图谱+重排）：路由 34ms + 检索 197ms（图谱 +4）+ 重排 77ms + 生成 1.7s ≈ **2.0s**，引用 3/3 通过。

## 五、LangSmith 验证

用户提供 key 后已启用：`/api/health` → `langsmith: enabled · rag-m`；
通过 LangSmith SDK 查到项目 `rag-m` 下的 trace（示例）：

    LangGraph (chain) → classify / retrieve / generate / verify (chain)
      └─ hybrid_search (retriever)
      └─ llm_chat (llm)

全部 `success`，可到 LangSmith 项目页按 run 查看耗时、输入输出与 token。

## 六、已知限制

1. 评测是"hybrid top-20 → rerank"的离线指标，未做端到端生成质量评测（citation 质量仍靠抽样）；
2. 重排只接在 simple/multihop，agentic 工具内部未接；
3. qwen3-reranker 的 OOM 是 8GB 显存下与 API 进程共存导致，单独跑没问题（已记录）；
4. 中文语料仅 2,403 篇，指标区分度有限。

## 七、复现

    .venv/bin/python scripts/fetch_rerankers.py
    .venv/bin/python scripts/bench_rerankers.py --langs en,zh
    # 结果：reports/reranker_benchmark.json
