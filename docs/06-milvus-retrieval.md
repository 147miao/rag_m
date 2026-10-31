# Milvus 索引与混合检索（M1/M2，2026-10-31）

## 一、语料与索引

| 项 | 值 |
|---|---|
| 语料 | HotpotQA 段落 37,368 + CMRC 文章切块 2,651 = **40,019 chunks** |
| 平均长度 | 548 字符（EN 段落级、ZH 800 字/块 + 100 字重叠） |
| Collection | rag_chunks |
| Dense | google/embeddinggemma-300m，768 维，HNSW（M=16, efConstruction=200, COSINE） |
| Sparse | Milvus 3.0 BM25 Function（text 字段 analyzer=chinese），SPARSE_INVERTED_INDEX |
| 元数据 | chunk_id / doc_id / title / source / lang / chunk_index / text |
| 入库耗时 | 630 秒（67 chunks/s，40,019 行，bf16 编码 + 批量 insert） |

## 二、检索评测（与 embedding 对比同一套 query/gold）

| 语言 | 模式 | R@5 | R@10 | R@20 | nDCG@10 | MRR@10 | 延迟 |
|---|---|---|---|---|---|---|---|
| EN（300 query） | dense | 0.790 | 0.838 | 0.863 | 0.7982 | 0.9005 | ~31ms |
| EN | sparse(BM25) | 0.720 | 0.815 | 0.875 | 0.7464 | 0.8687 | ~32ms |
| EN | **hybrid(RRF)** | 0.802 | **0.888** | **0.940** | **0.8032** | **0.8907** | ~32ms |
| ZH（100 query） | dense | 0.880 | 0.900 | 0.900 | 0.8715 | 0.8624 | ~31ms |
| ZH | sparse(BM25) | 0.970 | 0.980 | 0.980 | **0.9610** | **0.9548** | ~31ms |
| ZH | hybrid(RRF) | 0.970 | **1.000** | **1.000** | 0.9532 | 0.9380 | ~32ms |

结论：

- **EN 用 hybrid**：nDCG@10 比 dense 高 0.005、R@20 高 0.077，BM25 补足专有名词/年份类查询；
- **ZH 的 BM25 极强**（nDCG 0.961），hybrid 召回最高（R@10 1.0）但 nDCG 略降；当前默认 hybrid，可接受；
- **query prompt 很关键**：给 embeddinggemma 加 "task: search result | query: " 后，EN dense nDCG 0.751→0.798、ZH 0.715→0.872。

## 三、API

    POST /api/search
    {"query": "范廷颂是什么时候被任为主教的？", "k": 3, "mode": "hybrid"}   # mode: dense|sparse|hybrid
    → {"query": "...", "mode": "hybrid", "latency_ms": 37.7, "hits": [{chunk_id, doc_id, title, source, lang, chunk_index, score, text}]}

- 服务启动时加载 embeddinggemma（1.2GB）+ 分类器（1.2GB），Milvus 不可用时 /api/search 返回 503 并降级；
- 单条查询延迟：预热后 ~30-38ms（hybrid，含 query 编码 + 检索）。

## 四、复现

    python scripts/build_rag_corpus.py                 # 生成 data/rag/corpus.jsonl
    python scripts/ingest_milvus.py --recreate         # 编码 + 入库（约 10 分钟）
    python scripts/eval_retrieval_milvus.py            # dense/sparse/hybrid 对比
    python scripts/search_demo.py "query" --k 5        # 单条演示

## 五、待办

- reranker 消融（Qwen3-Reranker-0.6B，需空出 GPU）；
- 用 hybrid 检索替换 DemoRagService 的 simple 链路（LangGraph 接入）；
- 多跳/图谱链路与 Agentic 链路。
