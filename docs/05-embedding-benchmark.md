# Embedding 模型横向对比（2026-10-31）

## 一、评测协议

- 语料：
  - EN：HotpotQA 段落池 37,368 段（4,000 题 context 并集，按标题去重）；
  - ZH：CMRC2018 文章池 2,403 篇（按正文去重）。
- Query：EN 300 条（HotpotQA，gold = supporting_facts 段落，平均 2.0 个）；ZH 100 条（CMRC，gold = 所属文章）。
- 检索：精确 top-50（L2 归一化 + COSINE），指标为段落级 Recall@5/10/20、nDCG@10、MRR@10。
- 编码：max_seq_length=512、batch=32、bf16、RTX 4060 8GB；同一脚本、同一 pooling 规则（bge 系列 CLS、Qwen3 last-token、embeddinggemma mean）。
- 数据与脚本：scripts/build_retrieval_eval.py、scripts/bench_embeddings.py、scripts/report_embeddings.py。

## 二、结果

### 英文（HotpotQA，37,368 段 / 300 query）

| 模型 | 维度 | R@5 | R@10 | R@20 | nDCG@10 | MRR@10 | docs/s | 峰值显存 |
|---|---|---|---|---|---|---|---|---|
| bge-m3 | 1024 | 0.840 | 0.885 | 0.912 | 0.8481 | 0.9489 | 66.0 | 1.44 GiB |
| embeddinggemma-300m | 768 | 0.845 | 0.900 | 0.950 | 0.8476 | 0.9424 | 66.5 | 1.18 GiB |
| qwen3-embedding-0.6b | 1024 | 0.767 | 0.828 | 0.867 | 0.7872 | 0.9149 | 24.6 | 3.31 GiB |
| bge-large-zh-v1.5 | 1024 | 0.560 | 0.615 | 0.658 | 0.5888 | 0.7728 | 45.4 | 0.99 GiB |

### 中文（CMRC2018，2,403 篇 / 100 query）

| 模型 | 维度 | R@5 | R@10 | R@20 | nDCG@10 | MRR@10 | docs/s | 峰值显存 |
|---|---|---|---|---|---|---|---|---|
| embeddinggemma-300m | 768 | 0.990 | 0.990 | 0.990 | 0.9765 | 0.9720 | 39.8 | 1.18 GiB |
| bge-m3 | 1024 | 0.980 | 0.990 | 0.990 | 0.9594 | 0.9498 | 38.3 | 1.44 GiB |
| qwen3-embedding-0.6b | 1024 | 0.960 | 0.960 | 0.980 | 0.9476 | 0.9448 | 15.3 | 3.31 GiB |
| bge-large-zh-v1.5 | 1024 | 0.970 | 0.970 | 0.980 | 0.9379 | 0.9274 | 36.4 | 0.99 GiB |

## 三、结论

1. 默认选 google/embeddinggemma-300m：中文 nDCG@10 0.977 第一，英文 0.848 与 bge-m3 并列（R@10 0.900 更高），768 维、显存 1.18GB、吞吐与 bge-m3 持平，综合最优。
2. bge-m3 作为备选：英文 nDCG 略高 0.0005（无统计意义），优势是 8192 长上下文与自带 sparse/ColBERT（本项目的稀疏检索由 Milvus BM25 Function 承担，暂不需要）。
3. Qwen3-Embedding-0.6B 在本评测中明显低于预期（EN nDCG 0.787、ZH 0.948），且吞吐最低（24.6 docs/s）、显存最高（3.31GB）。已排查：last-token pooling 实现正确（mini 集 R@5 0.925，与 bge-m3 持平）、加 EOS 无显著变化、instruction 与官方一致；结论是"在这个语料/任务上它不占优"，不作为默认。
4. bge-large-zh-v1.5 只适合纯中文：英文 nDCG 0.589，跨语能力弱，但中文仍可用（0.938）。
5. Reranker 消融暂缓：Qwen3-Reranker-0.6B 在 8GB 显存下与 bge-m3 同时驻留会 OOM，且 API 服务常驻占用 1.2GB；需要单独空出 GPU 再跑（待办）。

## 四、对配置的影响

    # .env
    EMBED_MODEL=google/embeddinggemma-300m
    EMBED_DIM=768
    EMBED_MODEL_PATH=/run/media/myj/D/rag_m/models/embeddings/embeddinggemma-300m
    # 备选：BAAI/bge-m3（1024 维，长上下文 + 自带 sparse）

## 五、复现

    python scripts/build_retrieval_eval.py
    python scripts/bench_embeddings.py --out reports/embedding_benchmark.json --batch 32 --max-len 512
    python scripts/report_embeddings.py

## 六、已知局限

- EN 语料来自 HotpotQA 段落（Wikipedia 风格），与项目最终语料分布可能不同；
- ZH 语料仅 2,403 篇，指标偏高、区分度有限；
- 只比较 dense 检索，未比较 sparse/多向量（bge-m3 的 sparse 能力未启用）；
- reranker 消融缺失（显存限制）；
- Qwen3 结果基于手动 pooling 实现，未用官方 sentence-transformers 配置（ST 6.2 与该模型 modules.json 不兼容）。
