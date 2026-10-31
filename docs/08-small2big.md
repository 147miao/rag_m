# Small-to-Big 检索（小块精确检索 + 大块上下文）

> 思想：为检索精确性索引小块（child），为上下文丰富性返回大块（parent）。

## 一、实现

- **child 索引**：Milvus collection `rag_children`（46,142 条）
  - ZH（CMRC 文章）：200 字 / overlap 40；
  - EN（HotpotQA 段落）：段落本身就是自然段（~550 字），child = parent（1:1）。
- **parent 存储**：`data/rag/parents.jsonl`（39,771 条），服务启动时加载进内存（~22MB），parent = 整篇文章或 1600 字切片；
- **检索流程**：query → child hybrid 检索 top-50 → 按 parent_id 去重 → 取前 k 个 parent → 返回 parent 全文 + 命中 child 片段；
- **API**：`POST /api/search {"query": "...", "k": 4, "mode": "small2big"}`；
- **代码**：`scripts/build_corpus_small2big.py`、`scripts/ingest_milvus.py --with-parent`、`SearchService.search_small2big`。

## 二、短文档语料（CMRC 文章，avg ~600 字）：没有收益

| 方案 | doc hit@8 | answer coverage@8 | 平均上下文 | 延迟 |
|---|---|---|---|---|
| baseline：800 字 chunk hybrid top-8 | 1.00 | 1.00 | 3,591 字 | 28.9ms |
| small2big：200 字 child → parent top-8 | 0.98 | 0.98 | 4,282 字 | 30.5ms |

原因：文章本身只有几百字，800 字 chunk 基本等于整篇，parent 没有更多信息；反而多返回了上下文。

## 三、长文档语料（PubMedQA 摘要，1,000 篇，avg ~1.2k 字）：收益显著

| 方案 | doc hit@1 | evidence coverage@8 | 平均上下文字符 |
|---|---|---|---|
| baseline：800 字 chunk top-8 | 0.97 | **0.70** | 5,543 |
| small2big：200 字 child → parent **k=2** | 0.94 | **0.98** | **2,784** |
| small2big：k=4 | 0.94 | 0.99 | 5,612 |
| small2big：k=6 | 0.94 | 0.99 | 8,433 |
| small2big：k=8 | 0.94 | 0.99 | 11,341 |

- 同样约 5.5k 上下文字符：coverage 从 0.70 → 0.99（+29 点）；
- 只要 2.8k 字符（baseline 的一半）就能到 0.98 coverage；
- 代价：doc hit@1 从 0.97 → 0.94（child 检索排序 + parent 去重的轻微损失）；延迟持平（~29ms）。

## 四、结论

1. **small-to-big 的价值取决于文档长度**：短文档（<800 字）没有收益，长文档（>1k 字）收益巨大；
2. **推荐配置**：child 200 字 / overlap 40，parent = 整篇或 1600 字切片，**k_parents=4**（约等于 baseline 的上下文字符数，coverage +29 点）；
3. 如果生成模型上下文预算紧张，k_parents=2 即可（2.8k 字符，coverage 0.98）；
4. 下一步接入 PDF/DOCX 长文档时，默认走 small-to-big；当前 CMRC/HotpotQA 短语文料保持 800 字 chunk 即可。

## 五、复现

    python scripts/build_corpus_small2big.py
    python scripts/ingest_milvus.py --corpus .../children.jsonl --collection rag_children --with-parent --recreate
    python scripts/bench_small2big_long.py        # 长文档对照（PubMedQA）
    python scripts/bench_small2big_sweep.py       # k_parents 权衡曲线

结果：reports/small2big_long.json、reports/small2big_sweep.json。
