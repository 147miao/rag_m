# 火影语料横向测试矩阵（2026-11-01）

测试集：28 道火影问题（16 事实 + 12 多跳），gold = 条目名（如 "波风水门"、"自来也(火影忍者)"），语料 220 篇 / 117 万字。

## 一、检索（Recall@5/10、nDCG@10、MRR@10、p50）

### Embedding 模型

| 模型 | R@5 | R@10 | nDCG@10 | MRR | p50 |
|---|---|---|---|---|---|
| **embeddinggemma-300m** | 0.518 | 0.589 | **0.511** | **0.588** | 29.5ms |
| bge-m3 | 0.518 | 0.625 | 0.490 | 0.526 | 12.3ms |
| qwen3-embedding-0.6b | 0.536 | 0.625 | 0.475 | 0.525 | 19.4ms |
| bge-large-zh-v1.5 | 0.429 | 0.500 | 0.395 | 0.443 | 12.0ms |

结论：与 HotpotQA/CMRC 评测一致，**embeddinggemma-300m 仍是 nDCG 第一**；bge-m3 R@10 略高但排序质量差一点。

### 检索模式（embeddinggemma）

| 模式 | R@5 | R@10 | nDCG@10 | MRR |
|---|---|---|---|---|
| **hybrid（dense+BM25+RRF）** | **0.518** | **0.589** | **0.511** | **0.588** |
| dense | 0.375 | 0.500 | 0.371 | 0.388 |
| sparse（BM25） | 0.393 | 0.446 | 0.369 | 0.420 |

结论：中文语料上 **hybrid 显著优于单路**（nDCG +14 点），与 CMRC 评测一致。

### Reranker

| Reranker | R@5 | R@10 | nDCG@10 | MRR | p50 |
|---|---|---|---|---|---|
| 无重排 | 0.518 | 0.589 | **0.511** | **0.588** | 30.8ms |
| bge-reranker-base | 0.554 | 0.589 | 0.456 | 0.473 | 81.9ms |
| bge-reranker-v2-m3 | 0.554 | 0.589 | 0.450 | 0.454 | 210ms |
| qwen3-reranker-0.6b | 0.500 | 0.589 | 0.444 | 0.482 | 849ms |

**结论：在本测试集上重排反而降低 nDCG（0.511 → 0.456）**。原因：gold 是"条目名"，hybrid 的标题匹配已经很强，
reranker 按段落语义重排会打乱标题命中的顺序；R@5 反而略升（0.518→0.554），说明重排把正确文档挤进了前 5 但排序更差。
这提醒：**重排是否有效取决于评测口径**（标题级 vs 段落级），不能只看一种指标。

### Chunk size（固定 embeddinggemma）

| chunk 字符 | chunks | R@5 | R@10 | nDCG@10 | MRR |
|---|---|---|---|---|---|
| 200 | 8,395 | 0.500 | 0.554 | 0.380 | 0.382 |
| 400 | 5,018 | 0.518 | 0.607 | 0.438 | 0.473 |
| **800** | 1,782 | 0.500 | 0.589 | 0.522 | 0.604 |
| **1600** | 1,782 | 0.536 | 0.589 | **0.524** | **0.605** |

结论：长文档语料上 **大 chunk（800–1600）明显更好**（nDCG 0.38→0.52），与 HotpotQA 短段落语料"小块更精确"的结论相反——
因为这里 gold 是整篇条目，块越大越容易覆盖答案；也说明 chunk size 没有普适最优，取决于文档形态和评测粒度。

## 二、生成质量（28 题，含 12 多跳）

| 配置 | 正确率 | faithfulness | context_recall | citation_valid | p50 |
|---|---|---|---|---|---|
| rerank off | **1.000** | **0.955** | 1.000 | 1.000 | 1,262ms |
| rerank on | **1.000** | 0.948 | 1.000 | 1.000 | 1,545ms |

结论：生成层 **28/28 全对、引用 100%**；重排对生成质量没有增益（faithfulness 略降 0.7 点、延迟 +283ms），
与检索矩阵的结论一致——在本语料/评测口径下重排不划算。

## 三、汇总：火影语料 vs 之前 HotpotQA/CMRC 评测

| 维度 | 之前结论 | 火影语料结论 | 是否一致 |
|---|---|---|---|
| 最佳 embedding | embeddinggemma-300m | embeddinggemma-300m | ✅ 一致 |
| hybrid vs 单路 | hybrid 最优 | hybrid 最优（+14 点） | ✅ 一致 |
| reranker | 英文检索 +11 点、生成 +7.5 | 检索 -5.5 点、生成 -0.7 | ❌ 相反（评测口径不同） |
| chunk size | 中文短文章 200 略优（噪声内） | 长文档 800–1600 明显更优 | ❌ 相反（文档形态不同） |

## 四、说明与局限

1. 本矩阵的检索 gold 是**条目名**（标题级），与之前的段落级 gold 不同，导致 reranker/chunk 结论出现反转；
2. 生成质量里的 context_precision 指标在脚本中有 bug（值 >1），已从报告剔除，只保留 faithfulness/recall/citation；
3. 28 题样本偏小（CI 宽），结论用于方向判断；
4. 语料 220 篇 / 117 万字，单源（萌娘百科），无 fandom 数据（不可达）。

## 五、复现

    .venv/bin/python scripts/naruto_retrieval_matrix.py    # 检索矩阵（约 6 分钟）
    .venv/bin/python scripts/report_naruto_matrix.py       # 渲染报告
    # 生成矩阵：API 指向火影语料，RERANK_ENABLED=false/true 各跑一次
    .venv/bin/python scripts/naruto_generation_matrix.py rerank_off
    .venv/bin/python scripts/naruto_generation_matrix.py rerank_on
