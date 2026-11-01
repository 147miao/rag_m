# 测试加速 + 真实长文档（2026-11-01）

## 一、为什么之前慢、利用率低

1. **单卡 8GB**：embedding / reranker 模型各约 1.2GB，同一时刻只能加载一个（之前多进程并发直接 OOM）；
2. **任务串行**：GPU 阶段（embedding、rerank）和网络阶段（LLM judge）本来互不依赖，之前却串行执行，
   导致 GPU 空转等 LLM、CPU 空转等 GPU；
3. **重复计算**：每个轴重新检索、重新加载模型；systemd 的 `ragm-api.service` 常驻占 2.4GB 显存，测试时互相抢显存。

## 二、加速方案（已实现）

| 组件 | 作用 |
|---|---|
| `scripts/bench_common.py` | GPU 文件锁（多进程串行用卡）、候选结果缓存、CPU/GPU 利用率监控、API 服务停止/恢复 |
| `scripts/bench_all.py` | 并行编排：停止常驻 API → 启动轻量 API → GPU 矩阵与 LLM 评测**同时跑** → 恢复服务 |
| 候选缓存 | 同一 query/collection 的 top-20 只算一次，reranker/chunk 轴复用 |
| 资源监控 | 每秒采样 `/proc/stat` + `nvidia-smi`，输出 cpu_avg/max、gpu_avg/max |

用法：

    .venv/bin/python scripts/bench_all.py              # 并行跑 span 矩阵 + 生成评测
    .venv/bin/python scripts/bench_all.py --skip-gen   # 只跑 GPU 矩阵

## 三、实测效果

| 指标 | 串行（之前） | 并行（现在） |
|---|---|---|
| 两个测试套件总耗时 | 约 4~5 分钟 | **98.8 秒** |
| GPU 平均利用率 | 10~20% | **79.6%**（峰值 100%） |
| CPU 峰值利用率 | 低 | **78.3%**（平均 17.7%，主要是 LLM 等待期） |

结论：**GPU 已接近吃满（~80%）**；CPU 平均利用率低是因为大部分时间在等云端 LLM 返回（网络 I/O），
这不是 CPU 瓶颈——如果要提高整体吞吐，应增加并发 judge 请求（当前 3 路）而不是加 CPU 线程。

## 四、真实长文档：arXiv 论文

- 来源：arXiv PDF（8 篇 RAG 相关论文，`https://arxiv.org/pdf/<id>` 可直连）；
- 解析：`pypdf` 抽取文本（8 页论文约 3.2 万字符，0.1 秒/篇）；
  Unstructured `partition_pdf(strategy="fast")` 在本机 120 秒返回 0 个元素，暂不使用；
- 入库：6 篇有效（2 篇提取字符 <3000 被跳过），704 chunks / 3,072 children / 310 parents；
- 集合：`rag_papers_chunks`、`rag_papers_children`；
- 测试：10 道 RAG 领域问题（RAG 组成、RAGAS 指标、Self-RAG、GraphRAG、Corrective RAG、RAG 三阶段…），
  **9/10 正确**，p50 1,074ms，引用全部有效。

失败题：GraphRAG 优势（该论文未在 6 篇有效文档中，或检索未命中），系统如实拒答。

## 五、发现的问题（如实记录）

1. **火影 chunk-size 矩阵有数据构造 bug**：之前的实现是对 800 字 chunk 再切块（而不是对原文切块），
   导致同一文档多个块共享 `chunk_id`，标题级指标不受影响，但 chunk 级 gold 无法对齐；
   因此 docs/17 里火影 chunk-size 的数值不可信，需要修脚本重跑；
2. span 级 gold 矩阵脚本已写（`scripts/naruto_span_matrix.py`），待修完上述构造后重跑；
3. 语料扩充（目标 300+ 篇）本轮未完成，抓取脚本在 `/tmp/crawl_more.py`，下次继续。

## 六、复现

    # 并行基准
    .venv/bin/python scripts/bench_all.py
    # 论文语料
    .venv/bin/python scripts/build_papers_corpus.py
    .venv/bin/python scripts/ingest_milvus.py --corpus .../papers/chunks.jsonl --collection rag_papers_chunks --recreate
    .venv/bin/python scripts/ingest_milvus.py --corpus .../papers/children.jsonl --collection rag_papers_children --with-parent --recreate
    .venv/bin/python scripts/test_papers_rag.py
