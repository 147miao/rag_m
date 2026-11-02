# 全量语料统一加载（2026-11-02）

问题：每次只能切一个知识库（demo / 火影 / 论文），能不能都加载进去、一个服务全答？

## 一、做法

1. 语料合并：默认演示语料 40,019 + 火影 1,782 + arXiv 704 = **42,505 chunks**；parents 合并 40,915；
2. 索引：新建集合 `rag_all_chunks`（768 维 embeddinggemma + BM25），一次入库；
3. 图谱合并：`KGStore` 支持多图谱（`label=Entity,NarutoEntity`、`relation=CO_EVIDENCE,MENTIONS`），
   节点合并 105,788 个，实体链接/扩展时同时查两张图；
4. 切换：`make use-all`（写 systemd drop-in 后重启服务）。

## 二、实测：一个服务回答三类语料

| 问题 | 路由 | 正确 | 证据来源 | 引用 |
|---|---|---|---|---|
| 鸣人父亲是谁 | simple | ✅ 波风水门 | moegirl | 2/2 |
| Which US documentary was made first… | multihop | ✅ Scared Straight! | HotpotQA | 3/3 |
| What metrics does RAGAS use? | simple | ✅ faithfulness 等 | arxiv | 4/4 |
| 自来也的弟子中谁成为了四代火影？ | simple | ✅ 波风水门 | moegirl | 1/1 |
| What is the GraphRAG approach? | simple | ⚠️ 证据混入 HotpotQA，答案不完整 | HotpotQA+arxiv | 8/8 |

结论：**统一索引可用**，4/5 问题正确且证据来源正确；跨语料问题（GraphRAG）会混入无关语料。

## 三、代价：检索稀释（Naruto 28 题，标题级 gold）

| 索引 | R@5 | R@10 | nDCG@10 | MRR |
|---|---|---|---|---|
| 火影专用 `rag_naruto_chunks` | 0.554 | 0.661 | **0.553** | 0.639 |
| 全量 `rag_all_chunks` | 0.500 | 0.625 | 0.469 | 0.514 |

合并后火影检索 **nDCG@10 下降 8.4 点**（0.553 → 0.469）：其他语料的 4 万多个 chunk 成为干扰项。

## 四、建议

1. **体验优先**（一个服务答所有问题）：用全量索引，接受 5~10 点检索精度损失；
2. **精度优先**：保留独立集合 + 路由（当前 `use_corpus.sh` 切换），或加**按来源过滤**：
   - Milvus 标量字段 `source`（hotpotqa / cmrc / moegirl / arxiv）已存在，可给 `/api/search`、`/api/query` 加 `sources` 参数；
   - 或按问题语言/关键词自动选源（如检测到火影实体 → 只查火影+全量兜底）；
3. 折中：全量索引 + 查询时按 `source` 软加权（先全量召回，再给匹配来源的候选加分）。

## 五、复现

    # 合并语料
    python3 - <<PY  # 见本轮脚本，输出 data/all/chunks.jsonl + parents.jsonl
    # 入库
    .venv/bin/python scripts/ingest_milvus.py --corpus /run/media/myj/D/rag_m/data/all/chunks.jsonl --collection rag_all_chunks --recreate
    # 切换
    make use-all      # 全量
    make use-default  # 默认演示语料
    # 稀释评估
    .venv/bin/python scripts/eval_all_dilution.py
    # 三语料问答
    .venv/bin/python scripts/test_all_corpus.py
