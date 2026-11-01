# Graph RAG（多跳路由，2026-11-01）

## 一、图谱构建

- **实体**：HotpotQA 段落标题（105,570 个实体节点，lang=en）；
- **关系**：同一道 HotpotQA 题的 `supporting_facts` 标题两两建 `CO_EVIDENCE` 有向边，边属性 = 问题 id / 问题类型（bridge/comparison）/ 问题原文；
- **规模**：180,894 条边，来自 90,447 道题；
- **存储**：Neo4j 5.26（`ragm-neo4j` 容器，bolt://127.0.0.1:7687），Entity.name 唯一约束；
- **构建/导入脚本**：`scripts/build_kg.py`（约 20s）、`scripts/load_kg_neo4j.py`（约 40s）。

为什么用 co-evidence 边：HotpotQA 的多跳题本身就是"两个实体由一道题连起来"，这条边正好编码了多跳关系；
相比 LLM 逐段抽三元组（37k 段 × 1.3s ≈ 13 小时），这个方案 1 分钟建完且覆盖全部实体。

## 二、多跳检索流程（仅 multihop 路由）

    question
      → hybrid 检索 top-12（dense + BM25 + RRF）
      → 实体链接：先匹配问题中直接出现的实体名（长名优先），再用检索结果标题兜底（最多 6 个）
      → Neo4j 1 跳扩展（最多 40 条边）→ 邻居实体（最多 12 个）
      → 按邻居标题从 Milvus 补证据 chunk（最多 8 条）
      → 图谱证据 + hybrid 证据合并（图谱在前，去重，cap top-16）
      → prompt 注入【图谱关系】+ 编号资料 → 生成 → 引用校验
      → 响应带 graph 节点/边（前端"图谱"面板）

实现：`src/ragm/graph/kg.py`（KGStore）、`RagGraph._retrieve_node`、`SearchService.chunks_by_titles`。

## 三、实测（40 道 HotpotQA 多跳题：bridge + comparison）

| 指标 | hybrid-only | hybrid + Graph RAG |
|---|---|---|
| gold 文档命中率（top-12 去重） | 1.000 | 1.000 |
| **gold 文档召回** | 0.875 | **1.000** |
| 图谱额外带回的 gold 文档/题 | — | **0.25** |
| 检索延迟 | 43.4ms | +23.8ms（图谱扩展） |

结论：**多跳题的两跳证据召回从 87.5% 补到 100%**，代价是 ~24ms 的图谱扩展；
单跳题不受影响（只有 multihop 路由走图谱）。

端到端示例（截图 12-graph-rag.png）：

> Q: Which US documentary was made first: Scared Straight! or Soldier Girls?
> 实体链接：Scared Straight! / Soldier Girls → 图谱邻居（Front Lines、Dark Girls、Winter Soldier…）
> A: **Scared Straight! was made first [5][6][7].**（1978 vs 1981，3/3 引用通过，875ms，图谱 28 条边）

## 四、API 与健康检查

- `GET /api/health` 新增 `kg` 检查：`105570 entities`；Neo4j 不可用时自动禁用图谱扩展（不阻塞主链路）；
- `POST /api/query` 的 multihop 响应带 `graph: {nodes, edges}`，前端"图谱"面板直接渲染实体路径；
- 启动方式：`docker compose --profile graph up -d neo4j`（或 `make up` 带 profile）。

## 五、已知限制

1. **实体链接是字符串匹配**：问题里没直接出现实体名、且 hybrid 首屏也没命中时，图谱扩展会偏；后续可用 embedding 相似度做实体链接；
2. **图谱只覆盖英文 HotpotQA**：中文语料（CMRC）没有实体图；
3. **关系语义弱**：co_evidence 边只表示"同题共现"，没有明确谓词；后续可对高频实体用 LLM 抽三元组补语义关系；
4. 多跳答案仍由一次生成完成（没有显式多步推理链），下一步可做 iterative retrieval。

## 六、复现

    docker compose --profile graph up -d neo4j
    .venv/bin/python scripts/build_kg.py
    .venv/bin/python scripts/load_kg_neo4j.py
    .venv/bin/python scripts/eval_graph_rag.py    # 对比 hybrid-only vs graph
    curl -s -X POST localhost:8000/api/query -H "Content-Type: application/json" \
         -d '{"question": "Which US documentary was made first: Scared Straight! or Soldier Girls?"}'
