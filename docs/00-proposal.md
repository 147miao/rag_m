# RAG-M 项目方案 v0.2（商议稿）

> 目标：一个可运行、可评测、可写进简历的"路由式混合 RAG"学习项目。
> 技术栈：LangGraph + Milvus + Docker + git；文档解析 Unstructured；embedding 做横向对比。
> 本文所有环境结论均来自 2026-10-30 在本机的实测（命令输出为准），版本号来自 PyPI/GitHub Release 线上核实。

---

## 一、需求理解

用户提问 → **查询分类器** → 三条链路：

| 链路 | 触发条件 | 核心能力 |
|---|---|---|
| 传统 RAG | 单跳事实型 | Milvus 混合检索（dense + BM25）→ 重排 → 带引用生成 |
| Graph RAG | 多跳关系型 | 实体链接 → 知识图谱 1~2 跳遍历 → 子图证据 |
| Agentic RAG | 复杂推理型 | LangGraph ReAct 子图，按需调用 vector_search / kg_query / web_search |

真正的难点不是"三条链路各自能跑"，而是：
1. **路由器可评测**（准确率、混淆矩阵、误路由代价），否则架构没有说服力；
2. **三条链路共享一套索引与 chunk 元数据**，否则维护三份数据；
3. **统一评测口径**（检索/答案/系统三层），能回答"路由到底值不值"。

---

## 二、版本基线（2026-10-30 线上核实 + 本机实测）

| 组件 | 锁定版本 | 说明 |
|---|---|---|
| Python | **3.13.15**（uv 已自带） | 系统 3.14 装不了 unstructured（要求 >=3.11,<3.14） |
| langgraph | **1.2.18** | 1.x GA；`create_react_agent` 已弃用 |
| langchain | **1.4.7** | Agent 入口改为 `langchain.agents.create_agent` |
| langgraph-checkpoint-sqlite | 3.1.2 | 会话记忆/时间旅行 |
| pymilvus | **3.0.2** | 与 Milvus 3.0.x 配套 |
| Milvus | **v3.0.4**（本地镜像已有） | standalone = etcd + minio + milvus |
| milvus-lite | 3.2.2 | 可选：无 Docker 的开发/CI 后端 |
| unstructured | **0.27.36** | 开源版只有 auto/fast/ocr_only/hi_res，**无 vlm** |
| datasets | 5.1.0 | 4.0 起移除 loading script，只能走 parquet |
| ragas | 0.4.3 | **必须 pin `langchain-community<0.4`**，否则 import 崩 |
| sentence-transformers | 6.2.0 | embedding 对比用 |
| torch | 2.15.0（CPU 轮子） | 解析链路不装 CUDA 栈，省 3~5 GB 磁盘 |
| Neo4j | 5.26-community（镜像可拉） | 图库候选（可选） |

**已实测通过的关键能力（本次会话）**
- Milvus 3.0.4 standalone：3 容器启动，`healthz` OK；
- pymilvus 3.0.2：中文 analyzer + BM25 Function + RRF hybrid_search 全部返回正确结果；
- Ollama 免费云模型 `gemma4:31b-cloud`：中文路由分类 6/6 合理，输出 JSON，单次 0.7~3.4 s；**tool calling 正常**；
- bge-m3（Ollama）：batch 32 编码约 125 texts/s（GPU），1024 维；
- hf-mirror 下载 HotpotQA val parquet 26.7 MB，约 3.8 MB/s；
- Docker Desktop 已启动（VM 8.3 GB / 20 CPU），本地已有 Milvus/etcd/minio 镜像；
- Neo4j 5.26-community 镜像可通过 `docker.1panel.live` 拉取（manifest 校验通过）。

**不可用/受限**
- huggingface.co、github.com、Docker Hub、OpenAI API 直连不通 → 走 hf-mirror / ModelScope / 国内 registry 镜像；
- DeepSeek/GLM 的 Ollama cloud 模型需要付费 credits（当前只有 gemma4 免费）；DeepSeek API 需用户充值 key；
- 本机 sudo 需密码 → 不能自行 apt 安装 tesseract（Unstructured 的 fast 策略可纯 pip 跑通）。

---

## 三、总体架构

```
【离线索引】
 raw docs ──► Unstructured 解析 ──► 清洗 ──► chunk_by_title(1200/150) ──► embedding ──► Milvus
 (PDF/MD/DOCX/HTML/QA语料)              │                        │
                                        │                        └─ dense(1024) + sparse(BM25) + 标量元数据
                                        └──► 实体/关系抽取 ──► 知识图谱(Neo4j) + 实体别名索引

【在线问答】(LangGraph 主图)
 START ─► route(分类器) ─┬─ simple    ─► hybrid_retrieve ─► rerank ─► generate(引用) ─┐
                         ├─ multihop  ─► entity_link ─► kg_query(1~2跳) ─► generate ──┤─► finalize ─► END
                         └─ reasoning ─► agentic 子图(ReAct) ────────────────────────┘
                                            tools: vector_search / kg_query / web_search
```

**设计要点**
- 分类器 = LLM 结构化输出（`with_structured_output` 约束 Literal）+ 规则先验 + 低置信度兜底到 agentic + 结果缓存；
- 三条链路共用 Milvus chunk 表；KG 每条边带 `source_chunk_id`，回答可溯源到原文；
- Agentic 子图用 `create_agent`（LangGraph 1.x 官方入口），作为子图节点挂进主图；限 max_steps=6、token 预算；
- 拒答不做成第 4 条链路，而是 finalize 前的统一守卫（检索为空/证据不足/不可答判定）。

---

## 四、数据方案（已实测下载并解析，总下载约 130 MB）

**数据集选型（每类主 + 备）**

| 链路 | 主数据集 | 实测规模 | 许可 | 备注 |
|---|---|---|---|---|
| 简单事实 | SQuAD v1.1 val | 10,570 问 / 2,067 篇 | CC-BY-SA-4.0 | 首选 baseline |
| 简单事实（中） | CMRC2018 / DRCD | 3,219 / 3,524 问 | 源 Apache-2.0 | 中文覆盖 |
| 简单事实（检索真值） | MS MARCO v1.1 val | 10,047 问 × 6 passage | CC-BY-4.0 | **唯一有真检索标注**；需剔除 341 条无正例 |
| 多跳关系 | 2WikiMultiHopQA val | 12,576 问 | Apache-2.0 | 自带 evidences 推理链；context 是字符串 JSON |
| 多跳关系 | HotpotQA fullwiki val | 7,405 问 / 66,540 段 | CC-BY-SA-4.0 | 段落级 supporting facts；val 全为 hard |
| 多跳关系 | MuSiQue val | 2,417 问（2/3/4-hop） | 卡片未标 | question_decomposition 可做跳数分层 |
| 多跳（新闻） | MultiHop-RAG | 609 篇 / 2,556 问 | ODC-BY | **301 条 null_query 可做拒答负样本** |
| 复杂推理 | FRAMES | 824 问 | Apache-2.0 | 自带 gold 维基链接，适合多文档推理 |
| 复杂推理 | GraphRAG-Bench | medical 2,062 / novel 2,010 | MIT | 自带评测脚本 |
| 复杂推理（冒烟） | Bamboogle | 125 问 | 卡片未标 | 8 KB，秒下 |

**语料（corpus）**：QA 自带 context 直接当语料，gold 天然对齐。
- `corpus_small` ≈ 2.4 万 chunk（MuSiQue 1.4w + MultiHop-RAG 9k + GraphRAG-Bench 1.5k）→ 日常迭代；
- `corpus_full` ≈ 5 万 chunk（+ HotpotQA 5.2w 抽样）→ 出最终数字；
- `corpus_zh` ≈ 4.5 万 chunk（zhwiki 分片 1 抽 1.5w 页）→ 中文与 Unstructured 实验；
- 最小档 SQuAD-val ≈ 2.3k chunk → 单元测试。

**评测集**
- 路由：300 条（simple / multi_hop / reasoning 各 100），另加 100 条不可答（MultiHop-RAG null + 人工构造）测拒答；
- 检索：500 条 query（MS MARCO 真标注 + 多跳段落级 gold），指标 Recall@5/10/20、nDCG@10、MRR；
- 答案：200 条，EM/F1 + ROUGE-L（免 key）+ LLM judge（正确性/忠实度/引用支撑）。

---

## 五、Embedding 模型对比方案

**最终候选（4 个实测 + 1 个基线，均已确认可下载：ModelScope / hf-mirror）**

| 模型 | 参数 | 维度 | 最大输入 | fp16 权重 | License | 8 GB 4060 判定 |
|---|---|---|---|---|---|---|
| BAAI/bge-m3 | 568M | 1024 | 8192 | 1.14 G | MIT | ✅ 混合检索主力（dense+sparse+ColBERT 一次前向） |
| Qwen/Qwen3-Embedding-0.6B | 596M | 1024（MRL） | 32768 | 1.19 G | Apache-2.0 | ✅ 精度/长文候选（query 需加 instruction） |
| TencentBAC/Conan-embedding-v2 | ~1.4B | 3584 | 32768 | ~2.8 G（待实测） | Apache-2.0 | ⚠️ 中文上限候选，需先验证 bf16 是否 OOM |
| BAAI/bge-large-zh-v1.5 | 335M | 1024 | 512 | 0.67 G | MIT | ✅ 基线（没有基线的对比表没有说服力） |
| codefuse-ai/F2LLM-v2-0.6B（观望） | 596M | 1024（MRL 至 8 维） | 40960 | 1.19 G | Apache-2.0 | ✅ 2026 新秀，缺 C-MTEB 分项，列后续评估 |

**明确排除（写进报告的排除理由）**
- Qwen3-Embedding-4B / 8B、gte-Qwen2-7B：bf16 权重 8.04 / 15.13 / 15.2 G，8 GB 卡必 OOM，4bit 后精度不能引用官方分数；
- jina-embeddings-v3 / v5、Conan-embedding-v1：**CC-BY-NC-4.0**，学习可用但对外 demo/简历叙事不合适；
- embeddinggemma-2：模型卡与榜单 License 表述冲突，需核对原文；
- multilingual-e5-large：514 token 上限 + 中文偏弱。

**Reranker**：首选 **Qwen3-Reranker-0.6B**（跨语/长文明显强于 bge-reranker-v2-m3：MLDR 67.28 vs 59.51；纯中文二者接近）。两段式：向量召回 top-50 → rerank → top-8 进 LLM；只在"简单事实"和 Agentic 的 vector_search 开启；**与 embedder 分时加载**。

**公平对比协议（固定变量，否则对比无效）**
1. chunk=512 token / overlap=64 / 向量 L2 归一化 / max_length=1024（长文模型另测 8192 一组）/ batch 8、16、32 各测一次；
2. 两套数据：① C-MTEB 8 个检索任务（T2Retrieval / MMarco / DuRetrieval / Covid / Cmedqa / Ecom / Medical / VideoRetrieval）；② 自建 120 条 query（三条链路各 40，人工标 gold chunk），**自建集才是选型判据**；
3. 指标：Recall@{5,10,20}、nDCG@10、MRR；吞吐 docs/s 与 tokens/s；显存 `max_memory_allocated` + nvidia-smi 双记；单 query p50/p95；
4. 统计：每模型跑 3 次取中位数；nDCG@10 用 bootstrap(1000) 给 95% CI，**CI 重叠即视为并列**，再用吞吐/显存/许可决胜；
5. 验收线：nDCG@10 相对 bge-large-zh-v1.5 基线提升 ≥3 个点才"值得换"；峰值显存 ≤7.0 G；单 query p95 ≤150 ms（不含 LLM）；
6. 输出 `reports/embedding_benchmark.md`：模型 × {C-MTEB 均值、三链路 nDCG@10/Recall@10、p50/p95、docs/s、峰值显存、维度、索引体积} + 选型结论。

> 方法论提醒：MTEB 榜单分数**跨来源不可比**（Mean(Task) vs Mean(Type)、31 vs 35 数据集），官方分数只用于排候选；榜单入口用 https://leaderboard.mteb.org/ （可绕过 HF 直连）。

---

## 六、图谱（Graph RAG）选型

| 方案 | 成本 | 可解释性 | 结论 |
|---|---|---|---|
| Milvus 元数据"伪多跳"（entities 数组 + 标量过滤两轮检索） | 最低 | 中 | 阶段一兜底，不引新服务 |
| **Neo4j 5.26 community（Docker）** | 中（镜像已可拉，约 1~2 GB 内存） | **高**（Cypher 显式路径） | **推荐主线** |
| 微软 GraphRAG 3.3.0 | 极高（token 3~5 倍） | 低（黑盒） | 不适合 |
| LightRAG 1.5.7 | 中高 | 中 | 仅作 baseline 对照 |

**推荐**：Neo4j 作为主线（Docker 已有，与 Milvus 同栈），Milvus 伪多跳作为降级路径；LightRAG 只跑一次分数做对照。

Schema：
```cypher
(:Entity {name, type, desc, embedding})
(:Entity)-[:REL {type, desc, weight, source_chunk_id, doc_id}]->(:Entity)
```
多跳查询：种子实体（向量/别名匹配）→ 1~2 跳扩展（权重剪枝、条数上限）→ 收集 `source_chunk_id` 回 Milvus 取原文。

---

## 七、里程碑与验收

| 阶段 | 内容 | 验收标准 | 预估 |
|---|---|---|---|
| M0 | 环境与骨架 | `uv venv -p 3.13` + `docker compose up` + Milvus healthz OK + git 首次提交 + Makefile | 0.5 天 |
| M1 | 数据与评测集 | 9 份数据集落地；三档 corpus；路由 300+100 / 检索 500 / 答案 200 | 1 天 |
| M2 | 摄取与索引 | Unstructured 解析→chunk→Milvus(dense+sparse)；hybrid 检索跑通并出基线 | 1 天 |
| M3 | Embedding 对比 | 5~7 个模型完整对比报告；选定默认模型；reranker 消融 | 1 天 |
| M4 | 图谱与 Graph RAG | 实体/关系入库 Neo4j；实体链接；1~2 跳查询；多跳指标 | 1.5 天 |
| M5 | LangGraph 三路 | 路由器 + 三条链路 + 三工具 + checkpointer + 追踪；端到端可问答 | 1.5 天 |
| M6 | 评测与交付 | 三层评测 + 路由 vs 全 agentic 消融 + 报告 + README + demo | 1 天 |

**总验收**：一条命令可复现 setup/ingest/serve；报告含路由准确率/混淆矩阵、检索指标、答案指标、延迟与成本；≥10 条 badcase 归因；README 写清设计取舍。

---

## 八、需要拍板的决策点

1. **LLM 供应商**：(A) Ollama 免费云模型 gemma4:31b（零成本，已验证 tool calling，有限流风险）(B) DeepSeek API（需充值，稳定，单次约 0.03 元）(C) 抽象 provider，默认 A、可切 B。**推荐 C**。
2. **数据语言**：(A) 英文 benchmark 为主 + 中文检索对比 + 中文 demo 语料 (B) 纯中文 (C) 纯英文。**推荐 A**。
3. **图数据库**：(A) Neo4j Docker 主线（推荐）(B) 先 Milvus 伪多跳，后补 Neo4j (C) 不建图，纯元数据。**推荐 A 或 B**。
4. **web_search**：(A) SearXNG 自建 Docker（免费无 key，镜像可拉，Bing 实测可用）(B) Tavily API（需 key）(C) 先缓存/mock。**推荐 A**。
5. **交付形态**：(A) CLI + 评测报告 + 文档 (B) 再加 FastAPI + 网页 demo (C) 再加 LangGraph 可视化追踪。**推荐 B**。
6. **旧项目复用**：是否复用 `~/develop_项目/rag` 的 Milvus compose 与 Dify 中文语料？**推荐复用 compose，语料作中文 demo**。

---

## 九、风险与对策

| 风险 | 对策 |
|---|---|
| 免费云模型限流/下线 | provider 抽象 + 本地 qwen3:4b 兜底 + 路由/生成结果缓存 |
| web 搜索不稳定 | SearXNG 多引擎 + 缓存 + 失败降级为仅本地检索 |
| 8 GB 显存放不下 4B embedding / 重排 | 默认 0.6B；4B 用 GGUF 量化仅作参照；embedding 与 reranker 分时加载 |
| 数据下载截断（parquet 尾标丢失） | `--retry 5` + 下载后校验末 4 字节 `PAR1` + 断点续传 |
| ragas 0.4.3 依赖冲突 | pin `langchain-community<0.4` |
| 无 sudo 装不了 tesseract | 主链路 fast 策略（纯 pip）；hi_res 可选，需用户 apt |
| Docker Desktop 未开机自启 | Makefile 预检 `docker info` 并提示启动 |
| 评测集泄漏（corpus=评测文档） | corpus_full 混入干扰段落；报告标注 gold 粒度（段落级/doc 级） |

---

## 十、目录结构（草案）

```
rag_m/
├── pyproject.toml            # uv 管理，python 3.13，锁版本
├── docker/compose.yaml       # Milvus standalone + neo4j + searxng
├── data/{raw,processed}      # gitignore
├── src/ragm/
│   ├── config.py             # pydantic-settings
│   ├── ingest/               # unstructured 解析、清洗、分块、元数据
│   ├── store/{milvus.py,graph.py}
│   ├── llm/                  # provider 抽象（ollama / openai-compatible）
│   ├── embeddings/           # 可插拔 embedder + benchmark harness
│   ├── rag/                  # LangGraph: state/router/simple/graph/agentic
│   ├── tools/                # vector_search / kg_query / web_search
│   ├── eval/                 # 数据集加载、指标、报告
│   └── api/                  # FastAPI + 轻量网页
├── scripts/                  # ingest / build_kg / bench_embeddings / eval / serve
├── tests/
├── reports/                  # 评测报告与图表
└── docs/                     # 架构、决策记录、badcase
```
