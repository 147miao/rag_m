# RAG-M v1.0 进化计划（2026-11-08，rev.2）

> 定位：学习 + 作品集。本轮把"实验原型"推进到"有真实文档摄取、有评测门禁、可一键部署"的 v1.0。
> rev.2 吸收独立审核意见：修正 CI/容器/摄取契约/验收标准/并行边界。

## 0. 现状与问题

- 语料是公开数据集，没有"用户文档"接入链路；`src/ragm/ingest`、`src/ragm/eval` 仍是空模块；
- 评测脚本散落在 `scripts/`（bench_*.py / eval_*.py），无统一入口、无回归门禁；
- 无 CI、无应用 Dockerfile、无鉴权/限流；Milvus/Neo4j 容器当前停止，API degraded；
- Graph RAG 仅 1 跳；Agentic 已是 plan→tools 循环，但无预算/重试；
- re`ruff check src tests scripts` 有 22 个历史错误（scripts 为主），需先收敛范围。

## 1. 验收标准（可测、可证伪）

| # | 标准 | 证据/判定 |
|---|---|---|
| A1 | 摄取 PDF/DOCX/EPUB/MD 各 ≥2 个文件，一条命令生成 parent/child/chunk | `data/docs/` 三个 jsonl + `reports/ingest_report.md` |
| A2 | 幂等：同目录重复执行 second run inserted=0、skipped=N；文件修改后旧 chunk 清理（按 file_hash 删除旧 doc_id） | 两次运行报告 + Milvus 行数 |
| A3 | 长文档（>2000 字）自动 parent/child；查询默认 small-to-big（父块作上下文） | 集合统计 + 查询命中 parent 文本 |
| A4 | 统一评测入口：`python scripts/eval_all.py --suite all` 输出 ≥80 题的检索+生成报告 | `reports/eval_report.md` |
| A5 | 门禁：确定性检索门禁（无 LLM/GPU，固定 fixture）在 CI 运行，相对基线下降 >0.02 则失败 | CI 日志 / `make gate` |
| A6 | 容器：`docker compose --profile app up` 起 API（含 torch，GPU 直通）+ web 静态；`/readyz` 返回 200 且 status=ok | 容器日志 + curl |
| A7 | CI：push 触发 ruff（src tests）+ pytest + web build，全绿 | Actions 链接 |
| A8 | Graph RAG 2.0：2 跳+迭代检索；在 **40 题 held-out** 多跳集上相对 hybrid-only 非劣（ΔnDCG ≥ -0.01）且准确率不低于 v1 | 对照报告 `reports/graph_v2_vs_v1.md` |
| A9 | Agentic 2.0：预算（≤4 次工具、≤8000 tokens）+ 失败重试；15 题 accuracy ≥ v1、p50 ≤3.5s、异常=0 | 报告 `reports/agentic_v2.md` |
| A10 | 文档同步：README 增加 INGEST/EVAL/OPS 小节；`plan.md` 与本表一致 | README diff |

## 2. 关键设计决策（冻结接口，避免并行冲突）

1. **Milvus schema 不改**：沿用 `chunk_id/doc_id/title/source/lang/chunk_index/text/dense`；
   `page/section/file_hash` 写入 manifest（`data/docs/manifest.json`），`title` 形如 `file.pdf · p3 · 第2章`；
   这样 W1 不需要改 `milvus_store.py`（避免与 W3/W5 冲突）。
2. **摄取入口**：`scripts/ingest_docs.py --src <dir>` 只做解析+切块+写 JSONL（无 GPU）；
   `--embed --collection rag_docs` 时才编码入库（复用现有 Embedder 与 `ingest_milvus.py` 逻辑）。
3. **数据目录**：`data/docs/{parents,children,chunks}.jsonl` + `manifest.json`；数据仍走 D 盘软链。
4. **评测基线**：`tests/fixtures/eval_baseline.json`（随仓库提交），CI 只跑检索门禁；生成评测在本地 `make eval`。
5. **容器数据**：数据/模型放 `$HOME/ragm_data`（Docker Desktop 只共享 $HOME），compose 用服务名互连；
   `host.docker.internal` + `extra_hosts: host-gateway` 访问宿主 Ollama。
6. **默认小步快跑**：Graph 2.0 先复用现有 CO_EVIDENCE/MENTIONS 图做 2 跳+迭代，LLM 三元组作为可选开关（默认关）。

## 3. 工作流与文件所有权

| 工作流 | Owner | 文件范围 | 交付 |
|---|---|---|---|
| W1 摄取 | 子 agent A（general） | `src/ragm/ingest/*`、`scripts/ingest_docs.py`、`tests/test_ingest.py`、`data/docs/` | A1~A3 |
| W2 评测 | 子 agent B（general） | `src/ragm/eval/*`、`scripts/eval_all.py`、`tests/test_eval.py`、`tests/fixtures/eval_*` | A4~A5 |
| W3 工程 | 子 agent C（general） | `docker/Dockerfile.api`、`docker/compose.yaml`、`.github/workflows/ci.yml`、`.dockerignore`、`Makefile`、`scripts/ci.sh` | A6~A7 |
| W4 Graph | 主 agent | `src/ragm/graph/*`、`src/ragm/rag/graph.py`、`scripts/eval_graph_v2.py` | A8 |
| W5 Agentic | 主 agent | `src/ragm/agent/*`、`scripts/eval_agentic.py` | A9 |
| 集成/文档 | 主 agent | `README.md`、`docs/21-*`、`docs/22-*`、截图、提交/push | A10 |

共享文件（`config.py`、`milvus_store.py`、`api/routes.py`、`Makefile`）由主 agent 统一改，子 agent 只读或提 patch 建议。

## 4. 执行顺序

1. 恢复环境（Milvus/Neo4j/API）——主 agent 立即做；
2. 并行：W1、W2、W3（目录不重叠）；主 agent 同时做 W4 的 2 跳/迭代（不新增 LLM 抽取）；
3. W4/W5 完成 → 用 W1 新文档语料复测；
4. 集成：README/docs/截图；push；
5. 独立 acceptance + completeness 审核（fresh），按 A1~A10 逐条核验。

## 5. 风险与回退

| 风险 | 缓解 | 回退 |
|---|---|---|
| GPU 8GB 被占（当前 API 占 2.5GB） | 摄取只解析不编码；编码前停 systemd API；GPU 锁（复用 bench_common） | 用户释放 GPU 后再跑 |
| 新格式解析失败 | 逐文件 try/except，报告 skipped 清单 | 保留原文件，人工转 PDF |
| CI 装 torch 过重 | CI 只装 dev 依赖（不装 torch），只在 lint/test 用 rules+demo 后端 | 拆分 requirements-ci.txt |
| 门禁误报 | 基线固定 + 阈值 0.02 + `--no-gate` | 手动更新基线需 review |
| 子 agent 冲突 | 文件所有权表 + 合并前 git diff | 冲突时主 agent 手工合并 |

## 6. 验证

- 单测：`pytest tests -q`；新增 test_ingest（解析/幂等）、test_eval（指标）；
- E2E：真实 PDF/DOCX/EPUB → 解析 → 检索 → 问答；
- `make gate`（确定性检索门禁）；`make eval`（含生成，需 LLM）；
- `docker compose --profile app up` + `/readyz`；
- acceptance/completeness 两个 fresh 子 agent 独立核验。

## 7. 不在本轮范围

- 前端产品化（多会话/上传/设置/暗色）；鉴权体系（仅预留 API key）；
- 扫描版 PDF OCR（先支持文本层 PDF）；
- 大规模分布式索引。
