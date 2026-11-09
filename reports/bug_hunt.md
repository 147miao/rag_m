# Bug 清单（主 agent 审计，2026-11-09）

## B1（P0）后端依赖不可用时静默返回演示答案
- 位置：`src/ragm/api/main.py:66-105`
- 现象：`RAG_BACKEND=langgraph` 且 SearchService 初始化失败时，`app.state.service` 仍是 `DemoRagService`；`/api/query` 返回预制 demo 答案，只有 `/api/health` 显示 degraded。
- 复现：停掉 Milvus 后启动 API，`curl -X POST /api/query` 仍 200 且返回 demo 场景内容。
- 期望：配置为 langgraph 时依赖不可用应返回 503（或明确 `served_by=demo` 且前端提示），不能静默编造。

## B2（P0）子集合缺失导致整个检索服务不可用
- 位置：`src/ragm/store/search_service.py:16-35`（`info()` 调用 `child_store.count()`）
- 复现：`MILVUS_CHILD_COLLECTION=rag_does_not_exist` → `SearchService(settings)` 抛 DescribeCollectionException，API 启动报 search service unavailable。
- 期望：child 集合可选；`info()` 容错，`search_small2big` 缺失时返回明确错误。

## B3（P1）--embed 在集合不存在时不自动创建
- 位置：`scripts/ingest_docs.py::embed_chunks`
- 复现：`embed_chunks(tmp_out, "rag_probe_missing", recreate=False)` 抛 DescribeCollectionException。
- 期望：集合不存在时自动创建（recreate）再插入。

## B4（P1）文件被清空后旧 chunk 残留
- 位置：`src/ragm/ingest/pipeline.py::build`
- 复现：摄取 a.md（1 chunk）→ 清空 a.md → 再 build：报告 parsed=0/skipped=1，但 chunks.jsonl 仍有 1 行、manifest 仍是旧 hash。
- 期望：内容变空时删除旧 rows 并更新 manifest。

## B5（P2）feedback 日志路径依赖 cwd
- 位置：`src/ragm/api/routes.py:224` `Path("logs")`
- 期望：用 data_root 或仓库根解析绝对路径。

## B6（P2）agent 的 kg_query 忽略 hops 参数
- 位置：`src/ragm/agent/tools.py` 约 93 行，固定 `kg.expand(names, limit=30)`（1 跳）。
- 期望：按 hops 调 expand_hops，或移除该参数。

## 验证建议
- B1/B2 用环境变量 + curl 复现；B3/B4 用临时目录复现（无需 GPU）。
- 修复后 `pytest tests -q`、`ruff check src tests scripts` 全绿。

## B7（P1，前端）反馈状态跨答案残留
- 位置：`web/src/components/AnswerCard.tsx:67`
- 现象：`feedback` 是组件内 state，切换到下一条答案时不会重置；上一条点了「有帮助」后，下一条答案的按钮显示「已反馈」且被禁用。
- 修复：按 `scenario.id` 重置（useEffect），或给 AnswerCard 加 key。

## B8（P2，前端）证据高亮在清洗后失效
- 位置：`web/src/components/RightPanel.tsx:209` 与 `highlight()`
- 现象：`highlight(cleanEvidence(snippet), highlight)` 的 `highlight` 来自原始文本，清洗后 indexOf 找不到 → 高亮静默消失（或命中错误位置）。
- 修复：对 highlight 词也做同样清洗，或去掉后端 highlight、改为高亮首句。

## B9（P2，后端）/api/query 重复分类
- 位置：`src/ragm/api/routes.py:181-190` + `src/ragm/rag/langgraph_service.py:196-206`
- 现象：auto 路由时路由层先 `_classify` 一次，service.query 里又 classify 一次（约 2×36ms）；且 `route_override != auto` 时 service 仍会分类但结果被丢弃。
- 修复：把分类结果传入 service（或 service 只在 auto 时分类）。

## B10（P0，容器）Dockerfile 使用了不存在的 uv 参数
- 位置：`docker/Dockerfile.api:20`
- 现象：`uv pip install --system --no-dev -e .` → uv 0.12.7 报错 `unexpected argument '--no-dev'`（uv pip install 没有该参数），镜像构建必失败。
- 修复：去掉 `--no-dev`（uv pip install 默认不装 dependency-groups），或改用 `uv sync --no-dev`。

## B11（P0，容器）editable 安装在 COPY src 之前
- 位置：`docker/Dockerfile.api`
- 现象：先 `COPY pyproject.toml uv.lock README.md` 然后 `uv pip install -e .`，此时 `src/ragm` 还不存在，hatchling 构建 backend 找不到包目录 → 构建失败。
- 修复：先 COPY src（必要时 scripts），再安装；或改成非 editable 安装并最后 COPY 代码。
