# RAG-M：路由式混合 RAG

> 查询分类器 → 三条链路：**传统混合检索 RAG**（简单事实）/ **Graph RAG**（多跳关系）/ **Agentic RAG**（复杂推理，按需调用 vector_search / kg_query / web_search）。
> 技术栈：LangGraph 1.2 · Milvus 3.0 · Neo4j · Unstructured · Docker · Python 3.13

## 快速开始

```bash
cp .env.example .env
make check-data     # 校验 D 盘挂载
make install        # uv venv + 依赖
make up             # Milvus standalone
make health         # 期望 {"state":"OK"}
```

## 目录

```
docker/          Milvus / Neo4j / SearXNG compose
src/ragm/        应用代码（ingest / store / llm / embeddings / rag / tools / eval / api）
scripts/         数据下载、建索引、benchmark、评测
data/ -> D 盘    原始数据与语料（不入 git）
reports/         评测报告
docs/            方案与决策记录
```

## 文档

- [方案 v0.2](docs/00-proposal.md)
- 待补：ARCHITECTURE.md / EVAL_REPORT.md / DECISIONS.md

## 前端（web/）

```bash
cd web
pnpm install
pnpm dev          # http://127.0.0.1:5173
pnpm build        # 产物 web/dist
```

- 设计规范：[web/DESIGN.md](web/DESIGN.md)（遵循 OpenDesign 的 DESIGN.md 协议：色板 / 字体 / 间距 / 组件 / 状态 / 动效 / 无障碍）
- 演示深链：`?demo=multihop-1&tab=graph&fast=1`（demo = simple-1 / multihop-1 / agentic-1 / refusal-1；tab = evidence / graph / trace / debug；fast=1 直接呈现终态）
- 当前为**演示数据**，接口契约见 `web/src/lib/types.ts`，后端接入后由 `/api/query` 返回同结构 JSON
- 截图：[docs/screenshots](docs/screenshots)
