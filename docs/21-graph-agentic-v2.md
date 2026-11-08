# Graph RAG 2.0 与 Agentic 2.0（2026-11-08）

## Graph RAG 2.0

- `KGStore.expand_hops(names, hops, limit)`：逐跳 BFS，每跳配额 `limit//hops`，返回带 `hop` 字段的边；
  默认 `GRAPH_MAX_HOPS=2`、`GRAPH_HOP_LIMIT=40`；
- `RagGraph` 多跳分支：2 跳扩展 → 若图谱未补充证据且开启 `GRAPH_ITERATIVE`，用邻居实体改写查询做第二轮检索；
- 实测（合并语料，reranker 关闭）：问题「自来也的弟子中谁成为了四代火影？」→ 波风水门，引用 2/2，3.76s，图谱 32 节点；
- 对照脚本：`scripts/eval_graph_v2.py --hops 1 2`（需 GPU，未跑全量）。

## Agentic 2.0

- 工具预算：`AGENT_MAX_CALLS=4`、`AGENT_MAX_TOKENS=8000`、`AGENT_RETRY=1`；
- 路由条件：达到调用/令牌上限即停止取工具，进入 synthesize；
- 工具失败自动重试一次，`tool_log[].attempts` 记录尝试次数；
- 单测：`tests/test_agent_budget.py`（预算停止 + 重试恢复）；
- 实测：「What metrics does RAGAS use?」→ 1 次 vector_search 后正确回答（工具 1 次，未超预算）。

## 注意

- 本机 ComfyUI 占用约 2.6~4.9GB 显存，API 启动时若 OOM，已用 drop-in `~/.config/systemd/user/ragm-api.service.d/lowmem.conf`
  临时关闭 reranker（`RERANK_ENABLED=false`）；GPU 空闲后删除该 drop-in 并重启即可恢复重排。
