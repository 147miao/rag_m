# 生成质量评测 v2：修正口径后的可信基线（2026-11-01）

## 一、修正了什么

v1 有两个口径问题，导致结论失真：

1. **context 用的是 `Evidence.snippet`（260 字符截断）**：而送进 LLM 的上下文其实是最多 1200 字符。
   → 现在 `/api/query` 的 Evidence 增加 `full_text` 字段（与 prompt 一致），评测用它；
2. **answer_relevancy 用 LLM 直接打 0~1 分**：judge 过于宽松，全部 1.0，指标饱和。
   → 改为 ragas 原定义：由答案反推 3 个问题，与原问题做 embedding 余弦相似度取均值（Ollama bge-m3）。

## 二、v1 vs v2

| 指标 | v1 | v2 | 变化 |
|---|---|---|---|
| faithfulness | 0.795 | **0.978** | +0.183 |
| answer_relevancy | 1.000（饱和） | **0.767** | 指标恢复区分度 |
| context_precision | 0.244 | 0.247 | 基本不变 |
| context_recall | 0.750 | **0.975** | +0.225 |
| citation_valid | 0.975 | 0.975 | — |
| p50 延迟 | 1248ms | 1161ms | — |

分语言（v2）：

| 指标 | EN | ZH |
|---|---|---|
| faithfulness | 0.990 | 0.967 |
| answer_relevancy | 0.733 | 0.801 |
| context_precision | 0.338 | 0.156 |
| context_recall | 1.000 | 0.950 |
| citation_valid | 1.000 | 0.950 |

## 三、结论（修正 v1 的错误判断）

1. **v1 说"中文是短板"主要是测量误差**：用全文上下文后，中文 context_recall 0.55 → 0.95、faithfulness 0.67 → 0.967，
   与英文（1.00 / 0.99）差距很小，系统并没有 v1 显示的那么差；
2. **当前可信基线**：faithfulness 0.978、context_recall 0.975、citation 0.975、answer_relevancy 0.767；
3. **context_precision 0.25 仍然低，但不是缺陷**：多跳题 gold 只有 2 篇，top-8 必然包含 6 篇干扰文档；
   要衡量"精确性"应看 citation 对应的上下文（0.975 通过），而不是全量 precision；
4. answer_relevancy 0.77 是新的可优化点：答案切题但仍有提升空间（尤其英文 0.73）。

## 四、实现位置

- `/api/query`：`Evidence.full_text`（1200 字符，与 prompt 一致）——`src/ragm/api/schemas.py`、`langgraph_service.py`；
- 评测脚本：`scripts/eval_generation_v2.py`（full_text + embedding relevancy + 40 条中英混合，顺序执行）；
- 结果：`reports/generation_eval_v2.json`（含逐条明细）。

## 五、遗留

- context_precision 需要更合理的定义（按引用/按 gold 覆盖），当前指标对多跳不公平；
- 评测样本 40 条偏少（CI 较宽），后续可扩到 100 条；
- ragas 仍无法安装（langchain 1.x 兼容问题），指标为等价复刻。
