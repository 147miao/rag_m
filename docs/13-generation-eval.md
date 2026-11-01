# 生成质量评测（2026-11-01）

## 一、指标与实现

目标：把"看起来能用"变成可对比的数字。原计划用 ragas，但 **ragas 0.4.3 在本环境不可用**：
它 import 时依赖 `langchain_community.chat_models.vertexai`（langchain 1.x 已移除），
降级 langchain-community 又要求 `langchain_core.pydantic_v1`（1.x 也已移除），与项目 langchain 1.4.7 冲突。
因此在 `scripts/eval_generation.py` 里用 **LLM-as-judge 复刻了 ragas 的四个指标定义**（提示词与 ragas 对齐）：

| 指标 | 定义 | 判定方式 |
|---|---|---|
| faithfulness | 答案里的原子陈述是否被检索上下文支持 | LLM 拆 claim + 逐条 supported，取比例 |
| answer_relevancy | 答案是否直接回答该问题 | LLM 打 0~1 分 |
| context_precision | 检索上下文与问题的相关性 | LLM 逐条 0/1，取平均 |
| context_recall | gold 答案是否出现在上下文中 | 客观字符串匹配（归一化） |
| citation_valid | API 引用校验是否全通过 | 直接取 /api/query 的 stats |

样本：40 条（EN HotpotQA 20 + ZH CMRC 20），gold answer 来自数据集；单条 RAG 调用 + 1 次 judge 调用（顺序执行，避免并发限流）。

## 二、结果

| 指标 | EN | ZH | 全部 |
|---|---|---|---|
| faithfulness | 0.919 | 0.671 | 0.795 |
| answer_relevancy | 1.000 | 1.000 | 1.000 |
| context_precision | 0.338 | 0.150 | 0.244 |
| context_recall | 0.950 | 0.550 | 0.750 |
| citation_valid | 1.000 | 0.950 | 0.975 |
| p50 延迟 | 1304ms | 1248ms | 1248ms |

## 三、解读

1. **英文忠实度 0.92、引用 100% 有效**：生成基本 grounded，没有明显幻觉；
2. **中文忠实度 0.67、召回 0.55 明显偏弱**：
   - 检索侧：CMRC 文章切块后答案句常被切断；
   - 评测侧：API 返回的 evidence `snippet` 截断到 260 字符，gold 答案常落在截断之外 → context_recall 被低估；
3. **context_precision 0.24 偏低是预期**：多跳题 gold 只要 2 篇，top-8 里必然有 6 篇干扰文档；precision 低不代表检索错；
4. **answer_relevancy 全部 1.0，指标饱和**：judge 给分过于宽松（或题目确实都切题），这个指标目前区分度不足，需要更严格的提示词/更细粒度评分；
5. 引用校验 0.975 与前面人工抽样一致。

## 四、结论与待办

- 当前系统在英文上生成质量可用（faithfulness 0.92 / citation 1.0）；中文是短板（0.67 / 0.55）；
- 评测方法本身的三个已知偏差：snippet 截断、judge 宽松、context_precision 对多跳不公平；
- 下一步：① evidence 返回全文（或评测时走 /api/search 取全文）再评 recall；② answer_relevancy 改为"生成反向问题 + embedding 相似度"（ragas 原定义）；③ 加一组 rerank on/off 对照，量化重排对生成质量的影响。

## 五、复现

    .venv/bin/python scripts/eval_generation.py --n 40 --workers 1
    # 结果：reports/generation_eval.json

注：`ragas` 安装尝试记录在 `.venv-ragas/`（独立环境，未接入主流程）；主流程不依赖它。
