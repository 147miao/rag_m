# 路由分类器实验结果（进行中）

> 数据与实验均在 2026-10-31 本机完成。主评测集 = test_human（402 条双标注一致 + 98 条人工裁决）。

## 一、数据优化（两轮）

### 1.1 第一轮：按数据集来源打弱标签（已废弃）
- 来源映射：SQuAD/CMRC/DRCD→simple；2Wiki/HotpotQA/MuSiQue→multihop；FRAMES/GraphRAG-Bench→reasoning。
- 用 gemma4 + rubric 对 500 条测试集重新标注，与弱标签一致率仅 **65.9%**；
- 按来源看：FRAMES 10%、GraphRAG-Bench 20% —— 说明"按来源打标"把多跳事实题误标成 reasoning。

### 1.2 第二轮：用数据集自带问题类型元数据重标（v2）
- GraphRAG-Bench：Fact Retrieval→simple，Complex Reasoning/Contextual Summarize/Creative→reasoning；
- MultiHop-RAG：comparison→multihop，inference/temporal→reasoning，null→丢弃；
- FRAMES→multihop（多跳事实题，不是开放式推理）；其余来源不变。
- 重标后与 rubric 标注一致率升到 **80.4%**。
- 最终数据：15,790 条原始 → 去重 14,117 → 按类上限 3,000 平衡 → train 6,344 / dev 401 / test 500。

### 1.3 测试集：consensus + 人工裁决
- 402 条弱标签与 rubric 标注一致 → 取一致标签；
- 98 条分歧 → 由 lead 按 rubric 人工裁决（与弱标签一致 54%、与 LLM 标签一致 43%，是独立第三方判断）；
- 最终 test_human：500 条，simple 212 / multihop 188 / reasoning 100。

## 二、实验设置

- 基座：Qwen3-0.6B / Qwen3-1.7B（Apache-2.0，ModelScope）
- 任务头：Qwen3ForSequenceClassification（3 类），LoRA r=16、α=32、dropout 0.05、全 linear 层、modules_to_save=["score"]
- 训练：bf16、lr 2e-4（LoRA）/1e-3（头）、batch 8×grad-accum 2、max_len 128、epochs 8、早停按 dev macro-F1、label_smoothing 0.05、class_weight balanced
- 硬件：RTX 4060 8GB；0.6B 训练峰值 ~5.5GB，单轮约 25 分钟

## 三、结果（test_human，500 条）

| 方案 | Acc | macro-F1 | simple R | multihop R | reasoning R | p50 | 备注 |
|---|---|---|---|---|---|---|---|
| 规则基线 | 0.420 | 0.215 | 0.98 | 0.00 | 0.03 | <1ms | 关键词规则几乎不会判多跳 |
| gemma4 提示词 | 0.834 | 0.780 | 0.98 | 0.86 | 0.47 | ~760ms | reasoning 召回差 |
| bge-m3 + 逻辑回归 | 0.858 | 0.856 | 0.84 | 0.89 | 0.84 | ~2ms* | 训练 3 分钟，极强基线 |
| **LoRA Qwen3-0.6B plain** | **0.910** | **0.898** | 0.92 | 0.97 | 0.78 | 36ms | 当前最佳 |
| LoRA Qwen3-0.6B rsLoRA | 0.896 | 0.884 | 0.90 | 0.96 | 0.76 | 38ms | 本数据量下无增益 |

\* logreg 延迟只含推理（embedding 另计 ~10ms/条）。

## 四、关键发现：标签口径比模型重要

同一个 s1 模型在三个测试集上的表现：

| 测试集 | 标签来源 | Acc | macro-F1 |
|---|---|---|---|
| test_weak | 元数据弱标签（与训练同口径） | 0.988 | 0.989 |
| test_human | 双标注一致 + 人工裁决 | 0.910 | 0.898 |
| test_curated | gemma4 rubric 单标 | 0.804 | 0.742 |

结论：**0.74 vs 0.99 的差距主要来自标签定义，不是模型能力**。报告里必须同时给出三个口径，主判据用 test_human。

## 五、进行中

- s4 PiSSA 初始化（训练中）
- s5/s6 Qwen3-1.7B plain/rsLoRA（batch 2 + 梯度检查点）
- g1 生成式 A/B/C 对照
- DoRA 因 8GB 显存 OOM（DoRA 额外权重范数计算），将用 batch 4 + 梯度检查点重试

## 六、复现命令

```bash
# 数据
python scripts/relabel_route_dataset.py
python scripts/build_route_dataset.py --raw .../raw_v2 --cap-per-class 3000
python scripts/curate_test.py --in .../test_weak.jsonl --out .../test_curated.jsonl
# 训练 + 评测
bash scripts/run_matrix.sh
python scripts/eval_route_cls.py --run .../s1_06b_plain --data .../test_human.jsonl
python scripts/baselines.py --data .../test_human.jsonl --train .../train.jsonl
```
