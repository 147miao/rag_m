# Rerank on/off 的生成质量对照（2026-11-01）

问题：重排在检索层把英文 nDCG@10 从 0.810 提到 0.922（+11 点），但它对**最终生成质量**的贡献是多少？

## 一、设置

- 同一批 40 条评测样本（EN HotpotQA 20 + ZH CMRC 20，与 v2 基线一致）；
- 唯一变量：`RERANK_ENABLED=true/false`，其余（检索、图谱、prompt、生成模型）完全相同；
- 指标同 v2：faithfulness / answer_relevancy（embedding 反推）/ context_precision / context_recall / citation_valid。

## 二、结果

### 英文（EN）

| 指标 | rerank off | rerank on | Δ |
|---|---|---|---|
| faithfulness | 0.915 | **0.990** | **+0.075** |
| answer_relevancy | 0.7347 | 0.7334 | -0.001 |
| context_precision | 0.2875 | **0.3375** | +0.050 |
| context_recall | 0.950 | **1.000** | +0.050 |
| citation_valid | 1.000 | 1.000 | 0 |
| p50 延迟 | 1169ms | 1159ms | -10ms（重排 77ms，被生成波动抵消） |

### 中文（ZH）

| 指标 | rerank off | rerank on | Δ |
|---|---|---|---|
| faithfulness | 0.9667 | 0.9667 | 0 |
| answer_relevancy | 0.7962 | 0.8007 | +0.005 |
| context_precision | 0.1375 | 0.1562 | +0.019 |
| context_recall | 0.950 | 0.950 | 0 |
| citation_valid | 0.950 | 0.950 | 0 |
| p50 延迟 | 1132ms | 1198ms | +66ms |

### 全部

| 指标 | rerank off | rerank on | Δ |
|---|---|---|---|
| faithfulness | 0.9408 | **0.9783** | **+0.0375** |
| answer_relevancy | 0.7654 | 0.7670 | +0.002 |
| context_precision | 0.2125 | 0.2469 | +0.034 |
| context_recall | 0.950 | 0.975 | +0.025 |
| citation_valid | 0.975 | 0.975 | 0 |
| p50 延迟 | 1144ms | 1161ms | +17ms |

## 三、结论

1. **重排对英文生成质量有实打实的提升**：faithfulness +7.5 点、context_recall +5 点、precision +5 点；
   检索层 +11 nDCG 约有一半传导到生成层（LLM 对干扰文档有一定鲁棒性，不会 1:1 转化）；
2. **中文几乎没有变化**：检索本来接近饱和（recall 0.95），重排无增益，+66ms 是纯成本；
3. **延迟代价很小**：重排 77ms，端到端 p50 只 +17ms（生成波动大于重排开销）；
4. **建议**：保留重排（英文收益明显、成本低）；纯中文场景可关掉省 66ms。

## 四、复现

    # off：.env 设 RERANK_ENABLED=false 后重启 API
    .venv/bin/python scripts/eval_generation_v2.py --n 40 --out reports/generation_eval_v2_norerank.json
    # on：RERANK_ENABLED=true，同样命令输出 reports/generation_eval_v2.json

数据：reports/generation_eval_v2.json（on）、reports/generation_eval_v2_norerank.json（off）。
