# 路由分类器实验计划（v1）

> 目标：在"简单 / 多跳 / 推理"三分类上，跑通全部候选方案，用同一套 train/dev/test 选出**实际最好用**的分类器，接入现有 FastAPI。
> 环境实测：RTX 4060 8GB；Qwen3-0.6B 已下载（1.5GB）、1.7B 下载中（4.1GB）；peft 0.21.2 支持 SEQ_CLS / DoRA / rsLoRA / PiSSA。

## 一、数据设计（先优化数据，再谈模型）

### 1.1 标签定义（操作性）
| 类别 | 定义 | 边界规则 |
|---|---|---|
| simple | 单段落可答，一个事实/数值/定义 | 能直接引用一段话回答 → simple |
| multihop | 需串联 ≥2 实体/文档（bridge / comparison） | 需要两个段落且关系明确 → multihop |
| reasoning | 需多源聚合、比较、计算、规划或给建议 | ≥3 来源或需要推理过程 → reasoning |
| （拒答） | 不属于分类器职责，由答案守卫处理 | 不进入三分类 |

### 1.2 数据来源与弱标签
squad/cmrc/drcd→simple；wiki2/hotpot/musique→multihop；frames/graphrag_bench→reasoning；multihop_rag null→answerable=false（不参与三分类训练，留给拒答评测）。

### 1.3 优化步骤（按顺序）
1. **去重**：归一化问题 + 跨源去重；近重复（编辑距离/前缀）单独标记；
2. **平衡**：每类上限 2,500，按类别分层采样，避免 reasoning 被压死；
3. **难度分层**：按问题长度/实体数/问句类型打难度分，dev/test 保留 30% 高难度；
4. **增强**：对 reasoning 类用模板合成（把 multihop 问题改写成"对比/为什么/给建议"形式）500 条，单独标记 `synthetic=true`；
5. **测试集人工化**：500 条分层抽样，用 rubric 让 gemma4 重新标注 → 与弱标签比对 → 分歧项人工复核（lead 抽查 100 条）→ 产出 `test_curated.jsonl`；
6. **防泄漏**：按来源+问题哈希分组切分；同一原始文档的问题只进一个集合；报告里注明残留风险。

### 1.4 切分
| 集合 | 条数 | 用途 |
|---|---|---|
| train | ~5,000（含 500 合成） | 训练 |
| dev | 400 | 早停/选模/温度校准 |
| test_weak | 500 | 弱标签测试（有偏，仅参考） |
| test_curated | 500 | **最终判据**（rubric 标注 + 人工复核） |

## 二、实验矩阵

| 编号 | 方案 | 模型 | 任务头 | LoRA 变体 | 说明 |
|---|---|---|---|---|---|
| B0 | 规则基线 | — | — | — | 关键词/实体数/问句模式 |
| B1 | 提示词基线 | gemma4:31b-cloud | 生成 A/B/C | — | 当前实现 |
| B2 | embedding + 逻辑回归 | bge-m3 | sklearn | — | 10 分钟基线 |
| S1 | 分类头 | Qwen3-0.6B | SEQ_CLS | plain | |
| S2 | 分类头 | Qwen3-0.6B | SEQ_CLS | rsLoRA | |
| S3 | 分类头 | Qwen3-0.6B | SEQ_CLS | DoRA | |
| S4 | 分类头 | Qwen3-0.6B | SEQ_CLS | PiSSA init | |
| S5 | 分类头 | Qwen3-1.7B | SEQ_CLS | plain | |
| S6 | 分类头 | Qwen3-1.7B | SEQ_CLS | rsLoRA | |
| G1 | 生成式 | Qwen3-0.6B | CausalLM A/B/C | plain | 与分类头对照 |
| G2 | 生成式 | Qwen3-0.6B | CausalLM A/B/C | rsLoRA | 可选 |

训练配置（统一）：max_len 192；lr LoRA 2e-4 / 头 1e-3；batch 16（seq_cls）/ 8（gen）；epochs ≤10 + 早停（dev macro-F1，patience 2）；label_smoothing 0.05；class_weight balanced；bf16；固定 seed 42。

## 三、评测口径

- 指标：accuracy、macro-F1、每类 recall、混淆矩阵、ECE（10 bins）、p50/p95 单条延迟、训练耗时、峰值显存；
- 主判据：**test_curated 的 macro-F1**（类别不平衡下比 accuracy 稳）；
- 统计：同一测试集上做 paired bootstrap（1000 次）给 95% CI；两模型差值 CI 含 0 则视为并列，再用延迟/显存/训练成本决胜；
- 误路由代价分析：simple→reasoning（浪费）、reasoning→simple（答错）分开统计；
- 两级路由：小模型置信度阈值扫描（0.5~0.9），报升级率 vs 精度增益 vs 延迟增量。

## 四、交付物

1. `reports/classifier_report.md`：数据统计、矩阵结果表、混淆矩阵、误差分析、选型结论；
2. 最佳 adapter + 标签映射 + 温度参数，放到 `models/classifier/<run>/`；
3. `Classifier` 接口接入 API：`CLASSIFIER_BACKEND=lora|prompt|rules`，`/api/route` 端点，响应带 classifier 元数据；
4. 前端显示分类器来源与置信度。

## 五、时间预算

| 阶段 | 时间 |
|---|---|
| 数据下载/规范化（子 agent 并行） | 20~40 分钟 |
| 数据优化 + test_curated | 40~60 分钟 |
| 脚本与 smoke test | 40 分钟 |
| 矩阵训练（0.6B×6 + 1.7B×2 + gen） | 1.5~2.5 小时（GPU 串行） |
| 评测与报告 | 30 分钟 |
| API 集成 + 前端 | 40 分钟 |

## 六、风险

| 风险 | 对策 |
|---|---|
| 弱标签偏差 | 主判据用 test_curated；报告双列 |
| 数据泄漏（同文档问题） | 按来源分组切分 + 近重复标记 |
| 1.7B 显存边界 | 梯度检查点；OOM 则降 batch 或 4bit |
| 训练不稳定/过拟合 | 早停 + 多种子（至少 seed 42/43 各跑最佳配置） |
| 生成式训练太慢 | 只跑 G1，若明显落后则跳过 G2 |
