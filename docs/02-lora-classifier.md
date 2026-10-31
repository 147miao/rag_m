# LoRA 分类器方案（查询路由：简单 / 多跳 / 推理）

> 状态：待拍板。目标：用一个 LoRA 微调的本地小模型替代（或前置）现在的 gemma4 提示词分类器。
> 环境结论均来自本机实测（2026-10-31）。

## 一、先给结论

**可行，值得做，但要先摆正预期：**

- 3 分类任务的绝对精度上限不高，LoRA 小模型的价值主要在 **延迟（20ms vs 700~3400ms）、成本（0 vs API 计费）、离线可用、输出确定**；
- 建议做成**两级路由**：LoRA 小模型先判，置信度低于阈值时回退 gemma4 提示词（这正好是评审建议的 cascade 思路）；
- 建议同时做 **bge-m3 embedding + 逻辑回归** 基线：3 分类任务上它可能只要 10 分钟训练、精度接近 LoRA，如果差距 <2 个点，LoRA 的性价比就要重新算。

## 二、本机可行性（实测）

| 项 | 实测 | 含义 |
|---|---|---|
| GPU | RTX 4060 Laptop 8 GB，当前占用 ~1.2 GB | 0.6B 训练轻松；1.7B 需 bf16 + 梯度检查点，边界可行 |
| torch | 2.15.0+cu132，CUDA 可用 | 训练/推理都行 |
| 已装 | peft 0.21.2、accelerate 1.15.0、transformers 5.22.1、datasets 5.1.0 | 训练栈基本齐了 |
| 未装 | trl、bitsandbytes、unsloth | **可以都不装**：用 transformers Trainer + peft 即可；4bit 非必需 |
| 权重通道 | ModelScope 可下 Qwen3-0.6B（1.5 GB）/ 1.7B（4.1 GB），实测 ~5.3 MB/s，Apache-2.0 | 下载到 D 盘 models/ |
| Ollama | 0.33 的 Modelfile 只支持 FROM safetensors / GGUF 导入，**没有 ADAPTER 指令** | LoRA 想走 Ollama 需先 merge adapter 再导入；更简单的是 FastAPI 内进程推理 |

## 三、标签定义（最关键，先定这个）

三类按"回答所需的证据形态"定义，而不是按数据集来源：

| 类别 | 操作性定义 | 判定要点 | 例子 |
|---|---|---|---|
| simple 简单 | 单个段落即可回答，答案是一个事实/数值/定义 | 问题里通常只有一个实体；不需要串联 | "Milvus 的 HNSW 默认 M 是多少？" |
| multihop 多跳 | 需要连接 ≥2 个实体/文档才能回答（bridge / comparison） | 出现"谁的学生""A 的作者的出生地"等链式关系；或两个对象比较 | "bge-m3 的发布机构还发布了哪个 reranker？" |
| reasoning 推理 | 需要聚合/计算/多源对比/规划，单次检索无法给出答案 | 出现"对比/为什么/估算/步骤/权衡"；答案需要综合多个来源 | "对比 bge-m3 与 Qwen3-Embedding 在中英混合场景的优劣" |

**边界规则（写进标注手册，避免标注者各判各的）**：
1. 能用一个段落直接回答 → simple，即使需要常识；
2. 需要两个段落但关系明确（实体链）→ multihop；
3. 需要 ≥3 个来源、或需要比较/计算/给建议 → reasoning；
4. 拿不准时按"更贵的策略"标注（reasoning > multihop > simple），避免低估成本。

## 四、数据方案

| 集合 | 规模 | 来源 | 用途 |
|---|---|---|---|
| train | 2,000~3,000 | 现有数据集按来源给弱标签（SQuAD/CMRC→simple，2Wiki/HotpotQA/MuSiQue→multihop，FRAMES/GraphRAG-Bench→reasoning）+ 模板合成 + gemma4 标注 | 训练 |
| dev | 300 | 与 train 同分布但问题不重叠 | 选模/早停 |
| test-weak | 300 | 与 train 同来源、问题不重叠 | 参考（有偏） |
| **test-human** | **300** | **人工标注，来源与训练集切分（按文档/领域切）** | **最终判据** |

防泄漏：按 **文档/来源** 切分而不是按问题随机切分；同一篇文档的问题只能出现在一个集合里。报告里同时给 test-weak 和 test-human 两列，明确说明弱标签的偏差。

## 五、训练方案

**基座**：Qwen3-0.6B（主力）、Qwen3-1.7B（对照）；Apache-2.0，ModelScope 可下。

**数据格式**（关键：标签用单 token，不用中文词）：
```
<|im_start|>system
你是查询路由器。根据问题判断回答所需策略，只输出一个字母：A=简单，B=多跳，C=推理。<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
A<|im_end|>
```
理由：中文标签"简单/多跳/推理"可能被切成 1~2 个 token，取 logits 时会有歧义；A/B/C 在 Qwen 分词器里是单 token，可稳定取 3 个标签的 logits → softmax 得到置信度。

**LoRA 超参**：
- r=16, alpha=32, dropout=0.05；target: q/k/v/o + gate/up/down 全部 linear；
- lr=2e-4 cosine + 5% warmup；epochs=3；max_len=512；bf16；gradient_checkpointing；
- batch=1 × grad_accum=16（有效 16）；只对 assistant 段算 loss；
- 0.6B 预计 20~40 分钟，1.7B 约 1~1.5 小时（4060，2.5k 样本）。

**显存预算**：0.6B bf16 权重 1.2 GB + 激活 ~2-3 GB → 8 GB 富余；1.7B bf16 权重 3.4 GB + 激活 ~3 GB → 约 7 GB，需要梯度检查点；OOM 再上 4bit（bitsandbytes 可选装）。

## 六、服务与集成

**推荐：FastAPI 内进程推理**（最简单、可控、零额外服务）
- 加载 base + adapter（peft），`torch.inference_mode` 前向一次，取最后位置 3 个标签 token 的 logits → softmax；
- 0.6B 延迟预计 10~25 ms/条（短 prompt），显存 ~1.5 GB；
- 可选：`merge_and_unload()` 后 `ollama create` 导入，走 Ollama API（多一个进程，适合演示"部署"）。

**代码结构**：
```
src/ragm/classifier/
├── base.py        # Classifier Protocol: classify(question) -> RouteDecision
├── rules.py       # 规则基线（关键词/实体数/问句模式）
├── prompt_llm.py  # gemma4 提示词基线（当前实现）
├── lora.py        # 本地 Qwen3 + LoRA，单 token logits
└── factory.py     # 按 CLASSIFIER_BACKEND 选择
scripts/
├── build_route_dataset.py   # 弱标签 + 合成 + 人工集导入
├── train_lora.py            # peft + transformers Trainer
├── eval_classifier.py       # 五路对比 + 混淆矩阵 + 延迟
└── export_lora.py           # merge → safetensors / GGUF → Ollama
```

**API 变更（向后兼容）**：
- 新增 `POST /api/route`（只分类，便于评测与调试）；
- `POST /api/query` 在 `route_override=auto` 时调用分类器，响应增加 `classifier: {backend, model, confidence, latency_ms}`；
- `/api/health` 增加 classifier 状态；
- 前端路由徽章旁显示分类器来源（如 `lora-qwen3-0.6b · 0.93`），debug 面板展示 logits。

## 七、评测口径

五路对比（同一 test-human 集）：
1. 规则基线；
2. gemma4:31b 提示词（当前）；
3. bge-m3 embedding + 逻辑回归；
4. LoRA Qwen3-0.6B；
5. LoRA Qwen3-1.7B。

指标：accuracy、macro-F1、每类 recall、混淆矩阵、ECE（置信度校准）、p50/p95 延迟、单条成本；另报"两级路由"（小模型低置信 → gemma4 回退）的**升级率 / 精度增益 / 延迟增量**。

## 八、里程碑（约 3.5 天）

| 阶段 | 内容 | 产出 |
|---|---|---|
| C0 0.5d | Classifier Protocol + rules + prompt 基线 + `/api/route` + 评测脚本骨架 | 可切换的分类器接口 |
| C1 1d | 标签手册 + 数据集构建（弱标签/合成）+ 人工标注 300 | train/dev/test jsonl + 标注说明 |
| C2 1d | 训练 0.6B / 1.7B，选模 | adapter + 训练曲线 |
| C3 0.5d | 内进程服务 + API/前端集成 | `served_by` 之外新增 `classifier` 元数据 |
| C4 0.5d | 五路评测 + 误差分析 + 文档 | `reports/classifier_report.md` |

## 九、风险

| 风险 | 对策 |
|---|---|
| 标签循环论证（按数据集来源打标） | 训练可用弱标签，**最终判据必须是人工标注 test-human**；报告双列 |
| 类别边界模糊（simple vs multihop） | 标注手册 + 边界规则 + 双人抽检算 kappa（≥0.7 才继续） |
| 8 GB 显存 OOM（1.7B） | 先 0.6B；1.7B 用 bf16 + 梯度检查点；仍 OOM 再 4bit |
| 小样本过拟合 | 3 epochs 早停、数据增强（同义改写/模板）、对比 0.6B vs 1.7B |
| 单 GPU 串行 | embedding 批处理与训练错开；训练脚本加锁 |
| 版本冲突 | 不用 trl/unsloth，直接 transformers Trainer + peft；训练依赖单独 group |
| 服务路径改变精度 | 训练后、merge 后、服务端各评一次，记录同一测试集 |

## 十、待拍板

1. **基座**：只训 Qwen3-0.6B，还是 0.6B + 1.7B 都训做对比？（推荐都训，1.7B 只多 1 小时）
2. **数据来源**：先人工标 300 条测试集再训练，还是先用弱标签训起来再补人工集？（推荐先标测试集，避免"训完才发现标签有问题"）
3. **部署**：FastAPI 内进程（推荐）还是 merge 后导入 Ollama？
4. **是否加两级路由**（小模型 + 低置信回退 gemma4）？（推荐加，成本低、故事完整）
5. **是否加 embedding+逻辑回归基线**？（推荐加，10 分钟成本，用来证明 LoRA 的价值）
