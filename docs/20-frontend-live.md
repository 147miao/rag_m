# 前端去演示化与移动端修复（2026-11-02）

## 问题

控制台之前是“演示层”：左侧固定 4 个 demo 案例、答案/证据/轨迹来自 web/src/data/scenarios.ts 预制数据、
系统状态与评测数字硬编码、API 失败还落到本地演示动画。用户看到的永远是演示内容。

## 改动

| 模块 | 改动 |
|---|---|
| 示例问题 | 新增 GET /api/examples（按当前 MILVUS_COLLECTION 返回 all/naruto/papers/default 四套示例），前端拉取后点击即跑真实查询 |
| 空/运行态 | 未运行显示空状态引导；运行中显示阶段进度 + 骨架 + 流式片段；删除 runLocal() 演示兜底 |
| 系统状态 | 左侧 Milvus / Ollama / 分类器 / 知识图谱 / LangSmith 改读 GET /api/health，15s 轮询 |
| 评测卡 | 新增 GET /api/classifier/metrics（读 reports/classifier_metrics.json），显示真实 acc / macro-F1 / 样本数 |
| 反馈 | 「有帮助 / 没帮助」调用 POST /api/feedback，点击后置灰显示已反馈 |
| 错误处理 | 后端断开 / 查询失败显示错误条，不再静默降级 |
| 证据清洗 | 前端展示层去萌娘导航噪声；语料层另用 scripts/clean_naruto.py 清洗并重建索引 |
| 移动端 | TraceRail 从横向排列改为移动端纵向列表；右栏 w-full max-w-[400px]；引用编号与后续标点绑定避免孤立换行 |
| 细节 | 长标题截断、aria 标签、role=alert/status、空问题时运行按钮 disabled |

## 验证

- pnpm build 通过，浏览器 console 无报错；
- 真实查询「鸣人父亲是谁」→ 2.6~3.3s 回答「鸣人的父亲是四代目火影波风水门」，引用通过；
- 移动端 390×844 截图：TraceRail 纵向可读、答案卡片与统计网格正常；
- 后端 17 测试通过、ruff 全绿。
