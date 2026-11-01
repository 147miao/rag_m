# 真实文档测试：火影忍者语料（2026-11-01）

目标：用**真实网页文档**（不是数据集）走完整链路：抓取 → 解析 → small-to-big 切块 → 入 Milvus → 检索 → 生成 → 引用校验。

## 一、抓取

网络现实（实测）：

| 站点 | 结果 |
|---|---|
| zh.wikipedia.org / naruto.fandom.com | 不可达（000） |
| baike.baidu.com / zh.moegirl.org.cn / huijiwiki | HTTP 403（反爬） |
| baike.so.com（360百科） | 前几次可抓，随后搜索接口被限流 |
| **mzh.moegirl.org.cn（移动版萌娘百科）** | **HTML 可访问**（API 被禁 action-notallowed） |

最终方案：用 Playwright 渲染移动版萌娘百科页面，提取正文。

- 脚本：`scripts/crawl_moegirl.py`（58 个火影相关条目，成功 30 篇，250k 字符）
- 条目示例：火影忍者、漩涡鸣人、宇智波佐助、春野樱、旗木卡卡西、宇智波鼬、宇智波斑、千手柱间、大蛇丸、自来也、纲手、日向雏田、我爱罗、晓、木叶隐村、查克拉、尾兽、九尾、写轮眼、轮回眼、佩恩、长门、小南、赤砂之蝎、迪达拉、干柿鬼鲛、飞段、角都、药师兜、波风水门、漩涡玖辛奈、千手扉间、猿飞日斩、志村团藏、大筒木辉夜、六道仙人、十尾、须佐能乎、影分身之术、螺旋丸、千鸟、通灵之术、忍者联军、第四次忍界大战…

## 二、语料与索引

| 项 | 值 |
|---|---|
| 文档 | 30 篇 / 250k 字符（平均 8.3k 字/篇） |
| parent | 171 个（≤1600 字切片，平均 1466 字） |
| child | 1,579 个（200 字 / overlap 40） |
| flat chunk | 370 个（800 字 / overlap 100） |
| 索引 | `rag_naruto_chunks`（370）+ `rag_naruto_children`（1,579，带 parent_id） |
| 模型 | embeddinggemma-300m（768 维）+ BM25 sparse + bge-reranker-base |

## 三、端到端测试（12 道火影问题）

| 指标 | 结果 |
|---|---|
| 答案正确（含 gold 关键词） | **12/12 = 1.000** |
| 引用全部有效 | **12/12 = 1.000** |
| p50 延迟 | 1,482ms |

样例：

| 问题 | 答案（截断） |
|---|---|
| 漩涡鸣人的父亲是谁？ | 漩涡鸣人的父亲是波风水门 [1][2]。波风水门是火之国木叶忍者村的第四代火影… |
| 宇智波鼬为什么灭族？ | 为了保护村子、防止内乱引发战争以及守护弟弟的性命而灭族 [1][2]… |
| 晓组织的目的是什么？ | 收集 9 只尾兽并制造尾兽兵器 [1][2][3][4]… |
| 旗木卡卡西的写轮眼是从谁那里得到的？ | 从宇智波带土那里得到的 [4][5][7][8]… |
| 尾兽一共有几只？ | 九只 [1][2]… |
| 自来也有哪些弟子？ | 波风水门、长门和小南 [3][5]… |
| 木叶隐村是谁建立的？ | 由千手柱间与宇智波斑共同创建 [1][2][3][5]… |

## 四、结论

1. **真实网页文档链路已跑通**：抓取 → 切块 → 入库 → small-to-big → hybrid+rerank → 生成 → 引用校验；
2. 在真实语料上 **答案正确率 100%、引用有效率 100%**（12 条手写问题，样本小但方向明确）；
3. 这些火影问题都被分类器判为 simple（事实型），符合预期；多跳/agentic 路由未在该语料上单独验证；
4. 延迟 1.5s（生成占大头），与演示语料一致。

## 五、已知限制

1. 语料来源单一（萌娘百科），30 篇偏少；没有为该语料建知识图谱（multihop 的图谱扩展在火影语料上未启用）；
2. 360 百科/萌娘百科都有反爬，抓取脚本用了 Playwright，速度约 5s/页；
3. 评测是 12 条手写问题 + 关键词匹配，不是系统化评测集；
4. 前端演示仍指向原 HotpotQA/CMRC 语料，火影语料通过脚本测试（如需 UI 演示可切环境变量重启 API）。

## 六、复现

    python3 scripts/crawl_moegirl.py                 # Playwright 抓取（约 5 分钟）
    .venv/bin/python scripts/build_naruto_corpus.py  # parents/children/chunks
    .venv/bin/python scripts/ingest_milvus.py --corpus .../chunks.jsonl --collection rag_naruto_chunks --recreate
    .venv/bin/python scripts/ingest_milvus.py --corpus .../children.jsonl --collection rag_naruto_children --with-parent --recreate
    .venv/bin/python scripts/test_naruto_rag.py      # 12 题端到端测试

数据：`/run/media/myj/D/rag_m/data/naruto/`（raw/moegirl.jsonl、parents/children/chunks.jsonl），结果 `reports/naruto_test.json`。
