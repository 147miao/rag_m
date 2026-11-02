# 真实文档测试：火影忍者语料（2026-11-01）

## 一、为什么不用"整本火影小说"或盗版漫画站

1. 火影是漫画/动画（700 话），不存在"整本小说"；官方轻小说受版权保护，无合法免费全文；
2. 漫画是图片，需 OCR（1.26 万页 × 1~2s ≈ 4~7 小时），且对白碎片化，RAG 语料质量差；
3. 盗版聚合站（如 manhuazhan）本机不可达（http=000），且从版权角度不做。

## 二、可用数据源（实测）

| 站点 | 状态 |
|---|---|
| zh.wikipedia.org / naruto.fandom.com（中英） | 不可达（000） |
| baike.baidu.com / 萌娘百科桌面版 / huijiwiki | 403 / Cloudflare 验证 |
| naruto.miraheze.org | 可访问，但是波兰语 Naruto Wiki（非中文） |
| **mzh.moegirl.org.cn（移动版萌娘百科）** | 可访问（API 被禁，HTML 可用） |
| baike.so.com（360百科） | 前几次可用，随后限流 |

结论：中文可用来源仍是萌娘百科；fandom 中文维基在本网络不可达。

## 三、语料规模（三次扩充）

| 项 | v1 | v2 | v3（当前） |
|---|---|---|---|
| 文档 | 30 | 116 | 204（清洗后） |
| 字符 | 250k | 791k | 712k（清洗后） |
| parent | 171 | 548 | 546 |
| child | 1,579 | 5,005 | 4,548 |
| flat chunk | 370 | 1,186 | 1,112 |

扩充方式：种子页 → 680 链接 → 关键词过滤 153 → 抓 139；再从 5 个分类页（火影忍者/人物/忍术/组织/地点）发现 233 候选 → 抓 150。

## 四、实体图（Graph RAG for Naruto）

- 节点：218 个条目（人物/忍术/组织/地点/设定，按关键词自动分类）；
- 边：12,190 条 MENTIONS（条目 A 正文提到条目 B，边带来源问题文本）；
- 存储：Neo4j，label NarutoEntity，relation MENTIONS（与 HotpotQA 的 Entity/CO_EVIDENCE 隔离）；
- 接入：multihop 路由 → 实体链接（先匹配问题中的实体名，再用检索标题兜底）→ 1 跳扩展 → 邻居条目补证据 → hybrid+rerank → 生成。

## 五、端到端测试

### 事实型（18 题，v2 语料）

| 指标 | 结果 |
|---|---|
| 答案正确 | 17/18 = 0.944 |
| 引用有效 | 0.889 |
| p50 | 1,382ms |

两条失败为如实拒答（"火影是什么职位"证据不足、"忍者联军多少人"语料未覆盖）。

### 多跳型（12 题，v3 语料 + 实体图）

| 指标 | 结果 |
|---|---|
| 答案正确 | 12/12 = 1.000 |
| 引用有效 | 1.000 |
| 图谱额外证据/题 | 8.0 |
| p50 | 1,705ms |

样例：

| 问题 | 答案（截断） |
|---|---|
| 自来也的弟子中谁成为了四代火影？ | 波风水门 [1][3]（实体链接：自来也→波风水门，图谱 +8） |
| 晓组织里与宇智波鼬搭档的是谁？ | 干柿鬼鲛 [1][3] |
| 佩恩的本体是谁？ | 长门 [3][6][7][9] |
| 九尾人柱力中谁成为了七代火影？ | 漩涡鸣人 [1] |
| 木叶三忍除了自来也和纲手还有谁？ | 大蛇丸 [1][2] |
| 大蛇丸的弟子有哪些？ | 御手洗红豆和宇智波佐助 [1] |

## 六、结论

1. 语料扩到 220 篇 / 117 万字后，事实型 16/18、多跳型 12/12；
2. 实体图对多跳题有效：12 题全部命中，平均每题图谱补 8 条证据，延迟只增加 ~300ms；
3. 失败模式仍是拒答而非编造，符合设计；
4. 中文 fandom/wiki 源在本网络不可达，萌娘百科是当前最实用的中文来源。

## 七、复现

    python3 scripts/crawl_moegirl.py
    python3 /tmp/crawl_batch.py
    python3 /tmp/crawl_cats.py
    .venv/bin/python scripts/build_naruto_corpus.py
    .venv/bin/python scripts/ingest_milvus.py --corpus .../chunks.jsonl --collection rag_naruto_chunks --recreate
    .venv/bin/python scripts/ingest_milvus.py --corpus .../children.jsonl --collection rag_naruto_children --with-parent --recreate
    .venv/bin/python scripts/build_naruto_kg.py && .venv/bin/python scripts/load_naruto_kg_neo4j.py
    .venv/bin/python scripts/test_naruto_rag.py
    .venv/bin/python scripts/test_naruto_multihop.py
## 八、语料清洗（2026-11-02）

萌娘百科页面文本里混有大量导航/信息框/引用噪声（“跳转到正文 / Skip to content”、“贡献者：”、信息框字段、引用角标等）。
新增 scripts/clean_naruto.py 做规则清洗：丢弃导航行、引言行、信息框短字段，保留正文散文句（按句号重组、过滤 <18 字碎片），
并截断到 2 万字符。清洗后：220 → 204 篇、117 万 → 71 万字；重建语料 546 parents / 4,548 children / 1,112 chunks，
并同步更新全量索引 rag_all_chunks（删除旧 moegirl 行、插入清洗行）。

清洗后复测：事实型 17/18（+1）、多跳型 12/12（不变），证据片段不再包含导航噪声。
