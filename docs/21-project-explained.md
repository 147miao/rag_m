# RAG-M 项目详解：语言基础、外部库基础与理论基础

> 这篇文章从三个层次讲清楚整个项目：**Python 语言基础**（代码为什么这么写）、**外部库基础**（每个库负责什么）、**理论基础**（背后的算法与评估方法）。每个知识点都会指到项目里的真实文件和符号，方便边读边看代码。

- 项目：路由式混合 RAG（simple / multihop / agentic 三条链路）
- 技术栈：Python 3.13 · LangGraph 1.2 · Milvus 3.0 · Neo4j 5 · FastAPI · React 19 · Docker · Ollama
- 代码规模：src 约 40 个 Python 文件，scripts 约 40 个实验/工具脚本，web 7 个组件
- 阅读顺序建议：第 0 章 → 第 1/2/3 章任选 → 第 4 章（端到端）→ 第 5 章（文件地图）

---

## 第 0 章 一分钟看懂整个系统

一句话：**用户问题先被一个小分类器判定为"简单事实 / 多跳关系 / 复杂推理"，然后分别走混合检索 RAG、Graph RAG、Agentic RAG，最后统一做引用校验并以 SSE 流式输出。**

```text
用户问题
   │
   ▼
[分类器] Qwen3-0.6B + LoRA（36ms）  ── 失败降级规则分类器
   │  simple / multihop / reasoning(=agentic)
   ├───────────────┬────────────────────────┬─────────────────────┐
   ▼               ▼                        ▼                     │
[Simple RAG]   [Graph RAG]              [Agentic RAG]              │
hybrid 检索    实体链接→Neo4j 1 跳扩展    plan→tools 循环（≤2 轮）    │
+ bge 重排     + 邻居条目证据            vector_search/kg_query/    │
+ LLM 生成     + hybrid+重排+生成        web_search → 证据合并       │
   └───────────────┴────────────────────────┴─────────────────────┘
                                   ▼
                       [引用校验] [n] 编号范围检查
                                   ▼
                 SSE: stage(阶段) → token(逐字) → done(完整响应)
```

对应代码入口：

| 层次 | 入口 |
|---|---|
| HTTP 接口 | `src/ragm/api/main.py` → `create_app()` |
| 路由处理 | `src/ragm/api/routes.py` → `/api/query`、`/api/query/stream` |
| RAG 服务 | `src/ragm/rag/langgraph_service.py` → `LangGraphRagService` |
| 简单/多跳图 | `src/ragm/rag/graph.py` → `RagGraph`（LangGraph StateGraph） |
| Agentic 图 | `src/ragm/agent/agent_graph.py` → `AgentGraph` |
| 检索 | `src/ragm/store/search_service.py` → `SearchService` |
| 向量库 | `src/ragm/store/milvus_store.py` → `MilvusStore` |
| 图谱 | `src/ragm/graph/kg.py` → `KGStore` |
| 分类器 | `src/ragm/classifier/` |
| 前端 | `web/src/App.tsx` → SSE 流式渲染 |

---

## 第 1 章 语言基础：Python 3.13 在项目里怎么用

> 这一章不是 Python 教程，而是"项目里出现的每个语言特性，为什么用它、用在哪"。

### 1.1 类型注解与数据结构

**（1）dataclass：分类结果**

`src/ragm/classifier/base.py`：

```python
@dataclass
class RouteDecision:
    label: Label          # Literal["simple", "multihop", "reasoning"]
    confidence: float
    backend: str
    latency_ms: float
    reason: str = ""

    @property
    def route(self) -> Route:   # label → 对外路由名
        return LABEL_TO_ROUTE[self.label]
```

要点：`Literal` 让类型检查器能穷举取值；`@property` 把"标签"和"路由"两个概念解耦——分类器只认 3 个标签，API 对外只暴露 3 条路由。

**（2）TypedDict：LangGraph 的状态**

`src/ragm/rag/graph.py` 里 `RagState` 用 `TypedDict(total=False)` 描述图的"共享黑板"：

```python
class RagState(TypedDict, total=False):
    question: str
    route: str
    hits: list[dict]
    answer_lead: str
    citations: list[int]
    rerank_ms: float | None
```

为什么不用普通 dict？因为 LangGraph 依赖 state 的键集合来合并节点返回值；`total=False` 表示每个键都可选（节点只返回自己写的那部分）。注意一个坑：**没在 TypedDict 里声明的键，LangGraph 会静默丢掉**——项目里曾出现"trace 不显示重排模型名"，就是因为 `rerank_model` 没在 state 里声明。

**（3）pydantic：对外 API 契约**

`src/ragm/api/schemas.py` 里所有响应模型都继承 `APIModel`：

```python
class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
```

- `extra="forbid"`：后端多返回字段会直接报错，逼迫契约严格对齐前端；
- `alias_generator=to_camel`：`Stats` 用 camelCase 别名（`latencyMs`），其余保持 snake_case；
- 请求校验（`question` 长度、`k` 范围）由 pydantic 自动完成，路由函数不用手写 if。

**（4）Protocol：服务层接口**

`src/ragm/rag/service.py`：

```python
class RagService(Protocol):
    def scenarios(self) -> list[ScenarioSummary]: ...
    async def query(self, req: QueryRequest) -> QueryResponse: ...
    def stream(self, req: QueryRequest) -> AsyncIterator[StreamEvent]: ...
```

好处：API 层只依赖协议，`DemoRagService`（演示数据）和 `LangGraphRagService`（真实链路）可以无缝替换；支持结构化类型（不用继承）。

### 1.2 面向对象：组合、工厂与兜底

**工厂 + 兜底包装**（`src/ragm/classifier/factory.py`）：

```python
class FallbackClassifier:
    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback or RulesClassifier()

    def classify(self, question):
        if self.primary is not None:
            try:
                return self.primary.classify(question)
            except Exception:
                logger.warning("primary classifier failed, fallback", exc_info=True)
        return self.fallback.classify(question)
```

这是**包装器/装饰器模式**：主分类器（LoRA）失败时自动降级到规则分类器，API 永远拿得到结果。同样思路也用在 health 检查、搜索服务（Milvus 挂了返回 503 而不是崩溃）。

**可插拔模型注册表**（`src/ragm/embeddings/embedder.py`）：

```python
MODEL_SPECS = {
    "google/embeddinggemma-300m": {"kind": "mean", "dim": 768},
    "BAAI/bge-m3": {"kind": "cls", "dim": 1024},
    "Qwen/Qwen3-Embedding-0.6B": {"kind": "last", "dim": 1024,
        "query_instruction": "Instruct: ... Query: "},
}
```

不同模型的 pooling 方式不同（CLS / mean / last token），抽象成配置字段后一个 `Embedder` 类就能加载所有模型。

### 1.3 异步编程：为什么 API 能同时处理查询和流式输出

**（1）async def + await**：FastAPI 的路由函数是异步的（`src/ragm/api/routes.py`），等待 IO 时不会阻塞事件循环。

**（2）asyncio.to_thread：把同步的 GPU/网络阻塞调用丢进线程池**

`LangGraphRagService.query` 里：

```python
state = await asyncio.to_thread(self.rag.invoke, req.question, req.route_override)
```

`RagGraph.invoke` 内部是同步的（模型推理、Milvus 检索都是阻塞调用），如果直接 `await` 一个同步函数会卡住整个服务；`to_thread` 让它在线程里跑，事件循环继续服务其他请求。

**（3）异步生成器：SSE 流式输出**

`stream()` 用 `yield` 逐条产出事件：

```python
async def stream(self, req) -> AsyncIterator[StageEvent | TokenEvent | DoneEvent]:
    yield StageEvent(index=0, step=TraceStep(key="route", ...))
    async for token in self._aiter_tokens(messages):
        yield TokenEvent(text=token)
    yield DoneEvent(response=...)
```

`_aiter_tokens` 用 `threading.Thread + asyncio.Queue` 把同步的 `httpx` 流式响应桥接成异步迭代器（见 `langgraph_service.py:241`）。

### 1.4 生成器与上下文管理器

**生成器**：前端 `streamQuery` 和后端 `_aiter_tokens` 都是"边产边消费"的数据流；生成器让内存里不需要保存完整答案。

**上下文管理器**：`with self.driver.session() as s:`（Neo4j）、`with httpx.Client() as client:`（HTTP 连接池）、`@contextlib.contextmanager`（`bench_common.gpu_lock` 的 GPU 文件锁）。

### 1.5 装饰器：可观测性

`src/ragm/observability/tracing.py` 的 `@traceable` 是一个"懒加载装饰器"：

```python
def traceable(*, run_type="chain", name=None):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _enabled:
                return fn(*args, **kwargs)
            return _traced(fn, run_type, name)(*args, **kwargs)
        return wrapper
    return decorator
```

为什么不用 `@langsmith.traceable` 直接装饰？因为装饰发生在 import 时，而 LangSmith 的开关来自运行时 `.env`；**在调用时判断**才能让"启动后配置生效"成立。

### 1.6 文件、JSON、正则与路径

- `pathlib.Path`：所有路径都用 Path 拼接（`config.py` 的 `data_root / "data" / "rag"`），避免字符串拼接错误；
- `json`：JSONL 逐行读写（`for line in path.read_text().splitlines()`）；
- `re`：引用解析 `[n]`（`rag/citations.py`）、prompt 结构解析（`parse_structured_answer`）、语料清洗规则；
- `hashlib.sha1`：把标题映射成稳定 doc_id（`nr_<hash>`、`en_<hash>`），保证多次重建索引 id 不变。

### 1.7 异常处理与降级

项目里异常处理有一条统一原则：**核心链路失败必须降级，观测/可选功能失败不能影响主流程**。

- `try/except` + `logger.warning(exc_info=True)`：记录堆栈但不抛出；
- `FallbackClassifier`：分类失败 → 规则；
- `SearchService` 构建失败 → `app.state.search = None`，`/api/search` 返回 503；
- `KGStore` 连接失败 → 图谱扩展禁用，hybrid 检索继续；
- SSE 中 `onError` 回调 → 前端显示错误条，而不是播放假答案。

### 1.8 工程化：uv、pyproject、ruff、pytest

| 工具 | 作用 | 项目位置 |
|---|---|---|
| `uv` | 虚拟环境 + 依赖安装 | `make install`、`uv.lock` |
| `pyproject.toml` | 依赖、Python 版本（3.13）、ruff/pytest 配置 | 根目录 |
| `ruff` | 格式与静态检查 | `make lint` |
| `pytest` | 单元/契约测试（17 条） | `tests/` |
| `docker compose` | Milvus / Neo4j / SearXNG | `docker/compose.yaml` |
| `Makefile` | 常用命令入口 | 根目录 |

一个细节：`pyproject.toml` 用 `dependency-groups` 区分 `dev`（pytest/ruff）与 `eval`（ragas 等），避免生产依赖被评测库污染。

---
## 第 2 章 外部库基础：每个库负责什么

> 这一章按"数据流"的顺序介绍依赖：从 LLM、向量库、图数据库，到 Web 框架、模型库、数据处理与前端。

### 2.1 LangGraph：把 RAG 流程画成状态机

**它是什么**：LangGraph 是 LangChain 团队做的"有状态工作流"框架。你把流程写成**节点（函数）+ 边（跳转）**，它负责按顺序执行、在节点之间传递共享状态（state），并支持条件跳转和循环。

**项目怎么用**（`src/ragm/rag/graph.py`）：

```python
from langgraph.graph import END, START, StateGraph

g = StateGraph(RagState)
g.add_node("classify", self._classify_node)      # 分类
g.add_node("retrieve", self._retrieve_node)      # 检索 + 图谱扩展 + 重排
g.add_node("generate", self._generate_node)      # LLM 生成
g.add_node("verify", self._verify_node)          # 引用校验
g.add_edge(START, "classify")
g.add_edge("classify", "retrieve")
g.add_edge("retrieve", "generate")
g.add_edge("generate", "verify")
g.add_edge("verify", END)
graph = g.compile()
```

调用：`graph.invoke({"question": "...", "route_override": "auto"})` 返回合并后的 state。

**关键概念**

| 概念 | 项目中的对应 |
|---|---|
| State | `RagState`（TypedDict）；节点返回 dict 自动合并 |
| Node | `_classify_node` 等，参数是 state，返回要更新的键 |
| Edge | `add_edge` 固定顺序；`add_conditional_edges` 条件跳转 |
| Conditional edge | `AgentGraph._route_after_plan`：有 tool_calls → tools，否则 synthesize |
| 循环 | `tools → plan` 形成 ReAct 循环（`agent_graph.py`） |
| compile | `_build()` 返回可执行图；每次请求 invoke 一次 |

**Agentic 图的条件跳转**（`src/ragm/agent/agent_graph.py`）：

```python
g.add_conditional_edges("plan", self._route_after_plan,
                        {"tools": "tools", "synthesize": "synthesize"})
g.add_edge("tools", "plan")   # 工具执行完回到规划，形成循环
```

**一个坑**：state 里没声明的键会被丢弃（见 1.1 节），所以新增字段必须同时改 TypedDict。

### 2.2 LLM 接入：OpenAI 兼容协议 + httpx

LLM 不是用官方 SDK，而是直接用 **OpenAI 兼容的 HTTP 协议**（`/v1/chat/completions`），因为 Ollama、DeepSeek、vLLM、OpenAI 都支持它，换模型只改 `.env`。

`src/ragm/rag/llm.py` 提供三种调用：

```python
class LLMClient:
    def chat(self, messages, max_tokens=1024, json_mode=False) -> dict
        # 普通对话，返回 text + tokens_in/tokens_out

    def chat_tools(self, messages, tools, max_tokens=1024) -> dict
        # 带 tools 定义；返回 tool_calls（OpenAI 格式）

    def stream_chat(self, messages, max_tokens=1024) -> Iterator[str]
        # SSE 流式，逐 token yield
```

**工具调用格式**（OpenAI 标准）：

```json
{"role": "assistant", "tool_calls": [
  {"id": "call_1", "type": "function",
   "function": {"name": "vector_search", "arguments": "{\"query\": \"鸣人的父亲\"}"}}]}
```

项目处理：`AgentGraph._tools_node` 解析 `function.arguments`（JSON 字符串）→ 执行工具 → 把结果作为 `{"role": "tool", "tool_call_id": ...}` 消息塞回对话。

**流式解析**：SSE 每行以 `data:` 开头，内容是 JSON 增量（`choices[0].delta.content`）；`stream_chat` 负责拼装 token，`_aiter_tokens` 再桥接成异步迭代器。

### 2.3 Milvus / pymilvus：向量 + 稀疏混合检索

**Milvus 3.0 的核心能力**：一个 collection 里同时存稠密向量（FLOAT_VECTOR）、稀疏向量（SPARSE_FLOAT_VECTOR）和标量字段，并支持 **BM25 Function** 从文本字段自动生成稀疏向量。

项目的 schema（`src/ragm/store/milvus_store.py`）：

```python
schema.add_field("chunk_id", DataType.VARCHAR, max_length=128)
schema.add_field("text", DataType.VARCHAR, max_length=16384,
                 enable_analyzer=True, analyzer_params={"type": "chinese"})
schema.add_field("dense", DataType.FLOAT_VECTOR, dim=768)
schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
schema.add_function(Function(name="bm25_fn", function_type=FunctionType.BM25,
                             input_field_names=["text"], output_field_names=["sparse"]))
```

索引：

```python
idx.add_index(field_name="dense", index_type="HNSW", metric_type="COSINE",
              params={"M": 16, "efConstruction": 200})
idx.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
```

**关键 API 与坑**

| API | 用途 | 坑 |
|---|---|---|
| `create_collection` + `prepare_index_params` | 建集合与索引 | 建完要 `load_collection` 才能搜 |
| `insert` + `flush` | 批量写入 | 每 256 行一批，最后 flush 一次 |
| `search` | 单路检索 | 返回 distance 字段（COSINE 越大越相似） |
| `hybrid_search` + `AnnSearchRequest` + `RRFRanker` | 混合检索 | 各路 limit 与最终 limit 分开设置 |
| `query(filter=...)` | 按标量查 | **limit 上限 16384**，超过要分页 |
| `delete(filter=...)` | 按条件删 | 返回 delete_count；删除后需 flush |
| VARCHAR 长度 | 中文按字节算 | 16384 字节 ≈ 5461 个汉字，注意截断 |

项目封装 `MilvusStore`：`recreate()`、`insert()`、`flush()`、`load()`、`hybrid_search()`、`dense_search()`、`sparse_search()`、`delete_by_filter()`、`count()`。

### 2.4 Neo4j：实体图

**它是什么**：原生图数据库，数据模型是"节点 + 关系"，查询语言 Cypher。

项目图谱（`src/ragm/graph/kg.py`）：

```cypher
MATCH (a:Entity)-[r:CO_EVIDENCE]-(b:Entity)
WHERE a.name IN $names
RETURN a.name AS a, b.name AS b, r.qtype AS qtype, r.question AS question
LIMIT $limit
```

- 节点标签：HotpotQA 用 `Entity`，火影用 `NarutoEntity`（隔离不同图谱）；
- 关系：`CO_EVIDENCE`（同题共现）、`MENTIONS`（条目提到条目）；
- 唯一约束：`CREATE CONSTRAINT ... REQUIRE e.name IS UNIQUE`，保证 MERGE 不重复；
- 导入方式：`UNWIND $rows AS row MATCH ... MERGE ...` 批量写入（5000 行/批）。

**多图谱支持**：`KGStore` 接受逗号分隔的 `label=Entity,NarutoEntity`、`relation=CO_EVIDENCE,MENTIONS`，加载时合并实体名、扩展时分图查询，从而一个服务同时支持两套图谱。

### 2.5 FastAPI：HTTP 层

**应用生命周期**（`src/ragm/api/main.py`）：`lifespan` 里依次构建分类器、搜索服务（Milvus+embedding）、图谱、重排、LLM、LangGraph 服务；失败的部分置 None 并在 health 中体现。

**路由**（`src/ragm/api/routes.py`）：`APIRouter(prefix="/api")`，装饰器声明方法/路径/response_model；FastAPI 自动做参数校验、序列化、生成 OpenAPI 文档（`/docs`）。

**SSE 流式**：用 `StreamingResponse` 返回异步生成器：

```python
async def gen():
    async for ev in service.stream(req):
        yield f"event: {kind}\ndata: {payload}\n\n"
return StreamingResponse(gen(), media_type="text/event-stream")
```

前端用 `fetch + ReadableStream` 逐块解析（`web/src/lib/api.ts` 的 `streamQuery`）。

**中间件**：请求 ID + 耗时日志（`X-Request-ID`/`X-Elapsed-Ms`），写入 `logs/api.jsonl`。

### 2.6 pydantic-settings：配置管理

`src/ragm/config.py`：

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    data_root: Path = Path("/run/media/myj/D/rag_m")
    milvus_uri: str = "http://127.0.0.1:19530"
    embed_model_path: Path | None = None
    ...

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- `@lru_cache`：配置只解析一次（进程级单例）；
- 优先级：环境变量 > `.env` > 默认值（测试里用 `monkeypatch.setenv` 覆盖）；
- `extra="ignore"`：`.env` 里有多余变量不会报错。

### 2.7 transformers + peft：本地模型与 LoRA

**加载基座模型**（`src/ragm/embeddings/embedder.py`）：

```python
self.tok = AutoTokenizer.from_pretrained(path)
self.model = AutoModel.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0}).eval()
h = self.model(**enc).last_hidden_state      # [batch, seq, hidden]
```

- `bfloat16`：8GB 显存下比 fp32 省一半；`device_map={"": 0}` 表示全部放 GPU 0；
- `.eval()` + `torch.inference_mode()`：关闭 dropout、省显存。

**三种 pooling**（模型决定）：CLS（bge 系列取第 0 个 token）、mean（embeddinggemma 做 attention mask 加权平均）、last（Qwen3-Embedding 取最后一个非 pad token）。

**LoRA 微调**（`scripts/train_route_cls.py` + `src/ragm/classifier/lora.py`）：

```python
from peft import LoraConfig, get_peft_model
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                 modules_to_save=["score"])
model = get_peft_model(base, cfg)
```

- 任务头：`AutoModelForSequenceClassification.from_pretrained(..., num_labels=3)`（Qwen3 有 `Qwen3ForSequenceClassification`）；
- 变体：`use_rslora=True`（rsLoRA）、`use_dora=True`（DoRA）、`init_lora_weights="pissa"`（PiSSA）；
- 训练：HuggingFace `Trainer` + 早停（dev macro-F1）+ 类别权重 + label smoothing 0.05；
- 推理：`logits.softmax(-1)` 得到 3 类置信度，取 argmax。

### 2.8 数据处理：从 PDF/网页到语料

| 库 | 用途 | 项目位置 |
|---|---|---|
| `pyarrow` / `datasets` | 读 parquet 数据集（HotpotQA/CMRC） | `scripts/build_route_dataset.py`、`build_retrieval_eval.py` |
| `pypdf` | 抽取 PDF 文本（8 页论文 0.1 秒） | `scripts/build_papers_corpus.py` |
| `BeautifulSoup` | 解析 HTML | `scripts/crawl_naruto.py` |
| `Playwright` | 渲染 JS 页面（萌娘百科/360 百科） | `scripts/crawl_moegirl.py` |
| `unstructured` | 文档切分（PDF/DOCX） | 已安装，但 `partition_pdf(strategy="fast")` 在本机 120 秒返回 0 元素，改用 pypdf |
| `numpy` | 评测指标（mean、log2、argmax） | 所有 `eval_*.py` |

### 2.9 其他重要依赖

| 库/工具 | 作用 | 项目位置 |
|---|---|---|
| `httpx` | 同步 HTTP（Ollama/DeepSeek 调用、Bing RSS） | `rag/llm.py`、`agent/tools.py` |
| `langsmith` | 追踪（traceable、run 树） | `observability/tracing.py` |
| `pytest` + `pytest-asyncio` | 契约测试（httpx ASGI transport） | `tests/test_api.py` |
| `ruff` | lint/format | `make lint` |
| `uv` | 依赖解析与锁（uv.lock） | `make install` |
| `Ollama` | 本地/云 LLM 服务（OpenAI 兼容） | `.env` 的 `LLM_BASE_URL` |
| `React 19` + `Vite 8` + `Tailwind 4` + `TypeScript 7` | 前端 | `web/` |
| `Docker Compose` | Milvus/etcd/minio/Neo4j/SearXNG | `docker/compose.yaml` |

**版本矩阵（本项目实测）**

| 组件 | 版本 |
|---|---|
| Python | 3.13.15 |
| LangGraph / LangChain | 1.2.18 / 1.4.7 |
| Milvus / pymilvus | 3.0.4 / 3.0.2 |
| Neo4j | 5.26-community |
| FastAPI / pydantic | 0.162.0 / 2.14.2 |
| torch / transformers / peft | 2.15.0 / 5.22.1 / 0.21.2 |
| React / Vite / Tailwind | 19.3 / 8.3 / 4.3 |

---
## 第 3 章 理论基础：每个设计决策背后的原理

> 这一章回答"为什么这么做"：为什么分类、为什么混合检索、为什么重排、为什么 small-to-big、为什么需要图谱和工具、怎么评估才可信。

### 3.1 为什么要路由：RAG 的成本/精度三角

朴素 RAG 对所有问题都用同一套"检索 top-k → 生成"，但问题差异很大：

| 问题类型 | 例子 | 最优策略 |
|---|---|---|
| 单跳事实（simple） | "鸣人的父亲是谁？" | 一次 hybrid 检索 + 生成，成本最低 |
| 多跳关系（multihop） | "自来也的弟子中谁成为了四代火影？" | 需要先把"自来也→弟子"和"弟子→四代火影"两条链连起来，图谱扩展或多次检索 |
| 复杂推理（reasoning/agentic） | "对比 RAG 与 GraphRAG，各自适合什么场景？" | 需要多源检索、比较、归纳，甚至联网 |

**路由的收益**：简单题走轻量链路（36ms 分类 + 30ms 检索 + 1s 生成），复杂题才付出多跳/工具的额外成本；同时每类问题可以用最适合的检索策略，精度更高。代价是引入一个分类器，它有误差，所以需要**置信度**和**降级策略**。

### 3.2 文本分类：从 TF-IDF 到 LoRA 微调

**任务定义**：输入问题文本 x，输出 3 类之一 y ∈ {simple, multihop, reasoning}，以及置信度 p。

**模型演进（项目里都实现了）**

| 方案 | 原理 | 项目结果（500 条人工裁决测试集） |
|---|---|---|
| 规则 | 关键词/正则匹配 | acc 0.420 / macro-F1 0.215（只会判 simple） |
| LLM 提示词 | 让 gemma4 按 rubric 输出标签 | acc 0.834 / macro-F1 0.780 |
| bge-m3 + 逻辑回归 | 句向量 + 线性分类头 | acc 0.858 / macro-F1 0.856 |
| **Qwen3-0.6B + LoRA + SEQ_CLS** | 预训练语言模型 + 低秩适配 + 分类头 | **acc 0.910 / macro-F1 0.898** |

**LoRA 原理**：冻结原模型权重 W，只学习低秩增量 ΔW = B·A（A 是 r×d，B 是 d×r，r 很小）。可训练参数从 6 亿降到几百万，8GB 显存就能微调。分类任务再加一个线性头（hidden → 3），训练时用交叉熵。

**评估指标**：

- **Accuracy**：整体正确率，类别不平衡时会失真；
- **Macro-F1**：先算每类 F1 再平均，对少数类（reasoning）更敏感——本项目主指标；
- **ECE**（Expected Calibration Error）：置信度是否可信，影响"低置信度回退"策略。

**标签口径问题（重要教训）**：同一批数据，用"数据集来源"打标签和用"LLM rubric"打标签，一致率只有 65.9%；换成"数据集自带问题类型元数据"后升到 80.4%。说明**标签定义比模型选择更影响最终数字**——这也是为什么最终测试集采用"双标注一致 + 人工裁决"。

### 3.3 向量表示与语义检索

**双塔模型（bi-encoder）**：把问题和文档分别编码成向量，用余弦相似度比较：

```text
cos(q, d) = (q · d) / (||q|| × ||d||)
```

项目里所有 embedding 都做了 **L2 归一化**，所以内积等价于余弦相似度。

**为什么能语义匹配**：模型在海量文本对上训练过，"汽车"和"轿车"的向量距离很近，不需要字面重合。

**HNSW 近似最近邻**：精确检索要算 q 与所有文档的相似度（4 万条尚可，上亿条不可行）。HNSW 用多层图结构做近似搜索：

- `M`：每个节点的邻居数（16），越大越准越慢；
- `efConstruction`：建图时的候选队列（200）；
- `efSearch`：查询时的候选队列（由 Milvus 自动或参数控制）。

**Embedding 选型**：不能只看 MTEB 榜单，要在自己的语料上测。项目实测（EN nDCG@10 / ZH nDCG@10）：

| 模型 | EN | ZH | 维度 |
|---|---|---|---|
| embeddinggemma-300m | 0.848 | **0.977** | 768 |
| bge-m3 | **0.848** | 0.959 | 1024 |
| qwen3-embedding-0.6b | 0.787 | 0.948 | 1024 |
| bge-large-zh-v1.5 | 0.589 | 0.938 | 1024 |

结论：默认 embeddinggemma-300m（中文最好、维度小、显存省）；bge-m3 作为备选（长上下文、自带 sparse）。

### 3.4 稀疏检索与 BM25

**BM25 打分公式**：

```text
score(q, d) = Σ_t IDF(t) × (tf(t,d) × (k1+1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl))
```

- `tf`：词 t 在文档 d 中出现次数；
- `IDF`：越少见的词权重越高（"的"权重低，"九尾"权重高）；
- `k1`（1.2~2.0）：词频饱和速度；`b`（0.75）：文档长度惩罚。

**为什么 RAG 仍需要 BM25**：向量检索擅长语义相似，但对**精确匹配**（人名、专有名词、编号、数字）不敏感；BM25 正好相反。两者互补。

**Milvus 的实现**：`FunctionType.BM25` 挂在 `text` 字段上，写入时由 analyzer（中文分词）自动生成稀疏向量；检索时把自然语言 query 传给 sparse 字段即可，不需要自己算 tf-idf。

实测：英文 hybrid R@10 0.888 vs dense 0.838 / sparse 0.815；中文 BM25 单独就很强（nDCG 0.961）。

### 3.5 混合检索与 RRF 融合

dense 返回余弦相似度（0~1），sparse 返回 BM25 分数（0~30+），量纲完全不同，不能直接相加。**RRF（Reciprocal Rank Fusion）** 只看排名，不看分数：

```text
RRF(d) = Σ_{r ∈ 各检索器} 1 / (k + rank_r(d)),   k = 60
```

- 排名越靠前贡献越大；k=60 让头部差距平滑，避免某一路完全主导；
- 实现：Milvus 内置 `RRFRanker(60)`；项目里 `search_with_scores` 还做了一遍本地 RRF，为的是**同时保留 dense/sparse 原始分数**给前端展示。

### 3.6 Cross-Encoder 重排：两阶段检索

**Bi-encoder 的局限**：问题和文档分别编码，没有交互，细粒度匹配弱。

**Cross-encoder**：把 (query, document) 拼起来过一遍模型，输出相关性分数——精度高但慢（每个候选都要一次前向）。

所以采用**两阶段**：第一阶段用混合检索快速召回 top-20（保证召回率），第二阶段用 cross-encoder 精排到 top-8（提升 nDCG）。项目实测（EN nDCG@10）：

| 方案 | EN | ZH | p50 |
|---|---|---|---|
| 无重排 | 0.810 | 0.963 | — |
| **bge-reranker-base** | **0.922** | 0.995 | 59ms |
| bge-reranker-v2-m3 | 0.915 | 0.995 | 231ms |
| qwen3-reranker-0.6b | 0.866 | **0.996** | 552ms |

**重要方法论教训**：换一套评测集（火影，gold 是"条目名"而不是"段落"），重排反而让 nDCG 从 0.511 掉到 0.456——因为标题匹配本身很强，重排按段落语义重排会打乱顺序。**"重排是否有效"取决于评测口径**，不能一概而论。

### 3.7 切块与 Small-to-Big

**固定切块的矛盾**：

- 块太小：检索精准（主题集中），但上下文不完整，答案可能被切断；
- 块太大：上下文完整，但主题分散，检索精度下降。

**Small-to-Big（父文档检索）**：用**小块（child，200 字）**建索引负责"精确匹配"，命中后返回**所属大块（parent，整篇/1600 字）**负责"上下文完整"。

项目在 PubMedQA 长文档上的实测：

| 方案 | 证据覆盖率 | 上下文字符 |
|---|---|---|
| 800 字块 top-8 | 0.70 | 5,543 |
| small-to-big k=2 | **0.98** | **2,784** |
| small-to-big k=4 | 0.99 | 5,612 |

即：用一半的上下文，覆盖率从 0.70 提到 0.98。短文档（文章本身 <800 字）则没有收益——**chunk 策略要匹配文档形态**。

### 3.8 Graph RAG：把关系显式化

**动机**：多跳问题的答案分布在多个文档里，向量检索只能"各自召回"，缺少"实体之间的连接"。

**构图**：实体是节点，关系是边。

- HotpotQA：同一道题的 supporting_facts 标题两两连边（`CO_EVIDENCE`），边带问题类型与原文——相当于把"哪些实体在同一个问题里被一起用到"编码成图；
- 火影：条目正文提到其他条目就连边（`MENTIONS`），表示"这两个角色/概念相关"。

**检索流程**：问题 → 实体链接（先匹配问题里的实体名，再用检索结果的标题兜底）→ Neo4j 1 跳邻居 → 邻居条目的 chunk 补进证据 → 与 hybrid 证据合并 → 生成。

**效果**：HotpotQA 多跳题 gold 文档召回 0.875 → 1.0；火影多跳 12/12。代价是 ~24ms 图查询 + 可能的噪声（邻居不一定相关）。

### 3.9 Agentic RAG：工具循环（ReAct 思想）

ReAct = Reason + Act：LLM 先"想"（是否需要工具），再"做"（调用工具），观察结果后继续想，直到能回答。

项目实现（`src/ragm/agent/agent_graph.py`）：

```text
plan ──有 tool_calls──▶ tools ──▶ plan ──...──▶ synthesize ──▶ verify
  └──没有 tool_calls────────────────────────────┘
```

- 工具用 JSON Schema 描述（`vector_search` / `kg_query` / `web_search`），LLM 输出 `tool_calls`；
- 每轮工具结果作为 `role=tool` 消息回传；
- 最多 2 轮（避免无限循环与成本失控）；
- 最终 synthesize 把所有工具证据合并成编号上下文再生成。

**什么时候值得用**：问题需要"先查 A 再决定查 B"、或本地语料不够需要联网时；简单事实问题用 agent 反而更慢更贵。

### 3.10 生成、引用与拒答

**Grounded generation**：system prompt 强制"只依据检索资料回答、每个结论标 [n]、资料不足就明确说没有"。

**引用校验**（`src/ragm/rag/citations.py`）：正则提取所有 `[n]`，检查是否落在 evidence 范围内，输出 `2/2 通过` 这样的文案。局限：只校验编号范围，不校验语义支撑（下一步可做 NLI 校验）。

**拒答策略**：检索为空、或模型判定"资料中没有足够信息"时，返回 `refusal` 而不是编造。这是降低幻觉的关键手段——宁可说不知道。

### 3.11 评测理论：怎么判断"好不好"

**检索指标**（gold 是相关文档集合）：

```text
Recall@k = |top-k ∩ gold| / |gold|
DCG@k    = Σ_{i=1..k} rel_i / log2(i+1)
nDCG@k   = DCG@k / IDCG@k
MRR      = 1 / (第一个相关结果的名次)
```

nDCG 同时考虑"是否召回"和"排得靠不靠前"，是最常用的排序质量指标。

**生成指标**（本项目用 LLM-as-judge 复刻 ragas 定义）：

| 指标 | 定义 |
|---|---|
| faithfulness | 把答案拆成原子陈述，逐条判断是否被上下文支持，取比例 |
| answer_relevancy | 由答案反推 3 个问题，与原问题的 embedding 相似度均值 |
| context_precision | 逐条判断上下文是否与问题相关，取平均 |
| context_recall | gold 答案是否出现在上下文中（客观匹配） |
| citation_valid | 引用编号是否全部有效 |

实测（40 条中英混合）：faithfulness 0.978、context_recall 0.975、citation 0.975、answer_relevancy 0.767。

**LLM-as-judge 的偏差**：judge 偏宽松（v1 的 answer_relevancy 全是 1.0）、上下文截断会低估 recall（v1 用 260 字 snippet，v2 改用 1200 字全文后 recall 从 0.75 升到 0.975）。所以评测脚本本身也要被审视。

### 3.12 LangSmith：可观测性理论

一次查询是一棵 **span 树**：

```text
LangGraph (chain)
├── classify (chain) → llm_chat
├── retrieve (chain) → hybrid_search (retriever)
├── rerank (chain)
├── generate (chain) → llm_chat (llm)
└── verify (chain)
```

每个 span 记录输入、输出、耗时、token；可以回答"哪一步慢、哪个工具异常、模型返回什么"。LangSmith 未配置时装饰器直接透传，零开销。

---
## 第 4 章 一次请求的完整旅程（代码调用链）

以"鸣人的父亲是谁？"为例，从 HTTP 到答案，逐层看代码：

```text
① 前端 (web/src/App.tsx)
   fetch('/api/query/stream', {question, route_override:"auto"})
        │
② FastAPI (src/ragm/api/routes.py: query_stream)
   service.stream(req)  → StreamingResponse(text/event-stream)
        │
③ LangGraphRagService.stream (rag/langgraph_service.py)
   classify → LlmClient.classify()
   路由 = simple → RagGraph.prepare()
        │
④ RagGraph._retrieve_node (rag/graph.py)
   SearchService.search_with_scores(question, k=8)
     ├─ Embedder.encode_queries → 768 维向量
     ├─ Milvus dense_search + sparse_search
     ├─ 本地 RRF 融合（保留 dense/sparse 分数）
     └─ Reranker.rerank(top-20 → top-8)
        │
⑤ RagGraph._generate_node
   prompts.build_messages(hits) → LLMClient.chat(json/plain)
   解析出 answer_lead / answer_body
        │
⑥ RagGraph._verify_node
   citations.verify_citations → "1/1 通过"
        │
⑦ 组装 SSE 事件
   stage(route) → stage(retrieve) → token(逐字) → done(完整 QueryResponse)
        │
⑧ 前端渲染
   TraceRail / AnswerCard / RightPanel(evidence) / stats
```

**关键代码位置对照**

| 步骤 | 文件 | 函数/类 |
|---|---|---|
| 流式接口 | `api/routes.py` | `query_stream` |
| 服务编排 | `rag/langgraph_service.py` | `stream` / `query` |
| 图定义 | `rag/graph.py` | `RagGraph._build` |
| 检索 | `store/search_service.py` | `search_with_scores` |
| 向量库 | `store/milvus_store.py` | `hybrid_search` |
| 重排 | `embeddings/reranker.py` | `CrossEncoderReranker.rerank` |
| 生成 | `rag/llm.py` + `rag/prompts.py` | `chat` / `build_messages` |
| 引用 | `rag/citations.py` | `verify_citations` |
| 响应组装 | `api/schemas.py` | `QueryResponse` 等 |
| SSE 解析 | `web/src/lib/api.ts` | `streamQuery` |

**延迟构成（实测）**：分类 30~60ms + 检索 30~150ms + 重排 40~90ms + 生成 800~3000ms。生成占大头，所以分类器用 0.6B 小模型（36ms）是值得的。

---

## 第 5 章 项目文件地图

```text
rag_m/
├── src/ragm/
│   ├── config.py              pydantic-settings 配置中心
│   ├── api/
│   │   ├── main.py            FastAPI app 工厂、lifespan、中间件、静态文件
│   │   ├── routes.py          所有 /api 端点
│   │   ├── schemas.py         请求/响应 pydantic 模型（前后端契约）
│   │   └── examples.py        按语料返回示例问题
│   ├── classifier/            规则 / 提示词 / LoRA 三种分类器 + 工厂 + 降级
│   ├── embeddings/
│   │   ├── embedder.py        多模型 embedding（pooling 可配）
│   │   └── reranker.py        cross-encoder 与 Qwen3 生成式 reranker
│   ├── store/
│   │   ├── milvus_store.py    Milvus schema/索引/写入/检索/删除
│   │   └── search_service.py  检索服务（hybrid、small-to-big、RRF）
│   ├── graph/kg.py            Neo4j 实体图（实体链接、扩展、前端图数据）
│   ├── agent/
│   │   ├── tools.py           三个工具的定义与执行器
│   │   └── agent_graph.py     plan→tools→synthesize→verify 循环
│   ├── rag/
│   │   ├── graph.py           RagGraph（classify→retrieve→generate→verify）
│   │   ├── langgraph_service.py  服务实现（query/stream/响应组装）
│   │   ├── llm.py             OpenAI 兼容客户端
│   │   ├── prompts.py         提示词与上下文拼装
│   │   ├── citations.py       引用解析与校验
│   │   ├── service.py         RagService 协议 + Demo 实现
│   │   └── demo.py            演示数据（仅当 RAG_BACKEND=demo）
│   ├── observability/tracing.py  LangSmith 懒加载装饰器
│   └── ops/status.py          实验/训练进度看板
├── scripts/                   数据、索引、训练、评测（约 40 个）
├── web/                       React 19 前端
│   ├── src/App.tsx            主状态机（SSE/示例/空态/错误）
│   ├── src/components/        TopBar/SideRail/Composer/TraceRail/AnswerCard/RightPanel
│   ├── src/lib/api.ts         API 客户端（含 SSE 解析）
│   └── src/lib/types.ts       与后端契约对应的 TS 类型
├── tests/                     API 契约测试 + agent/tracing 单测（17 条）
├── docker/compose.yaml        Milvus/Neo4j/SearXNG
├── docs/                      方案与实验报告（本篇是第 21 篇）
├── reports/                   评测 JSON（gitignore）
└── Makefile                   常用命令
```

**scripts 分类速查**

| 类别 | 脚本 |
|---|---|
| 数据获取 | `crawl_moegirl.py`、`crawl_naruto.py`、`build_rag_corpus.py`、`build_naruto_corpus.py`、`build_papers_corpus.py`、`clean_naruto.py` |
| 索引 | `ingest_milvus.py`、`update_all_corpus.py`、`build_kg.py`、`load_kg_neo4j.py`、`build_corpus_small2big.py` |
| 分类器 | `build_route_dataset.py`、`relabel_route_dataset.py`、`curate_test.py`、`train_route_cls.py`、`eval_route_cls.py`、`baselines.py` |
| 评测 | `eval_retrieval_milvus.py`、`eval_graph_rag.py`、`eval_generation_v2.py`、`bench_embeddings.py`、`bench_rerankers.py`、`bench_chunk_size.py`、`bench_small2big_long.py` |
| 工程 | `bench_all.py`、`bench_common.py`、`status.py`、`use_corpus.sh`、`push_github.sh` |

---

## 第 6 章 实验结论一览（数字都是本项目实测）

### 6.1 分类器（test_human 500 条：402 双标注一致 + 98 人工裁决）

| 方案 | Acc | macro-F1 | p50 |
|---|---|---|---|
| 规则 | 0.420 | 0.215 | <1ms |
| gemma4 提示词 | 0.834 | 0.780 | ~760ms |
| bge-m3 + 逻辑回归 | 0.858 | 0.856 | ~2ms |
| **Qwen3-0.6B + LoRA + SEQ_CLS** | **0.910** | **0.898** | **36ms** |
| 对照：rsLoRA / PiSSA | 0.896 / 0.898 | 0.884 / 0.887 | 38 / 39ms |

### 6.2 检索（HotpotQA 37k 段 / CMRC 2.4k 篇）

| 语言 | 模式 | R@10 | nDCG@10 |
|---|---|---|---|
| EN | dense | 0.838 | 0.798 |
| EN | sparse | 0.815 | 0.746 |
| EN | **hybrid** | **0.888** | **0.803** |
| ZH | dense | 0.900 | 0.872 |
| ZH | sparse | 0.980 | **0.961** |
| ZH | hybrid | **1.000** | 0.953 |

### 6.3 其它结论

| 实验 | 结论 |
|---|---|
| Embedding | embeddinggemma-300m 综合最优（EN 0.848 / ZH 0.977） |
| Reranker | bge-reranker-base 最优（EN 0.922 / ZH 0.995 / 59ms） |
| Rerank 对生成 | 英文 faithfulness 0.915 → 0.990；中文无变化 |
| Chunk size | 短文档 200~1600 差异在噪声内；长文档 800~1600 更好 |
| Small-to-big | 长文档覆盖率 0.70 → 0.99（上下文字符减半） |
| Graph RAG | HotpotQA 多跳 gold 召回 0.875 → 1.000；火影多跳 12/12 |
| 生成质量 | faithfulness 0.978 / context_recall 0.975 / citation 0.975 |
| 真实文档 | 火影 204 篇：事实 17/18、多跳 12/12；arXiv 6 篇：10 题 9/10 |
| 性能 | 并行基准 GPU 平均利用率 79.6%（峰值 100%） |

---

## 第 7 章 学习路径（按顺序动手）

**Week 1：语言与工程基础**
1. 读 `src/ragm/config.py`、`api/schemas.py`，理解 pydantic 与配置；
2. 跑 `make install && make up && make api`，用 curl 调通 `/api/health`、`/api/route`；
3. 读 `tests/test_api.py`，理解契约测试。

**Week 2：表示学习与检索**
1. 读 `embeddings/embedder.py`，理解 pooling 与归一化；
2. 读 `store/milvus_store.py`，理解 schema/索引/hybrid_search；
3. 跑 `scripts/bench_embeddings.py`，自己换一个模型对比。

**Week 3：RAG 主链路**
1. 读 `rag/graph.py`，画一遍 classify→retrieve→generate→verify；
2. 读 `rag/prompts.py` 与 `citations.py`，理解引用与拒答；
3. 改 prompt 做一次 A/B，观察 faithfulness 变化。

**Week 4：Graph RAG 与 Agentic**
1. 读 `graph/kg.py` + `scripts/build_kg.py`，理解构图与扩展；
2. 读 `agent/tools.py` 与 `agent_graph.py`，理解 tool_calls 循环；
3. 加一个新工具（例如 calculator），跑通 agentic 路由。

**Week 5：评测**
1. 读 `eval_retrieval_milvus.py`、`eval_generation_v2.py`；
2. 理解 Recall/nDCG/MRR 公式，手动算一遍小例子；
3. 自己造 20 条问题，跑一次评测并解释结果。

**Week 6：前端与部署**
1. 读 `web/src/App.tsx` 的 SSE 状态机；
2. 改一个组件（例如证据面板）并理解数据流；
3. 用 Docker compose 部署，写一个 systemd 服务。

---

## 第 8 章 常见问题（FAQ）

**Q1：为什么不用 GPT-4/DeepSeek 做分类器？**
成本与延迟：每次提问都调大模型分类，延迟 700ms+、按量计费；小模型 LoRA 36ms、本地零成本，且在本任务上准确率更高（0.898 vs 0.780）。

**Q2：0.6B 够用吗？为什么不直接 7B？**
本任务输入是 30 字以内的问题、输出 3 类，0.6B + LoRA 已经到 0.898；7B 在 8GB 显卡上训练困难，收益边际。项目实测 1.7B 也未必更好（训练时间 2 小时/轮）。

**Q3：为什么既要 dense 又要 BM25？**
语义匹配 vs 精确匹配互补；中文 BM25 尤其强（nDCG 0.961）。混合检索 nDCG 比单路高 5~14 点。

**Q4：重排不是应该总是提升效果吗？**
不一定。重排优化的是"语义相关性排序"，如果评测 gold 是"标题匹配"，重排会打乱顺序（火影矩阵里 nDCG 0.511 → 0.456）。先明确评测口径，再决定要不要重排。

**Q5：为什么需要 Graph RAG？多查几次向量不行吗？**
多跳题的答案分布在多个文档，向量检索各自召回但缺少"连接"；图谱把实体关系显式化，1 跳扩展能把"邻居证据"带进来（召回 0.875 → 1.0）。

**Q6：为什么经常拒答？**
设计如此：证据不足时明确说"资料中没有足够信息"，比编造更安全。如果拒答过多，先检查语料是否覆盖问题、检索是否命中、prompt 是否过严。

**Q7：LangGraph 相比自己写 if/else 有什么好处？**
状态管理、条件跳转、循环、可观测性（span 树）、可组合（子图）。流程一复杂，手写 if/else 很快会失控。

**Q8：怎么接入自己的文档？**
把文档放到 `data/`，用 Unstructured/pypdf 抽文本 → 切成 parents/children → `scripts/ingest_milvus.py` 入库 → `make use-<corpus>` 切换。具体见 docs/16（火影）、docs/18（arXiv）。

**Q9：为什么有 17 个 Milvus collection？**
每个实验（embedding 对比、chunk size、naruto、papers、all）一个集合，互不干扰；生产环境只需要一个。

**Q10：项目能直接用于生产吗？**
不能直接。缺：鉴权、限流、多租户、语义引用校验、真正的文档摄取服务、监控告警。当前定位是"学习/实验 + 可演示的端到端系统"。

---

## 第 9 章 术语表

| 术语 | 含义 |
|---|---|
| RAG | Retrieval-Augmented Generation，检索增强生成 |
| Embedding | 把文本映射成稠密向量 |
| ANN / HNSW | 近似最近邻 / 分层可导航小世界图（向量索引算法） |
| BM25 | 基于词频与逆文档频率的稀疏检索打分 |
| RRF | Reciprocal Rank Fusion，按排名融合多路检索 |
| Cross-encoder | 把 query 与 doc 拼接后打分（重排） |
| LoRA | Low-Rank Adaptation，低秩适配微调 |
| rsLoRA / DoRA / PiSSA | LoRA 的变体（缩放、权重分解、SVD 初始化） |
| SEQ_CLS | Sequence Classification，序列分类任务头 |
| macro-F1 | 每类 F1 的平均，对少数类敏感 |
| ECE | 期望校准误差，衡量置信度是否可信 |
| nDCG | 归一化折损累计增益，排序质量指标 |
| MRR | 平均倒数排名 |
| faithfulness | 答案是否被证据支持（生成质量） |
| Graph RAG | 用知识图谱做检索增强 |
| ReAct | Reason + Act，LLM 工具调用循环 |
| SSE | Server-Sent Events，服务器推送流 |
| span | 追踪系统中的一次操作节点（LangSmith） |

---

## 附录：复现命令速查

```bash
# 环境
make install && make up && make health

# 数据与索引
.venv/bin/python scripts/fetch_models.py
.venv/bin/python scripts/build_rag_corpus.py
.venv/bin/python scripts/ingest_milvus.py --recreate

# 分类器
.venv/bin/python scripts/train_route_cls.py --out models/classifier/s1_06b_plain \
    --arch seq_cls --model models/Qwen3-0.6B --variant plain
.venv/bin/python scripts/eval_route_cls.py --run models/classifier/s1_06b_plain \
    --data data/classifier/test_human.jsonl

# 评测
.venv/bin/python scripts/eval_retrieval_milvus.py
.venv/bin/python scripts/eval_generation_v2.py --n 40
.venv/bin/python scripts/bench_all.py

# 服务
make api && make web
```

---

*这篇文档由项目维护者整理，数字均来自 `reports/` 与 `docs/` 中的实验记录；代码以仓库当前 `main` 分支为准。*
