"""API 契约：与 web/src/lib/types.ts 一一对应。

字段命名约定：
- 绝大多数字段与前端 TS 类型同名（snake_case）；
- Stats 使用 camelCase 别名（latencyMs 等），通过 alias_generator 自动转换，
  FastAPI 序列化默认 by_alias=True，前端无需改动。
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

Route = Literal["simple", "multihop", "agentic"]
RouteOverride = Literal["auto", "simple", "multihop", "agentic"]
ToolStatus = Literal["ok", "error"]
NodeKind = Literal["query", "entity", "doc", "answer"]
ServedBy = Literal["demo", "langgraph"]


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class Evidence(APIModel):
    id: str = Field(description="chunk 或图谱事实的确定性 ID")
    rank: int = Field(ge=1, description="重排后的序号")
    title: str
    source: str = Field(description="来源路径 / URL / kg:// 实体")
    dataset: str
    score: float = Field(ge=0, le=1)
    dense: float = Field(ge=0, le=1)
    sparse: float = Field(ge=0, le=1)
    snippet: str
    highlight: str | None = None


class TraceStep(APIModel):
    key: str
    label: str
    ms: int = Field(ge=0)
    detail: str


class ToolCall(APIModel):
    name: str
    args: str
    result: str
    ms: int = Field(ge=0)
    status: ToolStatus = "ok"


class GraphNode(APIModel):
    id: str
    label: str
    kind: NodeKind


class GraphEdge(APIModel):
    from_: str = Field(alias="from")
    to: str
    label: str


class GraphPayload(APIModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class Answer(APIModel):
    lead: str = ""
    body: str = ""


class Refusal(APIModel):
    title: str
    reason: str
    suggestion: str


class Stats(APIModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel, extra="forbid")

    latency_ms: int = Field(ge=0)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost_cny: float = Field(ge=0)
    citation_check: str


class ClassifierInfo(APIModel):
    backend: str
    label: Literal["simple", "multihop", "reasoning"]
    route: Route
    confidence: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    reason: str = ""


class RouteRequest(APIModel):
    question: str = Field(min_length=1, max_length=2000)


class RouteResponse(ClassifierInfo):
    pass


class QueryRequest(APIModel):
    question: str = Field(min_length=1, max_length=2000, description="用户问题")
    route_override: RouteOverride = Field(default="auto", description="auto 表示由分类器决定")
    top_k: int = Field(default=8, ge=1, le=50)
    stream: bool = False


class QueryResponse(APIModel):
    id: str
    question: str
    route: Route
    confidence: float = Field(ge=0, le=1)
    reason: str
    answer: Answer
    citations: list[int]
    evidence: list[Evidence]
    trace: list[TraceStep]
    tools: list[ToolCall] | None = None
    graph: GraphPayload | None = None
    stats: Stats
    refusal: Refusal | None = None
    classifier: ClassifierInfo | None = None
    served_by: ServedBy = "demo"
    generated_at: datetime


class ScenarioSummary(APIModel):
    id: str
    question: str
    route: Route
    confidence: float = Field(ge=0, le=1)
    title: str


class HealthCheck(APIModel):
    ok: bool
    detail: str = ""


class HealthResponse(APIModel):
    status: Literal["ok", "degraded"]
    version: str
    uptime_s: float
    checks: dict[str, HealthCheck]


class FeedbackRequest(APIModel):
    query_id: str
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=2000)


class FeedbackResponse(APIModel):
    ok: bool
    id: str


class ErrorBody(APIModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(APIModel):
    error: ErrorBody


# ---------- SSE ----------

class StageEvent(APIModel):
    """SSE: event=stage，进度阶段（用于前端 TraceRail 动画）。"""

    index: int
    step: TraceStep


class TokenEvent(APIModel):
    """SSE: event=token，答案增量文本。"""

    text: str


class DoneEvent(APIModel):
    """SSE: event=done，完整响应。"""

    response: QueryResponse
