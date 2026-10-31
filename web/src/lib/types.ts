export type Route = 'simple' | 'multihop' | 'agentic'

export type ToolStatus = 'ok' | 'error'

export interface Evidence {
  id: string
  rank: number
  title: string
  source: string
  dataset: string
  score: number
  dense: number
  sparse: number
  snippet: string
  highlight?: string
}

export interface TraceStep {
  key: string
  label: string
  ms: number
  detail: string
}

export interface ToolCall {
  name: string
  args: string
  result: string
  ms: number
  status: ToolStatus
}

export interface GraphNode {
  id: string
  label: string
  kind: 'query' | 'entity' | 'doc' | 'answer'
}

export interface GraphEdge {
  from: string
  to: string
  label: string
}

export interface ClassifierInfo {
  backend: string
  label: 'simple' | 'multihop' | 'reasoning'
  route: Route
  confidence: number
  latency_ms: number
  reason?: string
}

export interface Stats {
  latencyMs: number
  tokensIn: number
  tokensOut: number
  costCny: number
  citationCheck: string
  routeLabel?: string
}

export interface Scenario {
  id: string
  question: string
  route: Route
  confidence: number
  reason: string
  answer: { lead: string; body: string }
  citations: number[]
  evidence: Evidence[]
  trace: TraceStep[]
  tools?: ToolCall[]
  graph?: { nodes: GraphNode[]; edges: GraphEdge[] }
  stats: Stats
  classifier?: ClassifierInfo
  refusal?: { title: string; reason: string; suggestion: string }
}

export const ROUTE_META: Record<Route, { label: string; en: string; color: string; soft: string; desc: string }> = {
  simple: { label: '简单事实', en: 'SIMPLE', color: '#0F5C55', soft: '#E3EFEC', desc: '单跳检索即可回答' },
  multihop: { label: '多跳关系', en: 'MULTI-HOP', color: '#4338CA', soft: '#EAE8FB', desc: '需要连接多个实体' },
  agentic: { label: '复杂推理', en: 'AGENTIC', color: '#B45309', soft: '#F7EBDD', desc: '需要多工具推理' },
}
