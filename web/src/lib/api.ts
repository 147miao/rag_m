import type { Route, Scenario } from './types'

export interface ScenarioSummary {
  id: string
  question: string
  route: Route
  confidence: number
  title: string
}

export interface HealthCheck {
  ok: boolean
  detail: string
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  version: string
  uptime_s: number
  checks: Record<string, HealthCheck>
}

export type RouteOverride = Route | 'auto'

export interface QueryPayload {
  question: string
  route_override: RouteOverride
  top_k?: number
}

/** 后端 QueryResponse 与前端 Scenario 结构一致，仅多出 served_by / generated_at。 */
export type QueryResponse = Scenario & { served_by?: string; generated_at?: string }

const JSON_HEADERS = { 'Content-Type': 'application/json' }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(path, init)
  if (!resp.ok) {
    let message = 'HTTP ' + resp.status
    try {
      const body = await resp.json()
      message = body?.error?.message ?? message
    } catch {
      /* ignore */
    }
    throw new Error(message)
  }
  return (await resp.json()) as T
}

export function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>('/api/health', { signal })
}

export function fetchScenarios(signal?: AbortSignal): Promise<ScenarioSummary[]> {
  return request<ScenarioSummary[]>('/api/scenarios', { signal })
}

export function postQuery(payload: QueryPayload, signal?: AbortSignal): Promise<QueryResponse> {
  return request<QueryResponse>('/api/query', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
    signal,
  })
}

export interface StreamHandlers {
  onStage?: (index: number, step: { key: string; label: string; ms: number; detail: string }) => void
  onToken?: (text: string) => void
  onDone?: (response: QueryResponse) => void
  onError?: (message: string) => void
}

/** SSE：POST /api/query/stream，事件为 stage / token / done / error。 */
export async function streamQuery(payload: QueryPayload, handlers: StreamHandlers, signal?: AbortSignal): Promise<void> {
  const resp = await fetch('/api/query/stream', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
    signal,
  })
  if (!resp.ok || !resp.body) {
    handlers.onError?.('HTTP ' + resp.status)
    return
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const dispatch = (block: string) => {
    let event = 'message'
    const dataLines: string[] = []
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim()
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
    }
    if (dataLines.length === 0) return
    const raw = dataLines.join('\n')
    try {
      const data = JSON.parse(raw)
      if (event === 'stage') handlers.onStage?.(data.index, data.step)
      else if (event === 'token') handlers.onToken?.(data.text)
      else if (event === 'done') handlers.onDone?.(data.response)
      else if (event === 'error') handlers.onError?.(data?.error?.message ?? 'stream error')
    } catch {
      /* 忽略无法解析的帧 */
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx = buffer.indexOf('\n\n')
    while (idx >= 0) {
      const block = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      if (block.trim()) dispatch(block)
      idx = buffer.indexOf('\n\n')
    }
  }
  if (buffer.trim()) dispatch(buffer)
}

export function sendFeedback(payload: { query_id: string; rating: 'up' | 'down'; comment?: string }): Promise<{ ok: boolean; id: string }> {
  return request('/api/feedback', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  })
}
