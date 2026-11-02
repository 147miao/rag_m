import { useCallback, useEffect, useRef, useState } from 'react'
import type { Route, Scenario, TraceStep } from './lib/types'
import { ROUTE_META } from './lib/types'
import {
  fetchClassifierMetrics,
  fetchExamples,
  fetchHealth,
  fetchStatus,
  postQuery,
  sendFeedback,
  streamQuery,
  type ClassifierMetrics,
  type Example,
  type HealthResponse,
  type StatusResponse,
} from './lib/api'
import { AnswerCard } from './components/AnswerCard'
import { Composer } from './components/Composer'
import { RightPanel } from './components/RightPanel'
import { SideRail } from './components/SideRail'
import { TopBar, type ApiMode } from './components/TopBar'
import { TraceRail } from './components/TraceRail'
import { Icon, Stat, fmtCost, fmtMs } from './components/ui'

type Phase = 'idle' | 'running' | 'streaming' | 'done'
type Tab = 'evidence' | 'graph' | 'trace' | 'debug'

const PLACEHOLDER_STEPS: TraceStep[] = [
  { key: 'route', label: '路由分类', ms: 0, detail: '等待运行' },
  { key: 'retrieve', label: '混合检索', ms: 0, detail: 'dense + BM25' },
  { key: 'rerank', label: '重排', ms: 0, detail: 'reranker' },
  { key: 'generate', label: '生成', ms: 0, detail: 'LLM' },
  { key: 'verify', label: '引用校验', ms: 0, detail: 'citation' },
]

export default function App() {
  const [question, setQuestion] = useState('')
  const [override, setOverride] = useState<Route | 'auto'>('auto')
  const [phase, setPhase] = useState<Phase>('idle')
  const [stepIdx, setStepIdx] = useState(-1)
  const [streamed, setStreamed] = useState('')
  const [tab, setTab] = useState<Tab>('evidence')
  const [focus, setFocus] = useState<number | null>(null)
  const [panelOpen, setPanelOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [apiMode, setApiMode] = useState<ApiMode>('connecting')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [examples, setExamples] = useState<Example[] | null>(null)
  const [metrics, setMetrics] = useState<ClassifierMetrics | null>(null)
  const [result, setResult] = useState<Scenario | null>(null)
  const [activeExample, setActiveExample] = useState<string | null>(null)

  const timers = useRef<number[]>([])
  const abortRef = useRef<AbortController | null>(null)

  const clearAll = useCallback(() => {
    timers.current.forEach((t) => window.clearTimeout(t))
    timers.current = []
    abortRef.current?.abort()
    abortRef.current = null
  }, [])

  useEffect(() => clearAll, [clearAll])

  // 健康状态 + 训练进度轮询
  useEffect(() => {
    let alive = true
    const ctrl = new AbortController()
    const tick = () => {
      fetchHealth(ctrl.signal)
        .then((h) => {
          if (!alive) return
          setHealth(h)
          setApiMode('live')
        })
        .catch(() => {
          if (alive) setApiMode('demo')
        })
      fetchStatus(ctrl.signal)
        .then((s) => {
          if (alive) setStatus(s)
        })
        .catch(() => {})
    }
    tick()
    const timer = window.setInterval(tick, 15000)
    return () => {
      alive = false
      window.clearInterval(timer)
      ctrl.abort()
    }
  }, [])

  // 示例问题与分类器评测指标（均为后端实时数据）
  useEffect(() => {
    const ctrl = new AbortController()
    fetchExamples(ctrl.signal).then(setExamples).catch(() => setExamples([]))
    fetchClassifierMetrics(ctrl.signal).then(setMetrics).catch(() => setMetrics(null))
    return () => ctrl.abort()
  }, [])

  // 深链：?tab=…&live=1&q=…&route=…（截图/分享）；?run=1 自动运行
  useEffect(() => {
    const p = new URLSearchParams(window.location.search)
    const t = p.get('tab')
    if (t === 'evidence' || t === 'graph' || t === 'trace' || t === 'debug') setTab(t)
    const q = p.get('q') || ''
    const routeParam = p.get('route')
    const routeOverride: Route | 'auto' =
      routeParam === 'simple' || routeParam === 'multihop' || routeParam === 'agentic' ? routeParam : 'auto'
    if (q) setQuestion(q)
    if (p.get('live') === '1' && q) {
      postQuery({ question: q, route_override: routeOverride })
        .then((resp) => {
          setResult(resp as unknown as Scenario)
          setPhase('done')
          setStepIdx(resp.trace.length)
        })
        .catch(() => setPhase('idle'))
    } else if (p.get('run') === '1' && q) {
      const timer = window.setTimeout(() => runRef.current?.(q), 300)
      timers.current.push(timer)
    }
  }, [])

  const run = useCallback(
    (qOverride?: string) => {
      const q = (qOverride ?? question).trim()
      if (!q) {
        setError('请输入问题，或从左侧选择一个示例。')
        return
      }
      if (apiMode !== 'live') {
        setError('后端未连接：请先启动 FastAPI（make api），再进行查询。')
        return
      }
      clearAll()
      setError(null)
      setResult(null)
      setStreamed('')
      setStepIdx(0)
      setPhase('running')
      setFocus(null)
      const ctrl = new AbortController()
      abortRef.current = ctrl
      let acc = ''
      streamQuery(
        { question: q, route_override: override },
        {
          onStage: (index) => setStepIdx(index + 1),
          onToken: (text) => {
            acc += text
            setStreamed(acc)
            setPhase('streaming')
          },
          onDone: (resp) => {
            setResult(resp as unknown as Scenario)
            setStreamed('')
            setStepIdx(resp.trace.length)
            setPhase('done')
          },
          onError: (message) => {
            setError('查询失败：' + message)
            setPhase('idle')
          },
        },
        ctrl.signal,
      ).catch(() => {
        setError('查询失败：无法连接后端。')
        setPhase('idle')
      })
    },
    [apiMode, clearAll, override, question],
  )

  const runRef = useRef<((q?: string) => void) | null>(null)
  runRef.current = run

  const selectExample = (e: Example) => {
    setActiveExample(e.id)
    setQuestion(e.question)
    setOverride('auto')
    setTab('evidence')
    run(e.question)
  }

  const onCite = (n: number) => {
    setTab('evidence')
    setFocus(n)
    setPanelOpen(true)
  }

  const copyAnswer = async () => {
    if (!result) return
    const text = result.refusal
      ? result.refusal.title + '\n' + result.refusal.reason
      : result.answer.lead + '\n\n' + result.answer.body
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1400)
    } catch {
      setCopied(false)
    }
  }

  const sendRating = (rating: 'up' | 'down') => {
    if (!result) return
    sendFeedback({ query_id: result.id, rating }).catch(() => {})
  }

  const running = phase === 'running' || phase === 'streaming'
  const routeColor = result ? ROUTE_META[result.route].color : 'var(--color-teal)'

  return (
    <div className="min-h-screen">
      <TopBar apiMode={apiMode} health={health} status={status} />
      <div className="mx-auto flex w-full max-w-[1560px]">
        <SideRail examples={examples} activeId={activeExample} onSelect={selectExample} health={health} metrics={metrics} />

        <main className="min-w-0 flex-1 px-4 py-5 sm:px-6">
          <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
            <div>
              <h1 className="font-serif text-[26px] leading-tight tracking-[-0.01em] text-ink">检索控制台</h1>
              <p className="mt-1 text-[12px] text-muted">一个问题，三条链路。看清路由判断、检索证据与生成代价。</p>
            </div>
            <div className="flex items-center gap-2">
              <span className="hidden text-[10.5px] text-faint sm:inline">
                {apiMode === 'live' ? 'FastAPI 已连接 · SSE 流式' : apiMode === 'demo' ? '后端未连接 · 请启动 make api' : '正在连接后端…'}
              </span>
              <button
                type="button"
                onClick={() => setPanelOpen(true)}
                className="flex h-8 items-center gap-1.5 rounded-md border px-2.5 text-[11.5px] text-ink-2 xl:hidden"
                style={{ borderColor: 'var(--color-line)' }}
              >
                证据
              </button>
            </div>
          </div>

          <div className="flex flex-col gap-4">
            <Composer
              value={question}
              onChange={(v) => {
                setQuestion(v)
                setError(null)
              }}
              onRun={() => run()}
              running={running}
              override={override}
              onOverride={setOverride}
            />

            {error && (
              <div role="alert" className="flex items-start gap-2 rounded-md border-l-2 px-4 py-3 text-[12px]" style={{ borderColor: 'var(--color-danger)', background: '#FBF1EF' }}>
                <Icon name="alert" size={14} className="mt-[2px] shrink-0" />
                <span className="flex-1 text-ink-2">{error}</span>
                <button type="button" onClick={() => setError(null)} className="text-[11px] text-muted hover:text-ink">关闭</button>
              </div>
            )}

            {result ? (
              <>
                <TraceRail steps={result.trace} active={stepIdx} done={phase === 'done'} color={routeColor} />
                <AnswerCard
                  scenario={result}
                  question={question}
                  phase={phase}
                  streamed={streamed}
                  onCite={onCite}
                  onRerun={() => run()}
                  onCopy={copyAnswer}
                  onFeedback={sendRating}
                />
                <section className="card grid grid-cols-2 gap-4 px-5 py-4 sm:grid-cols-4">
                  <Stat label="端到端延迟" value={fmtMs(result.stats.latencyMs)} />
                  <Stat label="Tokens" value={String(result.stats.tokensIn)} unit={'in / ' + result.stats.tokensOut + ' out'} />
                  <Stat label="成本" value={fmtCost(result.stats.costCny)} />
                  <Stat label="引用校验" value={result.stats.citationCheck} />
                </section>
              </>
            ) : (
              <>
                <TraceRail steps={PLACEHOLDER_STEPS} active={running ? stepIdx : -1} done={false} color={routeColor} />
                {running ? (
                  <article className="card px-5 py-6" aria-busy="true">
                    <div className="flex items-center gap-2 text-[12px] text-muted">
                      <span className="h-2 w-2 animate-pulse rounded-full" style={{ background: 'var(--color-teal)' }} />
                      正在{phase === 'streaming' ? '生成答案' : '检索与推理'}…
                      {streamed && <span className="ml-2 flex-1 truncate text-[12px] text-ink-2">{streamed}</span>}
                    </div>
                    <div className="mt-4 space-y-2" aria-hidden="true">
                      <div className="h-3 w-2/3 animate-pulse rounded bg-surface-2" />
                      <div className="h-3 w-full animate-pulse rounded bg-surface-2" />
                      <div className="h-3 w-5/6 animate-pulse rounded bg-surface-2" />
                    </div>
                  </article>
                ) : (
                  <article className="card px-6 py-10 text-center">
                    <Icon name="spark" size={22} className="mx-auto text-faint" />
                    <p className="mt-3 text-[14px] text-ink">从左侧示例开始，或直接输入一个真实问题</p>
                    <p className="mx-auto mt-2 max-w-[520px] text-[12px] leading-relaxed text-muted">
                      答案会基于当前索引语料生成，并附上可点击的证据引用；跨语料的问题如果超出索引范围，系统会如实拒答，而不是编造。
                    </p>
                    <div className="mt-4 flex flex-wrap items-center justify-center gap-2 text-[10.5px] text-faint">
                      {(Object.keys(ROUTE_META) as Route[]).map((r) => (
                        <span key={r} className="flex items-center gap-1.5 rounded-full border px-2.5 py-1" style={{ borderColor: 'var(--color-line)' }}>
                          <span className="h-[5px] w-[5px] rounded-full" style={{ background: ROUTE_META[r].color }} />
                          {ROUTE_META[r].label}
                        </span>
                      ))}
                    </div>
                  </article>
                )}
              </>
            )}

            {copied && <p className="text-center text-[11px] text-success" role="status">已复制到剪贴板</p>}

            <p className="pb-6 text-center text-[10.5px] text-faint">
              RAG-M · FastAPI · LangGraph 1.2 · Milvus 3.0.4 · 设计规范见 web/DESIGN.md
            </p>
          </div>
        </main>

        <RightPanel
          scenario={result}
          tab={tab}
          onTab={setTab}
          focus={focus}
          onFocus={(n) => {
            setFocus(n)
            setTab('evidence')
          }}
          visible={panelOpen}
          onClose={() => setPanelOpen(false)}
        />
      </div>
    </div>
  )
}