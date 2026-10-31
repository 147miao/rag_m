import { useCallback, useEffect, useRef, useState } from 'react'
import { scenarios as demoScenarios } from './data/scenarios'
import type { Route, Scenario } from './lib/types'
import { ROUTE_META } from './lib/types'
import { fetchHealth, postQuery, streamQuery, type HealthResponse } from './lib/api'
import { AnswerCard } from './components/AnswerCard'
import { Composer } from './components/Composer'
import { RightPanel } from './components/RightPanel'
import { SideRail } from './components/SideRail'
import { TopBar, type ApiMode } from './components/TopBar'
import { TraceRail } from './components/TraceRail'
import { Stat, fmtCost, fmtMs } from './components/ui'

type Phase = 'idle' | 'running' | 'streaming' | 'done'
type Tab = 'evidence' | 'graph' | 'trace' | 'debug'

export default function App() {
  const params = new URLSearchParams(window.location.search)
  const initialDemo = demoScenarios.find((s) => s.id === params.get('demo')) ?? demoScenarios[0]
  const [activeId, setActiveId] = useState(initialDemo.id)
  const [question, setQuestion] = useState(initialDemo.question)
  const [override, setOverride] = useState<Route | 'auto'>('auto')
  const [phase, setPhase] = useState<Phase>('idle')
  const [stepIdx, setStepIdx] = useState(-1)
  const [streamed, setStreamed] = useState('')
  const [tab, setTab] = useState<Tab>('evidence')
  const [focus, setFocus] = useState<number | null>(null)
  const [panelOpen, setPanelOpen] = useState(false)
  const [copied, setCopied] = useState(false)

  // 后端状态：live = 走 FastAPI SSE；demo = 本地演示动画（后端不可用时降级）
  const [apiMode, setApiMode] = useState<ApiMode>('connecting')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [result, setResult] = useState<Scenario | null>(null)

  const localScenario = demoScenarios.find((s) => s.id === activeId) ?? demoScenarios[0]
  const scenario = result ?? localScenario
  const runRef = useRef<(() => void) | null>(null)
  const timers = useRef<number[]>([])
  const interval = useRef<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const clearAll = useCallback(() => {
    timers.current.forEach((t) => window.clearTimeout(t))
    timers.current = []
    if (interval.current !== null) {
      window.clearInterval(interval.current)
      interval.current = null
    }
    abortRef.current?.abort()
    abortRef.current = null
  }, [])

  useEffect(() => clearAll, [clearAll])

  // 启动时探测后端；失败则自动降级为演示模式
  useEffect(() => {
    const ctrl = new AbortController()
    fetchHealth(ctrl.signal)
      .then((h) => {
        setHealth(h)
        setApiMode('live')
      })
      .catch(() => setApiMode('demo'))
    return () => ctrl.abort()
  }, [])

  // 演示深链：?demo=<id>&run=1&tab=graph&fast=1
  useEffect(() => {
    const p = new URLSearchParams(window.location.search)
    const t = p.get('tab')
    if (t === 'evidence' || t === 'graph' || t === 'trace' || t === 'debug') setTab(t)
    if (p.get('live') === '1') {
      // API 驱动的静态终态（截图/分享）：直接 POST /api/query，不做动画
      postQuery({ question: initialDemo.question, route_override: 'auto' })
        .then((resp) => {
          setResult(resp as unknown as Scenario)
          setPhase('done')
          setStepIdx(resp.trace.length)
          setApiMode('live')
        })
        .catch(() => setPhase('idle'))
      return
    }
    if (p.get('fast') === '1') {
      const s = demoScenarios.find((x) => x.id === initialDemo.id) ?? demoScenarios[0]
      setPhase('done')
      setStepIdx(s.trace.length)
      setStreamed(s.answer.lead + '\n\n' + s.answer.body)
      return
    }
    if (p.get('run') === '1') {
      const timer = window.setTimeout(() => runRef.current?.(), 350)
      return () => window.clearTimeout(timer)
    }
  }, [initialDemo.id])

  /** 本地演示动画：后端不可用或 SSE 失败时的降级路径。 */
  const runLocal = useCallback(() => {
    clearAll()
    setStreamed('')
    setStepIdx(0)
    setPhase('running')
    setFocus(null)

    let t = 0
    localScenario.trace.forEach((s, i) => {
      t += Math.max(240, Math.min(560, s.ms * 0.8))
      timers.current.push(window.setTimeout(() => setStepIdx(i + 1), t))
    })
    timers.current.push(
      window.setTimeout(() => {
        if (localScenario.refusal) {
          setPhase('done')
          return
        }
        const full = localScenario.answer.lead + '\n\n' + localScenario.answer.body
        let i = 0
        setPhase('streaming')
        interval.current = window.setInterval(() => {
          i += 5
          setStreamed(full.slice(0, i))
          if (i >= full.length) {
            if (interval.current !== null) window.clearInterval(interval.current)
            interval.current = null
            setPhase('done')
          }
        }, 18)
      }, t + 220),
    )
  }, [clearAll, localScenario])

  const run = useCallback(() => {
    clearAll()
    setResult(null)
    setStreamed('')
    setStepIdx(0)
    setPhase('running')
    setFocus(null)

    if (apiMode !== 'live') {
      runLocal()
      return
    }

    const ctrl = new AbortController()
    abortRef.current = ctrl
    let acc = ''
    streamQuery(
      { question, route_override: override },
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
        onError: () => {
          setApiMode('demo')
          runLocal()
        },
      },
      ctrl.signal,
    ).catch(() => {
      setApiMode('demo')
      runLocal()
    })
  }, [apiMode, clearAll, override, question, runLocal])

  const selectScenario = (id: string) => {
    clearAll()
    const s = demoScenarios.find((x) => x.id === id)
    if (!s) return
    setActiveId(id)
    setQuestion(s.question)
    setOverride('auto')
    setPhase('idle')
    setStepIdx(-1)
    setStreamed('')
    setResult(null)
    setTab('evidence')
    setFocus(null)
  }

  const onCite = (n: number) => {
    setTab('evidence')
    setFocus(n)
    setPanelOpen(true)
  }

  const copyAnswer = async () => {
    const text = scenario.refusal
      ? scenario.refusal.title + '\n' + scenario.refusal.reason
      : scenario.answer.lead + '\n\n' + scenario.answer.body
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1400)
    } catch {
      setCopied(false)
    }
  }

  runRef.current = run

  const routeColor = ROUTE_META[scenario.route].color
  const running = phase === 'running' || phase === 'streaming'

  return (
    <div className="min-h-screen">
      <TopBar apiMode={apiMode} health={health} />
      <div className="mx-auto flex w-full max-w-[1560px]">
        <SideRail scenarios={demoScenarios} activeId={activeId} onSelect={selectScenario} />

        <main className="min-w-0 flex-1 px-4 py-5 sm:px-6">
          <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
            <div>
              <h1 className="font-serif text-[26px] leading-tight tracking-[-0.01em] text-ink">检索控制台</h1>
              <p className="mt-1 text-[12px] text-muted">一个问题，三条链路。看清路由判断、检索证据与生成代价。</p>
            </div>
            <div className="flex items-center gap-2">
              <span className="hidden text-[10.5px] text-faint sm:inline">
                {apiMode === 'live' ? 'FastAPI 已连接 · SSE 流式' : apiMode === 'demo' ? '后端未连接 · 本地演示数据' : '正在连接后端…'}
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
              onChange={setQuestion}
              onRun={run}
              running={running}
              override={override}
              onOverride={(v) => {
                setOverride(v)
                if (v !== 'auto') {
                  const target = demoScenarios.find((s) => s.route === v)
                  if (target && target.id !== activeId) selectScenario(target.id)
                }
              }}
            />

            <TraceRail steps={scenario.trace} active={stepIdx} done={phase === 'done'} color={routeColor} />

            <AnswerCard
              scenario={scenario}
              question={question}
              phase={phase}
              streamed={streamed}
              onCite={onCite}
              onRerun={run}
              onCopy={copyAnswer}
            />

            <section className="card grid grid-cols-2 gap-4 px-5 py-4 sm:grid-cols-4">
              <Stat label="端到端延迟" value={fmtMs(scenario.stats.latencyMs)} />
              <Stat label="Tokens" value={String(scenario.stats.tokensIn)} unit={'in / ' + scenario.stats.tokensOut + ' out'} />
              <Stat label="成本" value={fmtCost(scenario.stats.costCny)} />
              <Stat label="引用校验" value={scenario.stats.citationCheck} />
            </section>

            {copied && <p className="text-center text-[11px] text-success" role="status">已复制到剪贴板</p>}

            <p className="pb-6 text-center text-[10.5px] text-faint">
              RAG-M · FastAPI · LangGraph 1.2 · Milvus 3.0.4 · 设计规范见 web/DESIGN.md
            </p>
          </div>
        </main>

        <RightPanel
          scenario={scenario}
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
