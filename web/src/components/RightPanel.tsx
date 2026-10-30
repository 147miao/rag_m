import type { Evidence, GraphEdge, GraphNode, Scenario, ToolCall } from '../lib/types'
import { ROUTE_META } from '../lib/types'
import { Icon } from './ui'

type Tab = 'evidence' | 'graph' | 'trace' | 'debug'

export function RightPanel({
  scenario,
  tab,
  onTab,
  focus,
  onFocus,
  visible,
  onClose,
}: {
  scenario: Scenario
  tab: Tab
  onTab: (t: Tab) => void
  focus: number | null
  onFocus: (n: number) => void
  visible: boolean
  onClose: () => void
}) {
  const meta = ROUTE_META[scenario.route]
  const tabs: { key: Tab; label: string; count?: number }[] = [
    { key: 'evidence', label: '证据', count: scenario.evidence.length },
    { key: 'graph', label: '图谱' },
    { key: 'trace', label: '轨迹' },
    { key: 'debug', label: '调试' },
  ]

  return (
    <aside
      className={
        'z-40 flex w-[400px] shrink-0 flex-col border-l bg-paper ' +
        (visible ? 'fixed inset-y-0 right-0 shadow-2xl xl:static xl:shadow-none' : 'hidden xl:flex')
      }
      style={{ borderColor: 'var(--color-line)' }}
      aria-label="证据面板"
    >
      <div className="flex items-center gap-1 border-b px-3 py-2" style={{ borderColor: 'var(--color-line)' }}>
        {tabs.map((t) => {
          const active = tab === t.key
          return (
            <button
              key={t.key}
              type="button"
              onClick={() => onTab(t.key)}
              className={
                'flex h-7 items-center gap-1.5 rounded px-2.5 text-[11.5px] transition-colors ' +
                (active ? 'bg-surface text-ink' : 'text-muted hover:text-ink')
              }
              style={active ? { boxShadow: 'inset 0 0 0 1px var(--color-line)' } : undefined}
              aria-pressed={active}
            >
              {t.label}
              {t.count !== undefined && <span className="num text-[10px] text-faint">{t.count}</span>}
            </button>
          )
        })}
        <button
          type="button"
          onClick={onClose}
          className="ml-auto flex h-7 w-7 items-center justify-center rounded text-muted hover:bg-surface-2 xl:hidden"
          aria-label="关闭面板"
        >
          ✕
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
        {tab === 'evidence' && (
          <ul className="flex flex-col gap-2">
            {scenario.evidence.map((e) => (
              <li key={e.id}>
                <EvidenceCard evidence={e} color={meta.color} active={focus === e.rank} onClick={() => onFocus(e.rank)} />
              </li>
            ))}
          </ul>
        )}

        {tab === 'graph' && (
          scenario.graph ? (
            <div className="card p-3">
              <p className="eyebrow mb-2">实体路径 · {scenario.graph.edges.length} 条边</p>
              <GraphPath nodes={scenario.graph.nodes} edges={scenario.graph.edges} color={meta.color} />
              <ul className="mt-2 flex flex-col gap-1 border-t pt-2" style={{ borderColor: 'var(--color-line)' }}>
                {scenario.graph.edges.map((e, i) => (
                  <li key={i} className="flex items-center gap-1.5 text-[11px] text-muted">
                    <span className="num text-faint">{i + 1}</span>
                    <span className="text-ink-2">{scenario.graph?.nodes.find((n) => n.id === e.from)?.label ?? e.from}</span>
                    <span className="num text-faint">—{e.label}→</span>
                    <span className="text-ink-2">{scenario.graph?.nodes.find((n) => n.id === e.to)?.label ?? e.to}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <EmptyState title="这条链路没有图谱" desc="只有多跳关系型会走实体链接与子图查询。切换到「多跳关系」案例查看。" />
          )
        )}

        {tab === 'trace' && (
          <div className="flex flex-col gap-2">
            {scenario.tools ? (
              <ToolTimeline tools={scenario.tools} color={meta.color} />
            ) : (
              <div className="card p-3">
                <p className="eyebrow mb-2">执行阶段</p>
                <ol className="flex flex-col gap-2">
                  {scenario.trace.map((s, i) => (
                    <li key={s.key} className="flex items-start gap-2">
                      <span className="num mt-0.5 text-[10px] text-faint">{String(i + 1).padStart(2, '0')}</span>
                      <span className="min-w-0 flex-1">
                        <span className="flex items-baseline justify-between gap-2">
                          <span className="text-[12px] text-ink">{s.label}</span>
                          <span className="num text-[10.5px] text-muted">{s.ms} ms</span>
                        </span>
                        <span className="block truncate text-[10.5px] text-faint">{s.detail}</span>
                      </span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>
        )}

        {tab === 'debug' && (
          <pre className="card overflow-x-auto p-3 text-[10.5px] leading-relaxed text-ink-2">
            {JSON.stringify(
              {
                route: scenario.route,
                confidence: scenario.confidence,
                stats: scenario.stats,
                citations: scenario.citations,
                evidence: scenario.evidence.map((e) => ({ id: e.id, score: e.score, dense: e.dense, sparse: e.sparse })),
                trace: scenario.trace.map((t) => ({ stage: t.key, ms: t.ms })),
              },
              null,
              2,
            )}
          </pre>
        )}
      </div>

      <footer className="border-t px-3 py-2 text-[10.5px] text-faint" style={{ borderColor: 'var(--color-line)' }}>
        演示数据 · 接入后端后由 <span className="num">/api/query</span> 返回同结构 JSON
      </footer>
    </aside>
  )
}

function EvidenceCard({ evidence, color, active, onClick }: { evidence: Evidence; color: string; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="w-full rounded-md border p-3 text-left transition-colors hover:bg-surface"
      style={{
        borderColor: active ? color + '66' : 'var(--color-line)',
        background: active ? 'var(--color-surface)' : 'transparent',
      }}
    >
      <div className="flex items-start gap-2">
        <span
          className="num mt-[1px] flex h-[18px] w-[18px] shrink-0 items-center justify-center rounded-[3px] text-[10px]"
          style={{ background: color + '14', color }}
        >
          {evidence.rank}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[12px] font-medium text-ink">{evidence.title}</span>
          <span className="mt-0.5 flex items-center gap-1.5 text-[10px] text-faint">
            <span className="truncate">{evidence.source}</span>
            <span className="shrink-0 rounded-sm bg-surface-2 px-1 py-[1px]">{evidence.dataset}</span>
          </span>
        </span>
        <span className="num text-[10.5px]" style={{ color }}>{evidence.score.toFixed(2)}</span>
      </div>

      <p className="mt-2 line-clamp-3 text-[11.5px] leading-relaxed text-ink-2">
        {evidence.highlight ? highlight(evidence.snippet, evidence.highlight) : evidence.snippet}
      </p>

      <div className="mt-2 flex items-center gap-2">
        <div className="score-bar h-1 flex-1 overflow-hidden rounded-full" style={{ background: 'var(--color-surface-2)' }}>
          <i style={{ width: evidence.score * 100 + '%', background: color }} />
        </div>
        <span className="num text-[9.5px] text-faint">d {evidence.dense.toFixed(2)} · s {evidence.sparse.toFixed(2)}</span>
      </div>
    </button>
  )
}

function highlight(text: string, phrase: string) {
  const idx = text.indexOf(phrase)
  if (idx < 0) return text
  return (
    <>
      {text.slice(0, idx)}
      <mark style={{ background: '#F4E7C8', color: 'inherit', padding: '0 1px' }}>{phrase}</mark>
      {text.slice(idx + phrase.length)}
    </>
  )
}

function GraphPath({ nodes, edges, color }: { nodes: GraphNode[]; edges: GraphEdge[]; color: string }) {
  const rowH = 44
  const top = 22
  const height = top * 2 + Math.max(1, nodes.length - 1) * rowH + 10
  const label = (id: string) => nodes.find((n) => n.id === id)?.label ?? id
  return (
    <svg viewBox={'0 0 360 ' + height} className="w-full" role="img" aria-label="实体路径图">
      {nodes.map((n, i) => {
        const y = top + i * rowH
        const isEntity = n.kind === 'entity'
        return (
          <g key={n.id}>
            {i < nodes.length - 1 && (
              <line x1={26} y1={y} x2={26} y2={y + rowH} stroke={color + '55'} strokeWidth={1.2} strokeDasharray={isEntity ? undefined : '2 3'} />
            )}
            <circle cx={26} cy={y} r={isEntity ? 4.5 : 3.5} fill={isEntity ? color : '#17181B'} opacity={isEntity ? 1 : 0.5} />
            <text x={42} y={y + 3.5} fontSize={n.label.length > 22 ? 8 : 9} fill={isEntity ? '#17181B' : '#6B6F76'} fontFamily="monospace">
              {n.label}
            </text>
          </g>
        )
      })}
      {edges.map((e, i) => {
        const from = nodes.findIndex((n) => n.id === e.from)
        if (from < 0 || from >= nodes.length - 1) return null
        const y = top + from * rowH
        return (
          <text key={i} x={33} y={y + rowH / 2 + 2.5} fontSize={7.5} fill="#8B8F97" fontFamily="monospace">
            {e.label}
          </text>
        )
      })}
      <text x={42} y={height - 4} fontSize={7.5} fill="#9BA0A8" fontFamily="monospace">
        {label(nodes[0]?.id ?? '')} → {label(nodes[nodes.length - 1]?.id ?? '')}
      </text>
    </svg>
  )
}

function ToolTimeline({ tools, color }: { tools: ToolCall[]; color: string }) {
  return (
    <div className="card p-3">
      <p className="eyebrow mb-3">工具调用 · {tools.length} 次</p>
      <ol className="relative ml-1 border-l pl-4" style={{ borderColor: 'var(--color-line-strong)' }}>
        {tools.map((t, i) => (
          <li key={i} className="relative pb-3 last:pb-0">
            <span
              className="absolute -left-[21px] top-1 h-2 w-2 rounded-full ring-2"
              style={{ background: t.status === 'ok' ? color : 'var(--color-danger)', boxShadow: '0 0 0 2px var(--color-paper)' }}
            />
            <div className="flex items-baseline justify-between gap-2">
              <span className="num text-[11.5px] text-ink">{t.name}</span>
              <span className="num text-[10px] text-faint">{t.ms} ms</span>
            </div>
            <p className="num mt-0.5 truncate text-[10px] text-muted">{t.args}</p>
            <p className="mt-0.5 text-[10.5px] text-ink-2">{t.result}</p>
          </li>
        ))}
      </ol>
    </div>
  )
}

function EmptyState({ title, desc }: { title: string; desc: string }) {
  return (
    <div className="card flex flex-col items-center gap-2 px-6 py-10 text-center">
      <Icon name="graph" size={22} className="text-faint" />
      <p className="text-[12.5px] text-ink">{title}</p>
      <p className="max-w-[240px] text-[11px] leading-relaxed text-faint">{desc}</p>
    </div>
  )
}
