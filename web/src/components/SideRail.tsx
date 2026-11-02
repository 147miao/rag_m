import type { ClassifierMetrics, Example, HealthResponse } from '../lib/api'
import { ROUTE_META } from '../lib/types'
import { Icon } from './ui'

const ICONS: Record<'milvus' | 'ollama' | 'classifier' | 'kg' | 'langsmith', 'database' | 'cpu' | 'graph' | 'search' | 'spark'> = {
  milvus: 'database',
  ollama: 'cpu',
  classifier: 'spark',
  kg: 'graph',
  langsmith: 'search',
}

const LABELS: Record<string, string> = {
  milvus: 'Milvus',
  ollama: 'Ollama',
  classifier: '分类器',
  kg: '知识图谱',
  langsmith: 'LangSmith',
}

export function SideRail({
  examples,
  activeId,
  onSelect,
  health,
  metrics,
}: {
  examples: Example[] | null
  activeId: string | null
  onSelect: (e: Example) => void
  health: HealthResponse | null
  metrics: ClassifierMetrics | null
}) {
  const checks = health?.checks ?? {}
  const rows = Object.keys(LABELS).filter((k) => checks[k])
  const accuracy = metrics?.accuracy ?? null
  return (
    <aside className="hidden w-[232px] shrink-0 flex-col gap-6 border-r px-4 py-5 lg:flex" style={{ borderColor: 'var(--color-line)' }}>
      <section>
        <p className="eyebrow mb-2.5">示例问题</p>
        {examples === null ? (
          <ul className="flex flex-col gap-2" aria-hidden="true">
            {[0, 1, 2, 3].map((i) => (
              <li key={i} className="h-10 animate-pulse rounded-md bg-surface-2" />
            ))}
          </ul>
        ) : (
          <ul className="flex flex-col gap-1">
            {examples.map((e, i) => {
              const meta = ROUTE_META[e.route]
              const active = e.id === activeId
              return (
                <li key={e.id}>
                  <button
                    type="button"
                    onClick={() => onSelect(e)}
                    className={
                      'group flex w-full items-start gap-2.5 rounded-md px-2.5 py-2 text-left transition-colors ' +
                      (active ? 'bg-surface' : 'hover:bg-surface-2/60')
                    }
                    style={active ? { boxShadow: 'inset 0 0 0 1px var(--color-line)' } : undefined}
                    title={e.question}
                  >
                    <span className="num mt-[3px] text-[10px] text-faint">{String(i + 1).padStart(2, '0')}</span>
                    <span className="min-w-0 flex-1">
                      <span className="line-clamp-2 text-[12px] leading-snug text-ink">{e.question}</span>
                      <span className="mt-1 flex items-center gap-1.5">
                        <span className="h-[5px] w-[5px] rounded-full" style={{ background: meta.color }} />
                        <span className="text-[10.5px] text-muted">{e.hint}</span>
                      </span>
                    </span>
                  </button>
                </li>
              )
            })}
          </ul>
        )}
      </section>

      <section>
        <p className="eyebrow mb-2.5">系统</p>
        {health === null ? (
          <ul className="flex flex-col gap-2" aria-hidden="true">
            {[0, 1, 2].map((i) => (
              <li key={i} className="h-4 animate-pulse rounded bg-surface-2" />
            ))}
          </ul>
        ) : (
          <ul className="flex flex-col gap-2 text-[11.5px] text-muted">
            {rows.map((k) => (
              <StatusRow key={k} icon={ICONS[k as keyof typeof ICONS] ?? 'cpu'} label={LABELS[k]} value={trimDetail(checks[k].detail)} ok={checks[k].ok} />
            ))}
          </ul>
        )}
      </section>

      <section className="mt-auto">
        <p className="eyebrow mb-2">分类器评测</p>
        <div className="rounded-md border p-3" style={{ borderColor: 'var(--color-line)' }}>
          {metrics && accuracy !== null ? (
            <>
              <div className="flex items-baseline justify-between">
                <span className="text-[11.5px] text-muted">准确率</span>
                <span className="num text-[13px] text-ink">{accuracy.toFixed(3)}</span>
              </div>
              <div className="mt-2 h-1 overflow-hidden rounded-full" style={{ background: 'var(--color-surface-2)' }}>
                <div className="h-full rounded-full" style={{ width: Math.round(accuracy * 100) + '%', background: 'var(--color-teal)' }} />
              </div>
              <div className="mt-2 flex items-center justify-between text-[10.5px] text-faint">
                <span>macro-F1 {metrics.macro_f1.toFixed(3)}</span>
                <span>{metrics.n} 条</span>
              </div>
              <p className="mt-1 text-[10px] leading-relaxed text-faint">{metrics.test_set} · p50 {Math.round(metrics.p50_ms)}ms</p>
            </>
          ) : (
            <p className="text-[11px] text-faint">评测数据加载中…</p>
          )}
        </div>
      </section>
    </aside>
  )
}

function trimDetail(detail: string): string {
  const d = (detail || '').trim()
  if (!d) return ''
  return d.length > 18 ? d.slice(0, 17) + '…' : d
}

function StatusRow({ icon, label, value, ok }: { icon: 'database' | 'graph' | 'search' | 'cpu' | 'spark'; label: string; value: string; ok: boolean }) {
  return (
    <li className="flex items-center gap-2">
      <Icon name={icon} size={13} className="text-faint" />
      <span>{label}</span>
      <span className="num ml-auto max-w-[70px] truncate text-[10.5px] text-faint" title={value}>{value}</span>
      <span className="h-[5px] w-[5px] shrink-0 rounded-full" style={{ background: ok ? 'var(--color-success)' : 'var(--color-danger)' }} />
    </li>
  )
}