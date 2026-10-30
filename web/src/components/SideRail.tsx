import type { Scenario } from '../lib/types'
import { ROUTE_META } from '../lib/types'
import { Icon } from './ui'

export function SideRail({
  scenarios,
  activeId,
  onSelect,
}: {
  scenarios: Scenario[]
  activeId: string
  onSelect: (id: string) => void
}) {
  return (
    <aside className="hidden w-[232px] shrink-0 flex-col gap-6 border-r px-4 py-5 lg:flex" style={{ borderColor: 'var(--color-line)' }}>
      <section>
        <p className="eyebrow mb-2.5">演示案例</p>
        <ul className="flex flex-col gap-1">
          {scenarios.map((s, i) => {
            const meta = ROUTE_META[s.route]
            const active = s.id === activeId
            return (
              <li key={s.id}>
                <button
                  type="button"
                  onClick={() => onSelect(s.id)}
                  className={
                    'group flex w-full items-start gap-2.5 rounded-md px-2.5 py-2 text-left transition-colors ' +
                    (active ? 'bg-surface' : 'hover:bg-surface-2/60')
                  }
                  style={active ? { boxShadow: 'inset 0 0 0 1px var(--color-line)' } : undefined}
                >
                  <span className="num mt-[3px] text-[10px] text-faint">{String(i + 1).padStart(2, '0')}</span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[12.5px] text-ink">{s.question}</span>
                    <span className="mt-1 flex items-center gap-1.5">
                      <span className="h-[5px] w-[5px] rounded-full" style={{ background: meta.color }} />
                      <span className="text-[10.5px] text-muted">{meta.label}</span>
                    </span>
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      </section>

      <section>
        <p className="eyebrow mb-2.5">系统</p>
        <ul className="flex flex-col gap-2 text-[11.5px] text-muted">
          <StatusRow icon="database" label="Milvus" value="19530" ok />
          <StatusRow icon="graph" label="Neo4j" value="7687" ok />
          <StatusRow icon="search" label="SearXNG" value="8080" ok />
          <StatusRow icon="cpu" label="Embedding" value="bge-m3" ok />
        </ul>
      </section>

      <section className="mt-auto">
        <p className="eyebrow mb-2">评测</p>
        <div className="rounded-md border p-3" style={{ borderColor: 'var(--color-line)' }}>
          <div className="flex items-baseline justify-between">
            <span className="text-[11.5px] text-muted">路由准确率</span>
            <span className="num text-[13px] text-ink">0.91</span>
          </div>
          <div className="mt-2 h-1 overflow-hidden rounded-full" style={{ background: 'var(--color-surface-2)' }}>
            <div className="h-full rounded-full" style={{ width: '91%', background: 'var(--color-teal)' }} />
          </div>
          <p className="mt-2 text-[10.5px] leading-relaxed text-faint">300 条标注 · 95% CI ±0.04</p>
        </div>
      </section>
    </aside>
  )
}

function StatusRow({ icon, label, value, ok }: { icon: 'database' | 'graph' | 'search' | 'cpu'; label: string; value: string; ok: boolean }) {
  return (
    <li className="flex items-center gap-2">
      <Icon name={icon} size={13} className="text-faint" />
      <span>{label}</span>
      <span className="num ml-auto text-[10.5px] text-faint">{value}</span>
      <span className="h-[5px] w-[5px] rounded-full" style={{ background: ok ? 'var(--color-success)' : 'var(--color-danger)' }} />
    </li>
  )
}
