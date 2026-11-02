import type { TraceStep } from '../lib/types'
import { fmtMs } from './ui'

export function TraceRail({ steps, active, done, color }: { steps: TraceStep[]; active: number; done: boolean; color: string }) {
  return (
    <div className="card px-4 py-3">
      <div className="flex flex-col gap-2.5 sm:flex-row sm:items-start sm:gap-0">
        {steps.map((s, i) => {
          const state = done ? 'done' : i < active ? 'done' : i === active ? 'running' : 'idle'
          return (
            <div key={s.key} className="flex min-w-0 items-start sm:flex-1">
              <div className="flex min-w-0 flex-1 flex-col gap-1">
                <div className="flex items-center gap-2">
                  <span
                    className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-[10px]"
                    style={{
                      borderColor: state === 'idle' ? 'var(--color-line-strong)' : color,
                      background: state === 'idle' ? 'transparent' : state === 'running' ? color + '14' : color,
                      color: state === 'done' ? '#fff' : state === 'running' ? color : 'var(--color-faint)',
                    }}
                  >
                    {state === 'done' ? '✓' : i + 1}
                  </span>
                  <span className={'whitespace-nowrap text-[12px] ' + (state === 'idle' ? 'text-faint' : 'text-ink')}>{s.label}</span>
                  <span className="num ml-auto whitespace-nowrap pr-1 text-[10.5px] text-muted">{state === 'idle' ? '—' : fmtMs(s.ms)}</span>
                </div>
                <p className="truncate pl-7 text-[10.5px] text-faint">{s.detail}</p>
              </div>
              {i < steps.length - 1 && (
                <div
                  className="mt-[9px] hidden h-px w-4 shrink-0 sm:block"
                  style={{ background: i < active || done ? color + '66' : 'var(--color-line)' }}
                />
              )}
            </div>
          )
        })}
      </div>
      {!done && active >= 0 && (
        <div className="mt-2 flex items-center gap-2 text-[10.5px] text-muted">
          <span className="pulse-dot h-1.5 w-1.5 rounded-full" style={{ background: color }} />
          正在执行 · {steps[Math.min(active, steps.length - 1)]?.label}
        </div>
      )}
    </div>
  )
}