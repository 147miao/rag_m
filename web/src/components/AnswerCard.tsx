import { useState } from 'react'
import type { ReactNode } from 'react'
import type { Scenario } from '../lib/types'
import { ROUTE_META } from '../lib/types'
import { Icon, RouteBadge, fmtCost, fmtMs } from './ui'

function renderWithCitations(text: string, onCite: (n: number) => void) {
  const parts = text.split(/(\[\d+\])/g)
  const out: ReactNode[] = []
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i]
    const m = p.match(/^\[(\d+)\]$/)
    if (!m) {
      out.push(<span key={i}>{p}</span>)
      continue
    }
    const n = Number(m[1])
    const badge = (
      <button
        type="button"
        onClick={() => onCite(n)}
        className="num mx-0.5 inline-flex h-[16px] min-w-[16px] items-center justify-center rounded-[3px] border px-1 align-[2px] text-[9.5px] leading-none transition-colors hover:bg-teal-soft"
        style={{ borderColor: 'var(--color-line-strong)', color: 'var(--color-teal)' }}
        title={'查看证据 ' + n}
      >
        {n}
      </button>
    )
    // 引用后紧跟标点时绑定在一起，避免移动端换行成孤立的「。」行
    const next = parts[i + 1]
    if (next && /^[。，、；：！？]/.test(next)) {
      out.push(
        <span key={i} className="whitespace-nowrap">
          {badge}
          {next[0]}
        </span>,
      )
      parts[i + 1] = next.slice(1)
    } else {
      out.push(<span key={i}>{badge}</span>)
    }
  }
  return out
}

export function AnswerCard({
  scenario,
  question,
  phase,
  streamed,
  onCite,
  onRerun,
  onCopy,
  onFeedback,
}: {
  scenario: Scenario
  question: string
  phase: 'idle' | 'running' | 'streaming' | 'done'
  streamed: string
  onCite: (n: number) => void
  onRerun: () => void
  onCopy: () => void
  onFeedback?: (rating: 'up' | 'down') => void
}) {
  const meta = ROUTE_META[scenario.route]
  const showBody = phase === 'streaming' || phase === 'done'
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null)

  return (
    <article className="card overflow-hidden">
      <header className="flex flex-wrap items-center gap-2.5 border-b px-5 py-3" style={{ borderColor: 'var(--color-line)' }}>
        <RouteBadge route={scenario.route} confidence={scenario.confidence} />
        <span className="text-[11px] text-muted">{meta.desc}</span>
        {scenario.classifier && (
          <span className="num rounded-full border px-2 py-[2px] text-[9.5px] text-muted" style={{ borderColor: 'var(--color-line)' }} title={scenario.classifier.backend}>
            分类器 {scenario.classifier.backend.startsWith('lora') ? 'LoRA' : scenario.classifier.backend} · {scenario.classifier.confidence.toFixed(2)}
          </span>
        )}
        <span className="num ml-auto text-[10.5px] text-muted">{fmtMs(scenario.stats.latencyMs)}</span>
        <span className="num text-[10.5px] text-faint">{fmtCost(scenario.stats.costCny)}</span>
      </header>

      <div className="px-5 py-4">
        <p className="text-[12px] leading-relaxed text-muted">
          <span className="eyebrow mr-2">ROUTE</span>
          {scenario.reason}
        </p>

        {phase === 'idle' && (
          <div className="mt-4 space-y-2" aria-hidden="true">
            <div className="h-3 w-2/3 rounded bg-surface-2" />
            <div className="h-3 w-full rounded bg-surface-2" />
            <div className="h-3 w-5/6 rounded bg-surface-2" />
            <p className="pt-2 text-[11px] text-faint">点击「运行」查看这条链路如何得出结论。</p>
          </div>
        )}

        {phase === 'running' && (
          <div className="mt-4 space-y-2" aria-hidden="true">
            <div className="h-3 w-1/2 animate-pulse rounded bg-surface-2" />
            <div className="h-3 w-full animate-pulse rounded bg-surface-2" />
            <div className="h-3 w-4/5 animate-pulse rounded bg-surface-2" />
          </div>
        )}

        {showBody && scenario.refusal && (
          <div className="mt-4 rounded-md border-l-2 py-3 pl-4 pr-3" style={{ borderColor: 'var(--color-danger)', background: '#FBF1EF' }}>
            <p className="flex items-center gap-1.5 text-[13px] font-medium" style={{ color: 'var(--color-danger)' }}>
              <Icon name="alert" size={15} />
              {scenario.refusal.title}
            </p>
            <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-2">{scenario.refusal.reason}</p>
            <p className="mt-1.5 text-[12px] text-muted">{scenario.refusal.suggestion}</p>
          </div>
        )}

        {showBody && !scenario.refusal && (() => {
          // 流式阶段用增量文本；完成后用服务端返回的完整答案（SSE done / POST /api/query）
          const text =
            phase === 'streaming'
              ? streamed
              : scenario.answer.lead + (scenario.answer.body ? '\n\n' + scenario.answer.body : '')
          const [lead, ...rest] = text.split('\n\n')
          return (
            <div className="mt-4">
              <div className="font-serif text-[20px] leading-[1.45] tracking-[-0.01em] text-ink" aria-live="polite">
                {renderWithCitations(lead ?? '', onCite)}
                {phase === 'streaming' && rest.length === 0 && <span className="stream-caret" />}
              </div>
              {rest.length > 0 && (
                <div className="mt-3 text-[13.5px] leading-[1.75] text-ink-2">
                  {renderWithCitations(rest.join('\n\n'), onCite)}
                  {phase === 'streaming' && <span className="stream-caret" />}
                </div>
              )}
            </div>
          )
        })()}

        {showBody && (
          <div className="mt-5 flex flex-wrap items-center gap-2 border-t pt-3" style={{ borderColor: 'var(--color-line)' }}>
            <span className="eyebrow">引用</span>
            {scenario.citations.length === 0 ? (
              <span className="text-[11.5px] text-faint">无</span>
            ) : (
              scenario.citations.map((n) => (
                <button
                  key={n}
                  type="button"
                  onClick={() => onCite(n)}
                  className="num rounded border px-1.5 py-0.5 text-[10.5px] text-ink-2 transition-colors hover:border-line-strong hover:bg-surface-2"
                  style={{ borderColor: 'var(--color-line)' }}
                >
                  [{n}]
                </button>
              ))
            )}
            <span className="num ml-2 text-[10.5px] text-faint">校验 {scenario.stats.citationCheck}</span>

            <div className="ml-auto flex items-center gap-1">
              {onFeedback && (
                <>
                  <button type="button" onClick={() => { setFeedback('up'); onFeedback?.('up') }} disabled={feedback !== null} className="rounded border px-2 py-1 text-[10.5px] text-muted transition-colors hover:bg-surface-2 hover:text-ink disabled:opacity-50" style={{ borderColor: 'var(--color-line)' }} title="答案有帮助">{feedback === 'up' ? '已反馈' : '有帮助'}</button>
                  <button type="button" onClick={() => { setFeedback('down'); onFeedback?.('down') }} disabled={feedback !== null} className="rounded border px-2 py-1 text-[10.5px] text-muted transition-colors hover:bg-surface-2 hover:text-ink disabled:opacity-50" style={{ borderColor: 'var(--color-line)' }} title="答案没帮助">{feedback === 'down' ? '已反馈' : '没帮助'}</button>
                </>
              )}
              <IconButton icon="copy" label="复制答案" onClick={onCopy} />
              <IconButton icon="refresh" label="重新生成" onClick={onRerun} />
            </div>
          </div>
        )}
      </div>

      {phase === 'idle' && (
        <footer className="border-t px-5 py-3 text-[11px] text-faint" style={{ borderColor: 'var(--color-line)' }}>
          问题：{question}
        </footer>
      )}

      {showBody && !scenario.refusal && (
        <footer className="border-t px-5 py-2.5 text-[10.5px] text-faint" style={{ borderColor: 'var(--color-line)' }}>
          答案由 {ROUTE_META[scenario.route].label} 链路生成 · {scenario.trace.length} 个阶段 · 引用均可回溯到 chunk
        </footer>
      )}
    </article>
  )
}

function IconButton({ icon, label, onClick }: { icon: 'copy' | 'refresh'; label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className="flex h-8 w-8 items-center justify-center rounded-md text-muted transition-colors hover:bg-surface-2 hover:text-ink"
    >
      <Icon name={icon} size={14} />
    </button>
  )
}

export { renderWithCitations }