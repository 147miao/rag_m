import { useEffect, useRef } from 'react'
import type { Route } from '../lib/types'
import { Icon } from './ui'

const OPTIONS: { value: Route | 'auto'; label: string }[] = [
  { value: 'auto', label: '自动' },
  { value: 'simple', label: '简单' },
  { value: 'multihop', label: '多跳' },
  { value: 'agentic', label: '推理' },
]

export function Composer({
  value,
  onChange,
  onRun,
  running,
  override,
  onOverride,
}: {
  value: string
  onChange: (v: string) => void
  onRun: () => void
  running: boolean
  override: Route | 'auto'
  onOverride: (v: Route | 'auto') => void
}) {
  const ref = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault()
        onRun()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onRun])

  return (
    <div className="card p-4">
      <textarea
        ref={ref}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={3}
        placeholder="问一个问题。系统会自动判断该走哪条链路，也可以手动指定。"
        className="w-full resize-none bg-transparent text-[15px] leading-relaxed text-ink outline-none placeholder:text-faint"
        aria-label="问题输入"
      />

      <div className="mt-3 flex flex-wrap items-center gap-3 border-t pt-3" style={{ borderColor: 'var(--color-line)' }}>
        <div className="flex items-center gap-2">
          <span className="eyebrow">路由</span>
          <div className="flex rounded-md border p-0.5" style={{ borderColor: 'var(--color-line)' }}>
            {OPTIONS.map((o) => {
              const active = override === o.value
              return (
                <button
                  key={o.value}
                  type="button"
                  onClick={() => onOverride(o.value)}
                  className={
                    'h-7 rounded-[4px] px-2.5 text-[11.5px] transition-colors ' +
                    (active ? 'bg-ink text-paper' : 'text-muted hover:text-ink')
                  }
                  aria-pressed={active}
                >
                  {o.label}
                </button>
              )
            })}
          </div>
        </div>

        <span className="hidden text-[11px] text-faint sm:inline">⌘ / Ctrl + ↵ 运行</span>

        <button
          type="button"
          onClick={onRun}
          disabled={running || value.trim().length === 0}
          className="ml-auto flex h-9 items-center gap-2 rounded-md bg-teal px-4 text-[13px] font-medium text-white transition-opacity disabled:opacity-40"
        >
          {running ? <Icon name="refresh" size={14} className="animate-spin" /> : <Icon name="bolt" size={14} />}
          {running ? '运行中' : '运行'}
        </button>
      </div>
    </div>
  )
}
