import { Icon } from './ui'

export function TopBar() {
  return (
    <header className="sticky top-0 z-30 border-b bg-paper/85 backdrop-blur-[6px]" style={{ borderColor: 'var(--color-line)' }}>
      <div className="mx-auto flex h-14 max-w-[1560px] items-center gap-6 px-6">
        <div className="flex items-baseline gap-2.5">
          <span className="font-serif text-[19px] tracking-tight text-ink">RAG-M</span>
          <span className="hidden text-[11.5px] text-muted sm:inline">路由式混合检索增强生成 · 控制台</span>
        </div>

        <div className="ml-2 hidden items-center gap-3 md:flex">
          <Legend color="#0F5C55" label="简单事实" />
          <Legend color="#4338CA" label="多跳关系" />
          <Legend color="#B45309" label="复杂推理" />
        </div>

        <div className="ml-auto flex items-center gap-4">
          <span className="hidden items-center gap-1.5 text-[11.5px] text-muted lg:flex">
            <span className="h-1.5 w-1.5 rounded-full bg-success" />
            Milvus 3.0.4 · 已连接
          </span>
          <span className="hidden items-center gap-1.5 text-[11.5px] text-muted lg:flex">
            <Icon name="cpu" size={13} />
            gemma4:31b-cloud
          </span>
          <button
            type="button"
            className="flex h-8 items-center gap-1.5 rounded-md border px-2.5 text-[12px] text-ink-2 transition-colors hover:bg-surface-2"
            style={{ borderColor: 'var(--color-line)' }}
            title="设置（演示占位）"
          >
            <Icon name="layers" size={14} />
            设置
          </button>
        </div>
      </div>
    </header>
  )
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5 text-[11.5px] text-muted">
      <span className="h-[6px] w-[6px] rounded-full" style={{ background: color }} />
      {label}
    </span>
  )
}
