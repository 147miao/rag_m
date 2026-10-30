import type { Route } from '../lib/types'
import { ROUTE_META } from '../lib/types'

export function RouteBadge({ route, confidence, size = 'md' }: { route: Route; confidence?: number; size?: 'sm' | 'md' }) {
  const meta = ROUTE_META[route]
  return (
    <span
      className={
        'inline-flex items-center gap-1.5 rounded-full border ' +
        (size === 'sm' ? 'px-2 py-[2px]' : 'px-2.5 py-1')
      }
      style={{ borderColor: meta.color + '2E', background: meta.soft, color: meta.color }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: meta.color }} />
      <span className="text-[11px] font-medium tracking-wide">{meta.label}</span>
      {confidence !== undefined && <span className="num text-[10.5px] opacity-75">{confidence.toFixed(2)}</span>}
    </span>
  )
}

export function Icon({ name, size = 16, className = '' }: { name: IconName; size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6}
      strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">
      {paths[name]}
    </svg>
  )
}

export type IconName = keyof typeof paths

const paths = {
  search: (<><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></>),
  graph: (<><circle cx="6" cy="6" r="2.4" /><circle cx="18" cy="8" r="2.4" /><circle cx="12" cy="18" r="2.4" /><path d="m8.2 7.2 7.6 1.2M7.4 8.2l3.4 7.4M16.4 10.2l-3.2 5.6" /></>),
  tool: (<><path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L4 17v3h3l5.3-5.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2.1-2.1z" /></>),
  copy: (<><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V5a1 1 0 0 1 1-1h10" /></>),
  refresh: (<><path d="M20 11a8 8 0 1 0-2.3 6.7" /><path d="M20 5v6h-6" /></>),
  check: (<><path d="m4 12.5 5 5L20 6.5" /></>),
  alert: (<><path d="M12 3 2.5 20h19L12 3Z" /><path d="M12 9v5M12 17.2v.1" /></>),
  chevron: (<><path d="m9 6 6 6-6 6" /></>),
  database: (<><ellipse cx="12" cy="5.5" rx="8" ry="3" /><path d="M4 5.5v13c0 1.7 3.6 3 8 3s8-1.3 8-3v-13" /><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></>),
  cpu: (<><rect x="6" y="6" width="12" height="12" rx="2" /><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" /></>),
  clock: (<><circle cx="12" cy="12" r="8.5" /><path d="M12 7.5V12l3 2" /></>),
  bolt: (<><path d="M13 2 4.5 13.5H11L10 22l8.5-11.5H12L13 2Z" /></>),
  external: (<><path d="M14 4h6v6" /><path d="M20 4 10 14" /><path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" /></>),
  spark: (<><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M18 6l-2.5 2.5M8.5 15.5 6 18" /></>),
  layers: (<><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 13 9 5 9-5" /></>),
} as const

export function Stat({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="eyebrow">{label}</span>
      <span className="num text-[15px] text-ink">
        {value}
        {unit && <span className="ml-0.5 text-[11px] text-muted">{unit}</span>}
      </span>
    </div>
  )
}

export function fmtMs(ms: number): string {
  return ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : ms + ' ms'
}

export function fmtCost(cny: number): string {
  return '¥' + cny.toFixed(4)
}
