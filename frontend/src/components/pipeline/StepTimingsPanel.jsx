import { useMemo, useState } from 'react'
import { Clock, ArrowDown, ArrowUp, Hash, Info } from 'lucide-react'
import { STEP_LABELS, STEP_DESCRIPTIONS, UMBRELLA_STEPS } from './PipelineNode'

const STATUS_DOT = {
  completed: '#4edea3',
  running:   '#ffd480',
  failed:    '#ffb4ab',
  pending:   '#45464d',
  skipped:   '#5a5a63',
}

function formatDuration(ms) {
  if (ms == null) return '—'
  if (ms < 1) return `${ms.toFixed(2)}ms`
  if (ms < 1000) return `${Math.round(ms)}ms`
  const secs = ms / 1000
  if (secs < 60) return `${secs.toFixed(secs < 10 ? 2 : 1)}s`
  const mins = Math.floor(secs / 60)
  const rem = Math.floor(secs % 60)
  return `${mins}m ${rem}s`
}

/**
 * Step timings panel — sortable, compact view of where the pipeline spent
 * its time. Each row is a horizontal bar showing the step's contribution
 * to total duration.
 *
 * Props:
 *   steps      — sorted pipeline steps (canonical order from PipelineFlow)
 *   onJump     — optional (idx) => void, jumps the canvas to that step
 */
export default function StepTimingsPanel({ steps, onJump }) {
  const [sortKey, setSortKey] = useState('order') // 'order' | 'time'
  const [sortDir, setSortDir] = useState('asc')   // 'asc' | 'desc'

  // Total excludes umbrella rows (ml_inference, complete) — those duplicate
  // the wall-clock of all children and would inflate the sum.
  const total = useMemo(
    () => steps.reduce(
      (sum, s) => sum + (UMBRELLA_STEPS.has(s.name) ? 0 : (s.duration_ms || 0)),
      0,
    ),
    [steps],
  )
  // Max also excludes umbrellas so the bars compare like-for-like.
  const max = useMemo(
    () => Math.max(
      1,
      ...steps.filter(s => !UMBRELLA_STEPS.has(s.name)).map(s => s.duration_ms || 0),
    ),
    [steps],
  )

  const ranked = useMemo(() => {
    const withIdx = steps.map((s, i) => ({ ...s, originalIdx: i }))
    if (sortKey === 'time') {
      withIdx.sort((a, b) => (b.duration_ms || 0) - (a.duration_ms || 0))
      if (sortDir === 'asc') withIdx.reverse()
    } else if (sortDir === 'desc') {
      withIdx.reverse()
    }
    return withIdx
  }, [steps, sortKey, sortDir])

  const toggleSort = (key) => {
    if (sortKey === key) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir(key === 'time' ? 'desc' : 'asc')
    }
  }

  if (!steps.length) return null

  return (
    <div className="obsidian-glass rounded-2xl border border-outline-variant/10 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 px-5 py-3 border-b border-outline-variant/10">
        <div className="flex items-center gap-2">
          <Clock size={14} className="text-primary" />
          <h3 className="font-headline text-sm font-bold text-on-surface tracking-tight">Step Timings</h3>
          <span className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant/60">
            · Total {formatDuration(total)} · {steps.length} steps
          </span>
        </div>

        <div className="flex bg-surface-container-low rounded-lg p-0.5 border border-outline-variant/15">
          <SortBtn active={sortKey === 'order'} dir={sortDir} onClick={() => toggleSort('order')} icon={Hash} label="Order" />
          <SortBtn active={sortKey === 'time'}  dir={sortDir} onClick={() => toggleSort('time')}  icon={Clock} label="Time" />
        </div>
      </div>

      {/* Rows */}
      <div className="max-h-[420px] overflow-y-auto scrollbar-thin">
        <table className="w-full text-[12px]">
          <tbody>
            {ranked.map((step) => {
              const status = step.status || 'pending'
              const dotColor = STATUS_DOT[status]
              const ms = step.duration_ms || 0
              const isUmbrella = UMBRELLA_STEPS.has(step.name)
              const pctOfMax = isUmbrella ? 0 : (ms / max) * 100
              const pctOfTotal = !isUmbrella && total > 0 ? (ms / total) * 100 : 0
              const label = STEP_LABELS[step.name] || step.name
              const description = STEP_DESCRIPTIONS[step.name]

              return (
                <tr
                  key={step.name}
                  onClick={() => onJump && onJump(step.originalIdx)}
                  className={`cursor-pointer transition-colors hover:bg-surface-container-high/40 border-b border-outline-variant/5 last:border-b-0 ${
                    status === 'running' ? 'bg-tertiary/[0.04]' : ''
                  } ${isUmbrella ? 'opacity-70' : ''}`}
                  title={description || undefined}
                >
                  {/* index */}
                  <td className="py-2 pl-4 pr-2 text-[10px] font-mono text-on-surface-variant/60 w-[34px] align-top">
                    #{step.originalIdx + 1}
                  </td>

                  {/* status dot */}
                  <td className="py-2 pr-2 w-[14px] align-top">
                    <span
                      className={`inline-block w-2 h-2 rounded-full mt-1.5 ${status === 'running' ? 'live-dot' : ''}`}
                      style={{ backgroundColor: dotColor }}
                    />
                  </td>

                  {/* name + description */}
                  <td className="py-2 pr-3 w-[260px]">
                    <div className="flex items-center gap-1.5">
                      <span className="text-on-surface font-medium truncate">{label}</span>
                      {isUmbrella && (
                        <span
                          className="text-[8.5px] font-label uppercase tracking-widest px-1 py-px rounded bg-on-surface-variant/15 text-on-surface-variant/70"
                          title="Wall-clock total of all child steps — excluded from the overall total."
                        >
                          umbrella
                        </span>
                      )}
                    </div>
                    {description && (
                      <p className="text-[10px] text-on-surface-variant/60 leading-tight truncate mt-0.5">
                        {description}
                      </p>
                    )}
                  </td>

                  {/* horizontal bar */}
                  <td className="py-2 pr-3 align-middle">
                    <div className="relative h-[14px] bg-surface-container-low rounded-full overflow-hidden">
                      {isUmbrella ? (
                        // Umbrella: hatched marker — visually distinct, no comparable width
                        <div
                          className="absolute inset-0 opacity-50 rounded-full"
                          style={{
                            background:
                              'repeating-linear-gradient(45deg, rgba(123,208,255,0.18) 0 6px, transparent 6px 12px)',
                          }}
                        />
                      ) : (
                        <div
                          className="absolute left-0 top-0 bottom-0 rounded-full transition-all duration-300"
                          style={{
                            width: `${Math.max(2, pctOfMax)}%`,
                            background: status === 'running'
                              ? 'linear-gradient(90deg, rgba(78,222,163,0.4), #ffd480)'
                              : status === 'failed'
                                ? '#ffb4ab'
                                : status === 'skipped'
                                  ? 'rgba(144,144,151,0.35)'
                                  : 'linear-gradient(90deg, rgba(78,222,163,0.5), rgba(78,222,163,0.85))',
                          }}
                        />
                      )}
                    </div>
                  </td>

                  {/* time */}
                  <td className="py-2 pr-2 text-right font-mono text-on-surface tabular-nums w-[80px] align-middle">
                    {ms > 0 ? formatDuration(ms) : (status === 'running' ? '…' : status === 'skipped' ? 'skipped' : '—')}
                  </td>

                  {/* % of total — blank for umbrellas */}
                  <td className="py-2 pr-4 text-right font-mono text-[10px] text-on-surface-variant/70 w-[50px] align-middle">
                    {!isUmbrella && ms > 0 && total > 0 ? `${pctOfTotal.toFixed(1)}%` : ''}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function SortBtn({ active, dir, onClick, icon: Icon, label }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-[10px] font-label font-bold uppercase tracking-widest transition-all ${
        active ? 'bg-surface-container-high text-on-surface' : 'text-on-surface-variant/70 hover:text-on-surface'
      }`}
    >
      <Icon size={10} />
      {label}
      {active && (dir === 'asc' ? <ArrowUp size={9} /> : <ArrowDown size={9} />)}
    </button>
  )
}
