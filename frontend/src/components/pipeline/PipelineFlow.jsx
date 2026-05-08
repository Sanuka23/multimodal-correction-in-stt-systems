import { useMemo, useState, useRef, useLayoutEffect } from 'react'
import {
  CheckCircle2, Loader2, XCircle, Clock3, SkipForward,
  Inbox, ScanText, BookText, Search, AudioLines, Eye, Wand2, Send, Sparkles,
  Cpu, Layers, Globe, ShieldCheck, Tags, Mic, FileSearch, Database,
} from 'lucide-react'
import { STEP_LABELS, STEP_DESCRIPTIONS, UMBRELLA_STEPS } from './PipelineNode'
import { canonicalSteps } from './stepUtils'
import { formatTime } from '../../utils/datetime'

/* ── Step icon registry ────────────────────────────────────────────── */

const STEP_ICONS = {
  request_received:      Inbox,
  model_load:            Cpu,
  vocab_merge:           BookText,
  candidate_detection:   Search,
  topic_classification:  Tags,
  web_vocab_enrichment:  Globe,
  candidate_validation:  ShieldCheck,
  ocr_extraction:        ScanText,
  ocr_vocab_extraction:  FileSearch,
  whisper_pass2:         Mic,
  avsr_extraction:       AudioLines,
  avsr_pass2:            AudioLines,
  llm_reconciliation:    Wand2,
  ml_inference:          Layers,
  data_collection:       Database,
  apply_corrections:     Send,
  complete:              Sparkles,
}

const STATUS_COLOR = {
  completed: '#4edea3',
  running:   '#ffd480',
  failed:    '#ffb4ab',
  pending:   '#45464d',
  skipped:   '#5a5a63',
}

const STATUS_ICON = {
  completed: CheckCircle2,
  running:   Loader2,
  failed:    XCircle,
  pending:   Clock3,
  skipped:   SkipForward,
}

/* ── Helpers ──────────────────────────────────────────────────────── */

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

// Canonical step order + dedup live in `./stepUtils` so the timings panel
// and the canvas agree on indices.

/**
 * Serpentine layout that wraps gracefully:
 *   - ≤6  nodes: single row
 *   - ≤12 nodes: two rows (top L→R, bottom R→L)
 *   - >12 nodes: three rows (top L→R, mid R→L, bottom L→R)
 *
 * Returns: array of { x, y } in viewBox coords.
 */
function layoutNodes(n, vbW, vbH, padX = 130, padY = 110) {
  if (n <= 0) return []
  const innerW = vbW - 2 * padX

  let rows
  if (n <= 6) rows = 1
  else if (n <= 12) rows = 2
  else rows = 3

  // Distribute nodes across rows as evenly as possible
  const baseCount = Math.ceil(n / rows)
  const counts = []
  let remaining = n
  for (let r = 0; r < rows; r++) {
    const c = Math.min(baseCount, remaining)
    counts.push(c)
    remaining -= c
  }

  const ys =
    rows === 1
      ? [vbH / 2]
      : rows === 2
        ? [padY, vbH - padY]
        : [padY, vbH / 2, vbH - padY]

  const positions = []
  for (let r = 0; r < rows; r++) {
    const rowCount = counts[r]
    if (rowCount <= 0) continue
    const reversed = r % 2 === 1
    const y = ys[r]
    for (let i = 0; i < rowCount; i++) {
      const t = rowCount === 1 ? 0.5 : i / (rowCount - 1)
      const x = padX + innerW * (reversed ? 1 - t : t)
      positions.push({ x, y })
    }
  }
  return positions
}

/**
 * Build a smooth cubic Bezier between two points.
 *
 * For a row transition we use a tight U-curve that stays INSIDE the canvas
 * (never overshoots the right padding) — fixes the "swooshes off-screen"
 * problem with serpentine layouts.
 */
function buildPath(a, b, vbW) {
  const dx = b.x - a.x
  const dy = b.y - a.y
  const sameRow = Math.abs(dy) < 4
  if (sameRow) {
    const cx = Math.abs(dx) * 0.45
    const bow = (a.x + b.x) / 2 > vbW / 2 ? -8 : 8
    return `M ${a.x} ${a.y} C ${a.x + cx} ${a.y + bow}, ${b.x - cx} ${b.y + bow}, ${b.x} ${b.y}`
  }
  // U-curve. Direction is determined by which side of the canvas the U should
  // bow toward. Both endpoints sit near the same x (typically the row edge).
  // Clamp the swing so the path never leaves the viewBox.
  const midX = (a.x + b.x) / 2
  const towardsRight = midX > vbW / 2 ? 1 : -1
  const maxSwingRight = Math.max(20, vbW - 40 - Math.max(a.x, b.x))
  const maxSwingLeft = Math.max(20, Math.min(a.x, b.x) - 40)
  const maxSwing = towardsRight > 0 ? maxSwingRight : maxSwingLeft
  const swing = Math.min(140, Math.abs(dy) * 0.55, maxSwing)
  return `M ${a.x} ${a.y} ` +
         `C ${a.x + swing * towardsRight} ${a.y}, ` +
         `  ${b.x + swing * towardsRight} ${b.y}, ` +
         `  ${b.x} ${b.y}`
}

/* ── Node visual ──────────────────────────────────────────────────── */

function NodeChip({ step, label, status, isHovered, isPinned, index, total }) {
  const color = STATUS_COLOR[status] || STATUS_COLOR.pending
  const Icon = STEP_ICONS[step.name] || STATUS_ICON[status] || Clock3
  const StatusIcon = STATUS_ICON[status] || Clock3

  const isRunning = status === 'running'
  const isCompleted = status === 'completed'
  const isFailed = status === 'failed'
  const stepNumber = (index ?? 0) + 1

  return (
    <div className={`flex flex-col items-center pointer-events-auto ${isRunning ? 'node-bob-running' : ''}`}>
      {/* Halo */}
      <div className="relative">
        {isRunning && (
          <span
            className="absolute inset-0 rounded-full node-halo-running"
            style={{
              boxShadow: `0 0 0 12px ${color}33`,
              backgroundColor: `${color}22`,
            }}
          />
        )}

        {/* Core circle — slightly smaller for better breathing room */}
        <div
          className={`
            relative w-[56px] h-[56px] rounded-full flex items-center justify-center
            transition-all duration-300
            ${isHovered ? 'scale-110' : ''}
            ${isPinned ? 'ring-2 ring-primary/60' : ''}
          `}
          style={{
            background: isCompleted
              ? `radial-gradient(circle at 30% 30%, #1c3a35 0%, #0e2422 70%, #08171a 100%)`
              : isRunning
                ? `radial-gradient(circle at 30% 30%, #3a3324 0%, #1f1a10 70%, #0d0a04 100%)`
                : isFailed
                  ? `radial-gradient(circle at 30% 30%, #3a1f1f 0%, #1f1010 70%, #0d0606 100%)`
                  : `radial-gradient(circle at 30% 30%, #1c2236 0%, #0f1422 70%, #060912 100%)`,
            boxShadow: isCompleted
              ? `inset 0 0 0 1.5px ${color}88, 0 6px 18px -8px ${color}aa`
              : isRunning
                ? `inset 0 0 0 1.5px ${color}, 0 8px 22px -6px ${color}cc, 0 0 24px -4px ${color}88`
                : `inset 0 0 0 1px ${color}55`,
          }}
        >
          <Icon
            size={22}
            color={color}
            className={isRunning ? 'animate-spin [animation-duration:2.4s]' : ''}
            strokeWidth={2}
          />

          {/* Step-number badge — top-left */}
          <span
            className="absolute -top-1.5 -left-1.5 min-w-[20px] h-[20px] px-1 rounded-full flex items-center justify-center text-[10px] font-bold font-mono tabular-nums"
            style={{
              backgroundColor: '#0b1326',
              color,
              boxShadow: `0 0 0 1.5px ${color}`,
            }}
            title={`Step ${stepNumber} of ${total}`}
          >
            {stepNumber}
          </span>

          {/* Status badge — bottom-right */}
          <span
            className="absolute -bottom-1 -right-1 w-[18px] h-[18px] rounded-full flex items-center justify-center"
            style={{ backgroundColor: '#0b1326', boxShadow: `0 0 0 1.5px ${color}` }}
          >
            <StatusIcon
              size={10}
              color={color}
              className={isRunning ? 'animate-spin' : ''}
              strokeWidth={2.4}
            />
          </span>
        </div>
      </div>

      {/* Caption — narrower so neighbours don't overlap, two-line wrap */}
      <div className="mt-2.5 text-center w-[118px]">
        <p
          className={`text-[10.5px] font-medium leading-[1.15] ${
            isCompleted || isRunning || isFailed ? 'text-on-surface' : 'text-on-surface-variant/70'
          }`}
          style={{
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
            minHeight: '24px',
          }}
        >
          {isRunning ? <span className="shimmer-text">{label}</span> : label}
        </p>
        <p
          className="mt-0.5 font-mono text-[9.5px] tracking-tight"
          style={{ color: isCompleted || isRunning || isFailed ? color : '#6b6c75' }}
        >
          {step.duration_ms != null
            ? formatDuration(step.duration_ms)
            : isRunning
              ? '…'
              : status === 'pending' ? '·' : '—'}
        </p>
      </div>
    </div>
  )
}

/* ── Hover preview card ───────────────────────────────────────────── */

function detailPreview(step) {
  const { name, details } = step
  if (details == null) return null

  if (name === 'web_vocab_enrichment' || name === 'ocr_vocab_extraction') {
    const terms = details.terms || []
    if (!terms.length) {
      return <p className="text-[10px] text-on-surface-variant/60 italic">No terms.</p>
    }
    return (
      <div className="flex flex-wrap gap-1">
        {terms.slice(0, 12).map((t, i) => (
          <span key={i} className="text-[10px] font-mono bg-primary/10 text-primary px-1.5 py-0.5 rounded">
            {typeof t === 'string' ? t : t.term}
          </span>
        ))}
        {terms.length > 12 && (
          <span className="text-[10px] text-on-surface-variant/60">+ {terms.length - 12}</span>
        )}
      </div>
    )
  }

  if (name === 'topic_classification') {
    return (
      <div className="space-y-1 text-[11px] font-mono">
        <div><span className="text-on-surface-variant/70">field: </span><span className="text-on-surface">{details.field}</span></div>
        <div><span className="text-on-surface-variant/70">topic: </span><span className="text-on-surface">{details.topic}</span></div>
        {details.description && (
          <p className="text-[10px] text-on-surface-variant/80 leading-relaxed mt-1">{details.description}</p>
        )}
      </div>
    )
  }

  if (name === 'avsr_extraction' || name === 'avsr_pass2') {
    return (
      <div className="space-y-1 text-[10px] font-mono">
        <div><span className="text-on-surface-variant/70">mode: </span>{details.mode}</div>
        <div><span className="text-on-surface-variant/70">hints: </span>{details.hints}</div>
        {(details.samples || []).slice(0, 3).map((s, i) => (
          <p key={i} className="text-on-surface-variant truncate">
            <span className="text-tertiary">[{(s.start || 0).toFixed?.(1) ?? s.start}s]</span> {s.hint}
          </p>
        ))}
      </div>
    )
  }

  if (name === 'whisper_pass2') {
    return (
      <div className="space-y-1 text-[10px] font-mono">
        <div><span className="text-on-surface-variant/70">segments: </span>{details.segments_re_transcribed}</div>
        <div><span className="text-on-surface-variant/70">model: </span>{details.model} · {details.device}/{details.compute_type}</div>
      </div>
    )
  }

  if (name === 'llm_reconciliation') {
    const swaps = details.sample_swaps || []
    return (
      <div className="space-y-1">
        <div className="text-[10px] font-mono"><span className="text-on-surface-variant/70">swaps: </span><span className="text-secondary">{details.swaps}</span></div>
        {swaps.slice(0, 4).map((s, i) => (
          <p key={i} className="text-[10px] font-mono text-on-surface-variant truncate">{s}</p>
        ))}
      </div>
    )
  }

  if (name === 'candidate_detection') {
    const items = Array.isArray(details) ? details : details?.candidates || []
    if (!items.length) return null
    return (
      <div className="space-y-1">
        {items.slice(0, 4).map((it, i) => (
          <div key={i} className="flex items-center gap-2 text-[11px] font-mono">
            <span className="text-error">{it.error_found || '—'}</span>
            <span className="text-on-surface-variant">→</span>
            <span className="text-secondary">{it.likely_correct || '—'}</span>
            {it.category && (
              <span className="text-[9px] text-on-surface-variant/60 ml-auto">{it.category}</span>
            )}
          </div>
        ))}
        {items.length > 4 && (
          <p className="text-[10px] text-on-surface-variant/60">+ {items.length - 4} more</p>
        )}
      </div>
    )
  }

  if (name === 'ml_inference') {
    const items = Array.isArray(details) ? details : details?.corrections || []
    if (!items.length) return null
    return (
      <div className="space-y-1">
        {items.slice(0, 4).map((it, i) => (
          <div key={i} className="flex items-center gap-2 text-[11px] font-mono">
            <span className="text-error">{it.error || '—'}</span>
            <span className="text-on-surface-variant">→</span>
            <span className="text-secondary">{it.corrected || '—'}</span>
            {it.confidence != null && (
              <span className="text-[9px] text-primary ml-auto">
                {(it.confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>
        ))}
        {items.length > 4 && (
          <p className="text-[10px] text-on-surface-variant/60">+ {items.length - 4} more</p>
        )}
      </div>
    )
  }

  if (name === 'ocr_extraction') {
    const samples = details?.text_samples || details?.samples || (typeof details === 'string' ? [details] : null)
    if (!samples) return null
    const arr = Array.isArray(samples) ? samples : [samples]
    return (
      <div className="space-y-1">
        {arr.slice(0, 4).map((s, i) => (
          <p key={i} className="text-[10px] font-mono text-on-surface-variant truncate">
            • {typeof s === 'string' ? s : JSON.stringify(s).slice(0, 80)}
          </p>
        ))}
        {arr.length > 4 && (
          <p className="text-[10px] text-on-surface-variant/60">+ {arr.length - 4} more lines</p>
        )}
      </div>
    )
  }

  if (typeof details === 'object') {
    const entries = Object.entries(details).slice(0, 6)
    return (
      <div className="space-y-1">
        {entries.map(([k, v]) => (
          <div key={k} className="flex items-baseline gap-2 text-[10px] font-mono">
            <span className="text-on-surface-variant/70">{k}:</span>
            <span className="text-on-surface truncate">
              {typeof v === 'object' ? JSON.stringify(v).slice(0, 60) : String(v).slice(0, 60)}
            </span>
          </div>
        ))}
      </div>
    )
  }

  return (
    <p className="text-[10px] font-mono text-on-surface-variant whitespace-pre-wrap">
      {String(details).slice(0, 220)}
    </p>
  )
}

function HoverCard({ step, label, status, x, y, side }) {
  const color = STATUS_COLOR[status] || STATUS_COLOR.pending
  const StatusIcon = STATUS_ICON[status] || Clock3

  const placeBelow = side === 'below'

  // Clamp horizontal position so the card never sticks past the canvas edges.
  // Card width 320px; container width unknown at render time but we work in %.
  // Anchoring the card to a clamped left% keeps it inside.
  const clampedX = Math.max(13, Math.min(87, x))
  const anchorOffset = x - clampedX // for tail/arrow calculations later

  return (
    <div
      className="absolute z-30 pointer-events-none hover-card-enter"
      style={{
        left: `${clampedX}%`,
        top: `${y}%`,
        transform: `translate(-50%, ${placeBelow ? '64px' : 'calc(-100% - 64px)'})`,
        width: 320,
      }}
    >
      <div
        className="rounded-xl p-3 shadow-2xl"
        style={{
          background: 'rgba(11, 19, 38, 0.96)',
          backdropFilter: 'blur(14px)',
          border: `1px solid ${color}55`,
          boxShadow: `0 24px 48px -12px ${color}33, 0 0 0 1px ${color}22`,
        }}
      >
        <div className="flex items-center justify-between mb-1 gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <StatusIcon size={13} color={color} className={status === 'running' ? 'animate-spin' : ''} />
            <h4 className="font-headline text-[13px] font-bold text-on-surface truncate">{label}</h4>
            {UMBRELLA_STEPS.has(step.name) && (
              <span
                className="text-[8.5px] font-label uppercase tracking-widest px-1 py-px rounded bg-on-surface-variant/15 text-on-surface-variant/70 flex-shrink-0"
                title="Wall-clock total of all child steps — excluded from the overall total."
              >
                umbrella
              </span>
            )}
          </div>
          <span
            className="text-[9px] font-label uppercase tracking-widest px-1.5 py-0.5 rounded-full flex-shrink-0"
            style={{ color, backgroundColor: `${color}18` }}
          >
            {status}
          </span>
        </div>

        {STEP_DESCRIPTIONS[step.name] && (
          <p className="text-[10.5px] text-on-surface-variant/80 leading-snug mb-2">
            {STEP_DESCRIPTIONS[step.name]}
          </p>
        )}

        <div className="grid grid-cols-2 gap-2 mb-2 text-[10px]">
          <div>
            <p className="text-on-surface-variant/60 font-label">Duration</p>
            <p className="font-mono" style={{ color }}>{formatDuration(step.duration_ms)}</p>
          </div>
          <div>
            <p className="text-on-surface-variant/60 font-label">Updated · LK</p>
            <p className="font-mono text-on-surface-variant">
              {step.updated_at ? formatTime(step.updated_at) : '—'}
            </p>
          </div>
        </div>

        <div className="border-t border-outline-variant/15 pt-2">
          <p className="text-[9px] font-label uppercase tracking-widest text-on-surface-variant/60 mb-1.5">
            Step Logs / Details
          </p>
          {detailPreview(step) || (
            <p className="text-[10px] text-on-surface-variant/60 italic">
              {status === 'running' ? 'Streaming…' : status === 'pending' ? 'Awaiting upstream…' : 'No detail payload.'}
            </p>
          )}
        </div>

        {/* Hint footer */}
        <p className="mt-2 pt-2 border-t border-outline-variant/10 text-[9px] text-on-surface-variant/50 text-center">
          Click node to pin full log panel
        </p>
      </div>
    </div>
  )
}

/* ── Main canvas ──────────────────────────────────────────────────── */

export default function PipelineFlow({
  steps: rawSteps,
  selectedIdx,
  onSelectStep,
}) {
  const [hovered, setHovered] = useState(null)
  const containerRef = useRef(null)

  const VB_W = 1400
  const VB_H = 580

  // Canonical execution order so the visual flow always matches what
  // the pipeline actually ran (regardless of MongoDB insertion order).
  const steps = useMemo(() => canonicalSteps(rawSteps || []), [rawSteps])

  const positions = useMemo(
    () => layoutNodes(steps.length, VB_W, VB_H),
    [steps.length],
  )

  const overallStatus = steps.some(s => s.status === 'failed')
    ? 'failed'
    : steps.some(s => s.status === 'running')
      ? 'running'
      : steps.length > 0 && steps.every(s => s.status === 'completed')
        ? 'completed'
        : 'pending'

  // Persist running ring across re-renders
  const runningIdx = steps.findIndex(s => s.status === 'running')

  return (
    <div
      ref={containerRef}
      className="relative w-full canvas-bg-grid"
      style={{ aspectRatio: `${VB_W} / ${VB_H}`, minHeight: 380 }}
    >
      {/* SVG path layer */}
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id="grad-completed" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#4edea3" stopOpacity="0.85" />
            <stop offset="100%" stopColor="#4edea3" stopOpacity="0.55" />
          </linearGradient>
          <linearGradient id="grad-running" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#4edea3" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#ffd480" stopOpacity="0.95" />
          </linearGradient>
          <linearGradient id="grad-pending" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#45464d" stopOpacity="0.55" />
            <stop offset="100%" stopColor="#45464d" stopOpacity="0.35" />
          </linearGradient>
          <filter id="glow-running" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {/* Arrowheads — one variant per connector status */}
          {[
            { id: 'arrow-completed', color: '#4edea3', opacity: 0.95 },
            { id: 'arrow-running',   color: '#ffd480', opacity: 1.0  },
            { id: 'arrow-pending',   color: '#45464d', opacity: 0.55 },
            { id: 'arrow-failed',    color: '#ffb4ab', opacity: 0.9  },
          ].map(a => (
            <marker
              key={a.id}
              id={a.id}
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill={a.color} opacity={a.opacity} />
            </marker>
          ))}
        </defs>

        {/* Curves between consecutive nodes */}
        {positions.slice(0, -1).map((a, i) => {
          const b = positions[i + 1]
          const path = buildPath(a, b, VB_W)
          const target = steps[i + 1]
          const source = steps[i]
          const tStatus = target?.status || 'pending'
          const sStatus = source?.status || 'pending'

          const isActive = tStatus === 'running'
          const isCompleted = tStatus === 'completed' || (sStatus === 'completed' && tStatus !== 'pending')
          const isFailed = tStatus === 'failed'

          let stroke = 'url(#grad-pending)'
          let arrow = 'arrow-pending'
          let opacity = 0.5
          let className = ''
          if (isActive) {
            stroke = 'url(#grad-running)'
            arrow = 'arrow-running'
            opacity = 1
            className = 'flow-running-path flow-glow-running'
          } else if (isCompleted) {
            stroke = 'url(#grad-completed)'
            arrow = 'arrow-completed'
            opacity = 0.95
            className = 'flow-glow-completed'
          } else if (isFailed) {
            stroke = '#ffb4ab'
            arrow = 'arrow-failed'
            opacity = 0.85
          }

          return (
            <g key={`edge-${i}`}>
              {/* Backdrop wide soft */}
              <path
                d={path}
                stroke={stroke}
                strokeWidth={isActive ? 8 : isCompleted ? 6 : 4}
                strokeOpacity={isActive ? 0.18 : isCompleted ? 0.12 : 0.08}
                fill="none"
                strokeLinecap="round"
              />
              {/* Main flow with arrowhead */}
              <path
                d={path}
                stroke={stroke}
                strokeOpacity={opacity}
                strokeWidth={2.4}
                fill="none"
                strokeLinecap="round"
                markerEnd={`url(#${arrow})`}
                className={className}
              />
              {/* Animated travelling dot for the active edge */}
              {isActive && (
                <circle r="4" fill="#ffd480" filter="url(#glow-running)">
                  <animateMotion
                    dur="1.6s"
                    repeatCount="indefinite"
                    path={path}
                    rotate="auto"
                  />
                  <animate
                    attributeName="opacity"
                    values="0;1;1;0"
                    keyTimes="0;0.15;0.85;1"
                    dur="1.6s"
                    repeatCount="indefinite"
                  />
                </circle>
              )}
            </g>
          )
        })}

        {/* Pulse ring around running node (drawn in SVG so it sits below DOM nodes) */}
        {runningIdx >= 0 && positions[runningIdx] && (
          <g
            className="ring-pulse-running"
            transform={`translate(${positions[runningIdx].x} ${positions[runningIdx].y})`}
          >
            <circle
              className="pulse-ring"
              cx="0" cy="0" r="28"
              fill="none"
              stroke="#ffd480"
              strokeWidth="1.5"
            />
          </g>
        )}
      </svg>

      {/* HTML node layer */}
      {steps.map((step, i) => {
        const pos = positions[i]
        if (!pos) return null
        const xPct = (pos.x / VB_W) * 100
        const yPct = (pos.y / VB_H) * 100
        const label = STEP_LABELS[step.name] || step.name
        const status = step.status || 'pending'

        return (
          <button
            key={step.name || i}
            type="button"
            className="absolute group focus:outline-none"
            style={{
              left: `${xPct}%`,
              top: `${yPct}%`,
              transform: 'translate(-50%, -50%)',
            }}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered((h) => (h === i ? null : h))}
            onFocus={() => setHovered(i)}
            onBlur={() => setHovered((h) => (h === i ? null : h))}
            onClick={() => onSelectStep(selectedIdx === i ? null : i)}
            aria-label={`${label} — ${status}`}
          >
            <NodeChip
              step={step}
              label={label}
              status={status}
              isHovered={hovered === i}
              isPinned={selectedIdx === i}
              index={i}
              total={steps.length}
            />
          </button>
        )
      })}

      {/* Hover card */}
      {hovered != null && positions[hovered] && (() => {
        const pos = positions[hovered]
        const xPct = (pos.x / VB_W) * 100
        const yPct = (pos.y / VB_H) * 100
        const step = steps[hovered]
        const label = STEP_LABELS[step.name] || step.name
        const status = step.status || 'pending'
        return (
          <HoverCard
            step={step}
            label={label}
            status={status}
            x={xPct}
            y={yPct}
            side={yPct < 35 ? 'below' : 'above'}
          />
        )
      })()}

      {/* Stage progress strip */}
      <div className="absolute left-6 right-6 bottom-3 flex items-center gap-3 pointer-events-none select-none">
        <div className="flex items-center gap-1.5">
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{ backgroundColor: STATUS_COLOR[overallStatus] }}
          />
          <span className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant/70">
            {overallStatus === 'running'
              ? `Running step ${(runningIdx >= 0 ? runningIdx + 1 : 0)}/${steps.length}`
              : overallStatus === 'completed'
                ? 'Pipeline complete'
                : overallStatus === 'failed'
                  ? 'Pipeline failed'
                  : 'Pipeline queued'}
          </span>
        </div>
        <div className="flex-1 h-px bg-outline-variant/20 ml-2" />
        <span className="text-[10px] font-mono text-on-surface-variant/60">
          {steps.filter(s => s.status === 'completed').length} / {steps.length} ✓
        </span>
      </div>
    </div>
  )
}
