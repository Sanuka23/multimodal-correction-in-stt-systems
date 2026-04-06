import { CheckCircle, Loader, XCircle, Clock, SkipForward } from 'lucide-react'

const STEP_LABELS = {
  request_received: 'Request Received',
  vocab_merge: 'Vocab Merge',
  candidate_detection: 'Error Detection',
  ocr_extraction: 'OCR Extraction',
  ml_inference: 'Whisper + Reconciliation',
  avsr_extraction: 'AVSR Extraction',
  apply_corrections: 'Apply Corrections',
  complete: 'Complete',
}

const STATUS_CONFIG = {
  completed: {
    color: '#4edea3',
    bg: 'rgba(78, 222, 163, 0.08)',
    icon: CheckCircle,
    border: 'border-l-[#4edea3]',
  },
  running: {
    color: '#ffd480',
    bg: 'rgba(255, 212, 128, 0.08)',
    icon: Loader,
    border: 'border-l-[#ffd480]',
  },
  failed: {
    color: '#ffb4ab',
    bg: 'rgba(255, 180, 171, 0.08)',
    icon: XCircle,
    border: 'border-l-[#ffb4ab]',
  },
  pending: {
    color: '#909097',
    bg: 'rgba(144, 144, 151, 0.05)',
    icon: Clock,
    border: 'border-l-[#909097]',
  },
  skipped: {
    color: '#909097',
    bg: 'rgba(144, 144, 151, 0.05)',
    icon: SkipForward,
    border: 'border-l-[#909097]',
  },
}

function formatDuration(ms) {
  if (!ms) return '\u2014'
  if (ms < 1000) return `${ms}ms`
  const secs = Math.floor(ms / 1000)
  if (secs < 60) return `${secs}s`
  const mins = Math.floor(secs / 60)
  const rem = secs % 60
  return `${mins}m ${rem}s`
}

export default function PipelineNode({ step, isSelected, onClick }) {
  const status = step.status || 'pending'
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.pending
  const Icon = cfg.icon
  const label = STEP_LABELS[step.name] || step.name

  return (
    <button
      onClick={onClick}
      className={`
        w-40 flex-shrink-0 flux-card rounded-xl border-l-[3px] ${cfg.border}
        p-4 text-left transition-all cursor-pointer
        hover:brightness-110
        ${status === 'running' ? 'step-running' : ''}
        ${isSelected ? 'node-active ring-1 ring-primary/30' : ''}
      `}
      style={{ backgroundColor: cfg.bg }}
    >
      <Icon
        size={20}
        color={cfg.color}
        className={status === 'running' ? 'animate-spin' : ''}
      />

      <p className="mt-3 text-xs font-medium text-on-surface leading-tight truncate">
        {label}
      </p>

      <p className="mt-1 font-mono text-[10px]" style={{ color: cfg.color }}>
        {formatDuration(step.duration_ms)}
      </p>
    </button>
  )
}

export { STEP_LABELS }
