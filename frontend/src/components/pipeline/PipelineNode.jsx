import { CheckCircle, Loader, XCircle, Clock, SkipForward } from 'lucide-react'

const STEP_LABELS = {
  request_received:      'Request Received',
  model_load:            'Model Load',
  vocab_merge:           'Vocab Merge',
  candidate_detection:   'Error Detection',
  topic_classification:  'Topic Classification',
  web_vocab_enrichment:  'Web Vocab Enrichment',
  candidate_validation:  'Candidate Validation',
  ocr_extraction:        'Screen OCR',
  ocr_vocab_extraction:  'OCR Term Mining',
  whisper_pass2:         'Whisper Pass 2',
  avsr_extraction:       'AVSR (Lip Reading)',
  avsr_pass2:            'AVSR Pass 2',
  llm_reconciliation:    'LLM Reconciliation',
  ml_inference:          'Pipeline Total',
  data_collection:       'Training-Data Collect',
  apply_corrections:     'Apply Corrections',
  complete:              'Complete',
}

/**
 * One-line description of what each step actually does.
 * Surfaced in the hover card and the timings panel so it's clear,
 * for example, that "Screen OCR" and "OCR Term Mining" are different stages.
 */
const STEP_DESCRIPTIONS = {
  request_received:      'Job accepted — vocab parsed, video URL captured.',
  model_load:            'Loads Qwen3.5-9B (MLX 4-bit) and tokenizer once per process.',
  vocab_merge:           'Combines built-in domain vocabulary with team / file custom terms.',
  candidate_detection:   'LLM scans the transcript and flags suspicious words as ASR errors.',
  topic_classification:  'LLM classifies the meeting field, topic and suggests domain vocab.',
  web_vocab_enrichment:  'DuckDuckGo searches the topic; LLM extracts a fresh glossary.',
  candidate_validation:  'Cross-chunk pooling + per-candidate web search to pick best target.',
  ocr_extraction:        'PaddleOCR reads raw text from ~15 sampled video frames.',
  ocr_vocab_extraction:  'LLM mines person / product / company names from the OCR text.',
  whisper_pass2:         'Re-transcribes flagged segments with vocab-biased Whisper small.',
  avsr_extraction:       'Lip-reading hints from MediaPipe / Auto-AVSR on AVSR-eligible candidates.',
  avsr_pass2:            'Optional second AVSR pass on uncertain segments.',
  llm_reconciliation:    'Reconciles original vs Whisper using vocab + OCR + AVSR evidence.',
  ml_inference:          'Umbrella — wall-clock for the full pipeline (sum of all children).',
  data_collection:       'Persists applied corrections as JSONL training pairs.',
  apply_corrections:     'Saves enhanced transcript + correction details to the dashboard DB.',
  complete:              'Umbrella — final job marker with totals and evidence rollup.',
}

/** Steps whose duration is the sum of other steps and would double-count. */
const UMBRELLA_STEPS = new Set(['ml_inference', 'complete'])

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

export { STEP_LABELS, STEP_DESCRIPTIONS, UMBRELLA_STEPS }
