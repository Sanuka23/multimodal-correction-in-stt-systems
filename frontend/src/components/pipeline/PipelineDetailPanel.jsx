import { useRef, useEffect, useState } from 'react'
import { X } from 'lucide-react'
import { STEP_LABELS } from './PipelineNode'

const STATUS_COLORS = {
  completed: '#4edea3',
  running: '#ffd480',
  failed: '#ffb4ab',
  pending: '#909097',
  skipped: '#909097',
}

function formatTimestamp(ts) {
  if (!ts) return '\u2014'
  try {
    return new Date(ts).toLocaleString()
  } catch {
    return ts
  }
}

/* ---------- Sub-renderers for specific step types ---------- */

function CandidateDetectionTable({ details }) {
  const items = Array.isArray(details) ? details : details?.candidates || []
  if (!items.length) return <JsonBlock data={details} />

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead>
          <tr className="border-b border-outline-variant/10">
            <th className="pb-2 pr-4 font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Error Found</th>
            <th className="pb-2 pr-4 font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Likely Correct</th>
            <th className="pb-2 pr-4 font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Category</th>
            <th className="pb-2 font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Timestamp</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-outline-variant/5">
          {items.map((item, i) => (
            <tr key={i}>
              <td className="py-2 pr-4 text-error font-mono">{item.error_found ?? '\u2014'}</td>
              <td className="py-2 pr-4 text-secondary font-mono">{item.likely_correct ?? '\u2014'}</td>
              <td className="py-2 pr-4 text-on-surface-variant">{item.category ?? '\u2014'}</td>
              <td className="py-2 text-on-surface-variant font-mono">{item.timestamp ?? '\u2014'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function MlInferenceTable({ details }) {
  const items = Array.isArray(details) ? details : details?.corrections || []
  if (!items.length) return <JsonBlock data={details} />

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead>
          <tr className="border-b border-outline-variant/10">
            <th className="pb-2 pr-4 font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Term</th>
            <th className="pb-2 pr-4 font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Error</th>
            <th className="pb-2 pr-4 font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Corrected</th>
            <th className="pb-2 pr-4 font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Confidence</th>
            <th className="pb-2 font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Applied</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-outline-variant/5">
          {items.map((item, i) => (
            <tr key={i}>
              <td className="py-2 pr-4 text-primary font-mono">{item.term ?? '\u2014'}</td>
              <td className="py-2 pr-4 text-error font-mono">{item.error ?? '\u2014'}</td>
              <td className="py-2 pr-4 text-secondary font-mono">{item.corrected ?? '\u2014'}</td>
              <td className="py-2 pr-4 text-on-surface-variant font-mono">
                {item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : '\u2014'}
              </td>
              <td className="py-2">
                {item.applied ? (
                  <span className="text-secondary text-[10px] font-label">YES</span>
                ) : (
                  <span className="text-on-surface-variant text-[10px] font-label">NO</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function OcrExtraction({ details }) {
  const samples = details?.text_samples || details?.samples || (typeof details === 'string' ? [details] : null)
  if (!samples) return <JsonBlock data={details} />

  return (
    <div className="space-y-2">
      <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant mb-2">OCR Text Samples</p>
      {(Array.isArray(samples) ? samples : [samples]).map((s, i) => (
        <pre
          key={i}
          className="p-3 rounded-lg text-xs text-on-surface-variant font-mono whitespace-pre-wrap break-words"
          style={{ backgroundColor: 'rgba(144, 144, 151, 0.05)' }}
        >
          {typeof s === 'string' ? s : JSON.stringify(s, null, 2)}
        </pre>
      ))}
    </div>
  )
}

function JsonBlock({ data }) {
  if (data == null) return <p className="text-xs text-on-surface-variant">No details available.</p>

  return (
    <pre
      className="p-3 rounded-lg text-xs text-on-surface-variant font-mono whitespace-pre-wrap break-words overflow-x-auto"
      style={{ backgroundColor: 'rgba(144, 144, 151, 0.05)' }}
    >
      {typeof data === 'string' ? data : JSON.stringify(data, null, 2)}
    </pre>
  )
}

/* ---------- Detail Renderer ---------- */

function StepDetails({ step }) {
  const { name, details } = step
  switch (name) {
    case 'candidate_detection':
      return <CandidateDetectionTable details={details} />
    case 'ml_inference':
      return <MlInferenceTable details={details} />
    case 'ocr_extraction':
      return <OcrExtraction details={details} />
    default:
      return <JsonBlock data={details} />
  }
}

/* ---------- Main Panel ---------- */

export default function PipelineDetailPanel({ step, onClose }) {
  const [visible, setVisible] = useState(false)
  const panelRef = useRef(null)

  useEffect(() => {
    if (step) {
      requestAnimationFrame(() => setVisible(true))
    } else {
      setVisible(false)
    }
  }, [step])

  if (!step) return null

  const status = step.status || 'pending'
  const color = STATUS_COLORS[status] || STATUS_COLORS.pending
  const label = STEP_LABELS[step.name] || step.name

  return (
    <div
      ref={panelRef}
      className="overflow-hidden transition-all duration-300 ease-in-out"
      style={{
        maxHeight: visible ? '600px' : '0px',
        opacity: visible ? 1 : 0,
      }}
    >
      <div className="flux-card rounded-xl p-6 mt-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
            <h3 className="font-headline text-sm font-bold text-on-surface">{label}</h3>
            <span
              className="text-[10px] font-label uppercase tracking-wider px-2 py-0.5 rounded-full"
              style={{ color, backgroundColor: `${color}15` }}
            >
              {status}
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-surface-bright transition-colors text-on-surface-variant"
          >
            <X size={16} />
          </button>
        </div>

        {/* Timestamp */}
        <p className="text-[10px] text-on-surface-variant mb-4 font-mono">
          Updated: {formatTimestamp(step.updated_at)}
        </p>

        {/* Step-specific content */}
        <StepDetails step={step} />
      </div>
    </div>
  )
}
