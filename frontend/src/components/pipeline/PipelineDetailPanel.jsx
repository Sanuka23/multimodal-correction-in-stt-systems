import { useRef, useEffect, useState } from 'react'
import { X } from 'lucide-react'
import { STEP_LABELS } from './PipelineNode'
import { formatFullTimestamp } from '../../utils/datetime'

const STATUS_COLORS = {
  completed: '#4edea3',
  running: '#ffd480',
  failed: '#ffb4ab',
  pending: '#909097',
  skipped: '#909097',
}

function formatTimestamp(ts) {
  return formatFullTimestamp(ts) || '\u2014'
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

/* ---------- New step renderers ---------- */

function PillList({ items, color = '#7bd0ff' }) {
  if (!items || !items.length) {
    return <p className="text-xs text-on-surface-variant/60 italic">None.</p>
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((it, i) => (
        <span
          key={i}
          className="text-[11px] font-mono px-2 py-0.5 rounded-md"
          style={{ color, backgroundColor: `${color}18`, border: `1px solid ${color}33` }}
        >
          {typeof it === 'string' ? it : it.term || JSON.stringify(it)}
          {it && it.category ? <span className="opacity-60 ml-1">· {it.category}</span> : null}
        </span>
      ))}
    </div>
  )
}

function KeyValueGrid({ pairs }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
      {pairs.filter(([, v]) => v != null && v !== '').map(([k, v]) => (
        <div key={k} className="bg-surface-container-low rounded-lg p-2.5 border border-outline-variant/10">
          <p className="text-[9px] font-label uppercase tracking-widest text-on-surface-variant/70">{k}</p>
          <p className="text-[12px] font-mono text-on-surface mt-0.5 break-words">
            {typeof v === 'object' ? JSON.stringify(v) : String(v)}
          </p>
        </div>
      ))}
    </div>
  )
}

function VocabMergeView({ details }) {
  return (
    <KeyValueGrid pairs={[
      ['Domain terms', details.domain_terms],
      ['Custom terms', details.custom_terms],
      ['Merged total', details.merged_terms],
    ]} />
  )
}

function ModelLoadView({ details }) {
  return (
    <KeyValueGrid pairs={[
      ['Base model', details.base_model],
      ['Backend', details.backend],
      ['Adapter', details.adapter_path],
      ['Loaded', details.loaded ? 'yes' : 'no'],
    ]} />
  )
}

function TopicClassificationView({ details }) {
  return (
    <div className="space-y-3">
      <KeyValueGrid pairs={[
        ['Field', details.field],
        ['Topic', details.topic],
      ]} />
      {details.description && (
        <div>
          <p className="text-[9px] font-label uppercase tracking-widest text-on-surface-variant/70 mb-1">Description</p>
          <p className="text-xs text-on-surface-variant leading-relaxed">{details.description}</p>
        </div>
      )}
      {details.suggested_vocab?.length > 0 && (
        <div>
          <p className="text-[9px] font-label uppercase tracking-widest text-on-surface-variant/70 mb-1.5">
            Suggested vocab ({details.suggested_vocab.length})
          </p>
          <PillList items={details.suggested_vocab} color="#ffb95f" />
        </div>
      )}
    </div>
  )
}

function WebVocabView({ details }) {
  return (
    <div className="space-y-3">
      <KeyValueGrid pairs={[['Terms added', details.terms_added]]} />
      {details.terms?.length > 0 ? (
        <PillList items={details.terms} color="#7bd0ff" />
      ) : (
        <p className="text-xs text-on-surface-variant/60 italic">No web terms accepted.</p>
      )}
    </div>
  )
}

function OcrVocabView({ details }) {
  return (
    <div className="space-y-3">
      <KeyValueGrid pairs={[['Terms extracted', details.terms_extracted]]} />
      {details.terms?.length > 0 ? (
        <PillList items={details.terms} color="#4edea3" />
      ) : (
        <p className="text-xs text-on-surface-variant/60 italic">No structured vocab extracted.</p>
      )}
    </div>
  )
}

function WhisperPass2View({ details }) {
  return (
    <KeyValueGrid pairs={[
      ['Segments re-transcribed', details.segments_re_transcribed],
      ['Model', details.model],
      ['Device', details.device],
      ['Compute', details.compute_type],
    ]} />
  )
}

function AvsrView({ details }) {
  return (
    <div className="space-y-3">
      <KeyValueGrid pairs={[
        ['Mode', details.mode],
        ['Hints emitted', details.hints],
        ['Lip transcripts', details.lip_transcripts],
      ]} />
      {details.samples?.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-[9px] font-label uppercase tracking-widest text-on-surface-variant/70">Samples</p>
          {details.samples.map((s, i) => (
            <div key={i} className="text-[11px] font-mono bg-surface-container-low rounded-lg p-2 border border-outline-variant/10">
              <span className="text-tertiary">[{s.start?.toFixed?.(1) ?? s.start}–{s.end?.toFixed?.(1) ?? s.end}s]</span>{' '}
              <span className="text-on-surface-variant">{s.hint}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ReconciliationView({ details }) {
  return (
    <div className="space-y-3">
      <KeyValueGrid pairs={[['Swaps', details.swaps]]} />
      {details.sample_swaps?.length > 0 && (
        <div className="space-y-1">
          <p className="text-[9px] font-label uppercase tracking-widest text-on-surface-variant/70">Sample swaps</p>
          {details.sample_swaps.map((s, i) => (
            <p key={i} className="text-[11px] font-mono text-on-surface-variant">{s}</p>
          ))}
        </div>
      )}
    </div>
  )
}

function CompletionView({ details }) {
  return (
    <div className="space-y-3">
      <KeyValueGrid pairs={[
        ['Total time', details.total_time_ms ? `${Math.round(details.total_time_ms)}ms` : null],
        ['Attempted', details.corrections_attempted],
        ['Applied', details.corrections_applied],
      ]} />
      {details.evidence_sources && (
        <div>
          <p className="text-[9px] font-label uppercase tracking-widest text-on-surface-variant/70 mb-1.5">Evidence sources</p>
          <KeyValueGrid pairs={Object.entries(details.evidence_sources)} />
        </div>
      )}
    </div>
  )
}

/* ---------- Detail Renderer ---------- */

function StepDetails({ step }) {
  const { name, details } = step
  if (!details) return <p className="text-xs text-on-surface-variant">No details available.</p>

  switch (name) {
    case 'candidate_detection':
      return <CandidateDetectionTable details={details} />
    case 'ml_inference':
      return <MlInferenceTable details={details} />
    case 'ocr_extraction':
      return <OcrExtraction details={details} />
    case 'model_load':
      return <ModelLoadView details={details} />
    case 'vocab_merge':
      return <VocabMergeView details={details} />
    case 'topic_classification':
      return <TopicClassificationView details={details} />
    case 'web_vocab_enrichment':
      return <WebVocabView details={details} />
    case 'ocr_vocab_extraction':
      return <OcrVocabView details={details} />
    case 'whisper_pass2':
      return <WhisperPass2View details={details} />
    case 'avsr_extraction':
    case 'avsr_pass2':
      return <AvsrView details={details} />
    case 'llm_reconciliation':
      return <ReconciliationView details={details} />
    case 'complete':
      return <CompletionView details={details} />
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
