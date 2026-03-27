import { useState } from 'react'
import {
  ShieldCheck,
  Bot,
  FileText,
  FileSpreadsheet,
  Lightbulb,
  ChevronDown,
} from 'lucide-react'
import { usePolling } from '../hooks/usePolling'
import api from '../api/client'

/* ── static placeholder data (wire to API later) ── */

const PLACEHOLDER_METRICS = {
  wer: { value: '4.82%', delta: '0.4%', bar: 4.82 },
  cer: { value: '98.1%', label: 'STABLE', bar: 98.1 },
  tter: [
    { key: 'person_name', pct: 12 },
    { key: 'tech_term', pct: 48 },
    { key: 'product_name', pct: 72 },
    { key: 'tech_acronym', pct: 35 },
    { key: 'compliance', pct: 18 },
  ],
}

const PLACEHOLDER_GROUND_TRUTH = [
  {
    ts: '00:12',
    text: 'The patient presents with acute myocardial infarction symptoms. Vital signs are stable but tachycardia is noted.',
  },
  {
    ts: '00:45',
    text: 'Proceed with intravenous administration of aspirin 325mg and call for cardiology consult immediately.',
  },
  {
    ts: '01:10',
    text: 'Patient history indicates no prior allergies to beta blockers or ACE inhibitors.',
  },
  {
    ts: '01:35',
    text: 'Oxygen saturation is 94% on room air, suggesting mild hypoxemia.',
  },
]

const PLACEHOLDER_INFERENCE = [
  {
    ts: '00:12',
    segments: [
      { type: 'text', value: 'The patient presents with ' },
      { type: 'removed', value: 'acute myocardial infarction' },
      { type: 'added', value: 'a cute myo-cardial infarction' },
      { type: 'hint', value: 'OCR: ScreenApp at 0:12' },
      { type: 'text', value: ' symptoms. Vital signs are stable but ' },
      { type: 'removed', value: 'tachycardia' },
      { type: 'added', value: 'tack-y-cardia' },
      { type: 'text', value: ' is noted.' },
    ],
  },
  {
    ts: '00:45',
    segments: [
      { type: 'text', value: 'Proceed with ' },
      { type: 'removed', value: 'intravenous' },
      { type: 'added', value: 'intra-venous' },
      { type: 'vocab', value: '[ScreenApp | 0.94]' },
      { type: 'text', value: ' administration of ' },
      { type: 'removed', value: 'aspirin' },
      { type: 'added', value: 'as-pirin' },
      { type: 'text', value: ' 325mg and call for cardiology consult immediately.' },
    ],
  },
  {
    ts: '01:10',
    segments: [
      { type: 'text', value: 'Patient history indicates no prior allergies to ' },
      { type: 'removed', value: 'beta blockers' },
      { type: 'added', value: 'better blockers' },
      { type: 'vocab', value: '[PharmaDB | 0.88]' },
      { type: 'text', value: ' or ACE inhibitors.' },
    ],
  },
  {
    ts: '01:35',
    segments: [
      { type: 'text', value: 'Oxygen saturation is 94% on room air, suggesting ' },
      { type: 'removed', value: 'mild hypoxemia' },
      { type: 'added', value: 'mildly hypoxemic' },
      { type: 'hint', value: 'AVSR: Mouth Tracking' },
      { type: 'text', value: '.' },
    ],
  },
]

const PLACEHOLDER_ERRORS = [
  { label: 'Substitutions', count: 12, pct: 64, color: 'bg-primary' },
  { label: 'Deletions', count: 4, pct: 22, color: 'bg-secondary' },
  { label: 'Insertions', count: 3, pct: 14, color: 'bg-tertiary' },
]

const PLACEHOLDER_METADATA = [
  { key: 'ALGORITHM', value: 'Levenshtein Dist.' },
  { key: 'NORMALIZATION', value: 'Lower + Punct.' },
  { key: 'SEGMENTS', value: '1,402 Blocks' },
  { key: 'EXECUTION', value: '0.042ms' },
]

const PLACEHOLDER_EVALUATIONS = [
  { id: '#8821', title: 'E-Commerce Live Stream', wer: '3.2%', werColor: 'text-primary', meta: '2 hours ago \u2022 PCM 44kHz' },
  { id: '#8819', title: 'Technical Lecture: AI', wer: '1.8%', werColor: 'text-secondary', meta: 'Yesterday \u2022 Opus 16kHz' },
  { id: '#8815', title: 'Street Noise Stress Test', wer: '12.4%', werColor: 'text-error', meta: '2 days ago \u2022 AAC 48kHz', border: 'border-error/20' },
  { id: '#8812', title: 'Corporate Meeting A', wer: '4.1%', werColor: 'text-primary', meta: '3 days ago \u2022 PCM 44kHz' },
]

const HISTORY_OPTIONS = [
  'FILE_ID: A8821_MED',
  'FILE_ID: B4412_ENG',
  'FILE_ID: C0091_LEG',
]

/* ── inline segment renderer ── */

function InferenceSegment({ seg }) {
  switch (seg.type) {
    case 'text':
      return <>{seg.value}</>
    case 'removed':
      return <span className="diff-removed">{seg.value}</span>
    case 'added':
      return <span className="diff-added">{seg.value}</span>
    case 'hint':
      return <span className="hint-tag">{seg.value}</span>
    case 'vocab':
      return <span className="vocab-context">{seg.value}</span>
    default:
      return <>{seg.value}</>
  }
}

/* ── main component ── */

export default function Compare() {
  const [metrics] = useState(PLACEHOLDER_METRICS)
  const [groundTruth] = useState(PLACEHOLDER_GROUND_TRUTH)
  const [inference] = useState(PLACEHOLDER_INFERENCE)
  const [errors] = useState(PLACEHOLDER_ERRORS)
  const [metadata] = useState(PLACEHOLDER_METADATA)
  const [evaluations] = useState(PLACEHOLDER_EVALUATIONS)
  const [historyOpen, setHistoryOpen] = useState(false)

  /*
   * Data fetching stub -- uncomment when the compare endpoint is ready:
   *
   * const fetchData = async () => {
   *   try {
   *     const [corrRes, evalRes] = await Promise.all([
   *       api.get('/api/corrections?limit=8').catch(() => ({ data: [] })),
   *       api.get('/api/evaluations').catch(() => ({ data: [] })),
   *     ])
   *     // setCorrections(corrRes.data)
   *     // setEvaluations(evalRes.data)
   *   } catch {}
   * }
   * usePolling(fetchData, 15000)
   */

  return (
    <div>
      {/* ── Header Section ── */}
      <section className="mb-8 flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4">
        <div>
          <h2 className="font-headline text-4xl font-extrabold text-on-surface tracking-tight mb-2">
            Transcript Diff
          </h2>
          <p className="text-on-surface-variant font-body max-w-xl">
            Deep linguistic comparison between Model Inference (v2.4-beta) and Ground Truth (Human Verified).
          </p>
        </div>
        <div className="flex gap-3">
          <button className="flex items-center gap-2 px-4 py-2 bg-surface-container-high hover:bg-surface-bright text-on-surface-variant rounded-xl text-xs font-label tracking-widest uppercase transition-all">
            <FileText size={14} /> Export PDF
          </button>
          <button className="flex items-center gap-2 px-4 py-2 bg-surface-container-high hover:bg-surface-bright text-on-surface-variant rounded-xl text-xs font-label tracking-widest uppercase transition-all">
            <FileSpreadsheet size={14} /> Export CSV
          </button>
        </div>
      </section>

      {/* ── Metrics Overview ── */}
      <section className="grid grid-cols-12 gap-6 mb-8">
        {/* Global WER */}
        <div className="col-span-12 md:col-span-3 bg-surface-container-low p-6 rounded-xl border-l-4 border-primary">
          <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant mb-1">Global WER</p>
          <div className="flex items-baseline gap-2">
            <span className="text-4xl font-label font-bold text-primary">{metrics.wer.value}</span>
            <span className="text-xs text-secondary font-mono">&darr; {metrics.wer.delta}</span>
          </div>
          <div className="w-full bg-surface-container-highest h-1 mt-4 rounded-full overflow-hidden">
            <div className="bg-primary h-full rounded-full transition-all duration-300" style={{ width: `${metrics.wer.bar}%` }} />
          </div>
        </div>

        {/* Accuracy (CER) */}
        <div className="col-span-12 md:col-span-3 bg-surface-container-low p-6 rounded-xl border-l-4 border-secondary">
          <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant mb-1">Accuracy (CER)</p>
          <div className="flex items-baseline gap-2">
            <span className="text-4xl font-label font-bold text-secondary">{metrics.cer.value}</span>
            <span className="text-xs text-slate-500 font-mono">{metrics.cer.label}</span>
          </div>
          <div className="w-full bg-surface-container-highest h-1 mt-4 rounded-full overflow-hidden">
            <div className="bg-secondary h-full rounded-full transition-all duration-300" style={{ width: `${metrics.cer.bar}%` }} />
          </div>
        </div>

        {/* TTER Breakdown */}
        <div className="col-span-12 md:col-span-6 bg-surface-container-low p-6 rounded-xl">
          <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant mb-4">
            Per-Category TTER Breakdown
          </p>
          <div className="flex items-end gap-4 h-16">
            {metrics.tter.map((cat) => (
              <div key={cat.key} className="flex-1 flex flex-col items-center gap-2">
                <div
                  className="w-full bg-primary/20 rounded-t-lg relative"
                  style={{ height: `${cat.pct}%` }}
                >
                  <div
                    className="absolute bottom-0 w-full bg-primary rounded-t-lg transition-all duration-300"
                    style={{ height: '100%' }}
                  />
                </div>
                <span className="text-[9px] font-mono text-slate-500 uppercase">{cat.key}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Main Comparison View ── */}
      <section className="grid grid-cols-12 gap-8 mb-12">
        {/* Left: Side-by-side comparison */}
        <div className="col-span-12 xl:col-span-8 space-y-4">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-xs font-label uppercase tracking-widest text-slate-400">
              Analysis Workspace
            </h3>
            <div className="flex gap-2">
              {/* History dropdown */}
              <div className="relative">
                <button
                  onClick={() => setHistoryOpen(!historyOpen)}
                  className="flex items-center gap-2 text-[10px] font-mono bg-surface-container px-4 py-2 rounded-xl border border-outline-variant/30 text-on-surface-variant hover:text-primary hover:border-primary/50 transition-all"
                >
                  LOAD FROM HISTORY
                  <ChevronDown size={12} className={`transition-transform ${historyOpen ? 'rotate-180' : ''}`} />
                </button>
                {historyOpen && (
                  <div className="absolute right-0 top-full mt-2 w-48 bg-surface-variant border border-outline-variant rounded-xl shadow-2xl z-10 py-2">
                    {HISTORY_OPTIONS.map((opt) => (
                      <button
                        key={opt}
                        onClick={() => setHistoryOpen(false)}
                        className="block w-full text-left px-4 py-2 text-[10px] font-mono text-on-surface-variant hover:bg-surface-container-highest hover:text-primary transition-colors"
                      >
                        {opt}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <button className="text-[10px] font-mono bg-primary/10 px-4 py-2 rounded-xl border border-primary/30 text-primary hover:bg-primary/20 transition-all">
                PASTE NEW
              </button>
            </div>
          </div>

          {/* Two-column diff panel */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-outline-variant/50 rounded-xl overflow-hidden border border-outline-variant/50 shadow-2xl">
            {/* Source Column (Ground Truth) */}
            <div className="bg-surface-container-low p-8">
              <div className="flex items-center gap-2 mb-6">
                <ShieldCheck size={14} className="text-slate-500" />
                <span className="text-[10px] font-label uppercase tracking-widest text-slate-500">
                  Ground Truth (Reference)
                </span>
              </div>
              <div className="space-y-8 text-sm font-body leading-relaxed text-on-surface-variant">
                {groundTruth.map((block) => (
                  <p key={block.ts}>
                    <button className="text-primary hover:underline font-mono text-xs mr-2 transition-all">
                      [{block.ts}]
                    </button>
                    {block.text}
                  </p>
                ))}
              </div>
            </div>

            {/* Inference Column (Diff View) */}
            <div className="bg-surface-container-high p-8 border-l border-outline-variant/50">
              <div className="flex items-center gap-2 mb-6">
                <Bot size={14} className="text-primary" />
                <span className="text-[10px] font-label uppercase tracking-widest text-primary">
                  Model Inference (Hypothesis)
                </span>
              </div>
              <div className="space-y-8 text-sm font-body leading-relaxed text-on-surface-variant">
                {inference.map((block) => (
                  <div key={block.ts}>
                    <button className="text-primary hover:underline font-mono text-xs mr-2 transition-all">
                      [{block.ts}]
                    </button>
                    {block.segments.map((seg, i) => (
                      <InferenceSegment key={i} seg={seg} />
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Right sidebar */}
        <div className="col-span-12 xl:col-span-4 space-y-6">
          {/* Error Distribution */}
          <div className="bg-surface-container-low p-6 rounded-xl">
            <h3 className="text-xs font-label uppercase tracking-widest text-slate-400 mb-6">
              Error Distribution
            </h3>
            <div className="space-y-4">
              {errors.map((err) => (
                <div key={err.label}>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs font-mono text-slate-500">{err.label}</span>
                    <span className="text-xs font-mono text-on-surface">
                      {err.count} ({err.pct}%)
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-surface-container-highest rounded-full">
                    <div
                      className={`h-full ${err.color} rounded-full transition-all duration-300`}
                      style={{ width: `${err.pct}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Comparison Metadata */}
          <div className="bg-surface-variant p-6 rounded-xl border border-outline-variant/50">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xs font-label uppercase tracking-widest text-slate-400">
                Comparison Metadata
              </h3>
              <span className="text-[10px] font-mono text-secondary bg-secondary/10 px-2 py-0.5 rounded">
                ALIGNED
              </span>
            </div>
            <ul className="space-y-3 font-mono text-[11px]">
              {metadata.map((item, i) => (
                <li
                  key={item.key}
                  className={`flex justify-between ${
                    i < metadata.length - 1 ? 'border-b border-outline-variant pb-2' : ''
                  }`}
                >
                  <span className="text-slate-500">{item.key}</span>
                  <span className="text-on-surface-variant">{item.value}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Precision Alert */}
          <div className="bg-primary-container p-5 rounded-xl flex gap-4 items-start border border-primary/20">
            <Lightbulb size={20} className="text-primary shrink-0 mt-0.5" />
            <div>
              <p className="text-xs font-headline font-bold text-primary mb-1">
                Precision Alert: Phonetic Ambiguity
              </p>
              <p className="text-[11px] text-on-surface-variant leading-relaxed">
                Model v2.4-beta exhibits consistent confusion between homophones in high-noise environments.{' '}
                <strong className="text-on-surface">Example:</strong> &quot;Better blockers&quot; vs &quot;Beta blockers&quot;.
                Suggested mitigation: Increase Beam Search width to 10 or enable Specialized Vocabulary
                &apos;Medical-Standard-2024&apos;.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Recent Evaluations ── */}
      <section className="mb-12">
        <h3 className="text-xs font-label uppercase tracking-widest text-slate-400 mb-6">
          Recent Evaluations
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {evaluations.map((ev) => (
            <div
              key={ev.id}
              className={`bg-surface-container-low p-5 rounded-xl hover:bg-surface-container transition-all group cursor-pointer border ${
                ev.border || 'border-transparent hover:border-outline-variant'
              }`}
            >
              <div className="flex justify-between items-start mb-3">
                <span className="text-[10px] font-mono text-slate-500">Job {ev.id}</span>
                <span className={`text-xs font-bold ${ev.werColor}`}>{ev.wer} WER</span>
              </div>
              <p className="text-sm font-headline font-bold text-on-surface mb-1">{ev.title}</p>
              <p className="text-[10px] text-slate-500 uppercase font-label">{ev.meta}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
