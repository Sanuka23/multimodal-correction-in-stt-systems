import { useState, useCallback } from 'react'
import {
  ShieldCheck,
  Bot,
  FileText,
  FileSpreadsheet,
  Lightbulb,
  ChevronDown,
  Loader2,
} from 'lucide-react'
import { useCorrections } from '../api/queries'
import { useEvaluate } from '../api/mutations'

/* ── helpers ── */

function fmtDate(utc) {
  if (!utc) return ''
  return new Date(utc).toLocaleString()
}

function pct(n, d) {
  if (!d) return 0
  return Math.round((n / d) * 100)
}

/* ── build diff segments from evaluate response ── */

function buildDiffSegments(diff) {
  if (!Array.isArray(diff) || diff.length === 0) return { refBlocks: [], hypBlocks: [] }

  const refBlocks = []
  const hypBlocks = []

  for (const entry of diff) {
    switch (entry.type) {
      case 'equal':
        refBlocks.push({ type: 'text', value: entry.ref_word })
        hypBlocks.push({ type: 'text', value: entry.hyp_word || entry.ref_word })
        break
      case 'substitute':
        refBlocks.push({ type: 'removed', value: entry.ref_word })
        hypBlocks.push({ type: 'added', value: entry.hyp_word })
        break
      case 'delete':
        refBlocks.push({ type: 'removed', value: entry.ref_word })
        break
      case 'insert':
        hypBlocks.push({ type: 'added', value: entry.hyp_word })
        break
      default:
        refBlocks.push({ type: 'text', value: entry.ref_word || '' })
        hypBlocks.push({ type: 'text', value: entry.hyp_word || '' })
    }
  }

  return { refBlocks, hypBlocks }
}

/* ── inline segment renderer ── */

function DiffWord({ seg }) {
  switch (seg.type) {
    case 'removed':
      return <span className="diff-removed">{seg.value} </span>
    case 'added':
      return <span className="diff-added">{seg.value} </span>
    default:
      return <>{seg.value} </>
  }
}

/* ── main component ── */

export default function Compare() {
  /* --- state --- */
  const [selectedId, setSelectedId] = useState('__manual__')
  const [referenceText, setReferenceText] = useState('')
  const [hypothesisText, setHypothesisText] = useState('')
  const [evalResult, setEvalResult] = useState(null)
  const [selectedCorrection, setSelectedCorrection] = useState(null)
  const [historyOpen, setHistoryOpen] = useState(false)

  /* --- API hooks --- */
  const { data: corrections, isLoading: correctionsLoading } = useCorrections(50)
  const evaluate = useEvaluate()

  /* --- run evaluation --- */
  const runEvaluation = useCallback(
    (refText, hypText) => {
      if (!refText || !hypText) return
      evaluate.mutate(
        { reference_text: refText, hypothesis_text: hypText },
        {
          onSuccess: (data) => setEvalResult(data),
        },
      )
    },
    [evaluate],
  )

  /* --- handle correction selection --- */
  const handleSelect = useCallback(
    (value) => {
      setHistoryOpen(false)
      if (value === '__manual__') {
        setSelectedId('__manual__')
        setReferenceText('')
        setHypothesisText('')
        setEvalResult(null)
        setSelectedCorrection(null)
        return
      }
      const corr = (corrections || []).find((c) => c.file_id === value)
      if (!corr) return
      setSelectedId(value)
      setSelectedCorrection(corr)
      setReferenceText(corr.original_text || '')
      setHypothesisText(corr.enhanced_text || '')
      runEvaluation(corr.original_text, corr.enhanced_text)
    },
    [corrections, runEvaluation],
  )

  /* --- derived values --- */
  const werValue = evalResult ? `${(evalResult.wer * 100).toFixed(2)}%` : '--'
  const cerValue = evalResult ? `${(evalResult.cer * 100).toFixed(2)}%` : '--'
  const werBar = evalResult ? evalResult.wer * 100 : 0
  const cerBar = evalResult ? evalResult.cer * 100 : 0

  const totalErrors =
    evalResult
      ? (evalResult.substitutions || 0) + (evalResult.deletions || 0) + (evalResult.insertions || 0)
      : 0

  const errors = evalResult
    ? [
        {
          label: 'Substitutions',
          count: evalResult.substitutions || 0,
          pct: pct(evalResult.substitutions || 0, totalErrors),
          color: 'bg-primary',
        },
        {
          label: 'Deletions',
          count: evalResult.deletions || 0,
          pct: pct(evalResult.deletions || 0, totalErrors),
          color: 'bg-secondary',
        },
        {
          label: 'Insertions',
          count: evalResult.insertions || 0,
          pct: pct(evalResult.insertions || 0, totalErrors),
          color: 'bg-tertiary',
        },
      ]
    : []

  const metadata = evalResult
    ? [
        { key: 'ALGORITHM', value: 'Levenshtein Dist.' },
        { key: 'REF WORDS', value: String(evalResult.ref_words ?? '--') },
        { key: 'HYP WORDS', value: String(evalResult.hyp_words ?? '--') },
        { key: 'HITS', value: String(evalResult.hits ?? '--') },
      ]
    : []

  const { refBlocks, hypBlocks } = evalResult?.diff
    ? buildDiffSegments(evalResult.diff)
    : { refBlocks: [], hypBlocks: [] }

  /* --- recent corrections as evaluation cards --- */
  const recentEvaluations = (corrections || []).slice(0, 4).map((c, idx) => {
    const applied = c.corrections_applied ?? 0
    const attempted = c.corrections_attempted ?? 0
    const ratio = attempted > 0 ? Math.round((applied / attempted) * 100) : 0
    return {
      id: c._id || `${c.file_id}-${idx}`,
      title: c.file_id,
      wer: `${applied}/${attempted}`,
      werColor: ratio >= 80 ? 'text-secondary' : ratio >= 50 ? 'text-primary' : 'text-error',
      meta: fmtDate(c.created_at) + (c.processing_time_ms ? ` \u2022 ${c.processing_time_ms}ms` : ''),
      border: ratio < 50 ? 'border-error/20' : 'border-transparent hover:border-outline-variant',
    }
  })

  /* --- dropdown options --- */
  const correctionOptions = (corrections || []).map((c) => ({
    value: c.file_id,
    label: `${c.file_id}  \u2014  ${fmtDate(c.created_at)}  (${c.corrections_applied ?? 0} applied)`,
  }))

  return (
    <div>
      {/* -- Header Section -- */}
      <section className="mb-8 flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4">
        <div>
          <h2 className="font-headline text-4xl font-extrabold text-on-surface tracking-tight mb-2">
            Transcript Diff
          </h2>
          <p className="text-on-surface-variant font-body max-w-xl">
            Compare original and enhanced transcripts side-by-side with WER / CER evaluation.
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

      {/* -- Metrics Overview -- */}
      <section className="grid grid-cols-12 gap-6 mb-8">
        {/* Global WER */}
        <div className="col-span-12 md:col-span-3 bg-surface-container-low p-6 rounded-xl border-l-4 border-primary">
          <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant mb-1">
            Global WER
          </p>
          <div className="flex items-baseline gap-2">
            <span className="text-4xl font-label font-bold text-primary">{werValue}</span>
            {evaluate.isPending && <Loader2 size={14} className="animate-spin text-primary" />}
          </div>
          <div className="w-full bg-surface-container-highest h-1 mt-4 rounded-full overflow-hidden">
            <div
              className="bg-primary h-full rounded-full transition-all duration-300"
              style={{ width: `${Math.min(werBar, 100)}%` }}
            />
          </div>
        </div>

        {/* Accuracy (CER) */}
        <div className="col-span-12 md:col-span-3 bg-surface-container-low p-6 rounded-xl border-l-4 border-secondary">
          <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant mb-1">
            CER
          </p>
          <div className="flex items-baseline gap-2">
            <span className="text-4xl font-label font-bold text-secondary">{cerValue}</span>
          </div>
          <div className="w-full bg-surface-container-highest h-1 mt-4 rounded-full overflow-hidden">
            <div
              className="bg-secondary h-full rounded-full transition-all duration-300"
              style={{ width: `${Math.min(cerBar, 100)}%` }}
            />
          </div>
        </div>

        {/* TTER (from evaluate response, if present) */}
        <div className="col-span-12 md:col-span-6 bg-surface-container-low p-6 rounded-xl">
          <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant mb-4">
            Error Breakdown
          </p>
          {evalResult ? (
            <div className="flex items-end gap-4 h-16">
              {[
                { key: 'substitutions', val: evalResult.substitutions || 0 },
                { key: 'deletions', val: evalResult.deletions || 0 },
                { key: 'insertions', val: evalResult.insertions || 0 },
                { key: 'hits', val: evalResult.hits || 0 },
              ].map((cat) => {
                const maxVal = Math.max(
                  evalResult.substitutions || 0,
                  evalResult.deletions || 0,
                  evalResult.insertions || 0,
                  evalResult.hits || 0,
                  1,
                )
                const barH = Math.max((cat.val / maxVal) * 100, 4)
                return (
                  <div key={cat.key} className="flex-1 flex flex-col items-center gap-2">
                    <div
                      className="w-full bg-primary/20 rounded-t-lg relative"
                      style={{ height: `${barH}%` }}
                    >
                      <div
                        className="absolute bottom-0 w-full bg-primary rounded-t-lg transition-all duration-300"
                        style={{ height: '100%' }}
                      />
                    </div>
                    <span className="text-[9px] font-mono text-slate-500 uppercase">
                      {cat.key} ({cat.val})
                    </span>
                  </div>
                )
              })}
            </div>
          ) : (
            <p className="text-xs text-slate-500">Select a correction or run a comparison to see breakdown.</p>
          )}
        </div>
      </section>

      {/* -- Main Comparison View -- */}
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
                  {correctionsLoading ? 'LOADING...' : 'LOAD FROM HISTORY'}
                  <ChevronDown
                    size={12}
                    className={`transition-transform ${historyOpen ? 'rotate-180' : ''}`}
                  />
                </button>
                {historyOpen && (
                  <div className="absolute right-0 top-full mt-2 w-80 bg-surface-variant border border-outline-variant rounded-xl shadow-2xl z-10 py-2 max-h-64 overflow-y-auto">
                    <button
                      onClick={() => handleSelect('__manual__')}
                      className="block w-full text-left px-4 py-2 text-[10px] font-mono text-on-surface-variant hover:bg-surface-container-highest hover:text-primary transition-colors"
                    >
                      MANUAL INPUT
                    </button>
                    {correctionOptions.map((opt) => (
                      <button
                        key={opt.value}
                        onClick={() => handleSelect(opt.value)}
                        className={`block w-full text-left px-4 py-2 text-[10px] font-mono hover:bg-surface-container-highest hover:text-primary transition-colors ${
                          selectedId === opt.value
                            ? 'text-primary bg-primary/5'
                            : 'text-on-surface-variant'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                    {correctionOptions.length === 0 && !correctionsLoading && (
                      <p className="px-4 py-2 text-[10px] text-slate-500">No corrections found.</p>
                    )}
                  </div>
                )}
              </div>
              <button
                onClick={() => handleSelect('__manual__')}
                className="text-[10px] font-mono bg-primary/10 px-4 py-2 rounded-xl border border-primary/30 text-primary hover:bg-primary/20 transition-all"
              >
                PASTE NEW
              </button>
            </div>
          </div>

          {/* Manual input fields (only in manual mode) */}
          {selectedId === '__manual__' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-[10px] font-label uppercase tracking-widest text-slate-400 mb-2">
                  Reference Text
                </label>
                <textarea
                  value={referenceText}
                  onChange={(e) => setReferenceText(e.target.value)}
                  rows={4}
                  placeholder="Paste reference (ground truth) text here..."
                  className="w-full bg-surface-container-low border border-outline-variant/50 rounded-xl p-4 text-sm font-body text-on-surface-variant placeholder:text-slate-600 resize-none focus:outline-none focus:border-primary/50"
                />
              </div>
              <div>
                <label className="block text-[10px] font-label uppercase tracking-widest text-slate-400 mb-2">
                  Hypothesis Text
                </label>
                <textarea
                  value={hypothesisText}
                  onChange={(e) => setHypothesisText(e.target.value)}
                  rows={4}
                  placeholder="Paste hypothesis (ASR output) text here..."
                  className="w-full bg-surface-container-low border border-outline-variant/50 rounded-xl p-4 text-sm font-body text-on-surface-variant placeholder:text-slate-600 resize-none focus:outline-none focus:border-primary/50"
                />
              </div>
              <div className="md:col-span-2">
                <button
                  onClick={() => runEvaluation(referenceText, hypothesisText)}
                  disabled={!referenceText || !hypothesisText || evaluate.isPending}
                  className="flex items-center gap-2 px-6 py-2 bg-primary text-on-primary rounded-xl text-xs font-label tracking-widest uppercase transition-all disabled:opacity-40 disabled:cursor-not-allowed hover:bg-primary/90"
                >
                  {evaluate.isPending && <Loader2 size={14} className="animate-spin" />}
                  Compare
                </button>
              </div>
            </div>
          )}

          {/* Two-column diff panel */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-outline-variant/50 rounded-xl overflow-hidden border border-outline-variant/50 shadow-2xl">
            {/* Source Column (Ground Truth / Reference) */}
            <div className="bg-surface-container-low p-8">
              <div className="flex items-center gap-2 mb-6">
                <ShieldCheck size={14} className="text-slate-500" />
                <span className="text-[10px] font-label uppercase tracking-widest text-slate-500">
                  Ground Truth (Reference)
                </span>
              </div>
              <div className="space-y-8 text-sm font-body leading-relaxed text-on-surface-variant">
                {refBlocks.length > 0 ? (
                  <p>
                    {refBlocks.map((seg, i) => (
                      <DiffWord key={i} seg={seg} />
                    ))}
                  </p>
                ) : referenceText ? (
                  <p>{referenceText}</p>
                ) : (
                  <p className="text-slate-600 italic">
                    Select a correction from history or paste text to compare.
                  </p>
                )}
              </div>
            </div>

            {/* Inference Column (Hypothesis / Diff View) */}
            <div className="bg-surface-container-high p-8 border-l border-outline-variant/50">
              <div className="flex items-center gap-2 mb-6">
                <Bot size={14} className="text-primary" />
                <span className="text-[10px] font-label uppercase tracking-widest text-primary">
                  Model Inference (Hypothesis)
                </span>
              </div>
              <div className="space-y-8 text-sm font-body leading-relaxed text-on-surface-variant">
                {hypBlocks.length > 0 ? (
                  <p>
                    {hypBlocks.map((seg, i) => (
                      <DiffWord key={i} seg={seg} />
                    ))}
                  </p>
                ) : hypothesisText ? (
                  <p>{hypothesisText}</p>
                ) : (
                  <p className="text-slate-600 italic">
                    Enhanced text will appear here after comparison.
                  </p>
                )}
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
            {errors.length > 0 ? (
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
            ) : (
              <p className="text-xs text-slate-500">No evaluation data yet.</p>
            )}
          </div>

          {/* Comparison Metadata */}
          {metadata.length > 0 && (
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
          )}

          {/* Correction Details (when loaded from history) */}
          {selectedCorrection?.corrections && selectedCorrection.corrections.length > 0 && (
            <div className="bg-surface-variant p-6 rounded-xl border border-outline-variant/50">
              <h3 className="text-xs font-label uppercase tracking-widest text-slate-400 mb-4">
                Correction Details
              </h3>
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {selectedCorrection.corrections.map((c, i) => (
                  <div
                    key={i}
                    className={`p-3 rounded-lg text-[11px] font-mono ${
                      c.applied
                        ? 'bg-secondary/10 border border-secondary/20'
                        : 'bg-surface-container-highest border border-outline-variant/30'
                    }`}
                  >
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-bold text-on-surface">{c.term}</span>
                      <span
                        className={`text-[9px] px-1.5 py-0.5 rounded ${
                          c.applied
                            ? 'bg-secondary/20 text-secondary'
                            : 'bg-slate-700 text-slate-400'
                        }`}
                      >
                        {c.applied ? 'APPLIED' : 'SKIPPED'}
                      </span>
                    </div>
                    {c.category && (
                      <p className="text-slate-500 mb-1">
                        Category: <span className="text-on-surface-variant">{c.category}</span>
                      </p>
                    )}
                    {c.error_found && (
                      <p className="text-slate-500 mb-1">
                        Error: <span className="text-on-surface-variant">{c.error_found}</span>
                      </p>
                    )}
                    {c.changes && (
                      <p className="text-slate-500 mb-1">
                        Changes: <span className="text-on-surface-variant">{c.changes}</span>
                      </p>
                    )}
                    {c.confidence != null && (
                      <p className="text-slate-500">
                        Confidence:{' '}
                        <span className="text-on-surface-variant">
                          {(c.confidence * 100).toFixed(0)}%
                        </span>
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Precision Alert */}
          {evalResult && totalErrors > 0 && (
            <div className="bg-primary-container p-5 rounded-xl flex gap-4 items-start border border-primary/20">
              <Lightbulb size={20} className="text-primary shrink-0 mt-0.5" />
              <div>
                <p className="text-xs font-headline font-bold text-primary mb-1">
                  Evaluation Summary
                </p>
                <p className="text-[11px] text-on-surface-variant leading-relaxed">
                  Found <strong className="text-on-surface">{totalErrors} errors</strong> across{' '}
                  {evalResult.ref_words ?? '?'} reference words:{' '}
                  {evalResult.substitutions ?? 0} substitutions,{' '}
                  {evalResult.deletions ?? 0} deletions,{' '}
                  {evalResult.insertions ?? 0} insertions.
                </p>
              </div>
            </div>
          )}

          {evaluate.isError && (
            <div className="bg-error/10 border border-error/20 p-4 rounded-xl">
              <p className="text-xs text-error font-mono">
                Evaluation failed: {evaluate.error?.message || 'Unknown error'}
              </p>
            </div>
          )}
        </div>
      </section>

      {/* -- Recent Evaluations -- */}
      <section className="mb-12">
        <h3 className="text-xs font-label uppercase tracking-widest text-slate-400 mb-6">
          Recent Corrections
        </h3>
        {recentEvaluations.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {recentEvaluations.map((ev) => (
              <div
                key={ev.id}
                onClick={() => handleSelect(ev.id)}
                className={`bg-surface-container-low p-5 rounded-xl hover:bg-surface-container transition-all group cursor-pointer border ${
                  ev.border || 'border-transparent hover:border-outline-variant'
                }`}
              >
                <div className="flex justify-between items-start mb-3">
                  <span className="text-[10px] font-mono text-slate-500 truncate max-w-[60%]">
                    {ev.id}
                  </span>
                  <span className={`text-xs font-bold ${ev.werColor}`}>
                    {ev.wer} applied
                  </span>
                </div>
                <p className="text-sm font-headline font-bold text-on-surface mb-1 truncate">
                  {ev.title}
                </p>
                <p className="text-[10px] text-slate-500 uppercase font-label">{ev.meta}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-slate-500">
            {correctionsLoading ? 'Loading corrections...' : 'No recent corrections found.'}
          </p>
        )}
      </section>
    </div>
  )
}
