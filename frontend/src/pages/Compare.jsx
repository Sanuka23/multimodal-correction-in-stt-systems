import { useState, useCallback, useEffect, useMemo, useRef, forwardRef } from 'react'
import {
  ShieldCheck,
  Bot,
  FileText,
  FileSpreadsheet,
  Lightbulb,
  Loader2,
  GitCompare,
  Sparkles,
  Search,
  History,
  Database,
  Pencil,
  ArrowRight,
  Check,
  X,
  Filter,
  Columns2,
  AlignLeft,
  Inbox,
  Hash,
  Percent,
  Calculator,
  ChevronDown,
} from 'lucide-react'
import { useCorrections } from '../api/queries'
import api from '../api/client'
import { useEvaluate } from '../api/mutations'
import PageHeader from '../components/ui/PageHeader'
import StatCard from '../components/ui/StatCard'
import SectionCard from '../components/ui/SectionCard'
import { formatDateTime, timeAgo } from '../utils/datetime'

/* ── helpers ─────────────────────────────────────────────────────── */

const fmtDate = formatDateTime

function pct(n, d) {
  if (!d) return 0
  return Math.round((n / d) * 100)
}

function fmtPctValue(v) {
  if (v == null || isNaN(v)) return '—'
  return `${(v * 100).toFixed(1)}%`
}

/* ── build diff segments from evaluate response ──────────────────── */

function buildDiffSegments(diff) {
  if (!Array.isArray(diff) || diff.length === 0) return { refBlocks: [], hypBlocks: [], unifiedBlocks: [] }

  const refBlocks = []
  const hypBlocks = []
  const unifiedBlocks = []
  let groupIdx = 0

  for (const entry of diff) {
    switch (entry.type) {
      case 'equal':
        refBlocks.push({ type: 'text', value: entry.ref_word })
        hypBlocks.push({ type: 'text', value: entry.hyp_word || entry.ref_word })
        unifiedBlocks.push({ type: 'text', value: entry.ref_word })
        break
      case 'substitute':
        refBlocks.push({ type: 'removed', value: entry.ref_word, group: groupIdx })
        hypBlocks.push({ type: 'added', value: entry.hyp_word, group: groupIdx })
        unifiedBlocks.push({ type: 'sub', from: entry.ref_word, to: entry.hyp_word, group: groupIdx })
        groupIdx++
        break
      case 'delete':
        refBlocks.push({ type: 'removed', value: entry.ref_word, group: groupIdx })
        unifiedBlocks.push({ type: 'del', value: entry.ref_word, group: groupIdx })
        groupIdx++
        break
      case 'insert':
        hypBlocks.push({ type: 'added', value: entry.hyp_word, group: groupIdx })
        unifiedBlocks.push({ type: 'ins', value: entry.hyp_word, group: groupIdx })
        groupIdx++
        break
      default:
        refBlocks.push({ type: 'text', value: entry.ref_word || '' })
        hypBlocks.push({ type: 'text', value: entry.hyp_word || '' })
        unifiedBlocks.push({ type: 'text', value: entry.ref_word || '' })
    }
  }

  return { refBlocks, hypBlocks, unifiedBlocks }
}

/* ── inline diff word ────────────────────────────────────────────── */

function DiffWord({ seg, isHighlighted, onClick }) {
  const cls =
    seg.type === 'removed'
      ? `inline-block rounded px-1 mx-px text-error line-through decoration-error/60 bg-error/10 ${isHighlighted ? 'ring-2 ring-error/60 bg-error/25' : ''} ${onClick ? 'cursor-pointer hover:bg-error/20' : ''}`
      : seg.type === 'added'
        ? `inline-block rounded px-1 mx-px text-secondary bg-secondary/10 font-semibold ${isHighlighted ? 'ring-2 ring-secondary/60 bg-secondary/25' : ''} ${onClick ? 'cursor-pointer hover:bg-secondary/20' : ''}`
        : ''
  if (seg.type === 'text') return <>{seg.value} </>
  return (
    <span
      data-group={seg.group}
      className={cls}
      onClick={onClick}
    >
      {seg.value}
    </span>
  )
}

/* ── source-picker tab content ───────────────────────────────────── */

function SourceTab({ active, label, icon: Icon, count, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2 rounded-xl text-[12px] font-label tracking-wide transition-all ${
        active
          ? 'bg-primary/15 text-primary border border-primary/30 shadow-sm shadow-primary/10'
          : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high border border-transparent'
      }`}
    >
      <Icon size={13} />
      <span>{label}</span>
      {count != null && (
        <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded-full ${active ? 'bg-primary/25 text-primary' : 'bg-surface-container-high text-on-surface-variant/70'}`}>
          {count}
        </span>
      )}
    </button>
  )
}

/* ── main component ──────────────────────────────────────────────── */

export default function Compare() {
  /* --- state --- */
  const [selectedId, setSelectedId] = useState('__manual__')
  const [referenceText, setReferenceText] = useState('')
  const [hypothesisText, setHypothesisText] = useState('')
  const [evalResult, setEvalResult] = useState(null)
  const [selectedCorrection, setSelectedCorrection] = useState(null)
  const [sourceTab, setSourceTab] = useState('recent')          // recent | datasets | manual
  const [viewMode, setViewMode] = useState('corrections')        // corrections | transcript
  const [diffLayout, setDiffLayout] = useState('side-by-side')   // side-by-side | inline
  const [highlightedGroup, setHighlightedGroup] = useState(null)
  const [searchQ, setSearchQ] = useState('')
  const [filterApplied, setFilterApplied] = useState('all')      // all | applied | skipped

  const refScrollRef = useRef(null)
  const hypScrollRef = useRef(null)

  /* --- API hooks --- */
  const { data: corrections, isLoading: correctionsLoading } = useCorrections(50)
  const [amiMeetings, setAmiMeetings] = useState([])
  const [earningsCalls, setEarningsCalls] = useState([])
  const [slideavsrVideos, setSlideavsrVideos] = useState([])
  const [amiV2Meetings, setAmiV2Meetings] = useState([])
  const evaluate = useEvaluate()

  useEffect(() => {
    api.get('/api/ami/list').then(r => setAmiMeetings(r.data?.meetings || [])).catch(() => {})
    api.get('/api/earnings/list').then(r => setEarningsCalls(r.data?.meetings || [])).catch(() => {})
    api.get('/api/slideavsr/list').then(r => setSlideavsrVideos(r.data?.meetings || [])).catch(() => {})
    api.get('/api/ami_v2/list').then(r => setAmiV2Meetings(r.data?.meetings || [])).catch(() => {})
  }, [])

  /* --- run evaluation --- */
  const runEvaluation = useCallback(
    (refText, hypText) => {
      if (!refText || !hypText) return
      evaluate.mutate(
        { reference_text: refText, hypothesis_text: hypText },
        { onSuccess: (data) => setEvalResult(data) },
      )
    },
    [evaluate],
  )

  /* --- source loaders --- */
  const loadDataset = useCallback(async (kind, id) => {
    const url = `/api/${kind}/compare/${id}`
    setSelectedId(`${kind}::${id}`)
    setSelectedCorrection(null)
    try {
      const res = await api.get(url)
      const d = res.data
      setReferenceText(d.ground_truth || '')
      setHypothesisText(d.screenapp_text || '')
      runEvaluation(d.ground_truth, d.screenapp_text)
    } catch { /* ignore */ }
  }, [runEvaluation])

  const loadCorrection = useCallback((corr) => {
    if (!corr) return
    setSelectedId(corr.file_id)
    setSelectedCorrection(corr)
    setReferenceText(corr.original_text || '')
    setHypothesisText(corr.enhanced_text || '')
    runEvaluation(corr.original_text, corr.enhanced_text)
  }, [runEvaluation])

  const resetManual = useCallback(() => {
    setSelectedId('__manual__')
    setReferenceText('')
    setHypothesisText('')
    setEvalResult(null)
    setSelectedCorrection(null)
    setSourceTab('manual')
  }, [])

  /* --- derived values --- */
  const werValue = evalResult ? `${(evalResult.wer * 100).toFixed(2)}%` : '—'
  const cerValue = evalResult ? `${(evalResult.cer * 100).toFixed(2)}%` : '—'
  const werBar = evalResult ? evalResult.wer * 100 : 0
  const cerBar = evalResult ? evalResult.cer * 100 : 0

  const totalErrors = evalResult
    ? (evalResult.substitutions || 0) + (evalResult.deletions || 0) + (evalResult.insertions || 0)
    : 0

  const errorMix = evalResult ? [
    { label: 'Sub',  key: 'substitutions', count: evalResult.substitutions || 0, color: 'bg-primary',   text: 'text-primary' },
    { label: 'Del',  key: 'deletions',     count: evalResult.deletions || 0,     color: 'bg-error',     text: 'text-error' },
    { label: 'Ins',  key: 'insertions',    count: evalResult.insertions || 0,    color: 'bg-tertiary',  text: 'text-tertiary' },
  ] : []

  const hitsPct = evalResult && evalResult.ref_words
    ? Math.round(((evalResult.hits || 0) / evalResult.ref_words) * 100)
    : null

  const { refBlocks, hypBlocks, unifiedBlocks } = useMemo(
    () => evalResult?.diff
      ? buildDiffSegments(evalResult.diff)
      : { refBlocks: [], hypBlocks: [], unifiedBlocks: [] },
    [evalResult],
  )

  const recentCorrections = corrections || []

  const filteredCorrectionItems = useMemo(() => {
    const items = selectedCorrection?.corrections || []
    return items.filter((c) => {
      if (filterApplied === 'applied' && !c.applied) return false
      if (filterApplied === 'skipped' && c.applied) return false
      if (searchQ) {
        const hay = `${c.term || ''} ${c.error_found || ''} ${c.category || ''}`.toLowerCase()
        if (!hay.includes(searchQ.toLowerCase())) return false
      }
      return true
    })
  }, [selectedCorrection, filterApplied, searchQ])

  /* --- jump-to-group: scroll both panes to a diff group --- */
  const jumpToGroup = useCallback((group) => {
    setHighlightedGroup(group)
    requestAnimationFrame(() => {
      ;[refScrollRef.current, hypScrollRef.current].forEach((root) => {
        if (!root) return
        const el = root.querySelector(`[data-group="${group}"]`)
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
      })
    })
    // un-highlight after a moment
    window.setTimeout(() => setHighlightedGroup(null), 2200)
  }, [])

  /* --- counts for tabs --- */
  const counts = {
    recent:   recentCorrections.length,
    datasets: amiV2Meetings.length + slideavsrVideos.length + earningsCalls.length + amiMeetings.length,
  }

  /* ====================================================================
     Render
     =================================================================== */

  const hasComparison = !!evalResult
  const corrItems = selectedCorrection?.corrections || []
  const correctionStats = {
    applied:  corrItems.filter(c => c.applied).length,
    skipped:  corrItems.filter(c => !c.applied).length,
    total:    corrItems.length,
  }

  return (
    <div>
      <PageHeader
        eyebrow="Quality Inspector"
        title="Transcript Diff"
        description="Compare original and enhanced transcripts side-by-side. Inspect WER / CER, browse every correction, and audit applied edits."
        icon={GitCompare}
        actions={
          <>
            <button className="flex items-center gap-2 px-3.5 py-2 bg-surface-container-high hover:bg-surface-bright text-on-surface-variant rounded-xl text-[11px] font-label tracking-widest uppercase transition-all">
              <FileText size={13} /> PDF
            </button>
            <button className="flex items-center gap-2 px-3.5 py-2 bg-surface-container-high hover:bg-surface-bright text-on-surface-variant rounded-xl text-[11px] font-label tracking-widest uppercase transition-all">
              <FileSpreadsheet size={13} /> CSV
            </button>
          </>
        }
      />

      {/* ── 1. SOURCE BAR ─────────────────────────────────────────── */}
      <SectionCard tone="glass" className="mb-6" bodyClassName="p-4">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-1.5 text-[10px] font-label uppercase tracking-[0.28em] text-on-surface-variant/70 mr-2">
            <Sparkles size={12} className="text-primary" /> Source
          </div>
          <SourceTab active={sourceTab === 'recent'}   label="Recent"   icon={History}  count={counts.recent}   onClick={() => setSourceTab('recent')} />
          <SourceTab active={sourceTab === 'datasets'} label="Datasets" icon={Database} count={counts.datasets} onClick={() => setSourceTab('datasets')} />
          <SourceTab active={sourceTab === 'manual'}   label="Paste"    icon={Pencil}                          onClick={() => setSourceTab('manual')} />

          <div className="flex-1" />

          {hasComparison && (
            <div className="flex items-center gap-2 text-[11px] text-on-surface-variant">
              <span className="font-label uppercase tracking-widest text-[9px] text-on-surface-variant/60">Loaded:</span>
              <span className="font-mono text-on-surface bg-surface-container-high px-2 py-0.5 rounded-md truncate max-w-[260px]">
                {selectedCorrection?.file_id || selectedId.replace(/^.*::/, '') || '—'}
              </span>
              {selectedCorrection?.created_at && (
                <span
                  className="text-on-surface-variant/60 font-mono text-[10.5px]"
                  title={timeAgo(selectedCorrection.created_at)}
                >
                  {formatDateTime(selectedCorrection.created_at)}
                </span>
              )}
              <button onClick={resetManual} className="ml-2 text-on-surface-variant/70 hover:text-error transition-colors" title="Clear">
                <X size={14} />
              </button>
            </div>
          )}
        </div>

        {/* Tab body */}
        <div className="mt-4">
          {sourceTab === 'recent' && (
            <div>
              {correctionsLoading ? (
                <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-2">
                  {Array.from({ length: 4 }).map((_, i) => (
                    <div key={i} className="h-[68px] rounded-xl bg-surface-container-high animate-pulse" />
                  ))}
                </div>
              ) : recentCorrections.length === 0 ? (
                <p className="text-xs text-on-surface-variant/70 italic px-1">No recent correction jobs yet. Run a correction via the ScreenApp API to populate this list.</p>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-2 max-h-[280px] overflow-y-auto scrollbar-thin pr-1">
                  {recentCorrections.slice(0, 16).map((c, i) => {
                    const isSelected = selectedId === c.file_id
                    const ratio = c.corrections_attempted > 0
                      ? Math.round((c.corrections_applied / c.corrections_attempted) * 100)
                      : null
                    return (
                      <button
                        key={c._id || c.file_id || i}
                        onClick={() => loadCorrection(c)}
                        className={`text-left rounded-xl p-3 transition-all border ${
                          isSelected
                            ? 'bg-primary/10 border-primary/40 ring-1 ring-primary/30'
                            : 'bg-surface-container-high/60 border-outline-variant/10 hover:border-primary/20 hover:bg-surface-container-high'
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2 mb-1">
                          <span className="font-mono text-[11px] text-on-surface truncate max-w-[150px]">
                            {(c.file_id || '').slice(0, 14)}…
                          </span>
                          {ratio != null && (
                            <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded-full ${
                              ratio >= 80 ? 'bg-secondary/15 text-secondary'
                                          : ratio >= 50 ? 'bg-primary/15 text-primary'
                                                       : 'bg-tertiary/15 text-tertiary'
                            }`}>
                              {c.corrections_applied}/{c.corrections_attempted}
                            </span>
                          )}
                        </div>
                        <p
                          className="text-[10px] text-on-surface-variant/70 font-mono"
                          title={timeAgo(c.created_at)}
                        >
                          {formatDateTime(c.created_at)}
                          {c.processing_time_ms ? ` · ${Math.round(c.processing_time_ms)}ms` : ''}
                        </p>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          )}

          {sourceTab === 'datasets' && (
            <DatasetGrid
              amiV2Meetings={amiV2Meetings}
              slideavsrVideos={slideavsrVideos}
              earningsCalls={earningsCalls}
              amiMeetings={amiMeetings}
              selectedId={selectedId}
              onLoad={loadDataset}
            />
          )}

          {sourceTab === 'manual' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <label className="block text-[10px] font-label uppercase tracking-widest text-on-surface-variant/70 mb-1.5">
                  Reference (Ground Truth)
                </label>
                <textarea
                  value={referenceText}
                  onChange={(e) => setReferenceText(e.target.value)}
                  rows={4}
                  placeholder="Paste reference text…"
                  className="w-full bg-surface-container-low border border-outline-variant/20 rounded-xl p-3 text-[13px] font-body text-on-surface-variant placeholder:text-on-surface-variant/40 resize-none focus:outline-none focus:border-primary/50"
                />
              </div>
              <div>
                <label className="block text-[10px] font-label uppercase tracking-widest text-on-surface-variant/70 mb-1.5">
                  Hypothesis (ASR Output)
                </label>
                <textarea
                  value={hypothesisText}
                  onChange={(e) => setHypothesisText(e.target.value)}
                  rows={4}
                  placeholder="Paste ASR output…"
                  className="w-full bg-surface-container-low border border-outline-variant/20 rounded-xl p-3 text-[13px] font-body text-on-surface-variant placeholder:text-on-surface-variant/40 resize-none focus:outline-none focus:border-primary/50"
                />
              </div>
              <div className="md:col-span-2 flex justify-end">
                <button
                  onClick={() => runEvaluation(referenceText, hypothesisText)}
                  disabled={!referenceText || !hypothesisText || evaluate.isPending}
                  className="flex items-center gap-2 px-5 py-2 bg-primary text-on-primary rounded-xl text-xs font-label font-bold tracking-widest uppercase transition-all disabled:opacity-40 disabled:cursor-not-allowed hover:brightness-110"
                >
                  {evaluate.isPending && <Loader2 size={13} className="animate-spin" />}
                  Compare
                </button>
              </div>
            </div>
          )}
        </div>
      </SectionCard>

      {/* ── 2. METRICS STRIP ──────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard
          label="Word Error Rate"
          icon={Percent}
          tone="primary"
          accent="primary"
          value={
            <div className="flex items-baseline gap-2">
              <span>{werValue}</span>
              {evaluate.isPending && <Loader2 size={14} className="animate-spin text-primary" />}
            </div>
          }
          delta={
            <div className="w-full mt-2 h-1 bg-surface-container-high rounded-full overflow-hidden">
              <div className="h-full bg-primary rounded-full transition-all duration-500" style={{ width: `${Math.min(werBar, 100)}%` }} />
            </div>
          }
        />
        <StatCard
          label="Character Error Rate"
          icon={Hash}
          tone="secondary"
          accent="secondary"
          value={cerValue}
          delta={
            <div className="w-full mt-2 h-1 bg-surface-container-high rounded-full overflow-hidden">
              <div className="h-full bg-secondary rounded-full transition-all duration-500" style={{ width: `${Math.min(cerBar, 100)}%` }} />
            </div>
          }
        />
        <StatCard
          label="Errors"
          icon={Calculator}
          tone="tertiary"
          accent="tertiary"
          value={hasComparison ? totalErrors : '—'}
          delta={
            errorMix.length > 0 ? (
              <div className="flex gap-1.5 mt-1.5 flex-wrap">
                {errorMix.map((e) => (
                  <span key={e.key} className={`text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-surface-container-high ${e.text}`}>
                    {e.label} {e.count}
                  </span>
                ))}
              </div>
            ) : '—'
          }
        />
        <StatCard
          label="Hit Rate"
          icon={ShieldCheck}
          tone="default"
          value={hitsPct != null ? `${hitsPct}%` : '—'}
          delta={hasComparison
            ? <span><span className="text-secondary font-mono">{evalResult.hits ?? 0}</span> / {evalResult.ref_words ?? 0} words</span>
            : '—'}
        />
      </div>

      {/* ── 3. EMPTY STATE ────────────────────────────────────────── */}
      {!hasComparison && !evaluate.isPending && (
        <SectionCard tone="muted">
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
              <Inbox size={28} className="text-primary" strokeWidth={1.6} />
            </div>
            <h3 className="font-headline text-lg font-bold text-on-surface mb-1">Pick a source to begin</h3>
            <p className="text-sm text-on-surface-variant/80 max-w-md mb-5">
              Choose a recent correction job, load a public benchmark video, or paste any two transcripts above to run a side-by-side comparison.
            </p>
            <div className="flex gap-2 flex-wrap justify-center">
              <button onClick={() => setSourceTab('recent')}   className="px-4 py-2 bg-primary/10 text-primary rounded-xl text-xs font-label hover:bg-primary/15 transition">View recent jobs</button>
              <button onClick={() => setSourceTab('datasets')} className="px-4 py-2 bg-surface-container-high text-on-surface-variant rounded-xl text-xs font-label hover:bg-surface-bright transition">Browse datasets</button>
              <button onClick={() => setSourceTab('manual')}   className="px-4 py-2 bg-surface-container-high text-on-surface-variant rounded-xl text-xs font-label hover:bg-surface-bright transition">Paste manually</button>
            </div>
          </div>
        </SectionCard>
      )}

      {/* ── 4. MAIN VIEW (only when we have comparison data) ─────── */}
      {hasComparison && (
        <div className="grid grid-cols-12 gap-6">
          {/* Left: tabbed main panel */}
          <div className="col-span-12 xl:col-span-8 space-y-5">
            {/* View switcher */}
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div className="flex bg-surface-container-low rounded-xl p-1 border border-outline-variant/15">
                <button
                  onClick={() => setViewMode('corrections')}
                  className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-[11px] font-label font-bold uppercase tracking-widest transition-all ${
                    viewMode === 'corrections' ? 'bg-primary text-on-primary' : 'text-on-surface-variant hover:text-on-surface'
                  }`}
                >
                  <Sparkles size={12} /> Corrections
                  {corrItems.length > 0 && (
                    <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded-full ${
                      viewMode === 'corrections' ? 'bg-on-primary/15 text-on-primary' : 'bg-surface-container-high text-on-surface-variant'
                    }`}>
                      {corrItems.length}
                    </span>
                  )}
                </button>
                <button
                  onClick={() => setViewMode('transcript')}
                  className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-[11px] font-label font-bold uppercase tracking-widest transition-all ${
                    viewMode === 'transcript' ? 'bg-primary text-on-primary' : 'text-on-surface-variant hover:text-on-surface'
                  }`}
                >
                  <FileText size={12} /> Transcript Diff
                </button>
              </div>

              {viewMode === 'transcript' && (
                <div className="flex bg-surface-container-low rounded-xl p-1 border border-outline-variant/15">
                  <button
                    onClick={() => setDiffLayout('side-by-side')}
                    className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-[10px] font-label font-bold uppercase tracking-widest transition-all ${
                      diffLayout === 'side-by-side' ? 'bg-surface-container-high text-on-surface' : 'text-on-surface-variant hover:text-on-surface'
                    }`}
                  >
                    <Columns2 size={11} /> Side-by-side
                  </button>
                  <button
                    onClick={() => setDiffLayout('inline')}
                    className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-[10px] font-label font-bold uppercase tracking-widest transition-all ${
                      diffLayout === 'inline' ? 'bg-surface-container-high text-on-surface' : 'text-on-surface-variant hover:text-on-surface'
                    }`}
                  >
                    <AlignLeft size={11} /> Inline
                  </button>
                </div>
              )}
            </div>

            {/* CORRECTIONS view */}
            {viewMode === 'corrections' && (
              <SectionCard tone="default" bodyClassName="p-0">
                {/* Filter bar */}
                <div className="flex items-center gap-2 flex-wrap p-3 border-b border-outline-variant/10">
                  <div className="flex items-center gap-2 flex-1 min-w-[200px] bg-surface-container-low rounded-lg px-2.5 py-1.5 border border-outline-variant/15">
                    <Search size={12} className="text-on-surface-variant/70" />
                    <input
                      value={searchQ}
                      onChange={(e) => setSearchQ(e.target.value)}
                      placeholder="Search term, error, or category…"
                      className="bg-transparent text-[12px] text-on-surface placeholder:text-on-surface-variant/40 outline-none flex-1"
                    />
                  </div>
                  <div className="flex bg-surface-container-low rounded-lg p-0.5 border border-outline-variant/15">
                    {[
                      { k: 'all',     l: `All (${correctionStats.total})` },
                      { k: 'applied', l: `Applied (${correctionStats.applied})` },
                      { k: 'skipped', l: `Skipped (${correctionStats.skipped})` },
                    ].map(({ k, l }) => (
                      <button
                        key={k}
                        onClick={() => setFilterApplied(k)}
                        className={`px-2.5 py-1 rounded-md text-[10px] font-label font-bold uppercase tracking-wider transition-all ${
                          filterApplied === k ? 'bg-surface-container-high text-on-surface' : 'text-on-surface-variant/70 hover:text-on-surface'
                        }`}
                      >
                        {l}
                      </button>
                    ))}
                  </div>
                </div>

                {/* List */}
                <div className="p-2 max-h-[640px] overflow-y-auto scrollbar-thin">
                  {filteredCorrectionItems.length === 0 ? (
                    <div className="text-center py-10 text-on-surface-variant/60 text-[13px]">
                      {selectedCorrection
                        ? 'No corrections match the current filter.'
                        : 'Pick a correction job from "Recent" to see the per-correction list.'}
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {filteredCorrectionItems.map((c, i) => (
                        <CorrectionRow
                          key={i}
                          item={c}
                          onJump={() => {
                            // try to find the diff group by matching the swap text
                            const target = String(c.changes || '').split('→').pop()?.trim()
                            if (!target) return
                            // Find first added block whose value matches
                            const added = unifiedBlocks.find(b => (b.type === 'sub' ? b.to : b.value) === target)
                            if (added && added.group != null) jumpToGroup(added.group)
                            setViewMode('transcript')
                          }}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </SectionCard>
            )}

            {/* TRANSCRIPT view */}
            {viewMode === 'transcript' && (
              <>
                {diffLayout === 'side-by-side' ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <TranscriptPane
                      ref={refScrollRef}
                      side="ref"
                      blocks={refBlocks}
                      fallback={referenceText}
                      highlightedGroup={highlightedGroup}
                      onWordClick={(g) => g != null && jumpToGroup(g)}
                    />
                    <TranscriptPane
                      ref={hypScrollRef}
                      side="hyp"
                      blocks={hypBlocks}
                      fallback={hypothesisText}
                      highlightedGroup={highlightedGroup}
                      onWordClick={(g) => g != null && jumpToGroup(g)}
                    />
                  </div>
                ) : (
                  <UnifiedDiffPane
                    blocks={unifiedBlocks}
                    highlightedGroup={highlightedGroup}
                    onWordClick={(g) => g != null && jumpToGroup(g)}
                  />
                )}
              </>
            )}
          </div>

          {/* Right: contextual panel */}
          <div className="col-span-12 xl:col-span-4 space-y-4">
            <SectionCard
              eyebrow="Comparison"
              title="Metadata"
              icon={Calculator}
              actions={
                <span className="text-[9px] font-mono text-secondary bg-secondary/10 px-2 py-0.5 rounded-md font-bold">ALIGNED</span>
              }
            >
              <ul className="space-y-2 text-[12px]">
                <MetaRow label="Algorithm"  value="Levenshtein" />
                <MetaRow label="Ref words"  value={evalResult.ref_words ?? '—'} mono />
                <MetaRow label="Hyp words"  value={evalResult.hyp_words ?? '—'} mono />
                <MetaRow label="Hits"       value={evalResult.hits ?? '—'}      mono valueColor="text-secondary" />
                <MetaRow label="Errors"     value={totalErrors}                  mono valueColor="text-tertiary" />
              </ul>
            </SectionCard>

            <SectionCard
              eyebrow="Errors"
              title="Distribution"
              icon={Filter}
            >
              <div className="space-y-3">
                {errorMix.length === 0 ? (
                  <p className="text-xs text-on-surface-variant/60">No errors found.</p>
                ) : errorMix.map((err) => {
                  const p = pct(err.count, totalErrors)
                  return (
                    <div key={err.key}>
                      <div className="flex justify-between text-[11px] font-label mb-1">
                        <span className="text-on-surface-variant">{err.key}</span>
                        <span className={`font-mono ${err.text}`}>{err.count} <span className="text-on-surface-variant/60">({p}%)</span></span>
                      </div>
                      <div className="h-1.5 bg-surface-container-high rounded-full overflow-hidden">
                        <div className={`h-full ${err.color} rounded-full transition-all duration-500`} style={{ width: `${p}%` }} />
                      </div>
                    </div>
                  )
                })}
              </div>
            </SectionCard>

            {hasComparison && totalErrors > 0 && (
              <SectionCard tone="glass" accent="primary">
                <div className="flex gap-3 items-start">
                  <div className="w-8 h-8 rounded-xl bg-primary/15 text-primary flex items-center justify-center flex-shrink-0">
                    <Lightbulb size={16} />
                  </div>
                  <div className="min-w-0">
                    <p className="font-headline text-[13px] font-bold text-primary mb-1">Evaluation summary</p>
                    <p className="text-[12px] text-on-surface-variant leading-relaxed">
                      Found <strong className="text-on-surface">{totalErrors} errors</strong> over{' '}
                      <strong className="text-on-surface">{evalResult.ref_words ?? '?'}</strong> reference words —{' '}
                      <span className="text-primary">{evalResult.substitutions ?? 0} sub</span>,{' '}
                      <span className="text-error">{evalResult.deletions ?? 0} del</span>,{' '}
                      <span className="text-tertiary">{evalResult.insertions ?? 0} ins</span>.
                    </p>
                  </div>
                </div>
              </SectionCard>
            )}

            {evaluate.isError && (
              <SectionCard tone="default" accent="error">
                <p className="text-xs text-error font-mono">
                  Evaluation failed: {evaluate.error?.message || 'Unknown error'}
                </p>
              </SectionCard>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

/* ── sub-components ──────────────────────────────────────────────── */

function MetaRow({ label, value, mono, valueColor = 'text-on-surface' }) {
  return (
    <li className="flex justify-between items-center py-1 border-b border-outline-variant/10 last:border-b-0">
      <span className="text-on-surface-variant/70 text-[11px] font-label uppercase tracking-wider">{label}</span>
      <span className={`${mono ? 'font-mono' : ''} ${valueColor}`}>{value}</span>
    </li>
  )
}

function CorrectionRow({ item, onJump }) {
  const applied = !!item.applied
  const conf = item.confidence != null
    ? typeof item.confidence === 'number' && item.confidence <= 1
      ? Math.round(item.confidence * 100)
      : Number(item.confidence)
    : null

  const errFrom = item.error_found || (item.changes ? String(item.changes).split('→')[0]?.trim() : null)
  const errTo = item.term || (item.changes ? String(item.changes).split('→').pop()?.trim() : null)

  return (
    <button
      onClick={onJump}
      className={`w-full text-left rounded-xl p-3 border transition-all group flex items-stretch gap-3 ${
        applied
          ? 'bg-secondary/[0.06] border-secondary/20 hover:bg-secondary/[0.1] hover:border-secondary/30'
          : 'bg-surface-container-high/40 border-outline-variant/10 hover:bg-surface-container-high'
      }`}
    >
      {/* Applied indicator strip */}
      <span className={`w-1 rounded-full flex-shrink-0 ${applied ? 'bg-secondary' : 'bg-on-surface-variant/30'}`} />

      <div className="flex-1 min-w-0">
        {/* Top row: error → corrected */}
        <div className="flex items-center gap-2 flex-wrap">
          {errFrom && (
            <span className="font-mono text-[12px] px-1.5 py-0.5 rounded-md bg-error/10 text-error line-through decoration-error/40">
              {errFrom}
            </span>
          )}
          <ArrowRight size={12} className="text-on-surface-variant/50" />
          {errTo && (
            <span className="font-mono text-[12px] px-1.5 py-0.5 rounded-md bg-secondary/10 text-secondary font-semibold">
              {errTo}
            </span>
          )}
          {item.category && (
            <span className="ml-auto text-[9px] font-label uppercase tracking-widest px-2 py-0.5 rounded-full bg-surface-container-high text-on-surface-variant/80">
              {item.category}
            </span>
          )}
        </div>

        {/* Bottom row: confidence + status */}
        <div className="flex items-center gap-3 mt-2">
          {conf != null && (
            <div className="flex items-center gap-1.5">
              <div className="w-16 h-1 bg-surface-container-high rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    conf >= 90 ? 'bg-secondary' : conf >= 70 ? 'bg-primary' : 'bg-tertiary'
                  }`}
                  style={{ width: `${Math.min(conf, 100)}%` }}
                />
              </div>
              <span className="font-mono text-[10px] text-on-surface-variant">{conf}%</span>
            </div>
          )}
          <span className={`flex items-center gap-1 text-[10px] font-label uppercase tracking-widest ${
            applied ? 'text-secondary' : 'text-on-surface-variant/60'
          }`}>
            {applied ? <Check size={11} /> : <X size={11} />}
            {applied ? 'Applied' : 'Skipped'}
          </span>
          <span className="ml-auto text-[10px] text-on-surface-variant/50 font-label tracking-widest uppercase opacity-0 group-hover:opacity-100 transition-opacity">
            View in transcript →
          </span>
        </div>
      </div>
    </button>
  )
}

const TranscriptPane = forwardRef(function TranscriptPane(
  { side, blocks, fallback, highlightedGroup, onWordClick },
  ref,
) {
  const isRef = side === 'ref'
  return (
    <SectionCard
      tone={isRef ? 'muted' : 'default'}
      accent={isRef ? undefined : 'primary'}
      eyebrow={isRef ? 'Original Transcript' : 'Enhanced (Hypothesis)'}
      title={isRef ? 'Source' : 'Inference Output'}
      icon={isRef ? ShieldCheck : Bot}
      bodyClassName="p-0"
    >
      <div
        ref={ref}
        className="p-5 max-h-[640px] overflow-y-auto scrollbar-thin text-[13px] leading-relaxed text-on-surface font-body"
      >
        {blocks.length > 0 ? (
          <p>
            {blocks.map((seg, i) => (
              <DiffWord
                key={i}
                seg={seg}
                isHighlighted={seg.group != null && seg.group === highlightedGroup}
                onClick={seg.group != null ? () => onWordClick(seg.group) : null}
              />
            ))}
          </p>
        ) : fallback ? (
          <p>{fallback}</p>
        ) : (
          <p className="text-on-surface-variant/50 italic">Nothing here yet.</p>
        )}
      </div>
    </SectionCard>
  )
})

function UnifiedDiffPane({ blocks, highlightedGroup, onWordClick }) {
  return (
    <SectionCard
      tone="default"
      accent="primary"
      eyebrow="Inline Diff"
      title="Unified View"
      icon={AlignLeft}
      bodyClassName="p-0"
    >
      <div className="p-5 max-h-[640px] overflow-y-auto scrollbar-thin text-[13px] leading-loose text-on-surface font-body">
        {blocks.length > 0 ? (
          <p>
            {blocks.map((b, i) => {
              if (b.type === 'text') return <span key={i}>{b.value} </span>
              const high = b.group != null && b.group === highlightedGroup
              if (b.type === 'sub') {
                return (
                  <span
                    key={i}
                    data-group={b.group}
                    onClick={() => onWordClick(b.group)}
                    className={`inline-block mx-0.5 cursor-pointer ${high ? 'ring-2 ring-primary/60 rounded-md' : ''}`}
                    title="Substitution"
                  >
                    <span className="rounded-l px-1 bg-error/10 text-error line-through decoration-error/50">{b.from}</span>
                    <span className="rounded-r px-1 bg-secondary/10 text-secondary font-semibold">{b.to}</span>{' '}
                  </span>
                )
              }
              if (b.type === 'del') {
                return (
                  <span key={i} data-group={b.group} onClick={() => onWordClick(b.group)}
                    className={`inline-block mx-px px-1 rounded cursor-pointer bg-error/10 text-error line-through decoration-error/50 ${high ? 'ring-2 ring-error/50' : ''}`}
                    title="Deletion">
                    {b.value}{' '}
                  </span>
                )
              }
              return (
                <span key={i} data-group={b.group} onClick={() => onWordClick(b.group)}
                  className={`inline-block mx-px px-1 rounded cursor-pointer bg-secondary/10 text-secondary font-semibold ${high ? 'ring-2 ring-secondary/50' : ''}`}
                  title="Insertion">
                  {b.value}{' '}
                </span>
              )
            })}
          </p>
        ) : (
          <p className="text-on-surface-variant/50 italic">No diff data.</p>
        )}
      </div>
    </SectionCard>
  )
}

/* ── DatasetGrid sub-component ───────────────────────────────────── */

function DatasetGrid({ amiV2Meetings, slideavsrVideos, earningsCalls, amiMeetings, selectedId, onLoad }) {
  const buckets = [
    { kind: 'ami_v2',    label: 'AMI v2',     accent: 'primary',   items: amiV2Meetings.map(m => ({ id: m.meeting_id, sub: `WER ${m.baseline_wer}%` })) },
    { kind: 'slideavsr', label: 'SlideAVSR',  accent: 'secondary', items: slideavsrVideos.map(m => ({ id: m.file_id || m.meeting_id, sub: `WER ${m.baseline_wer}%` })) },
    { kind: 'earnings',  label: 'Earnings22', accent: 'tertiary',  items: earningsCalls.map(m => ({ id: m.meeting_id, sub: `${m.accent} · WER ${m.baseline_wer}%` })) },
    { kind: 'ami',       label: 'AMI',        accent: 'primary',   items: amiMeetings.map(m => ({ id: m.meeting_id, sub: `WER ${m.baseline_wer}%` })) },
  ]

  const accentText = { primary: 'text-primary', secondary: 'text-secondary', tertiary: 'text-tertiary' }
  const accentBg   = { primary: 'bg-primary',   secondary: 'bg-secondary',   tertiary: 'bg-tertiary' }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 max-h-[320px] overflow-y-auto scrollbar-thin pr-1">
      {buckets.map(({ kind, label, accent, items }) => (
        <div key={kind} className="bg-surface-container-low rounded-xl p-3 border border-outline-variant/10">
          <div className="flex items-center gap-2 mb-2">
            <span className={`w-1.5 h-1.5 rounded-full ${accentBg[accent]}`} />
            <span className="font-label text-[10px] uppercase tracking-[0.22em] text-on-surface-variant/80">{label}</span>
            <span className="ml-auto text-[10px] font-mono text-on-surface-variant/60">{items.length}</span>
          </div>
          {items.length === 0 ? (
            <p className="text-[11px] text-on-surface-variant/50 italic">Empty.</p>
          ) : (
            <div className="space-y-1">
              {items.slice(0, 8).map((it) => {
                const sel = selectedId === `${kind}::${it.id}`
                return (
                  <button
                    key={it.id}
                    onClick={() => onLoad(kind, it.id)}
                    className={`w-full text-left rounded-lg px-2 py-1.5 text-[11px] transition-all ${
                      sel ? `bg-${accent}/10 ${accentText[accent]} ring-1 ring-${accent}/30` :
                            'text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface'
                    }`}
                  >
                    <div className="font-mono truncate">{it.id}</div>
                    <div className={`text-[9px] font-label uppercase tracking-widest ${sel ? accentText[accent] : 'text-on-surface-variant/60'}`}>{it.sub}</div>
                  </button>
                )
              })}
              {items.length > 8 && (
                <p className="text-[10px] text-on-surface-variant/50 px-2">+ {items.length - 8} more…</p>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
