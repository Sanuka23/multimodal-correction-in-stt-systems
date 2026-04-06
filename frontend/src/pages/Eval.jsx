import { useState, useMemo } from 'react'
import { BarChart3, ArrowUpRight, ArrowDownRight, Clock, Video, ChevronUp, ChevronDown } from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { useEvalResults } from '../api/queries'

function fmt(v, decimals = 2) {
  if (v == null || isNaN(v)) return '\u2014'
  return Number(v).toFixed(decimals)
}

function fmtPct(v, decimals = 2) {
  if (v == null || isNaN(v)) return '\u2014'
  return `${Number(v).toFixed(decimals)}%`
}

function fmtMs(v) {
  if (v == null || isNaN(v)) return '\u2014'
  return `${Math.round(Number(v))}ms`
}

function ImprovementCell({ value }) {
  if (value == null || isNaN(value)) return <span className="text-on-surface-variant">{'\u2014'}</span>
  const num = Number(value)
  const isPositive = num > 0
  return (
    <span className={`flex items-center gap-1 font-label font-bold ${isPositive ? 'text-secondary' : 'text-error'}`}>
      {isPositive ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
      {isPositive ? '+' : ''}
      {fmt(num)}%
    </span>
  )
}

const CHART_TOOLTIP_STYLE = {
  contentStyle: {
    background: '#131b2e',
    border: '1px solid rgba(144,144,151,0.2)',
    borderRadius: '8px',
    fontSize: '11px',
    color: '#dae2fd',
  },
}

export default function Eval() {
  const [version, setVersion] = useState('v1')
  const [sortKey, setSortKey] = useState(null)
  const [sortDir, setSortDir] = useState('asc')

  const { data, isLoading, error } = useEvalResults(version)

  const summary = data?.summary || null
  const perVideo = useMemo(() => {
    const raw = data?.per_video || data?.results || []
    return Array.isArray(raw) ? raw : []
  }, [data])

  const accentBreakdown = useMemo(() => {
    const raw = data?.accent_breakdown || []
    return Array.isArray(raw) ? raw : Object.entries(raw || {}).map(([k, v]) => ({ accent: k, ...v }))
  }, [data])

  const videoEval = useMemo(() => {
    const raw = data?.video_eval || data?.youtube_eval || []
    return Array.isArray(raw) ? raw : []
  }, [data])

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const sortedPerVideo = useMemo(() => {
    if (!sortKey) return perVideo
    return [...perVideo].sort((a, b) => {
      const av = a[sortKey] ?? 0
      const bv = b[sortKey] ?? 0
      if (typeof av === 'string') return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
      return sortDir === 'asc' ? av - bv : bv - av
    })
  }, [perVideo, sortKey, sortDir])

  const chartData = useMemo(() => {
    return perVideo.map((v) => ({
      name: (v.title || v.video_id || '').substring(0, 20),
      baseline: Number(v.baseline_wer ?? v.wer_baseline ?? 0),
      corrected: Number(v.corrected_wer ?? v.wer_corrected ?? 0),
    }))
  }, [perVideo])

  const SortIcon = ({ field }) => {
    if (sortKey !== field) return null
    return sortDir === 'asc' ? (
      <ChevronUp size={10} className="inline ml-0.5" />
    ) : (
      <ChevronDown size={10} className="inline ml-0.5" />
    )
  }

  const werImprovement = summary
    ? (summary.avg_baseline_wer ?? 0) - (summary.avg_corrected_wer ?? 0)
    : null

  return (
    <div>
      {/* Header */}
      <header className="mb-8 flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <BarChart3 size={20} className="text-primary" />
            <h2 className="font-headline text-3xl font-extrabold text-on-surface tracking-tight">
              Evaluation Results
            </h2>
          </div>
          <p className="text-on-surface-variant font-body max-w-xl">
            Compare model performance across versions with per-video WER analysis, accent breakdowns, and latency metrics.
          </p>
        </div>

        {/* Version Toggle */}
        <div className="flex bg-surface-container-high rounded-lg p-1 border border-outline-variant/20">
          {['v1', 'v2'].map((v) => (
            <button
              key={v}
              onClick={() => {
                setVersion(v)
                setSortKey(null)
              }}
              className={`px-5 py-2 text-xs font-label font-bold uppercase tracking-widest rounded-md transition-all ${
                version === v
                  ? 'bg-primary text-on-primary'
                  : 'text-on-surface-variant hover:text-on-surface'
              }`}
            >
              {v.toUpperCase()} Results
            </button>
          ))}
        </div>
      </header>

      {/* Loading / Error */}
      {isLoading && (
        <div className="flex items-center justify-center py-20">
          <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          <span className="ml-3 text-on-surface-variant text-sm font-label">Loading results...</span>
        </div>
      )}

      {error && !isLoading && (
        <div className="flux-card rounded-xl p-8 text-center">
          <p className="text-error text-sm font-label">Failed to load evaluation results.</p>
          <p className="text-on-surface-variant text-xs mt-2">
            {error?.message || 'Please check the API connection.'}
          </p>
        </div>
      )}

      {!isLoading && !error && !summary && perVideo.length === 0 && (
        <div className="flux-card rounded-xl p-12 text-center">
          <BarChart3 size={32} className="text-on-surface-variant mx-auto mb-4 opacity-40" />
          <p className="text-on-surface-variant font-body text-sm">
            No evaluation data available for {version.toUpperCase()}.
          </p>
        </div>
      )}

      {/* Summary Cards */}
      {summary && (
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
          {/* Videos Evaluated */}
          <div className="flux-card rounded-xl p-5">
            <div className="flex justify-between items-start mb-2">
              <span className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest">
                Videos Evaluated
              </span>
              <Video size={14} className="text-primary" />
            </div>
            <div className="text-2xl font-headline font-extrabold text-on-surface">
              {summary.n_videos ?? perVideo.length ?? '\u2014'}
            </div>
          </div>

          {/* Avg Baseline WER */}
          <div className="flux-card rounded-xl p-5">
            <span className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest block mb-2">
              Avg Baseline WER
            </span>
            <div className="text-2xl font-headline font-extrabold text-on-surface">
              {fmtPct(summary.avg_baseline_wer)}
            </div>
          </div>

          {/* Avg Corrected WER */}
          <div className="flux-card rounded-xl p-5">
            <span className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest block mb-2">
              Avg Corrected WER
            </span>
            <div className="text-2xl font-headline font-extrabold text-primary">
              {fmtPct(summary.avg_corrected_wer)}
            </div>
          </div>

          {/* WER Improvement */}
          <div className="flux-card rounded-xl p-5">
            <span className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest block mb-2">
              WER Improvement
            </span>
            <div className="text-2xl font-headline font-extrabold">
              <ImprovementCell value={werImprovement} />
            </div>
          </div>

          {/* Latency p50 */}
          <div className="flux-card rounded-xl p-5">
            <div className="flex justify-between items-start mb-2">
              <span className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest">
                Latency p50
              </span>
              <Clock size={14} className="text-tertiary" />
            </div>
            <div className="text-2xl font-headline font-extrabold text-on-surface">
              {fmtMs(summary.latency_p50 ?? summary.p50_latency_ms)}
            </div>
          </div>
        </section>
      )}

      {/* Per-Video Results Table */}
      {sortedPerVideo.length > 0 && (
        <section className="mb-8">
          <h3 className="font-headline text-lg font-bold text-on-surface mb-4">Per-Video Results</h3>
          <div className="flux-card rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest border-b border-outline-variant/20">
                    {[
                      { key: 'title', label: 'Title' },
                      { key: 'accent', label: 'Accent' },
                      { key: 'domain', label: 'Domain' },
                      { key: 'baseline_wer', label: 'Baseline WER' },
                      { key: 'corrected_wer', label: 'Corrected WER' },
                      { key: 'improvement', label: 'Improvement' },
                      { key: 'latency_ms', label: 'Latency' },
                    ].map((col) => (
                      <th
                        key={col.key}
                        onClick={() => handleSort(col.key)}
                        className="py-3 px-4 cursor-pointer hover:text-primary transition-colors select-none"
                      >
                        {col.label}
                        <SortIcon field={col.key} />
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="text-sm font-body">
                  {sortedPerVideo.map((row, idx) => {
                    const baselineWer = row.baseline_wer ?? row.wer_baseline
                    const correctedWer = row.corrected_wer ?? row.wer_corrected
                    const improvement = row.improvement ?? (baselineWer != null && correctedWer != null
                      ? baselineWer - correctedWer
                      : null)
                    const latency = row.latency_ms ?? row.latency

                    return (
                      <tr
                        key={row.video_id || idx}
                        className="border-b border-outline-variant/10 hover:bg-surface-container-high/50 transition-colors"
                      >
                        <td className="py-3 px-4 text-on-surface text-xs font-label font-medium truncate max-w-[200px]">
                          {row.title || row.video_id || '\u2014'}
                        </td>
                        <td className="py-3 px-4 text-on-surface-variant text-xs font-label">
                          {row.accent || '\u2014'}
                        </td>
                        <td className="py-3 px-4 text-on-surface-variant text-xs font-label">
                          {row.domain || '\u2014'}
                        </td>
                        <td className="py-3 px-4 text-on-surface-variant text-xs font-label">
                          {fmtPct(baselineWer)}
                        </td>
                        <td className="py-3 px-4 text-primary text-xs font-label font-bold">
                          {fmtPct(correctedWer)}
                        </td>
                        <td className="py-3 px-4">
                          <ImprovementCell value={improvement} />
                        </td>
                        <td className="py-3 px-4 text-on-surface-variant text-xs font-label">
                          {fmtMs(latency)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}

      {/* WER Comparison Bar Chart */}
      {chartData.length > 0 && (
        <section className="mb-8">
          <h3 className="font-headline text-lg font-bold text-on-surface mb-4">WER Comparison</h3>
          <div className="flux-card rounded-xl p-6">
            <ResponsiveContainer width="100%" height={Math.max(300, chartData.length * 40)}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 20, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3449" horizontal={false} />
                <XAxis
                  type="number"
                  tick={{ fontSize: 10, fill: '#c6c6cd' }}
                  axisLine={{ stroke: '#2d3449' }}
                  tickLine={false}
                  unit="%"
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fontSize: 10, fill: '#c6c6cd' }}
                  axisLine={false}
                  tickLine={false}
                  width={120}
                />
                <Tooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => `${Number(v).toFixed(2)}%`} />
                <Legend
                  wrapperStyle={{ fontSize: '11px', color: '#c6c6cd' }}
                  iconType="rect"
                  iconSize={10}
                />
                <Bar dataKey="baseline" name="Baseline WER" fill="#2d3449" radius={[0, 4, 4, 0]} barSize={14} />
                <Bar dataKey="corrected" name="Corrected WER" fill="#7bd0ff" radius={[0, 4, 4, 0]} barSize={14} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      {/* Accent Breakdown */}
      {accentBreakdown.length > 0 && (
        <section className="mb-8">
          <h3 className="font-headline text-lg font-bold text-on-surface mb-4">Accent Breakdown</h3>
          <div className="flux-card rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest border-b border-outline-variant/20">
                    <th className="py-3 px-4">Accent</th>
                    <th className="py-3 px-4">N</th>
                    <th className="py-3 px-4">Avg Baseline WER</th>
                    <th className="py-3 px-4">Avg Corrected WER</th>
                    <th className="py-3 px-4">Improvement</th>
                  </tr>
                </thead>
                <tbody className="text-sm font-body">
                  {accentBreakdown.map((row, idx) => {
                    const baseWer = row.avg_baseline_wer ?? row.baseline_wer
                    const corrWer = row.avg_corrected_wer ?? row.corrected_wer
                    const imp = row.improvement ?? (baseWer != null && corrWer != null ? baseWer - corrWer : null)

                    return (
                      <tr
                        key={row.accent || idx}
                        className="border-b border-outline-variant/10 hover:bg-surface-container-high/50 transition-colors"
                      >
                        <td className="py-3 px-4 text-on-surface text-xs font-label font-medium capitalize">
                          {(row.accent || '\u2014').replace(/_/g, ' ')}
                        </td>
                        <td className="py-3 px-4 text-on-surface-variant text-xs font-label">
                          {row.n ?? row.count ?? '\u2014'}
                        </td>
                        <td className="py-3 px-4 text-on-surface-variant text-xs font-label">
                          {fmtPct(baseWer)}
                        </td>
                        <td className="py-3 px-4 text-primary text-xs font-label font-bold">
                          {fmtPct(corrWer)}
                        </td>
                        <td className="py-3 px-4">
                          <ImprovementCell value={imp} />
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}

      {/* YouTube Eval Section */}
      {videoEval.length > 0 && (
        <section className="mb-8">
          <h3 className="font-headline text-lg font-bold text-on-surface mb-4">YouTube Evaluation</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {videoEval.map((v, idx) => {
              const imp = v.improvement ?? (
                v.baseline_wer != null && v.corrected_wer != null
                  ? v.baseline_wer - v.corrected_wer
                  : null
              )
              const isPositive = imp != null && imp > 0

              return (
                <div
                  key={v.video_id || v.title || idx}
                  className={`flux-card rounded-xl p-5 border-l-4 ${isPositive ? 'border-l-secondary' : 'border-l-error'}`}
                >
                  <h4 className="font-label text-sm font-bold text-on-surface mb-3 truncate">
                    {v.title || v.video_id || 'Unknown Video'}
                  </h4>
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <span className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest block mb-0.5">
                        Baseline WER
                      </span>
                      <span className="font-label font-bold text-on-surface">{fmtPct(v.baseline_wer)}</span>
                    </div>
                    <div>
                      <span className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest block mb-0.5">
                        Corrected WER
                      </span>
                      <span className="font-label font-bold text-primary">{fmtPct(v.corrected_wer)}</span>
                    </div>
                    <div>
                      <span className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest block mb-0.5">
                        Improvement
                      </span>
                      <ImprovementCell value={imp} />
                    </div>
                    <div>
                      <span className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest block mb-0.5">
                        Corrections
                      </span>
                      <span className="font-label font-bold text-on-surface">
                        {v.corrections_count ?? v.n_corrections ?? '\u2014'}
                      </span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </section>
      )}
    </div>
  )
}
