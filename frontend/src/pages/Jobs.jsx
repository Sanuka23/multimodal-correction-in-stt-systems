import { useState, useCallback } from 'react'
import { Link } from 'react-router-dom'
import {
  Filter,
  Trash2,
  TrendingUp,
  Cpu,
  AlertTriangle,
  AlertCircle,
  Lightbulb,
  ChevronDown,
  ChevronRight,
} from 'lucide-react'
import api from '../api/client'
import { usePolling } from '../hooks/usePolling'
import Badge from '../components/ui/Badge'

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function formatDuration(ms) {
  if (!ms) return '\u2014'
  const secs = Math.floor(ms / 1000)
  if (secs < 60) return `${secs}s`
  const mins = Math.floor(secs / 60)
  const remSecs = secs % 60
  return `${mins}m ${remSecs}s`
}

const STATUS_DOT = {
  completed: 'bg-secondary shadow-[0_0_8px_rgba(78,222,163,0.4)]',
  running: 'bg-primary animate-pulse shadow-[0_0_8px_rgba(123,208,255,0.4)]',
  failed: 'bg-error shadow-[0_0_8px_rgba(255,180,171,0.4)]',
}

const STATUS_TEXT = {
  completed: 'text-on-surface',
  running: 'text-on-surface',
  failed: 'text-error',
}

const PIPELINE_LABEL = {
  completed: 'VIEW_FLOW',
  running: 'MONITOR',
  failed: 'LOGS',
}

const TYPE_OPTIONS = [
  { label: 'Type: All Jobs', value: '' },
  { label: 'Type: Correction', value: 'correction' },
  { label: 'Type: Eval', value: 'eval' },
  { label: 'Type: Train', value: 'train' },
]

const STATUS_OPTIONS = [
  { label: 'Status: Any', value: '' },
  { label: 'Status: Completed', value: 'completed' },
  { label: 'Status: Running', value: 'running' },
  { label: 'Status: Failed', value: 'failed' },
]

const DATE_OPTIONS = [
  { label: 'Date: Last 24 Hours', value: '24h' },
  { label: 'Date: Last 7 Days', value: '7d' },
  { label: 'Date: Last 30 Days', value: '30d' },
]

const LIMIT = 20

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function Jobs() {
  /* --- state --- */
  const [jobs, setJobs] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [stats, setStats] = useState([])
  const [activeRuns, setActiveRuns] = useState(0)
  const [failureRate, setFailureRate] = useState('0%')
  const [lastIncident, setLastIncident] = useState(null)

  const [filterType, setFilterType] = useState('')
  const [filterStatus, setFilterStatus] = useState('')
  const [filterDate, setFilterDate] = useState('24h')
  const [showFilters, setShowFilters] = useState(false)

  const totalPages = Math.max(1, Math.ceil(total / LIMIT))

  /* --- data fetching --- */
  const fetchJobs = useCallback(async () => {
    try {
      const params = { limit: LIMIT, page }
      if (filterType) params.job_type = filterType
      if (filterStatus) params.status = filterStatus
      if (filterDate) params.date_range = filterDate
      const { data } = await api.get('/api/jobs', { params })
      setJobs(data.jobs || [])
      setTotal(data.total || 0)
    } catch {
      /* silent */
    }
  }, [page, filterType, filterStatus, filterDate])

  const fetchStats = useCallback(async () => {
    try {
      const { data } = await api.get('/api/jobs/stats')
      setStats(Array.isArray(data) ? data : data.daily || [])
      if (data.active_runs !== undefined) setActiveRuns(data.active_runs)
      if (data.failure_rate !== undefined) setFailureRate(data.failure_rate)
      if (data.last_incident !== undefined) setLastIncident(data.last_incident)
    } catch {
      /* silent */
    }
  }, [])

  usePolling(fetchJobs, 8000, [page, filterType, filterStatus, filterDate])
  usePolling(fetchStats, 15000, [])

  /* --- chart helpers --- */
  const maxCount = Math.max(1, ...stats.map((s) => s.count || 0))

  /* --- pagination helpers --- */
  function buildPageNumbers() {
    const pages = []
    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) pages.push(i)
    } else {
      pages.push(1, 2, 3)
      if (page > 4) pages.push('...')
      if (page > 3 && page < totalPages - 2) pages.push(page)
      if (page < totalPages - 3) pages.push('...')
      pages.push(totalPages)
    }
    return [...new Set(pages)]
  }

  /* ================================================================ */
  /*  RENDER                                                           */
  /* ================================================================ */
  return (
    <div className="max-w-7xl mx-auto">
      {/* ── Page Header ───────────────────────────────────────────── */}
      <div className="flex justify-between items-end mb-12">
        <div className="space-y-1">
          <p className="font-label text-xs text-primary uppercase tracking-[0.3em]">
            System Intelligence
          </p>
          <h2 className="font-headline text-4xl font-extrabold text-on-surface tracking-tight">
            Correction Jobs
          </h2>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setShowFilters((v) => !v)}
            className="flex items-center gap-2 px-4 py-2 obsidian-glass hover:bg-surface-bright text-on-surface-variant text-sm font-medium rounded-xl transition-all"
          >
            <Filter size={18} />
            Filters
          </button>
          <button className="flex items-center gap-2 px-4 py-2 border border-error/20 text-error hover:bg-error/5 text-sm font-medium rounded-xl transition-all">
            <Trash2 size={18} />
            Bulk Delete
          </button>
        </div>
      </div>

      {/* ── Dashboard Bento Grid ──────────────────────────────────── */}
      <div className="grid grid-cols-12 gap-6 mb-10">
        {/* Execution Velocity – bar chart */}
        <div className="col-span-12 lg:col-span-8 obsidian-glass p-6 rounded-xl border border-outline-variant/10">
          <div className="flex justify-between items-start mb-6">
            <h3 className="font-headline font-bold text-lg text-on-surface">
              Execution Velocity
            </h3>
            <span className="font-label text-[10px] bg-secondary-container/20 text-secondary px-2 py-0.5 rounded-full">
              CORRECTIONS PER DAY
            </span>
          </div>

          <div className="h-48 flex items-end justify-between gap-2 px-2">
            {stats.length > 0
              ? stats.map((s, i) => {
                  const pct = Math.max(2, Math.round((s.count / maxCount) * 100))
                  const isWeekend = ['Sat', 'Sun'].includes(s.day)
                  return (
                    <div
                      key={i}
                      className={`flex-1 ${
                        isWeekend
                          ? 'bg-secondary/30 hover:bg-secondary/50'
                          : 'bg-primary/40 hover:bg-primary/60'
                      } rounded-t-lg transition-all`}
                      style={{ height: `${pct}%` }}
                      title={`${s.day}: ${s.count.toLocaleString()} corrections`}
                    />
                  )
                })
              : /* placeholder bars when no data */
                ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((d) => (
                  <div
                    key={d}
                    className="flex-1 bg-primary/20 rounded-t-lg transition-all h-[8%]"
                    title={`${d}: 0 corrections`}
                  />
                ))}
          </div>

          <div className="flex justify-between mt-4 px-1">
            {(stats.length > 0
              ? stats.map((s) => s.day)
              : ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ).map((d) => (
              <span key={d} className="font-label text-[10px] text-slate-500 uppercase">
                {d}
              </span>
            ))}
          </div>
        </div>

        {/* Right column – Active Runs + Failure Rate */}
        <div className="col-span-12 lg:col-span-4 grid grid-rows-2 gap-6">
          {/* Active Runs */}
          <div className="obsidian-glass p-6 rounded-xl flex flex-col justify-between">
            <p className="font-label text-xs text-on-surface-variant uppercase tracking-widest">
              Active Runs
            </p>
            <div className="flex items-baseline gap-3">
              <h4 className="font-label text-4xl font-bold text-primary">{activeRuns}</h4>
              <span className="text-secondary text-xs flex items-center">
                <TrendingUp size={14} className="mr-0.5" /> +2
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Cpu size={14} className="text-slate-500" />
              <p className="text-[10px] text-slate-400">Arch: Apple Silicon / CPU Inference</p>
            </div>
          </div>

          {/* Failure Rate */}
          <div className="obsidian-glass p-6 rounded-xl flex flex-col justify-between border-l-4 border-error/50">
            <p className="font-label text-xs text-on-surface-variant uppercase tracking-widest">
              Failure Rate (24h)
            </p>
            <div className="flex items-baseline gap-3">
              <h4 className="font-label text-4xl font-bold text-error">{failureRate}</h4>
              <span className="text-error text-xs flex items-center">
                <AlertTriangle size={14} className="mr-0.5" /> Critical
              </span>
            </div>
            <p className="text-[10px] text-slate-500">
              Last incident: {lastIncident ?? '\u2014'}
            </p>
          </div>
        </div>
      </div>

      {/* ── Job History Table ─────────────────────────────────────── */}
      <div className="obsidian-glass rounded-xl overflow-hidden border border-outline-variant/5">
        {/* Table Controls */}
        <div className="p-6 border-b border-outline-variant/10 flex flex-wrap gap-4 items-center justify-between">
          <div className="flex flex-wrap gap-4">
            {/* Type filter */}
            <div className="relative">
              <select
                value={filterType}
                onChange={(e) => { setFilterType(e.target.value); setPage(1) }}
                className="appearance-none obsidian-glass text-on-surface text-xs font-label px-4 py-2 pr-10 rounded-lg border-none focus:ring-1 focus:ring-primary"
              >
                {TYPE_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
              <ChevronDown size={16} className="absolute right-2 top-2 text-slate-500 pointer-events-none" />
            </div>

            {/* Status filter */}
            <div className="relative">
              <select
                value={filterStatus}
                onChange={(e) => { setFilterStatus(e.target.value); setPage(1) }}
                className="appearance-none obsidian-glass text-on-surface text-xs font-label px-4 py-2 pr-10 rounded-lg border-none focus:ring-1 focus:ring-primary"
              >
                {STATUS_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
              <ChevronDown size={16} className="absolute right-2 top-2 text-slate-500 pointer-events-none" />
            </div>

            {/* Date filter */}
            <div className="relative">
              <select
                value={filterDate}
                onChange={(e) => { setFilterDate(e.target.value); setPage(1) }}
                className="appearance-none obsidian-glass text-on-surface text-xs font-label px-4 py-2 pr-10 rounded-lg border-none focus:ring-1 focus:ring-primary"
              >
                {DATE_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
              <ChevronDown size={16} className="absolute right-2 top-2 text-slate-500 pointer-events-none" />
            </div>
          </div>

          <div className="text-[10px] text-slate-500 font-mono">
            Showing {jobs.length} of {total.toLocaleString()} records
          </div>
        </div>

        {/* Data Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-900/40">
                <th className="px-6 py-4 font-label text-[10px] uppercase tracking-widest text-slate-500">
                  Job Type
                </th>
                <th className="px-6 py-4 font-label text-[10px] uppercase tracking-widest text-slate-500">
                  Job UUID
                </th>
                <th className="px-6 py-4 font-label text-[10px] uppercase tracking-widest text-slate-500">
                  Status
                </th>
                <th className="px-6 py-4 font-label text-[10px] uppercase tracking-widest text-slate-500">
                  Duration
                </th>
                <th className="px-6 py-4 font-label text-[10px] uppercase tracking-widest text-slate-500">
                  Result Summary
                </th>
                <th className="px-6 py-4 font-label text-[10px] uppercase tracking-widest text-slate-500 text-right">
                  Pipeline
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/10">
              {jobs.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-6 py-16 text-center text-sm text-slate-500">
                    No jobs found.
                  </td>
                </tr>
              )}
              {jobs.map((job) => {
                const status = job.status || 'completed'
                return (
                  <tr
                    key={job._id}
                    className={`hover:bg-surface-bright/30 transition-colors group ${
                      status === 'failed' ? 'bg-error/5' : ''
                    }`}
                  >
                    {/* Job Type */}
                    <td className="px-6 py-5">
                      <span className="px-2 py-1 bg-surface-container-highest text-[10px] font-label rounded-lg text-on-surface-variant">
                        {job.job_type}
                      </span>
                    </td>

                    {/* UUID */}
                    <td className="px-6 py-5">
                      <span className="font-mono text-xs text-sky-400">
                        {job._id ? `${job._id.slice(0, 8)}...` : '\u2014'}
                      </span>
                    </td>

                    {/* Status */}
                    <td className="px-6 py-5">
                      <div className="flex items-center gap-2">
                        <div
                          className={`w-2 h-2 rounded-full ${STATUS_DOT[status] || STATUS_DOT.completed}`}
                        />
                        <span className={`text-xs font-medium capitalize ${STATUS_TEXT[status] || 'text-on-surface'}`}>
                          {status}
                        </span>
                      </div>
                    </td>

                    {/* Duration */}
                    <td className="px-6 py-5 font-mono text-xs text-on-surface-variant">
                      {formatDuration(job.duration_ms)}
                    </td>

                    {/* Result Summary */}
                    <td className="px-6 py-5">
                      {status === 'running' ? (
                        <div className="w-32 h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
                          <div className="h-full bg-primary w-[45%] animate-pulse" />
                        </div>
                      ) : status === 'failed' ? (
                        <p className="text-xs text-error/80 flex items-center gap-1">
                          <AlertCircle size={14} />
                          {typeof job.result_summary === 'string' ? job.result_summary : 'Process exited with an error'}
                        </p>
                      ) : (
                        <p className="text-xs text-slate-400 line-clamp-1">
                          {typeof job.result_summary === 'string' ? job.result_summary : job.result_summary ? JSON.stringify(job.result_summary) : '\u2014'}
                        </p>
                      )}
                    </td>

                    {/* Pipeline Link */}
                    <td className="px-6 py-5 text-right">
                      <Link
                        to={`/pipeline/${job._id}`}
                        className="text-primary hover:underline text-xs font-label tracking-tight"
                      >
                        {PIPELINE_LABEL[status] || 'VIEW_FLOW'}
                      </Link>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="p-6 border-t border-outline-variant/10 flex justify-between items-center">
          <button
            disabled={page <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            className="px-4 py-2 obsidian-glass text-on-surface-variant text-xs font-label rounded-lg hover:text-primary transition-colors disabled:opacity-50"
          >
            PREVIOUS
          </button>

          <div className="flex gap-2">
            {buildPageNumbers().map((p, i) =>
              p === '...' ? (
                <span key={`dots-${i}`} className="px-2 flex items-end text-slate-600">
                  ...
                </span>
              ) : (
                <button
                  key={p}
                  onClick={() => setPage(p)}
                  className={`w-8 h-8 flex items-center justify-center rounded-lg text-xs font-bold transition-all ${
                    p === page
                      ? 'bg-primary text-on-primary shadow-lg shadow-primary/20'
                      : 'hover:bg-slate-800 text-on-surface-variant'
                  }`}
                >
                  {p}
                </button>
              )
            )}
          </div>

          <button
            disabled={page >= totalPages}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            className="px-4 py-2 obsidian-glass text-on-surface-variant text-xs font-label rounded-lg hover:text-primary transition-colors disabled:opacity-50"
          >
            NEXT
          </button>
        </div>
      </div>

      {/* ── Contextual Data Footer ────────────────────────────────── */}
      <div className="mt-12 grid grid-cols-1 md:grid-cols-2 gap-8 p-8 obsidian-glass rounded-xl border border-outline-variant/10">
        {/* Pipeline Health */}
        <div>
          <h5 className="font-label text-[10px] text-slate-500 uppercase tracking-widest mb-4">
            Pipeline Health
          </h5>
          <div className="flex items-center gap-4">
            <div className="relative w-12 h-12">
              <svg className="w-full h-full transform -rotate-90">
                <circle
                  className="text-surface-container-highest"
                  cx="24"
                  cy="24"
                  r="20"
                  fill="transparent"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <circle
                  className="text-secondary"
                  cx="24"
                  cy="24"
                  r="20"
                  fill="transparent"
                  stroke="currentColor"
                  strokeWidth="4"
                  strokeDasharray="125.6"
                  strokeDashoffset="12.5"
                />
              </svg>
              <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold">
                92%
              </span>
            </div>
            <p className="text-xs text-on-surface-variant">
              Optimal throughput detected. System operating within nominal latency thresholds.
            </p>
          </div>
        </div>

        {/* System Alerts */}
        <div>
          <h5 className="font-label text-[10px] text-slate-500 uppercase tracking-widest mb-4">
            System Alerts
          </h5>
          <div className="flex items-start gap-3 bg-tertiary-container/20 p-3 rounded-lg">
            <Lightbulb size={18} className="text-tertiary flex-shrink-0 mt-0.5" />
            <p className="text-[10px] text-tertiary">
              Optimization: Refactoring the data loader in flow FL_99 could reduce execution
              duration by up to 12%.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
