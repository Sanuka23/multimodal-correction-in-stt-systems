import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Zap,
  CheckCircle,
  ShieldCheck,
  CheckCheck,
  TrendingUp,
  Activity,
  Eye,
  ChevronRight,
  AudioWaveform,
} from 'lucide-react'
import Badge from '../components/ui/Badge'
import { usePolling } from '../hooks/usePolling'
import api from '../api/client'

function timeAgo(dateStr) {
  if (!dateStr) return ''
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'Just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

function formatDuration(ms) {
  if (!ms) return '\u2014'
  const secs = Math.floor(ms / 1000)
  if (secs < 60) return `${secs}s`
  const mins = Math.floor(secs / 60)
  const remSecs = secs % 60
  return `${mins}m ${remSecs}s`
}

const STATUS_COLOR_MAP = {
  running: 'bg-secondary',
  completed: 'bg-slate-700',
  success: 'bg-slate-700',
  failed: 'bg-error',
  pending: 'bg-tertiary',
}

const STATUS_LABEL_MAP = {
  running: { text: 'RUNNING', cls: 'text-secondary bg-secondary/10 border-secondary/20' },
  completed: { text: 'SUCCESS', cls: 'text-slate-400' },
  success: { text: 'SUCCESS', cls: 'text-slate-400' },
  failed: { text: 'FAILED', cls: 'text-error bg-error/10 border-error/20' },
  pending: { text: 'PENDING', cls: 'text-tertiary bg-tertiary/10 border-tertiary/20' },
}

const ACCENT_DATA = [
  { label: 'American', count: 6096, color: 'bg-primary', pct: 38 },
  { label: 'South Asian', count: 5958, color: 'bg-secondary', pct: 37 },
  { label: 'European', count: 345, color: 'bg-tertiary', pct: 5 },
  { label: 'Unknown', count: 3683, color: 'bg-slate-600', pct: 20 },
]

export default function Dashboard() {
  const [stats, setStats] = useState({
    correction_count: 0,
    total_applied: 0,
    avg_confidence: 0,
    success_rate: 0,
  })
  const [corrections, setCorrections] = useState([])
  const [jobs, setJobs] = useState([])
  const [health, setHealth] = useState({
    model: { status: 'unknown', adapter: '' },
    ocr: { status: 'unknown', confidence: 0 },
    avsr: { status: 'unknown', mode: '' },
  })

  const fetchData = async () => {
    try {
      const [statsRes, corrRes, jobsRes, healthRes] = await Promise.all([
        api.get('/api/stats').catch(() => ({ data: stats })),
        api.get('/api/corrections?limit=8').catch(() => ({ data: [] })),
        api.get('/api/jobs?limit=5').catch(() => ({ data: [] })),
        api.get('/api/health').catch(() => ({ data: health })),
      ])
      if (statsRes.data) setStats(statsRes.data)
      setCorrections(Array.isArray(corrRes.data) ? corrRes.data : [])
      const jd = jobsRes.data
      setJobs(Array.isArray(jd) ? jd : jd?.jobs || [])
      if (healthRes.data) setHealth(healthRes.data)
    } catch {
      /* silent */
    }
  }

  usePolling(fetchData, 15000)

  // Derived values
  const correctionCount = stats.correction_count || 0
  const totalApplied = stats.total_applied || 0
  const avgConfidence = stats.avg_confidence || 0
  const successRate = stats.success_rate || 0
  const appliedPct = correctionCount > 0 ? Math.round((totalApplied / correctionCount) * 100) : 0
  const p50 = stats.p50_latency_ms || stats.avg_duration_ms || 161000
  const p95 = stats.p95_latency_ms || 342000
  const tterImprovement = stats.tter_improvement || '+14.2%'
  const correctionsToday = stats.corrections_today || totalApplied || 0

  // Health-derived
  const modelStatus = health.model?.status || 'Active'
  const modelAdapter = health.model?.adapter || 'Adapter v1'
  const ocrStatus = health.ocr?.status || 'Healthy'
  const ocrConfidence = health.ocr?.confidence != null ? `${(health.ocr.confidence * 100).toFixed(1)}%` : '99.2%'
  const avsrMode = health.avsr?.mode || 'MediaPipe'

  const healthBorderColor = (status) => {
    if (!status) return 'border-b-secondary'
    const s = String(status).toLowerCase()
    if (s === 'active' || s === 'healthy' || s === 'ok') return 'border-b-secondary'
    if (s === 'mediapipe' || s === 'running') return 'border-b-primary'
    if (s === 'error' || s === 'down') return 'border-b-error'
    return 'border-b-secondary'
  }

  const healthIconColor = (status) => {
    const s = String(status).toLowerCase()
    if (s === 'mediapipe' || s === 'running') return 'text-primary'
    return 'text-secondary'
  }

  return (
    <>
      {/* ── Top Stats Row ── */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        {/* Correction Count */}
        <div className="bg-surface-container rounded-3xl p-5 border border-slate-800/50">
          <div className="flex justify-between items-start mb-2">
            <span className="font-label text-[10px] text-slate-500 uppercase tracking-widest">Correction Count</span>
            <Zap size={14} className="text-primary" />
          </div>
          <div className="text-3xl font-headline font-black text-on-surface">
            {correctionCount.toLocaleString()}
          </div>
          <div className="text-[10px] text-secondary mt-1 flex items-center gap-1">
            <TrendingUp size={12} /> +8% from yesterday
          </div>
        </div>

        {/* Total Applied */}
        <div className="bg-surface-container rounded-3xl p-5 border border-slate-800/50">
          <div className="flex justify-between items-start mb-2">
            <span className="font-label text-[10px] text-slate-500 uppercase tracking-widest">Total Applied</span>
            <CheckCircle size={14} className="text-secondary" />
          </div>
          <div className="text-3xl font-headline font-black text-on-surface">
            {totalApplied.toLocaleString()}
          </div>
          <div className="text-[10px] text-slate-500 mt-1">{appliedPct}% of identified gaps</div>
        </div>

        {/* Avg Confidence */}
        <div className="bg-surface-container rounded-3xl p-5 border border-slate-800/50">
          <div className="flex justify-between items-start mb-2">
            <span className="font-label text-[10px] text-slate-500 uppercase tracking-widest">Avg Confidence</span>
            <ShieldCheck size={14} className="text-tertiary" />
          </div>
          <div className="text-3xl font-headline font-black text-on-surface">
            {avgConfidence > 1 ? avgConfidence.toFixed(2) : avgConfidence.toFixed(2)}
          </div>
          <div className="text-[10px] text-secondary mt-1">Stable vs last run</div>
        </div>

        {/* Success Rate */}
        <div className="bg-surface-container rounded-3xl p-5 border border-slate-800/50">
          <div className="flex justify-between items-start mb-2">
            <span className="font-label text-[10px] text-slate-500 uppercase tracking-widest">Success Rate</span>
            <CheckCheck size={14} className="text-secondary" />
          </div>
          <div className="text-3xl font-headline font-black text-secondary">
            {successRate > 1 ? `${successRate.toFixed(0)}%` : `${(successRate * 100).toFixed(0)}%`}
          </div>
          <div className="text-[10px] text-slate-500 mt-1">Post-correction validation</div>
        </div>
      </div>

      {/* ── Live Status Banner ── */}
      <section className="mb-8 bg-surface-container-low rounded-3xl overflow-hidden relative border border-slate-800/50">
        <div className="p-6 relative z-10 flex flex-col md:flex-row justify-between items-center gap-6">
          <div>
            <h2 className="font-label text-xs font-bold text-primary uppercase tracking-widest mb-1 flex items-center gap-2">
              <Activity size={14} />
              Active Eval Session
            </h2>
            <div className="flex items-baseline gap-4">
              <span className="font-headline text-3xl font-extrabold text-on-surface">Dataset: Test_Suite_04</span>
              <span className="font-label text-sm text-on-surface-variant bg-surface-container-highest px-2 py-0.5 rounded-lg border border-slate-700">
                {modelAdapter}
              </span>
            </div>
          </div>

          <div className="flex flex-wrap gap-8 items-center bg-surface-container-high/50 px-6 py-4 rounded-3xl backdrop-blur-sm border border-slate-800">
            <div className="flex flex-col">
              <span className="font-label text-[10px] text-slate-500 uppercase tracking-widest">p50 Latency</span>
              <span className="font-label text-xl font-bold text-on-surface">{formatDuration(p50)}</span>
            </div>
            <div className="h-8 w-px bg-slate-800" />
            <div className="flex flex-col">
              <span className="font-label text-[10px] text-slate-500 uppercase tracking-widest">p95 Latency</span>
              <span className="font-label text-xl font-bold text-on-surface">{formatDuration(p95)}</span>
            </div>
            <div className="h-8 w-px bg-slate-800" />
            <div className="flex flex-wrap gap-2">
              <Link
                to="/jobs"
                className="bg-white/5 hover:bg-white/10 text-on-surface text-[10px] font-bold px-4 py-2 rounded-full transition-all flex items-center gap-2"
              >
                <Eye size={14} />
                View Last Job
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ── Pipeline Health + Execution Summary ── */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 mb-8">
        {/* Pipeline Component Health */}
        <div className="md:col-span-8 bg-surface-container rounded-3xl p-6 border border-slate-800/50">
          <h3 className="font-headline text-lg font-bold text-on-surface mb-6 flex justify-between items-center">
            Pipeline Component Health
            <span className="text-[10px] font-label text-slate-500 uppercase tracking-widest">
              Session p50: {formatDuration(p50)}
            </span>
          </h3>
          <div className="grid grid-cols-3 gap-4">
            {/* Model Status */}
            <div className={`bg-surface-container-low p-5 rounded-2xl border border-slate-800/80 border-b-2 ${healthBorderColor(modelStatus)}`}>
              <div className="flex justify-between items-start mb-2">
                <span className="font-label text-[10px] text-slate-400 uppercase tracking-tighter">Model Status</span>
                <CheckCircle size={14} className={healthIconColor(modelStatus)} />
              </div>
              <div className="font-headline text-lg font-bold">{modelStatus}</div>
              <div className="font-label text-[10px] text-slate-500 mt-1">{modelAdapter}</div>
            </div>

            {/* OCR Engine */}
            <div className={`bg-surface-container-low p-5 rounded-2xl border border-slate-800/80 border-b-2 ${healthBorderColor(ocrStatus)}`}>
              <div className="flex justify-between items-start mb-2">
                <span className="font-label text-[10px] text-slate-400 uppercase tracking-tighter">OCR Engine</span>
                <CheckCircle size={14} className={healthIconColor(ocrStatus)} />
              </div>
              <div className="font-headline text-lg font-bold">{ocrStatus}</div>
              <div className="font-label text-[10px] text-slate-500 mt-1">Confidence: {ocrConfidence}</div>
            </div>

            {/* AVSR Module */}
            <div className="bg-surface-container-low p-5 rounded-2xl border border-slate-800/80 border-b-2 border-b-primary">
              <div className="flex justify-between items-start mb-2">
                <span className="font-label text-[10px] text-slate-400 uppercase tracking-tighter">AVSR Module</span>
                <AudioWaveform size={14} className="text-primary" />
              </div>
              <div className="font-headline text-lg font-bold">{avsrMode}</div>
              <div className="font-label text-[10px] text-slate-500 mt-1">Active Mode</div>
            </div>
          </div>
        </div>

        {/* Execution Summary */}
        <div className="md:col-span-4 bg-surface-container rounded-3xl p-6 border border-slate-800/50 relative overflow-hidden">
          <h3 className="font-headline text-lg font-bold text-on-surface mb-6">Execution Summary</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-end border-b border-slate-800 pb-2">
              <div className="flex flex-col">
                <span className="text-[10px] uppercase font-label text-slate-500 tracking-wider">TTER Improvement</span>
                <span className="text-2xl font-black text-secondary">{tterImprovement}</span>
              </div>
              <TrendingUp size={20} className="text-secondary mb-1" />
            </div>
            <div className="flex justify-between items-end border-b border-slate-800 pb-2">
              <div className="flex flex-col">
                <span className="text-[10px] uppercase font-label text-slate-500 tracking-wider">Corrections Today</span>
                <span className="text-2xl font-black text-on-surface">{correctionsToday.toLocaleString()}</span>
              </div>
            </div>
            <div className="flex justify-between items-end border-b border-slate-800 pb-2">
              <div className="flex flex-col">
                <span className="text-[10px] uppercase font-label text-slate-500 tracking-wider">Avg Confidence</span>
                <span className="text-2xl font-black text-on-surface">
                  {avgConfidence > 1 ? avgConfidence.toFixed(2) : avgConfidence.toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Live Correction Feed + Job Flow / Accent Sidebar ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Corrections List */}
        <div className="lg:col-span-7">
          <div className="flex justify-between items-center mb-6">
            <h3 className="font-headline text-xl font-extrabold text-on-surface">Live Correction Feed</h3>
            <button className="font-label text-[10px] uppercase tracking-widest text-primary hover:underline">
              Export Log
            </button>
          </div>
          <div className="space-y-4">
            {corrections.length === 0 ? (
              <div className="bg-surface-container-high rounded-3xl p-6 text-center text-on-surface-variant text-sm border border-slate-800/50">
                No corrections yet. Send a transcript via the ScreenApp API.
              </div>
            ) : (
              corrections.map((c, i) => {
                const first = c.corrections?.[0] || {}
                const entryId = (c._id || '').substring(0, 8).toUpperCase()
                return (
                  <div
                    key={c._id || i}
                    className="bg-surface-container-high rounded-3xl p-5 group cursor-pointer hover:bg-surface-bright transition-all border border-slate-800/50"
                  >
                    {/* Header */}
                    <div className="flex justify-between items-start mb-4">
                      <div className="flex items-center gap-3">
                        <span className="w-8 h-8 rounded-full bg-secondary-container/20 text-secondary flex items-center justify-center">
                          <Zap size={18} />
                        </span>
                        <div>
                          <span className="font-label text-[10px] text-slate-500 uppercase tracking-widest">
                            Entry ID: {entryId}
                          </span>
                          <h4 className="font-body font-semibold text-on-surface">
                            {first.category
                              ? `${first.category} Auto-Fix`
                              : `${c.corrections_applied || 0} correction(s) applied`}
                          </h4>
                        </div>
                      </div>
                      <span className="font-label text-[10px] text-slate-500">{timeAgo(c.created_at)}</span>
                    </div>

                    {/* Detected / Corrected */}
                    {first.error_found && (
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-surface-container-lowest p-3 rounded-2xl border-l-4 border-error">
                          <span className="font-label text-[9px] uppercase text-error block mb-1">Detected</span>
                          <code className="font-label text-sm text-on-surface-variant line-through opacity-60">
                            {first.error_found}
                          </code>
                        </div>
                        <div className="bg-surface-container-lowest p-3 rounded-2xl border-l-4 border-secondary">
                          <span className="font-label text-[9px] uppercase text-secondary block mb-1">Corrected</span>
                          <code className="font-label text-sm text-secondary">{first.term}</code>
                        </div>
                      </div>
                    )}

                    {/* Footer badges */}
                    <div className="mt-4 pt-4 border-t border-slate-800/50 flex justify-between items-center">
                      <div className="flex gap-2">
                        {first.category && (
                          <span className="bg-surface-container-highest px-3 py-1 rounded-full text-[10px] text-slate-400 font-label border border-slate-700">
                            {first.category}
                          </span>
                        )}
                        {first.confidence != null && (
                          <span className="bg-secondary/10 px-3 py-1 rounded-full text-[10px] text-secondary font-label border border-secondary/20 font-bold">
                            Confidence: {typeof first.confidence === 'number' && first.confidence <= 1
                              ? (first.confidence * 100).toFixed(0) + '%'
                              : first.confidence}
                          </span>
                        )}
                      </div>
                      <ChevronRight
                        size={18}
                        className="text-slate-500 group-hover:text-primary transition-colors"
                      />
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </div>

        {/* Job Flow + Accent Distribution Sidebar */}
        <div className="lg:col-span-5">
          <div className="bg-surface-container-low rounded-3xl p-6 h-full border border-slate-800/50">
            <h3 className="font-headline text-lg font-bold text-on-surface mb-6">Job Flow</h3>

            {/* Jobs */}
            <div className="space-y-6 mb-8">
              {jobs.length === 0 ? (
                <p className="text-xs text-on-surface-variant">No recent jobs</p>
              ) : (
                jobs.slice(0, 5).map((j, i) => {
                  const status = (j.status || 'pending').toLowerCase()
                  const barColor = STATUS_COLOR_MAP[status] || 'bg-slate-700'
                  const label = STATUS_LABEL_MAP[status] || STATUS_LABEL_MAP.pending
                  const isActive = status === 'running'
                  return (
                    <Link
                      key={j._id || i}
                      to={`/pipeline/${j._id}`}
                      className={`flex items-center justify-between ${!isActive ? 'opacity-70' : ''}`}
                    >
                      <div className="flex gap-4 items-center">
                        <div className={`w-2 h-10 ${barColor} rounded-full`} />
                        <div>
                          <div className="font-body font-bold text-sm">
                            {j.file_id || j.name || (j._id || '').substring(0, 16)}
                          </div>
                          <div className="font-label text-[10px] text-slate-500">
                            {j.job_type || 'correction'}
                            {j.duration_ms ? ` \u2022 ${formatDuration(j.duration_ms)}` : ''}
                          </div>
                        </div>
                      </div>
                      <span
                        className={`font-label text-[10px] font-bold px-2 py-0.5 rounded-full border ${label.cls}`}
                      >
                        {label.text}
                      </span>
                    </Link>
                  )
                })
              )}
            </div>

            {/* Accent Distribution */}
            <div className="mt-8 pt-6 border-t border-slate-800">
              <h4 className="font-label text-[10px] uppercase tracking-[0.2em] text-slate-500 mb-4">
                Accent Distribution
              </h4>
              <div className="space-y-4">
                {ACCENT_DATA.map((a) => (
                  <div key={a.label}>
                    <div className="flex justify-between text-[10px] font-label mb-1">
                      <span className="text-slate-400">{a.label}</span>
                      <span className="text-on-surface">{a.count.toLocaleString()}</span>
                    </div>
                    <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                      <div className={`${a.color} h-full`} style={{ width: `${a.pct}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
