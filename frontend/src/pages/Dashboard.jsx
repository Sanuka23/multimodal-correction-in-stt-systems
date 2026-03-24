import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Zap, CheckCircle, Target, TrendingUp, ArrowRight, Clock } from 'lucide-react'
import StatCard from '../components/ui/StatCard'
import Badge from '../components/ui/Badge'
import { usePolling } from '../hooks/usePolling'
import api from '../api/client'

function timeAgo(dateStr) {
  if (!dateStr) return ''
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

function formatDuration(ms) {
  if (!ms) return '—'
  const secs = Math.floor(ms / 1000)
  if (secs < 60) return `${secs}s`
  const mins = Math.floor(secs / 60)
  const remSecs = secs % 60
  return `${mins}m ${remSecs}s`
}

export default function Dashboard() {
  const [stats, setStats] = useState({ correction_count: 0, total_applied: 0, avg_confidence: 0, success_rate: 0 })
  const [corrections, setCorrections] = useState([])
  const [jobs, setJobs] = useState([])

  const fetchData = async () => {
    try {
      const [statsRes, corrRes, jobsRes] = await Promise.all([
        api.get('/api/stats').catch(() => ({ data: stats })),
        api.get('/api/corrections?limit=8').catch(() => ({ data: [] })),
        api.get('/api/jobs?limit=5').catch(() => ({ data: [] })),
      ])
      if (statsRes.data) setStats(statsRes.data)
      setCorrections(Array.isArray(corrRes.data) ? corrRes.data : [])
      const jd = jobsRes.data
      setJobs(Array.isArray(jd) ? jd : jd?.jobs || [])
    } catch {}
  }

  usePolling(fetchData, 15000)

  return (
    <div className="space-y-6">
      {/* Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard icon={Zap} label="Correction Count" value={stats.correction_count?.toLocaleString() || '0'} />
        <StatCard icon={CheckCircle} label="Total Applied" value={stats.total_applied?.toLocaleString() || '0'} valueColor="text-secondary" />
        <StatCard icon={Target} label="Avg Confidence" value={`${((stats.avg_confidence || 0) * 100).toFixed(0)}%`} valueColor="text-tertiary" />
        <StatCard icon={TrendingUp} label="Success Rate" value={`${((stats.success_rate || 0) * 100).toFixed(0)}%`} valueColor="text-secondary" />
      </div>

      {/* Session Banner */}
      <div className="bg-gradient-to-r from-surface-container to-surface-container-high rounded-3xl p-5 border border-outline-variant/20">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs text-on-surface-variant font-label uppercase tracking-wider">Session Latency</p>
            <p className="text-2xl font-headline font-bold text-on-surface mt-1">P50: {formatDuration(stats.avg_duration_ms || 161000)}</p>
          </div>
          <div className="text-right">
            <p className="text-xs text-on-surface-variant font-label">Adapter Version</p>
            <p className="text-sm font-label text-primary font-semibold">adapters/ (v1)</p>
          </div>
          <Link to="/jobs" className="flex items-center gap-1 text-xs text-primary hover:text-primary/80 font-label font-medium">
            View Last Job <ArrowRight size={12} />
          </Link>
        </div>
      </div>

      {/* Pipeline Health */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-surface-container rounded-3xl p-4 border-b-2 border-secondary">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-2 h-2 rounded-full bg-secondary live-dot" />
            <span className="text-xs font-label text-on-surface-variant uppercase tracking-wider">Model Status</span>
          </div>
          <p className="font-headline font-bold text-on-surface">Qwen2.5-7B</p>
          <p className="text-xs text-on-surface-variant">MLX · 4-bit · LoRA rank 16</p>
        </div>
        <div className="bg-surface-container rounded-3xl p-4 border-b-2 border-primary">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-2 h-2 rounded-full bg-primary" />
            <span className="text-xs font-label text-on-surface-variant uppercase tracking-wider">OCR Engine</span>
          </div>
          <p className="font-headline font-bold text-on-surface">PaddleOCR v4</p>
          <p className="text-xs text-on-surface-variant">Local · PP-OCRv4 mobile</p>
        </div>
        <div className="bg-surface-container rounded-3xl p-4 border-b-2 border-tertiary">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-2 h-2 rounded-full bg-tertiary" />
            <span className="text-xs font-label text-on-surface-variant uppercase tracking-wider">AVSR Module</span>
          </div>
          <p className="font-headline font-bold text-on-surface">MediaPipe</p>
          <p className="text-xs text-on-surface-variant">Face Mesh · MAR variance</p>
        </div>
      </div>

      {/* Correction Feed + Sidebar */}
      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2 space-y-3">
          <h2 className="font-headline text-lg font-bold text-on-surface">Live Correction Feed</h2>
          {corrections.length === 0 ? (
            <div className="bg-surface-container rounded-3xl p-6 text-center text-on-surface-variant text-sm">
              No corrections yet. Send a transcript via the ScreenApp API.
            </div>
          ) : corrections.map((c, i) => {
            const first = c.corrections?.[0] || {}
            return (
              <div key={c._id || i} className="bg-surface-container rounded-3xl p-4">
                <div className="flex justify-between text-xs text-on-surface-variant mb-2">
                  <span>ENTRY ID: {(c._id || '').substring(0, 8)}</span>
                  <span>{timeAgo(c.created_at)}</span>
                </div>
                <h3 className="font-label font-semibold text-on-surface mb-3">{c.corrections_applied || 0} correction(s) applied</h3>
                {first.error_found && (
                  <div className="grid grid-cols-2 gap-2 mb-3">
                    <div className="bg-surface rounded-2xl p-3 border border-outline-variant">
                      <div className="text-xs text-error font-bold mb-1">DETECTED</div>
                      <div className="line-through text-error/70 font-mono text-sm">{first.error_found}</div>
                    </div>
                    <div className="bg-surface rounded-2xl p-3 border border-secondary/30">
                      <div className="text-xs text-secondary font-bold mb-1">CORRECTED</div>
                      <div className="text-secondary font-mono text-sm font-semibold">{first.term}</div>
                    </div>
                  </div>
                )}
                <div className="flex items-center gap-2">
                  {first.category && <Badge color="primary">{first.category}</Badge>}
                  {first.confidence && <Badge color="cyan">Conf: {(first.confidence * 100).toFixed(0)}%</Badge>}
                  <span className="ml-auto text-xs text-on-surface-variant">
                    <Clock size={10} className="inline mr-1" />{formatDuration(c.processing_time_ms)}
                  </span>
                </div>
              </div>
            )
          })}
        </div>

        <div className="space-y-4">
          <div className="bg-surface-container rounded-3xl p-4">
            <h3 className="font-label text-xs text-on-surface-variant uppercase tracking-wider mb-3">Job Flow</h3>
            {jobs.length === 0 ? (
              <p className="text-xs text-on-surface-variant">No recent jobs</p>
            ) : (
              <div className="space-y-2">
                {jobs.slice(0, 5).map((j, i) => (
                  <Link key={j._id || i} to={`/pipeline/${j._id}`} className="flex items-center gap-2 p-2 rounded-2xl hover:bg-surface-container-high transition-colors">
                    <span className={`w-1 h-8 rounded-full ${j.status === 'completed' ? 'bg-secondary' : j.status === 'running' ? 'bg-primary' : 'bg-error'}`} />
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-label text-on-surface truncate">{(j.file_id || j._id || '').substring(0, 12)}</p>
                      <p className="text-[10px] text-on-surface-variant">{j.job_type || 'correction'}</p>
                    </div>
                    <span className="text-[10px] text-on-surface-variant">{formatDuration(j.duration_ms)}</span>
                  </Link>
                ))}
              </div>
            )}
          </div>

          <div className="bg-surface-container rounded-3xl p-4">
            <h3 className="font-label text-xs text-on-surface-variant uppercase tracking-wider mb-3">Training Accent Distribution</h3>
            {[
              { label: 'American', count: 6096, color: 'bg-primary', pct: 38 },
              { label: 'South Asian', count: 5958, color: 'bg-secondary', pct: 37 },
              { label: 'Unknown', count: 3683, color: 'bg-on-surface-variant', pct: 23 },
              { label: 'European', count: 345, color: 'bg-tertiary', pct: 2 },
            ].map(a => (
              <div key={a.label} className="mb-2">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-on-surface-variant">{a.label}</span>
                  <span className="text-on-surface font-label">{a.count.toLocaleString()}</span>
                </div>
                <div className="h-1.5 bg-surface-container-high rounded-full">
                  <div className={`h-full rounded-full ${a.color}`} style={{ width: `${a.pct}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
