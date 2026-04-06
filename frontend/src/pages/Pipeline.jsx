import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, Activity, Clock, FileText, AlertCircle } from 'lucide-react'
import { useJobSteps } from '../api/queries'
import PipelineNode from '../components/pipeline/PipelineNode'
import PipelineConnector from '../components/pipeline/PipelineConnector'
import PipelineDetailPanel from '../components/pipeline/PipelineDetailPanel'
import Badge from '../components/ui/Badge'

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function formatDuration(ms) {
  if (!ms) return '\u2014'
  if (ms < 1000) return `${ms}ms`
  const secs = Math.floor(ms / 1000)
  if (secs < 60) return `${secs}s`
  const mins = Math.floor(secs / 60)
  const rem = secs % 60
  return `${mins}m ${rem}s`
}

const STATUS_BADGE = {
  completed: 'green',
  running: 'cyan',
  failed: 'red',
  pending: 'gray',
}

/* ------------------------------------------------------------------ */
/*  Skeleton loader for nodes                                          */
/* ------------------------------------------------------------------ */

function SkeletonNodes() {
  return (
    <div className="flex items-center gap-0 overflow-x-auto py-4 px-2">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="contents">
          {i > 0 && (
            <svg width="40" height="24" viewBox="0 0 40 24" className="flex-shrink-0 self-center">
              <line x1="0" y1="12" x2="32" y2="12" stroke="#45464d" strokeWidth="2" />
              <polygon points="32,6 40,12 32,18" fill="#45464d" />
            </svg>
          )}
          <div className="w-40 h-24 flex-shrink-0 flux-card rounded-xl animate-pulse" />
        </div>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Error state                                                        */
/* ------------------------------------------------------------------ */

function ErrorState({ jobId }) {
  return (
    <div className="max-w-xl mx-auto mt-24 text-center">
      <AlertCircle size={48} className="mx-auto text-error mb-4" />
      <h2 className="font-headline text-2xl font-bold text-on-surface mb-2">Job Not Found</h2>
      <p className="text-on-surface-variant text-sm mb-6">
        No pipeline data found for job <span className="font-mono text-primary">{jobId}</span>.
        It may have been deleted or the ID is incorrect.
      </p>
      <Link
        to="/jobs"
        className="inline-flex items-center gap-2 px-5 py-2.5 bg-primary/10 text-primary text-sm font-medium rounded-xl hover:bg-primary/20 transition-colors"
      >
        <ArrowLeft size={16} /> Back to Jobs
      </Link>
    </div>
  )
}

/* ================================================================== */
/*  Pipeline Page                                                      */
/* ================================================================== */

export default function Pipeline() {
  const { jobId } = useParams()
  const { data, isLoading, isError } = useJobSteps(jobId)
  const [selectedIdx, setSelectedIdx] = useState(null)

  /* Normalize response shape — API returns { job_id, status, pipeline_steps } */
  const steps = data?.pipeline_steps || data?.steps || (Array.isArray(data) ? data : [])
  const job = data || {}

  /* Compute overall status */
  const overallStatus = steps.some((s) => s.status === 'failed')
    ? 'failed'
    : steps.some((s) => s.status === 'running')
      ? 'running'
      : steps.length > 0 && steps.every((s) => s.status === 'completed')
        ? 'completed'
        : 'pending'

  /* Total duration */
  const totalDuration = steps.reduce((sum, s) => sum + (s.duration_ms || 0), 0)

  /* Summary counts */
  const correctionsApplied = job.corrections_applied ?? job.applied ?? null
  const correctionsAttempted = job.corrections_attempted ?? job.attempted ?? null

  /* Not found */
  if (!isLoading && (isError || (!data && !steps.length))) {
    return <ErrorState jobId={jobId} />
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="mb-10">
        <Link
          to="/jobs"
          className="inline-flex items-center gap-1.5 text-on-surface-variant text-xs font-label hover:text-primary transition-colors mb-6"
        >
          <ArrowLeft size={14} /> Back to Jobs
        </Link>

        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="space-y-1">
            <p className="font-label text-xs text-primary uppercase tracking-[0.3em]">
              Pipeline Monitor
            </p>
            <h2 className="font-headline text-3xl font-extrabold text-on-surface tracking-tight">
              Pipeline Monitor
            </h2>
          </div>

          <div className="flex items-center gap-3">
            <span className="font-mono text-xs text-on-surface-variant bg-surface-container-high px-3 py-1.5 rounded-lg">
              {jobId ? `${jobId.slice(0, 12)}...` : '\u2014'}
            </span>
            <Badge color={STATUS_BADGE[overallStatus] || 'gray'}>
              {overallStatus}
            </Badge>
            <div className="flex items-center gap-1.5 text-on-surface-variant text-xs">
              <Clock size={14} />
              <span className="font-mono">{formatDuration(totalDuration)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Pipeline Flow ───────────────────────────────────────── */}
      <div className="obsidian-glass rounded-xl p-6 border border-outline-variant/10 mb-6">
        <div className="flex items-center gap-1.5 mb-5">
          <Activity size={16} className="text-primary" />
          <h3 className="font-label text-xs uppercase tracking-widest text-on-surface-variant">
            Execution Flow
          </h3>
        </div>

        {isLoading ? (
          <SkeletonNodes />
        ) : (
          <div className="flex items-center gap-0 overflow-x-auto py-4 px-2 scrollbar-thin">
            {steps.map((step, i) => (
              <div key={step.name || i} className="contents">
                {i > 0 && <PipelineConnector targetStatus={step.status} />}
                <PipelineNode
                  step={step}
                  isSelected={selectedIdx === i}
                  onClick={() => setSelectedIdx(selectedIdx === i ? null : i)}
                />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Detail Panel ────────────────────────────────────────── */}
      <PipelineDetailPanel
        step={selectedIdx != null ? steps[selectedIdx] : null}
        onClose={() => setSelectedIdx(null)}
      />

      {/* ── Job Summary ─────────────────────────────────────────── */}
      {!isLoading && steps.length > 0 && (
        <div className="obsidian-glass rounded-xl p-6 mt-6 border border-outline-variant/10">
          <h3 className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant mb-4">
            Job Summary
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {/* File ID */}
            <div>
              <p className="text-[10px] text-on-surface-variant font-label mb-1">File ID</p>
              <p className="text-xs text-on-surface font-mono truncate">
                {job.file_id || job.fileId || '\u2014'}
              </p>
            </div>

            {/* Corrections Applied */}
            <div>
              <p className="text-[10px] text-on-surface-variant font-label mb-1">Corrections Applied</p>
              <p className="text-xs font-mono">
                {correctionsApplied != null ? (
                  <span className="text-secondary">{correctionsApplied}</span>
                ) : (
                  '\u2014'
                )}
                {correctionsAttempted != null && (
                  <span className="text-on-surface-variant"> / {correctionsAttempted}</span>
                )}
              </p>
            </div>

            {/* Processing Time */}
            <div>
              <p className="text-[10px] text-on-surface-variant font-label mb-1">Processing Time</p>
              <p className="text-xs text-primary font-mono">{formatDuration(totalDuration)}</p>
            </div>

            {/* Steps */}
            <div>
              <p className="text-[10px] text-on-surface-variant font-label mb-1">Steps</p>
              <p className="text-xs text-on-surface font-mono">
                {steps.filter((s) => s.status === 'completed').length} / {steps.length} completed
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
