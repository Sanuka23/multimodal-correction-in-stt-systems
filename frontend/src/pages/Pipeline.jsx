import { useState, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, Clock, AlertCircle, Activity } from 'lucide-react'
import { useJobSteps } from '../api/queries'
import PipelineFlow from '../components/pipeline/PipelineFlow'
import PipelineDetailPanel from '../components/pipeline/PipelineDetailPanel'
import StepTimingsPanel from '../components/pipeline/StepTimingsPanel'
import { canonicalSteps } from '../components/pipeline/stepUtils'
import { UMBRELLA_STEPS } from '../components/pipeline/PipelineNode'
import Badge from '../components/ui/Badge'

/* ── Helpers ──────────────────────────────────────────────────────── */

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

function FlowSkeleton() {
  return (
    <div
      className="relative w-full canvas-bg-grid rounded-2xl overflow-hidden"
      style={{ aspectRatio: '1400 / 580', minHeight: 380 }}
    >
      <div className="absolute inset-0 flex items-center justify-around">
        {Array.from({ length: 7 }).map((_, i) => (
          <div
            key={i}
            className="w-16 h-16 rounded-full bg-surface-container animate-pulse"
            style={{ animationDelay: `${i * 100}ms` }}
          />
        ))}
      </div>
    </div>
  )
}

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

/* ── Page ─────────────────────────────────────────────────────────── */

export default function Pipeline() {
  const { jobId } = useParams()
  const { data, isLoading, isError } = useJobSteps(jobId)
  const [selectedIdx, setSelectedIdx] = useState(null)

  const rawSteps = data?.pipeline_steps || data?.steps || (Array.isArray(data) ? data : [])
  // Single source of truth for ordered+deduped steps — both the canvas and
  // the timings panel use this so their indices line up.
  const steps = useMemo(() => canonicalSteps(rawSteps || []), [rawSteps])
  const job = data || {}

  const overallStatus = steps.some((s) => s.status === 'failed')
    ? 'failed'
    : steps.some((s) => s.status === 'running')
      ? 'running'
      : steps.length > 0 && steps.every((s) => s.status === 'completed')
        ? 'completed'
        : 'pending'

  // Exclude umbrella steps (ml_inference, complete) so the header doesn't
  // triple-count: the umbrella's wall-clock IS the sum of the child steps.
  const totalDuration = steps.reduce(
    (sum, s) => sum + (UMBRELLA_STEPS.has(s.name) ? 0 : (s.duration_ms || 0)),
    0,
  )
  const correctionsApplied = job.corrections_applied ?? job.applied ?? null
  const correctionsAttempted = job.corrections_attempted ?? job.attempted ?? null
  const completedCount = steps.filter((s) => s.status === 'completed').length
  const runningStep = steps.find((s) => s.status === 'running')

  if (!isLoading && (isError || (!data && !steps.length))) {
    return <ErrorState jobId={jobId} />
  }

  return (
    /* Full-bleed layout — escape the page padding container */
    <div className="-mx-6 -my-6">
      {/* ── Slim header row ───────────────────────────────────────── */}
      <div className="px-6 pt-6 pb-3 sticky top-0 z-20 bg-background/80 backdrop-blur-md border-b border-outline-variant/10">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-4 min-w-0">
            <Link
              to="/jobs"
              className="inline-flex items-center gap-1.5 text-on-surface-variant text-xs font-label hover:text-primary transition-colors flex-shrink-0"
            >
              <ArrowLeft size={14} /> Back
            </Link>

            <div className="h-6 w-px bg-outline-variant/30" />

            <div className="flex items-center gap-2 min-w-0">
              <Activity size={14} className="text-primary flex-shrink-0" />
              <p className="font-label text-[10px] text-primary uppercase tracking-[0.3em] flex-shrink-0">
                Pipeline Monitor
              </p>
              {runningStep && (
                <>
                  <span className="text-on-surface-variant/40 mx-1">/</span>
                  <span className="text-[11px] text-on-surface-variant truncate shimmer-text">
                    {runningStep.name}
                  </span>
                </>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3 flex-shrink-0">
            <span className="font-mono text-[11px] text-on-surface-variant bg-surface-container-high px-3 py-1.5 rounded-lg">
              {jobId ? `${jobId.slice(0, 14)}…` : '\u2014'}
            </span>
            <Badge color={STATUS_BADGE[overallStatus] || 'gray'}>
              {overallStatus}
            </Badge>
            <div className="flex items-center gap-1.5 text-on-surface-variant text-[11px]">
              <Clock size={13} />
              <span className="font-mono">{formatDuration(totalDuration)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Full-width flow canvas ─────────────────────────────────── */}
      <div className="px-2 md:px-6 py-4">
        {isLoading ? (
          <FlowSkeleton />
        ) : (
          <PipelineFlow
            steps={steps}
            selectedIdx={selectedIdx}
            onSelectStep={setSelectedIdx}
          />
        )}
      </div>

      {/* ── Pinned step detail (only when explicitly clicked) ───── */}
      {selectedIdx != null && (
        <div className="px-6 pb-2">
          <PipelineDetailPanel
            step={steps[selectedIdx]}
            onClose={() => setSelectedIdx(null)}
          />
        </div>
      )}

      {/* ── Step timings (where time was spent) ──────────────────── */}
      {!isLoading && steps.length > 0 && (
        <div className="px-6 pb-4">
          <StepTimingsPanel
            steps={steps}
            onJump={(idx) => setSelectedIdx(idx)}
          />
        </div>
      )}

      {/* ── Live job summary strip ─────────────────────────────────── */}
      {!isLoading && steps.length > 0 && (
        <div className="px-6 pb-8">
          <div className="obsidian-glass rounded-2xl p-5 border border-outline-variant/10">
            <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
              <div>
                <p className="text-[10px] text-on-surface-variant/70 font-label uppercase tracking-widest mb-1">File ID</p>
                <p className="text-xs text-on-surface font-mono truncate">
                  {job.file_id || job.fileId || '\u2014'}
                </p>
              </div>

              <div>
                <p className="text-[10px] text-on-surface-variant/70 font-label uppercase tracking-widest mb-1">Corrections</p>
                <p className="text-xs font-mono">
                  {correctionsApplied != null ? (
                    <span className="text-secondary">{correctionsApplied}</span>
                  ) : (
                    <span className="text-on-surface-variant/60">—</span>
                  )}
                  {correctionsAttempted != null && (
                    <span className="text-on-surface-variant"> / {correctionsAttempted}</span>
                  )}
                </p>
              </div>

              <div>
                <p className="text-[10px] text-on-surface-variant/70 font-label uppercase tracking-widest mb-1">Total Time</p>
                <p className="text-xs text-primary font-mono">{formatDuration(totalDuration)}</p>
              </div>

              <div>
                <p className="text-[10px] text-on-surface-variant/70 font-label uppercase tracking-widest mb-1">Steps</p>
                <p className="text-xs text-on-surface font-mono">
                  <span className="text-secondary">{completedCount}</span>
                  <span className="text-on-surface-variant"> / {steps.length} done</span>
                </p>
              </div>

              <div>
                <p className="text-[10px] text-on-surface-variant/70 font-label uppercase tracking-widest mb-1">Active</p>
                <p className="text-xs font-mono">
                  {runningStep ? (
                    <span className="text-tertiary shimmer-text">{runningStep.name}</span>
                  ) : overallStatus === 'completed' ? (
                    <span className="text-secondary">idle</span>
                  ) : (
                    <span className="text-on-surface-variant/60">—</span>
                  )}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
