import { useEffect, useState } from 'react'
import { Sliders, Check, Loader2, AlertCircle } from 'lucide-react'
import api from '../../api/client'
import SectionCard from '../ui/SectionCard'

const STEP_TOGGLES = [
  {
    key: 'enable_topic_classification',
    label: 'Topic Classification',
    description: 'LLM classifies meeting field, topic, and suggests domain vocab.',
  },
  {
    key: 'enable_web_vocab_enrichment',
    label: 'Web Vocab Enrichment',
    description: 'DuckDuckGo searches the topic; LLM extracts a fresh glossary.',
  },
  {
    key: 'enable_candidate_validation',
    label: 'Candidate Validation',
    description: 'Cross-chunk pooling + per-candidate web search to pick best target.',
  },
  {
    key: 'enable_ocr_extraction',
    label: 'Screen OCR',
    description: 'PaddleOCR reads raw text from sampled video frames.',
  },
  {
    key: 'enable_ocr_vocab_extraction',
    label: 'OCR Term Mining',
    description: 'LLM mines person / product / company names from the OCR text.',
  },
  {
    key: 'enable_whisper_pass2',
    label: 'Whisper Pass 2',
    description: 'Re-transcribes flagged segments with a vocab-biased Whisper.',
  },
  {
    key: 'enable_avsr',
    label: 'AVSR (Lip Reading)',
    description: 'Lip-reading hints from MediaPipe / Auto-AVSR on flagged segments.',
  },
  {
    key: 'enable_data_collection',
    label: 'Training-Data Collect',
    description: 'Persists applied corrections as JSONL training pairs.',
  },
]

function Toggle({ checked, onChange, disabled }) {
  return (
    <button
      type="button"
      onClick={() => !disabled && onChange(!checked)}
      disabled={disabled}
      className={`
        relative inline-flex h-[22px] w-[42px] items-center rounded-full transition-colors flex-shrink-0
        ${checked ? 'bg-primary' : 'bg-surface-container-high'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
      `}
      aria-checked={checked}
      role="switch"
    >
      <span
        className={`
          inline-block h-[16px] w-[16px] rounded-full bg-on-primary transition-transform
          ${checked ? 'translate-x-[22px]' : 'translate-x-[3px]'}
        `}
      />
    </button>
  )
}

export default function PipelineStepsSettings() {
  const [settings, setSettings] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)
  const [savedFlash, setSavedFlash] = useState(false)

  useEffect(() => {
    let mounted = true
    api.get('/api/pipeline/settings')
      .then((r) => { if (mounted) setSettings(r.data) })
      .catch((e) => { if (mounted) setError(e?.message || 'Failed to load') })
      .finally(() => { if (mounted) setLoading(false) })
    return () => { mounted = false }
  }, [])

  const update = async (patch) => {
    setSaving(true)
    setError(null)
    try {
      const res = await api.post('/api/pipeline/settings', patch)
      setSettings(res.data)
      setSavedFlash(true)
      window.setTimeout(() => setSavedFlash(false), 1500)
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  const enabledCount = settings
    ? STEP_TOGGLES.filter(t => settings[t.key]).length
    : 0

  return (
    <SectionCard
      eyebrow="Customize"
      title="Pipeline Steps"
      icon={Sliders}
      actions={
        <div className="flex items-center gap-2 text-[10px] font-label uppercase tracking-widest">
          {saving ? (
            <span className="flex items-center gap-1 text-primary">
              <Loader2 size={11} className="animate-spin" /> Saving
            </span>
          ) : savedFlash ? (
            <span className="flex items-center gap-1 text-secondary">
              <Check size={11} /> Saved
            </span>
          ) : settings ? (
            <span className="text-on-surface-variant/70">
              <span className="text-secondary">{enabledCount}</span>
              <span className="text-on-surface-variant/40"> / {STEP_TOGGLES.length} active</span>
            </span>
          ) : null}
        </div>
      }
    >
      {loading && (
        <div className="text-xs text-on-surface-variant/60">Loading settings…</div>
      )}

      {error && (
        <div className="flex items-start gap-2 text-xs text-error mb-3 bg-error/10 border border-error/20 rounded-lg p-2.5">
          <AlertCircle size={13} className="mt-px" />
          <span>{error}</span>
        </div>
      )}

      {settings && (
        <div className="space-y-1.5">
          {STEP_TOGGLES.map((t) => {
            const enabled = !!settings[t.key]
            return (
              <div
                key={t.key}
                className={`
                  flex items-center gap-3 rounded-xl px-3 py-2.5 transition-colors
                  ${enabled
                    ? 'bg-surface-container-high/50 border border-outline-variant/15'
                    : 'bg-surface-container-low border border-outline-variant/10 opacity-75'}
                `}
              >
                <div className="flex-1 min-w-0">
                  <p className={`text-[13px] font-medium leading-tight ${enabled ? 'text-on-surface' : 'text-on-surface-variant'}`}>
                    {t.label}
                  </p>
                  <p className="text-[10.5px] text-on-surface-variant/70 mt-0.5 leading-tight">
                    {t.description}
                  </p>
                </div>
                <Toggle
                  checked={enabled}
                  onChange={(v) => update({ [t.key]: v })}
                  disabled={saving}
                />
              </div>
            )
          })}

          {/* AVSR sub-settings */}
          <div className="mt-3 pt-3 border-t border-outline-variant/10">
            <p className="text-[10px] font-label uppercase tracking-[0.22em] text-on-surface-variant/70 mb-2">
              AVSR Options
            </p>
            <div className="space-y-1.5">
              <div className="flex items-center gap-3 bg-surface-container-low rounded-xl px-3 py-2.5 border border-outline-variant/10">
                <div className="flex-1">
                  <p className="text-[12px] font-medium text-on-surface">Run on all flagged segments</p>
                  <p className="text-[10.5px] text-on-surface-variant/70 mt-0.5">
                    Bypass the person_name / content_word category gate — analyse every flagged segment.
                  </p>
                </div>
                <Toggle
                  checked={!!settings.avsr_run_on_all_flagged}
                  onChange={(v) => update({ avsr_run_on_all_flagged: v })}
                  disabled={saving || !settings.enable_avsr}
                />
              </div>

              <div className="flex items-center justify-between gap-3 bg-surface-container-low rounded-xl px-3 py-2.5 border border-outline-variant/10">
                <div>
                  <p className="text-[12px] font-medium text-on-surface">AVSR Mode</p>
                  <p className="text-[10.5px] text-on-surface-variant/70 mt-0.5">
                    `mediapipe` is lightweight (default). `auto_avsr` requires downloaded weights.
                  </p>
                </div>
                <select
                  value={settings.avsr_mode || 'mediapipe'}
                  onChange={(e) => update({ avsr_mode: e.target.value })}
                  disabled={saving || !settings.enable_avsr}
                  className="bg-surface-container-high text-[12px] text-on-surface px-3 py-1.5 rounded-lg border border-outline-variant/15 focus:outline-none focus:ring-2 focus:ring-primary/30 disabled:opacity-50"
                >
                  <option value="none">none</option>
                  <option value="mediapipe">mediapipe</option>
                  <option value="auto_avsr">auto_avsr</option>
                </select>
              </div>

              <div className="flex items-center gap-3 bg-surface-container-low rounded-xl px-3 py-2.5 border border-outline-variant/10">
                <div className="flex-1">
                  <p className="text-[12px] font-medium text-on-surface">
                    Min speaking confidence
                    <span className="ml-2 font-mono text-primary">
                      {Number(settings.avsr_min_speaking_confidence ?? 0.55).toFixed(2)}
                    </span>
                  </p>
                  <p className="text-[10.5px] text-on-surface-variant/70 mt-0.5">
                    Drop AVSR hints below this MediaPipe speaking confidence.
                  </p>
                </div>
                <input
                  type="range"
                  min="0.3"
                  max="0.9"
                  step="0.05"
                  value={settings.avsr_min_speaking_confidence ?? 0.55}
                  onChange={(e) => update({ avsr_min_speaking_confidence: Number(e.target.value) })}
                  disabled={saving || !settings.enable_avsr}
                  className="w-32 accent-primary"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </SectionCard>
  )
}
